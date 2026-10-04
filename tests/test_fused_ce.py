import unittest

import torch
import torch.nn.functional as F

import train


class FusedLinearCrossEntropyTest(unittest.TestCase):
    def setUp(self):
        if train.triton is None or not torch.cuda.is_available():
            self.skipTest("CUDA + triton are required")
        self.saved = (train.FUSED_CE, train.FUSED_CE_CHUNK)
        train.FUSED_CE_CHUNK = 1000  # T below is not a multiple of the chunk

    def tearDown(self):
        train.FUSED_CE, train.FUSED_CE_CHUNK = self.saved

    def _inputs(self, T=2500, D=256, V=5003):  # V not a multiple of the 4096 block
        torch.manual_seed(0)
        h = (torch.randn(T, D, device="cuda") * 2).to(torch.bfloat16)
        w = torch.randn(V, D, device="cuda") * D ** -0.5 * 3
        tgt = torch.randint(0, V, (T,), device="cuda")
        tgt[torch.rand(T, device="cuda") < 0.2] = train.LOSS_IGNORE_INDEX  # PAD filler
        return h, w, tgt

    def _grads(self, fused, h, w, tgt, fp32=False):
        train.FUSED_CE = fused
        hh = h.float().requires_grad_(True) if fp32 else h.clone().requires_grad_(True)
        ww = w.clone().requires_grad_(True)
        if fp32:
            loss = F.cross_entropy(hh @ ww.t(), tgt, ignore_index=train.LOSS_IGNORE_INDEX,
                                   reduction="sum")
        else:
            loss = train.lm_head_loss(hh, ww, tgt)
        dh, dw = torch.autograd.grad(loss * 0.37, [hh, ww])  # non-unit upstream gradient
        return loss.float(), dh.float(), dw.float()

    def test_matches_unfused_and_tracks_fp32(self):
        h, w, tgt = self._inputs()
        ref = self._grads(False, h, w, tgt, fp32=True)
        base = self._grads(False, h, w, tgt)
        fused = self._grads(True, h, w, tgt)
        for name, f, b, r in zip(("loss", "dh", "dW"), fused, base, ref):
            err_f = ((f - r).norm() / r.norm()).item()
            err_b = ((b - r).norm() / r.norm()).item()
            self.assertLess(err_f, 1.5 * err_b + 1e-4, f"{name}: fused {err_f:.2e} vs unfused {err_b:.2e}")
        torch.testing.assert_close(fused[0], base[0], rtol=1e-4, atol=0.0)  # same bf16 logits

    def test_vocab_smaller_than_one_block(self):
        # the byte vocabulary (259) leaves most lanes of the 4096-wide block empty
        h, w, tgt = self._inputs(V=train.VOCAB_SIZE)
        ref = self._grads(False, h, w, tgt, fp32=True)
        fused = self._grads(True, h, w, tgt)
        for name, f, r in zip(("loss", "dh", "dW"), fused, ref):
            self.assertTrue(torch.isfinite(f).all(), name)
            self.assertLess(((f - r).norm() / r.norm()).item(), 0.02, name)

    def test_no_grad_path_and_ignored_rows(self):
        h, w, tgt = self._inputs()
        train.FUSED_CE = True
        with torch.no_grad():
            loss = train.lm_head_loss(h, w, tgt)
        train.FUSED_CE = False
        with torch.no_grad():
            base = train.lm_head_loss(h, w, tgt)
        torch.testing.assert_close(loss, base, rtol=1e-4, atol=0.0)
        train.FUSED_CE = True
        all_ignored = torch.full_like(tgt, train.LOSS_IGNORE_INDEX)
        hh = h.clone().requires_grad_(True)
        loss = train.lm_head_loss(hh, w, all_ignored)
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(hh.grad.abs().max().item(), 0.0)

    def test_model_forward_with_targets_is_the_summed_loss(self):
        config = train.ModelConfig(vocab_size=5003, hidden_size=256, num_hidden_layers=1,
                                   intermediate_size=512, num_attention_heads=2,
                                   num_key_value_heads=1, head_dim=128)
        model = train.ByteLM(config).cuda()
        T = 1500
        ids = torch.randint(0, 5003, (T,), device="cuda")
        tgt = torch.randint(0, 5003, (T,), device="cuda")
        cu = torch.tensor([0, 700, T], device="cuda", dtype=torch.int32)
        pos = torch.cat([torch.arange(700), torch.arange(T - 700)]).cuda()
        train.FUSED_CE = True
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(ids, pos, cu, 800, targets=tgt)
            logits = model(ids, pos, cu, 800)
        torch.testing.assert_close(
            loss, F.cross_entropy(logits.float(), tgt, reduction="sum"), rtol=1e-4, atol=0.0)


if __name__ == "__main__":
    unittest.main()
