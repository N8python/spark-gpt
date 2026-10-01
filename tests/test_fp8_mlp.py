import unittest

import torch
import torch.nn.functional as F

import train


class FP8MLPBlockTest(unittest.TestCase):
    def test_fp8_mlp_block_tracks_fp32_reference(self):
        if train.triton is None or not torch.cuda.is_available():
            self.skipTest("CUDA + triton are required")
        torch.manual_seed(0)
        config = train.ModelConfig(
            vocab_size=train.VOCAB_SIZE, hidden_size=256, num_hidden_layers=1,
            intermediate_size=768, num_attention_heads=2, num_key_value_heads=1,
            head_dim=128, max_position_embeddings=512,
        )
        block = train.Block(config, 0).cuda().train()
        with torch.no_grad():
            block.post_attention_layernorm.weight.uniform_(0.5, 1.5)
            block.mlp.gate_up_proj.weight.normal_(0, 0.05)
            block.mlp.down_proj.weight.normal_(0, 0.05)
        rows = 1000  # not a multiple of any tile size
        h = torch.randn(rows, 256, device="cuda").to(torch.bfloat16)
        probe = torch.randn(rows, 256, device="cuda")
        params = [block.post_attention_layernorm.weight, block.mlp.gate_up_proj.weight,
                  block.mlp.down_proj.weight]

        def fp8_step():
            x = h.clone().requires_grad_(True)
            y = train._FP8MLPBlock.apply(x, *params, block.mlp.fp8_scale, block.mlp.fp8_amax,
                                         block.post_attention_layernorm.eps)
            return (y,) + torch.autograd.grad((y.float() * probe).sum(), [x] + params)

        fp8_step()  # prewarm: records amax under the initial scales
        self.assertTrue(bool((block.mlp.fp8_amax > 0).all()), block.mlp.fp8_amax)
        train.fp8_update_scales(block)
        self.assertTrue(bool((block.mlp.fp8_amax == 0).all()))
        out = fp8_step()

        x32 = h.float().requires_grad_(True)
        p32 = [p.detach().float().requires_grad_(True) for p in params]
        n = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6) * p32[0]
        g, u = (n @ p32[1].t()).chunk(2, dim=-1)
        y32 = x32 + (F.silu(g) * u) @ p32[2].t()
        ref = (y32,) + torch.autograd.grad((y32 * probe).sum(), [x32] + p32)
        for name, a, r in zip(("mlp out", "dh", "dnorm_w", "dW_gate_up", "dW_down"), out, ref):
            if name == "mlp out":  # compare the MLP contribution, not the residual
                a, r = a.float() - h.float(), r - h.float()
            err = ((a.float() - r).norm() / r.norm()).item()
            self.assertTrue(torch.isfinite(a).all(), name)
            self.assertLess(err, 0.12, f"{name}: rel err {err:.3e}")


class FP8ExpertsTest(unittest.TestCase):
    def test_fp8_experts_track_fp32_reference(self):
        if train.triton is None or not torch.cuda.is_available():
            self.skipTest("CUDA + triton are required")
        torch.manual_seed(0)
        config = train.ModelConfig(
            vocab_size=train.VOCAB_SIZE, hidden_size=256, num_hidden_layers=1,
            intermediate_size=768, num_attention_heads=2, num_key_value_heads=1,
            head_dim=128, max_position_embeddings=512, num_experts=4,
            num_experts_per_tok=2, moe_intermediate_size=128,
        )
        experts = train.Experts(config).cuda().train()
        with torch.no_grad():
            experts.gate_up_proj.normal_(0, 0.05)
            experts.down_proj.normal_(0, 0.05)
        T, k = 500, 2
        x = torch.randn(T, 256, device="cuda").to(torch.bfloat16)
        # expert 2 gets no tokens; each token picks two distinct experts from {0, 1, 3}
        choices = torch.tensor([[0, 1], [1, 3], [0, 3]], device="cuda")
        selected = choices[torch.randint(0, 3, (T,), device="cuda")]
        weights = torch.softmax(torch.randn(T, k, device="cuda"), -1).to(torch.bfloat16)
        probe = torch.randn(T, 256, device="cuda")
        saved = train.FP8_MLP
        train.FP8_MLP = True
        try:
            def fp8_step():
                xx = x.clone().requires_grad_(True)
                out = experts(xx, selected, weights)
                g = torch.autograd.grad((out.float() * probe).sum(),
                                        [xx, experts.gate_up_proj, experts.down_proj])
                return (out,) + g
            fp8_step()
            self.assertTrue(bool((experts.fp8_amax > 0).all()), experts.fp8_amax)
            train.fp8_update_scales(experts)
            out = fp8_step()
        finally:
            train.FP8_MLP = saved

        x32 = x.float().requires_grad_(True)
        wgu = experts.gate_up_proj.detach().float().requires_grad_(True)
        wd = experts.down_proj.detach().float().requires_grad_(True)
        y32 = torch.zeros(T, 256, device="cuda")
        for s in range(k):
            e = selected[:, s]
            g, u = torch.einsum("td,tod->to", x32, wgu[e]).chunk(2, dim=-1)
            y32 = y32 + weights[:, s, None].float() * torch.einsum(
                "ti,tdi->td", F.silu(g) * u, wd[e])
        ref = (y32,) + torch.autograd.grad((y32 * probe).sum(), [x32, wgu, wd])
        for name, a_, r in zip(("out", "dx", "dW_gate_up", "dW_down"), out, ref):
            err = ((a_.float() - r).norm() / r.norm()).item()
            self.assertTrue(torch.isfinite(a_).all(), name)
            self.assertLess(err, 0.12, f"{name}: rel err {err:.3e}")
        torch.testing.assert_close(out[2][2], torch.zeros_like(out[2][2]))  # unused expert
        torch.testing.assert_close(out[3][2], torch.zeros_like(out[3][2]))


class FP8GuardTest(unittest.TestCase):
    def test_moe_model_trains_with_and_without_fp8_flag(self):
        # regression: the fp8 guard must not touch sparse blocks' (absent) down_proj
        if train.triton is None or not torch.cuda.is_available():
            self.skipTest("CUDA + triton are required")
        torch.manual_seed(0)
        config = train.ModelConfig(
            vocab_size=train.VOCAB_SIZE, hidden_size=256, num_hidden_layers=2,
            intermediate_size=768, num_attention_heads=2, num_key_value_heads=1,
            head_dim=128, max_position_embeddings=512, num_experts=4,
            num_experts_per_tok=2, moe_intermediate_size=128, mlp_only_layers=(0,),
        )
        model = train.ByteLM(config).cuda().train()
        T = 256
        ids = torch.randint(0, 256, (T,), device="cuda")
        cu = torch.tensor([0, 100, T], device="cuda", dtype=torch.int32)
        pos = torch.cat([torch.arange(100), torch.arange(T - 100)]).cuda()
        saved = train.FP8_MLP
        try:
            for flag in (False, True):
                train.FP8_MLP = flag
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, router_logits, _ = model(ids, pos, cu, 156, output_router_logits=True)
                logits.float().sum().backward()
                self.assertEqual(len(router_logits), 1)
                self.assertTrue(torch.isfinite(logits).all())
                model.zero_grad(set_to_none=True)
        finally:
            train.FP8_MLP = saved


if __name__ == "__main__":
    unittest.main()
