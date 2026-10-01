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


if __name__ == "__main__":
    unittest.main()
