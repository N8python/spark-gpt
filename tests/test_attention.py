import unittest

import torch

import train


def tiny_attention_config(**overrides):
    values = dict(
        vocab_size=train.VOCAB_SIZE,
        hidden_size=256,
        num_hidden_layers=1,
        intermediate_size=768,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=512,
    )
    values.update(overrides)
    return train.ModelConfig(**values)


class FusedQKNormRoPETest(unittest.TestCase):
    def test_fused_qk_rope_matches_unfused_attention(self):
        if train.triton is None or not torch.cuda.is_available():
            self.skipTest("CUDA + triton are required")
        torch.manual_seed(0)
        config = tiny_attention_config()
        model = train.ByteLM(config).cuda()
        attn = model.layers[0].self_attn
        with torch.no_grad():  # non-trivial norm gains so their gradients matter
            attn.q_norm.weight.uniform_(0.5, 1.5)
            attn.k_norm.weight.uniform_(0.5, 1.5)
        # packed varlen window: documents plus a filler segment, positions restart per segment
        seg_lens = [300, 1, 457, 129, 113]
        total = sum(seg_lens)
        cu = torch.tensor([0] + torch.tensor(seg_lens).cumsum(0).tolist(),
                          device="cuda", dtype=torch.int32)
        pos = torch.cat([torch.arange(n) for n in seg_lens]).cuda()
        max_seqlen = max(seg_lens)
        qkv_dim = (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim
        qkv = (torch.randn(total, qkv_dim, device="cuda") * 2.0).to(torch.bfloat16)
        probe = torch.randn(total, config.num_attention_heads, config.head_dim, device="cuda")

        def unfused(qkv_, qw, kw, dtype):
            q, k, v = qkv_.split([4 * 128, 2 * 128, 2 * 128], dim=-1)
            cos = model.cos_cached[pos].to(dtype)
            sin = model.sin_cached[pos].to(dtype)

            def norm(x, w):
                xf = x.float()
                out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)
                return out.to(dtype) * w.to(dtype)

            q = attn._rope(norm(q.view(total, 4, 128), qw), cos, sin)
            k = attn._rope(norm(k.view(total, 2, 128), kw), cos, sin)
            v = v.reshape(total, 2, 128)
            if dtype == torch.float32:  # exact fp32 causal block-diagonal reference
                out = torch.empty(total, 4, 128, device="cuda")
                for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
                    qs, ks, vs = q[a:b], k[a:b].repeat_interleave(2, 1), v[a:b].repeat_interleave(2, 1)
                    s = torch.einsum("qhd,khd->hqk", qs, ks) * 128 ** -0.5
                    s = s.masked_fill(torch.ones(b - a, b - a, device="cuda").triu(1).bool(), float("-inf"))
                    out[a:b] = torch.einsum("hqk,khd->qhd", s.softmax(-1), vs)
                return out
            return train._varlen_attention(q, k, v, cu, max_seqlen)

        def grads(fn, dtype):
            x = qkv.detach().to(dtype).requires_grad_(True)
            qw = attn.q_norm.weight.detach().clone().requires_grad_(True)
            kw = attn.k_norm.weight.detach().clone().requires_grad_(True)
            out = fn(x, qw, kw)
            g = torch.autograd.grad((out.float() * probe).sum(), (x, qw, kw))
            return (out.float(),) + tuple(t.float() for t in g)

        ref = grads(lambda x, qw, kw: unfused(x, qw, kw, torch.float32), torch.float32)
        base = grads(lambda x, qw, kw: unfused(x, qw, kw, torch.bfloat16), torch.bfloat16)
        fused = grads(lambda x, qw, kw: train._QKNormRoPEAttention.apply(
            x, qw, kw, model.cos_cached, model.sin_cached, pos, cu, max_seqlen,
            4, 2, config.rms_norm_eps), torch.bfloat16)
        for name, f, u, r in zip(("out", "dqkv", "dq_norm", "dk_norm"), fused, base, ref):
            err_f = (f - r).abs().max().item()
            err_u = (u - r).abs().max().item()
            scale = r.abs().max().item()
            self.assertLess(err_f, 0.05 * scale + 1e-3, f"{name}: fused err {err_f} (scale {scale})")
            self.assertLess(err_f, 1.5 * err_u + 0.01 * scale + 1e-3,
                            f"{name}: fused err {err_f} vs unfused {err_u}")


if __name__ == "__main__":
    unittest.main()
