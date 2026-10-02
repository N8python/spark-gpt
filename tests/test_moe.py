import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

import train


def tiny_config(**overrides):
    values = dict(
        vocab_size=train.VOCAB_SIZE,
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=384,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        mlp_only_layers=(0,),
        norm_topk_prob=True,
    )
    values.update(overrides)
    return train.ModelConfig(**values)


class RouterTest(unittest.TestCase):
    def test_router_matches_qwen3_topk_math(self):
        config = tiny_config(hidden_size=3, num_hidden_layers=1,
                             mlp_only_layers=(), num_attention_heads=1,
                             num_key_value_heads=1, head_dim=3)
        router = train.TopKRouter(config)
        with torch.no_grad():
            router.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, -1.0, -1.0],
            ]))
        hidden = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
        logits, weights, selected = router(hidden)
        expected_logits = F.linear(hidden, router.weight)
        expected_probs = F.softmax(expected_logits, dtype=torch.float32, dim=-1)
        expected_weights, expected_selected = expected_probs.topk(2, dim=-1)
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(logits, expected_logits)
        torch.testing.assert_close(selected, expected_selected)
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))

    def test_expert_dispatch_matches_direct_reference(self):
        torch.manual_seed(0)
        config = tiny_config(hidden_size=4, num_hidden_layers=1,
                             intermediate_size=12, num_attention_heads=1,
                             num_key_value_heads=1, head_dim=4,
                             num_experts=3, moe_intermediate_size=3,
                             mlp_only_layers=())
        experts = train.Experts(config)
        with torch.no_grad():
            experts.gate_up_proj.normal_(0.0, 0.2)
            experts.down_proj.normal_(0.0, 0.2)
        hidden = torch.randn(5, 4, requires_grad=True)
        selected = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2], [1, 0]])
        weights = torch.tensor([
            [0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.55, 0.45], [0.9, 0.1]
        ])
        expected = torch.zeros_like(hidden)
        for token in range(hidden.shape[0]):
            for slot in range(selected.shape[1]):
                expert = int(selected[token, slot])
                gate, up = F.linear(
                    hidden[token], experts.gate_up_proj[expert]
                ).chunk(2, dim=-1)
                expert_output = F.linear(
                    F.silu(gate) * up, experts.down_proj[expert]
                )
                expected[token] += weights[token, slot] * expert_output
        actual = experts(hidden, selected, weights)
        torch.testing.assert_close(actual, expected)

        probe = torch.randn_like(actual)
        differentiable_inputs = (
            hidden, experts.gate_up_proj, experts.down_proj,
        )
        actual_grads = torch.autograd.grad(
            (actual * probe).sum(), differentiable_inputs, retain_graph=True
        )
        expected_grads = torch.autograd.grad(
            (expected * probe).sum(), differentiable_inputs
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)


class GatherTopKTest(unittest.TestCase):
    def test_gather_backward_matches_autograd_index(self):
        torch.manual_seed(0)
        num_tokens, top_k, dim, num_experts = 37, 2, 16, 5
        selected = torch.randint(0, num_experts, (num_tokens, top_k))
        _, permutation = torch.sort(selected.reshape(-1))
        token_index = permutation // top_k
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())
        hidden = torch.randn(num_tokens, dim, requires_grad=True)
        probe = torch.randn(num_tokens * top_k, dim)

        actual = train._GatherTopK.apply(hidden, token_index, inverse, top_k)
        expected = hidden[token_index]
        torch.testing.assert_close(actual, expected)
        actual_grad, = torch.autograd.grad((actual * probe).sum(), hidden)
        expected_grad, = torch.autograd.grad((expected * probe).sum(), hidden)
        torch.testing.assert_close(actual_grad, expected_grad)


class CombineTopKTest(unittest.TestCase):
    def test_combine_matches_scatter_reference(self):
        torch.manual_seed(0)
        num_tokens, top_k, dim, num_experts = 37, 2, 16, 5
        selected = torch.randint(0, num_experts, (num_tokens, top_k))
        _, permutation = torch.sort(selected.reshape(-1))
        token_index = permutation // top_k
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())
        rows = torch.randn(num_tokens * top_k, dim, requires_grad=True)
        weights = torch.rand(num_tokens * top_k, requires_grad=True)
        probe = torch.randn(num_tokens, dim)

        actual = train._CombineTopK.apply(rows, weights, inverse, token_index, top_k)
        ordered = torch.empty_like(rows)
        ordered[permutation] = rows * weights.unsqueeze(-1)
        expected = ordered.view(num_tokens, top_k, dim).sum(dim=1)
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad((actual * probe).sum(), (rows, weights))
        expected_grads = torch.autograd.grad((expected * probe).sum(), (rows, weights))
        for a, e in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, e)


class AuxiliaryLossTest(unittest.TestCase):
    def test_masked_loss_matches_formula_and_ignores_filler(self):
        logits = torch.tensor([
            [3.0, 1.0, 0.0, -1.0],
            [0.0, 2.0, 1.0, -2.0],
            [1.0, 0.0, 3.0, -1.0],
            [99.0, -99.0, -99.0, -99.0],
        ], requires_grad=True)
        selected = logits.detach().softmax(dim=-1).topk(2, dim=-1).indices
        valid = torch.tensor([True, True, True, False])
        loss, stats = train.global_load_balancing_loss(
            (logits,), (selected,), valid, num_experts=4, top_k=2
        )

        probs = logits[:3].float().softmax(dim=-1)
        counts = torch.bincount(selected[:3].reshape(-1), minlength=4).float()
        expected = 4 * torch.dot(counts / 3, probs.mean(dim=0))
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(stats["aux_loss"], expected.detach())

        base_loss, _ = train.global_load_balancing_loss(
            (logits[:3],), (selected[:3],), torch.ones(3, dtype=torch.bool),
            num_experts=4, top_k=2,
        )
        torch.testing.assert_close(loss, base_loss)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[-1], torch.zeros(4))

    def test_all_filler_is_finite_zero(self):
        logits = torch.randn(5, 4, requires_grad=True)
        selected = logits.detach().topk(2, dim=-1).indices
        loss, stats = train.global_load_balancing_loss(
            (logits,), (selected,), torch.zeros(5, dtype=torch.bool),
            num_experts=4, top_k=2,
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(stats["aux_loss"]), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


class WandbMetricTest(unittest.TestCase):
    @staticmethod
    def record(**overrides):
        values = {
            "step": 7,
            "loss": 1.25,
            "global_total_tokens": 1234,
            "real_tokens_per_second": 10.0,
            "window_tokens_per_second": 11.0,
            "observed_packing_utilization": 0.9,
            "lr": 1e-3,
            "peak_cuda_memory_gb": 2.0,
        }
        values.update(overrides)
        return values

    def test_dense_train_loss_is_lm_loss(self):
        metrics = train.wandb_train_metrics(self.record())
        self.assertEqual(metrics["train/loss"], 1.25)
        self.assertNotIn("train/aux_loss", metrics)
        self.assertNotIn("router/aux_loss", metrics)

    def test_moe_train_loss_and_balancing_loss_share_one_row(self):
        metrics = train.wandb_train_metrics(self.record(
            router_aux_loss=2.75,
            router_assignment_min_frac=0.1,
            router_assignment_max_frac=0.2,
            router_assignment_cv=0.3,
            router_entropy=1.5,
            router_unused_experts=0,
        ))
        self.assertEqual(metrics["train/loss"], 1.25)
        self.assertEqual(metrics["train/aux_loss"], 2.75)
        self.assertEqual(metrics["router/aux_loss"], 2.75)
        self.assertEqual(metrics["train/step"], 7)
        self.assertEqual(metrics["train/tokens"], 1234)


class InitializationOptimizerExportTest(unittest.TestCase):
    def test_experts_use_muon_and_router_uses_no_decay_adamw(self):
        model = train.ByteLM(tiny_config())
        train.apply_mup_init(
            model, base_dim=128, emb_std=0.02, hidden_std=0.02,
            router_std=0.02,
        )
        args = argparse.Namespace(
            mup_base_dim=128,
            model_dim=128,
            muon_lr=4e-3,
            muon_momentum=0.95,
            adamw_lr=5e-4,
            router_lr=None,
            weight_decay=0.01,
        )
        optimizer, summary = train.build_optimizer(model, args)
        expert_ids = {
            id(p) for name, p in model.named_parameters() if ".experts." in name
        }
        router_ids = {
            id(p) for name, p in model.named_parameters()
            if name.endswith("mlp.gate.weight")
        }
        self.assertTrue(expert_ids)
        self.assertTrue(router_ids)
        self.assertTrue(expert_ids <= {id(p) for p in optimizer.param_groups[0]["params"]})
        self.assertTrue(router_ids <= {id(p) for p in optimizer.param_groups[3]["params"]})
        self.assertEqual(optimizer.param_groups[3]["weight_decay"], 0.0)
        self.assertEqual(summary["router_param_count"], 4 * 128)

        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()  # exercises batched Newton-Schulz on 3-D expert tensors
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_hf_export_strictly_loads_as_qwen3_moe(self):
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            self.skipTest("transformers is optional")

        model = train.ByteLM(tiny_config())
        train.apply_mup_init(
            model, base_dim=128, emb_std=0.02, hidden_std=0.02,
            router_std=0.02,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            train.export_hf(model, out_dir)
            config_json = json.loads((out_dir / "config.json").read_text())
            self.assertEqual(config_json["model_type"], "qwen3_moe")
            self.assertEqual(config_json["architectures"], ["Qwen3MoeForCausalLM"])
            self.assertEqual(config_json["mlp_only_layers"], [0])
            loaded = AutoModelForCausalLM.from_pretrained(out_dir)
            self.assertEqual(loaded.__class__.__name__, "Qwen3MoeForCausalLM")

            exported = train.export_unfused_state_dict(model)
            expected = {
                (key if key == "lm_head.weight" else "model." + key): value
                for key, value in exported.items()
            }
            actual = loaded.state_dict()
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key])


@unittest.skipUnless(
    torch.cuda.is_available() and hasattr(F, "grouped_mm"),
    "CUDA grouped_mm is required",
)
class CudaGroupedMMTest(unittest.TestCase):
    def test_buffered_newton_schulz_matches_reference(self):
        def reference(x, steps=5):
            a, b, c = (3.4445, -4.7750, 2.0315)
            transpose_needed = x.shape[-2] > x.shape[-1]
            if transpose_needed:
                x = x.mT
            x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
            for _ in range(steps):
                gram = x @ x.mT
                polynomial = torch.baddbmm(
                    gram, gram, gram, beta=b, alpha=c
                )
                x = torch.baddbmm(
                    x, polynomial, x, beta=a, alpha=1.0
                )
            return x.mT if transpose_needed else x

        torch.manual_seed(0)
        for rows, columns in ((1536, 512), (512, 768)):
            with self.subTest(rows=rows, columns=columns):
                value = torch.randn(
                    2, rows, columns, device="cuda", dtype=torch.bfloat16
                )
                expected = reference(value.clone())
                actual = train._newtonschulz5_batched(value.clone())
                torch.testing.assert_close(
                    actual, expected, rtol=0.0, atol=0.0
                )

    def test_muon_update_fused_into_newton_schulz_matches_foreach_add(self):
        torch.manual_seed(0)
        shapes = ((1536, 512), (512, 768), (2, 256, 512))  # transposed, plain, 3-D experts

        def run(fused):
            torch.manual_seed(1)
            params = [torch.nn.Parameter(torch.randn(*s_, device="cuda")) for s_ in shapes]
            for p_ in params:
                p_.grad = torch.randn_like(p_)
            opt = train.MuonAdamWHybrid([{"params": params, "use_muon": True, "lr": 0.02}])
            orig = train._newtonschulz5_batched
            if not fused:
                train._newtonschulz5_batched = lambda x, steps=5, update=None: orig(x, steps)
            try:
                for _ in range(2):
                    opt.step()
            finally:
                train._newtonschulz5_batched = orig
            return [p_.detach() for p_ in params]

        for f, r in zip(run(True), run(False)):
            torch.testing.assert_close(f, r, rtol=0.0, atol=0.0)

    def test_output_scatter_matches_inverse_gather(self):
        torch.manual_seed(0)
        num_tokens, top_k, hidden_dim = 256, 2, 128
        selected = torch.randint(0, 8, (num_tokens, top_k), device="cuda")
        _, permutation = torch.sort(selected.reshape(-1))
        expert_output = torch.randn(
            num_tokens * top_k, hidden_dim,
            device="cuda", dtype=torch.bfloat16, requires_grad=True,
        )

        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(
            permutation.numel(), device="cuda"
        )
        expected = expert_output[inverse_permutation].view(
            num_tokens, top_k, hidden_dim
        ).sum(dim=1)
        ordered_output = torch.empty_like(expert_output)
        ordered_output[permutation] = expert_output
        actual = ordered_output.view(
            num_tokens, top_k, hidden_dim
        ).sum(dim=1)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        probe = torch.randn_like(actual)
        actual_grad, = torch.autograd.grad(
            (actual * probe).sum(), expert_output, retain_graph=True
        )
        expected_grad, = torch.autograd.grad(
            (expected * probe).sum(), expert_output
        )
        torch.testing.assert_close(
            actual_grad, expected_grad, rtol=0.0, atol=0.0
        )

    def test_fused_swiglu_matches_unfused_dense_and_grouped(self):
        if train.triton is None:
            self.skipTest("triton is required")
        torch.manual_seed(0)
        rows, dim, inter = 1000, 128, 192
        x = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        probe = torch.randn(rows, inter, device="cuda", dtype=torch.bfloat16)

        def unfused(x_, w_, offs_):
            parts, start = [], 0
            for e in range(w_.shape[0]):
                end = int(offs_[e])
                gate, up = (x_[start:end] @ w_[e].t()).chunk(2, dim=-1)
                parts.append(F.silu(gate) * up)
                start = end
            return torch.cat(parts)

        def check(w, offs):
            # fp32 reference; both bf16 paths must sit within bf16 noise of it,
            # and the fused path must not be measurably worse than the unfused one
            x32 = x.detach().float().requires_grad_(True)
            w32 = w.detach().float().requires_grad_(True)
            ref = unfused(x32, w32, offs)
            rx, rw = torch.autograd.grad((ref * probe.float()).sum(), (x32, w32))
            fused_h = train._FusedSwiGLU.apply(x, w, offs)
            fx, fw = torch.autograd.grad((fused_h * probe).sum(), (x, w))
            unfused_h = unfused(x, w, offs)
            ux, uw = torch.autograd.grad((unfused_h * probe).sum(), (x, w))
            for name, f, u, r in (("h", fused_h, unfused_h, ref), ("dx", fx, ux, rx), ("dw", fw, uw, rw)):
                err_f = (f.float() - r).abs().max().item()
                err_u = (u.float() - r).abs().max().item()
                scale = r.abs().max().item()
                self.assertLess(err_f, 0.05 * scale + 1e-3, f"{name}: fused err {err_f} (scale {scale})")
                self.assertLess(err_f, 1.5 * err_u + 0.01 * scale + 1e-3,
                                f"{name}: fused err {err_f} vs unfused {err_u}")
            return fw

        w = (torch.randn(1, 2 * inter, dim, device="cuda") * 0.05).to(torch.bfloat16).requires_grad_(True)
        check(w, torch.tensor([rows], device="cuda", dtype=torch.int32))
        counts = torch.tensor([300, 0, 700], device="cuda")
        w3 = (torch.randn(3, 2 * inter, dim, device="cuda") * 0.05).to(torch.bfloat16).requires_grad_(True)
        gw3 = check(w3, torch.cumsum(counts, 0).to(torch.int32))
        torch.testing.assert_close(gw3[1], torch.zeros_like(gw3[1]))

    def test_grouped_linear_matches_per_expert_reference(self):
        if train.triton is None:
            self.skipTest("triton is required")
        torch.manual_seed(1)
        rows, inter, dim = 1000, 192, 128
        h = torch.randn(rows, inter, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = (torch.randn(3, dim, inter, device="cuda") * 0.05).to(torch.bfloat16).requires_grad_(True)
        offs = torch.cumsum(torch.tensor([300, 0, 700], device="cuda"), 0).to(torch.int32)
        out = train._GroupedLinear.apply(h, w, offs)
        parts, start = [], 0
        for e in range(3):
            end = int(offs[e])
            parts.append(h[start:end] @ w[e].t())
            start = end
        expected = torch.cat(parts)
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)  # bf16: K-chunk order differs
        probe = torch.randn_like(out)
        gh, gw = torch.autograd.grad((out * probe).sum(), (h, w))
        eh, ew = torch.autograd.grad((expected * probe).sum(), (h, w))
        torch.testing.assert_close(gh, eh, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(gw, ew, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(gw[1], torch.zeros_like(gw[1]))

    def test_unused_experts_have_zero_finite_gradients(self):
        config = tiny_config(hidden_size=128, num_hidden_layers=1,
                             intermediate_size=384, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=64,
                             num_experts=4, moe_intermediate_size=64,
                             mlp_only_layers=())
        # SparkGPT keeps fp32 master parameters and casts grouped-GEMM compute
        # views to the bf16 residual dtype inside _grouped_linear.
        experts = train.Experts(config).cuda()
        with torch.no_grad():
            experts.gate_up_proj.normal_(0.0, 0.02)
            experts.down_proj.normal_(0.0, 0.02)
        hidden = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16,
                             requires_grad=True)
        selected = torch.tensor([[0, 1]], device="cuda").expand(256, -1)
        weights = torch.full((256, 2), 0.5, device="cuda", dtype=torch.bfloat16)
        output = experts(hidden, selected, weights)
        output.float().square().mean().backward()
        for gradient in (experts.gate_up_proj.grad, experts.down_proj.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            torch.testing.assert_close(gradient[2:], torch.zeros_like(gradient[2:]))


class FusedNormMLPTest(unittest.TestCase):
    def test_norm_swiglu_tracks_fp32_reference(self):
        if train.triton is None or not torch.cuda.is_available():
            self.skipTest("CUDA + triton are required")
        for hidden in (512, 1024):
            torch.manual_seed(0)
            norm = train.RMSNorm(hidden).cuda()
            with torch.no_grad():
                norm.weight.uniform_(0.5, 1.5)
            w_gu = (torch.randn(2 * 768, hidden, device="cuda") * hidden ** -0.5).requires_grad_(True)
            rows = 1000  # not a multiple of any tile size
            h0 = (torch.randn(rows, hidden, device="cuda") * 3.0).to(torch.bfloat16)
            probe = torch.randn(rows, 768, device="cuda")
            params = [norm.weight, w_gu]

            def run(fn, dtype):
                h = h0.to(dtype).requires_grad_(True)
                out = fn(h)
                g = torch.autograd.grad((out.float() * probe).sum(), [h] + params)
                return (out.float(),) + tuple(t.float() for t in g)

            def fp32(h):
                xn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight
                gate, up = (xn @ w_gu.t()).chunk(2, dim=-1)
                return torch.nn.functional.silu(gate) * up

            def unfused(h):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    offs = torch.full((1,), rows, device="cuda", dtype=torch.int32)
                    return train._FusedSwiGLU.apply(norm(h), w_gu.to(torch.bfloat16).unsqueeze(0), offs)

            ref = run(fp32, torch.float32)
            base = run(unfused, torch.bfloat16)
            fused = run(lambda h: train._NormSwiGLU.apply(h, norm.weight, w_gu, norm.eps), torch.bfloat16)
            for name, f, b, r in zip(("out", "dh", "dnorm_w", "dW_gu"), fused, base, ref):
                err_f = ((f - r).norm() / r.norm()).item()
                err_b = ((b - r).norm() / r.norm()).item()
                self.assertTrue(torch.isfinite(f).all(), name)
                self.assertLess(err_f, 1.5 * err_b + 1e-3,
                                f"hidden {hidden} {name}: fused {err_f:.2e} vs unfused {err_b:.2e}")


if __name__ == "__main__":
    unittest.main()
