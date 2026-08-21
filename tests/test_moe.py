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
        hidden = torch.randn(5, 4)
        selected = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2], [1, 0]])
        weights = torch.tensor([
            [0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.55, 0.45], [0.9, 0.1]
        ])
        actual = experts(hidden, selected, weights)
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
        torch.testing.assert_close(actual, expected)


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


if __name__ == "__main__":
    unittest.main()
