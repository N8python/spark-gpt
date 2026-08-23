"""sparkgpt: single-file DDP muP byte-level LM pretrainer for DGX Spark (GB10).

One file, one recipe. Qwen3-shaped decoder over raw UTF-8 bytes (vocab 259:
256 bytes + BOS/EOS/PAD), with optional stock-exportable Qwen3-MoE feed-forward
layers, trained with:

  * Whole-document varlen packing -- every batch has the same static
    --tokens-per-batch shape, but document boundaries are NEVER cut. Unused
    tail positions are PAD filler with ignored loss targets. Attention is
    flash-varlen and block-diagonal at document/filler boundaries: no
    inter-document attention, and RoPE restarts per segment.
  * muP width scaling (always on). Base hparams are defined at width
    --mup-base-dim (256) and transfer: hidden init 0.02*sqrt(base/width),
    residual writers (o/down) get an extra 1/sqrt(2L), embedding std constant,
    lm_head UNTIED + zero-init (init loss = ln 259), AdamW lm_head lr scaled
    by base/width. Muon lr needs NO width scaling (the orthogonalized update
    with the aspect-ratio factor is width-invariant). Verified 2026-07-01:
    optimal muon-lr 4e-3 at widths 256/512/1024. Keep head_dim fixed (128)
    across the width family.
  * Muon/AdamW hybrid (always). 2-D body and packed 3-D expert matrices ->
    Muon (NO weight decay); embedding, lm_head, norm gains, and routers ->
    AdamW (routers have no weight decay).
  * Optional Qwen3-MoE (--num-experts > 0): fp32 softmax top-k routing,
    selected-probability renormalization, packed BF16 grouped GEMM, and a
    filler-masked load-balancing loss aggregated globally across DDP ranks.
  * DDP across Spark nodes via torchrun; packed batches are sharded round-robin by
    rank. Single-process runs need no torchrun.

Defaults are a Chinchilla-optimal 50M run: 50M model (16L / 512d / 4Q+2KV
heads / head_dim 128 / MLP 1536) on 1B fineweb tokens (~72k tok/s single
node = ~4h; ~140k tok/s on both Sparks = ~2h).
The same hparams transfer to any width in the family (fix head_dim=128,
kv = heads/2, MLP = 3*dim); the Qwen3-0.6B shape (440.7M params) is:

  --model-layers 28 --model-dim 1024 --attention-heads 16 --kv-heads 8 \
  --intermediate-size 3072

Canonical runs:

  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # always, on Spark
  # single node (0.6B shape runs ~14.0k tok/s on one GB10):
  python train.py --run-name <name> --train-path lang_data/fineweb_2b.jsonl \
      --target-tokens 2000000000 --save-final \
      --val-path lang_data/fineweb_10m_val_fixed_seed0.jsonl
  # two nodes (run on each node with its own --node-rank; rank 0 is master):
  torchrun --nnodes 2 --node-rank <0|1> --nproc-per-node 1 \
      --master-addr <node0-ip> --master-port 29500 train.py ...

Hard-won GB10 (sm_121) facts -- do not relearn these:
  * compile mode "default" only: reduce-overhead's cudagraph pools OOM unified
    memory, and max-autotune is ~6.5% SLOWER than default (triton < cuBLAS).
  * fp8 is a net loss at dim 1024 (dynamic-scaling casts are bandwidth-bound;
    273 GB/s). Revisit at dim >= 2048.
  * Embedding is not autocast by PyTorch. Cast its output to the active
    autocast dtype once or every residual add promotes back to fp32; keeping
    the 0.6B residual stream in bf16 improved throughput by ~9% on GB10.
  * torch.compile traps: max_seqlen must be a CONSTANT python int and
    cu_seqlens' varying length must be mark_dynamic'd, else a silent
    recompile-limit eager fallback costs 2x throughput and +30 GB.
  * muon-lr 1e-2 (the old 20M-model default) diverges at 440M without muP.

Checkpoints: --save-final writes model_final.pt (native fused layout) plus a
READY-TO-LOAD HF directory checkpoints/<run>/hf/ (stock Qwen3ForCausalLM or
Qwen3MoeForCausalLM config + fp32 safetensors + byte tokenizer as a plain
tokenizer.json -- no custom code, no trust_remote_code). Periodic
--save-every checkpoints get an hf_step<N>/ twin on rank 0. RoPE is standard
rotate-half throughout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("WANDB_SILENT", "true")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_varlen_func

BOS, EOS, PAD = 256, 257, 258
VOCAB_SIZE = 259
LOSS_IGNORE_INDEX = -100
PACKING_FORMAT = "whole_document_static_v1"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    hidden_size: int = 512
    num_hidden_layers: int = 16
    intermediate_size: int = 1536
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 4096
    rope_theta: float = 1_000_000.0
    num_experts: int = 0
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 768
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 1e-3


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return out.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


def _varlen_attention(q, k, v, cu_seqlens, max_seqlen):
    """Block-diagonal causal attention over packed segments.

    flash-attn >= 2.7 registers its varlen kernels as torch custom ops, so
    this traces into the compiled graph (one graph per forward/backward
    instead of a graph break per layer). cu_seqlens has a data-dependent
    LENGTH (docs per window); the trainer marks that dim dynamic so no
    per-window recompile occurs."""
    return flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        causal=True,
    )


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.qkv_proj = nn.Linear(
            dim, (self.n_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    @staticmethod
    def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (T, heads, head_dim); cos/sin: (T, head_dim). HF/mlx qwen3
        # rotate-half. Cast HERE: under autocast x is bf16 while cos/sin ride
        # in fp32, and flash_attn rejects fp32.
        half = x.shape[-1] // 2
        rotate_half = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos.to(x.dtype)[:, None, :] + rotate_half * sin.to(x.dtype)[:, None, :]

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen):
        total = x.shape[0]
        q, k, v = self.qkv_proj(x).split(
            [self.n_heads * self.head_dim,
             self.n_kv_heads * self.head_dim,
             self.n_kv_heads * self.head_dim],
            dim=-1,
        )
        q = self._rope(self.q_norm(q.view(total, self.n_heads, self.head_dim)), cos, sin)
        k = self._rope(self.k_norm(k.view(total, self.n_kv_heads, self.head_dim)), cos, sin)
        v = v.view(total, self.n_kv_heads, self.head_dim)
        out = _varlen_attention(q, k, v, cu_seqlens, max_seqlen)
        return self.o_proj(out.reshape(total, self.n_heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TopKRouter(nn.Module):
    """Qwen3-MoE router: bias-free logits, fp32 softmax, then top-k."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        routing_weights, selected_experts = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        return (
            router_logits,
            routing_weights.to(dtype=router_logits.dtype),
            selected_experts,
        )


def _grouped_linear(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Jagged grouped linear for expert weights shaped (E, out, in).

    Torch's CUDA grouped GEMM is the production path.  The small eager
    fallback keeps CPU unit tests and older Torch builds functional.
    """
    grouped_mm = getattr(F, "grouped_mm", None)
    if hidden_states.is_cuda and grouped_mm is not None:
        # grouped_mm is not autocast-aware. SparkGPT retains fp32 master
        # parameters, so explicitly form the same bf16 compute view that
        # autocast supplies to ordinary dense linear layers.
        compute_weights = weights.to(dtype=hidden_states.dtype)
        return grouped_mm(
            hidden_states,
            compute_weights.transpose(-2, -1),
            offs=offsets,
        )

    outputs = []
    start = 0
    for expert_idx, end_tensor in enumerate(offsets):
        end = int(end_tensor.item())
        if end > start:
            outputs.append(F.linear(hidden_states[start:end], weights[expert_idx]))
        start = end
    if not outputs:
        return hidden_states.new_empty((0, weights.shape[1]))
    return torch.cat(outputs, dim=0)


class Experts(nn.Module):
    """Packed Qwen3-MoE experts in the stock HF checkpoint layout."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(
            self.num_experts, 2 * self.intermediate_dim, self.hidden_dim
        ))
        self.down_proj = nn.Parameter(torch.empty(
            self.num_experts, self.hidden_dim, self.intermediate_dim
        ))

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        top_k = selected_experts.shape[-1]

        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        sorted_experts, permutation = torch.sort(flat_experts)
        sorted_token_indices = permutation // top_k
        sorted_hidden = hidden_states[sorted_token_indices]
        sorted_weights = flat_weights[permutation]
        histc_input = (
            sorted_experts.int() if hidden_states.is_cuda
            else sorted_experts.float()
        )
        tokens_per_expert = torch.histc(
            histc_input,
            bins=self.num_experts,
            min=0,
            max=self.num_experts - 1,
        )
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

        gate, up = _grouped_linear(
            sorted_hidden, self.gate_up_proj, offsets
        ).chunk(2, dim=-1)
        expert_hidden = F.silu(gate) * up
        expert_output = _grouped_linear(
            expert_hidden, self.down_proj, offsets
        )
        expert_output = expert_output * sorted_weights.unsqueeze(-1)

        # Restore assignment order directly, avoiding an inverse-permutation
        # tensor followed by an indexed gather before the top-k reduction.
        ordered_output = torch.empty_like(expert_output)
        ordered_output[permutation] = expert_output
        return ordered_output.view(
            num_tokens, top_k, self.hidden_dim
        ).sum(dim=1).to(dtype=hidden_states.dtype)


class SparseMoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = TopKRouter(config)
        self.experts = Experts(config)

    def forward(self, hidden_states: torch.Tensor):
        router_logits, routing_weights, selected_experts = self.gate(hidden_states)
        output = self.experts(hidden_states, selected_experts, routing_weights)
        return output, router_logits, selected_experts


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config)
        self.is_sparse = config.num_experts > 0 and (
            layer_idx not in config.mlp_only_layers
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        self.mlp = (
            SparseMoE(config)
            if self.is_sparse
            else MLP(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, cu_seqlens, max_seqlen)
        mlp_input = self.post_attention_layernorm(h)
        if self.is_sparse:
            mlp_output, router_logits, selected_experts = self.mlp(mlp_input)
            return h + mlp_output, router_logits, selected_experts
        return h + self.mlp(mlp_input), None, None


class ByteLM(nn.Module):
    """Qwen3-shaped byte LM, flat packed (total_tokens,) layout, fused
    qkv/gate_up, untied lm_head (muP)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        freqs = torch.outer(
            torch.arange(config.max_position_embeddings, dtype=torch.float32), inv_freq
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_seqlen,
        output_router_logits: bool = False,
    ):
        x = self.embed_tokens(input_ids)
        autocast_enabled = torch.is_autocast_enabled(x.device.type)
        rope_dtype = (
            torch.get_autocast_dtype(x.device.type)
            if autocast_enabled else x.dtype
        )
        # Embedding is not autocast by PyTorch.  Keeping its fp32 output would
        # promote every residual add back to fp32, doubling activation traffic
        # despite the rest of training being bf16.  Enter the configured
        # autocast dtype once so the residual stream follows the requested
        # mixed-precision policy; non-autocast callers retain fp32 behavior.
        if autocast_enabled:
            x = x.to(dtype=rope_dtype)
        # Every attention layer consumes the same RoPE rows.  Cast them once
        # rather than recasting both tables for q and k in every layer.
        cos = self.cos_cached[position_ids].to(dtype=rope_dtype)
        sin = self.sin_cached[position_ids].to(dtype=rope_dtype)
        router_logits = []
        selected_experts = []
        for layer in self.layers:
            x, layer_router_logits, layer_selected_experts = layer(
                x, cos, sin, cu_seqlens, max_seqlen
            )
            if layer_router_logits is not None:
                router_logits.append(layer_router_logits)
                selected_experts.append(layer_selected_experts)
        logits = self.lm_head(self.norm(x))
        if output_router_logits:
            return logits, tuple(router_logits), tuple(selected_experts)
        return logits


def export_unfused_state_dict(model: ByteLM) -> dict:
    """Fused checkpoint -> separate q/k/v/gate/up layout (for HF/MLX export)."""
    cfg = model.config
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    out = {}
    for key, value in model.state_dict().items():
        if key.endswith("self_attn.qkv_proj.weight"):
            base = key[: -len("qkv_proj.weight")]
            out[base + "q_proj.weight"] = value[:q_dim]
            out[base + "k_proj.weight"] = value[q_dim : q_dim + kv_dim]
            out[base + "v_proj.weight"] = value[q_dim + kv_dim :]
        elif key.endswith("mlp.gate_up_proj.weight"):
            base = key[: -len("gate_up_proj.weight")]
            out[base + "gate_proj.weight"] = value[: cfg.intermediate_size]
            out[base + "up_proj.weight"] = value[cfg.intermediate_size :]
        else:
            out[key] = value
    return out


def _byte_tokenizer_json() -> dict:
    """A stock `tokenizers` fast-tokenizer spec that IS a raw-byte tokenizer:
    byte-level BPE with an EMPTY merge table, vocab = the GPT-2 byte alphabet
    at ids 0..255 (so token id == raw byte value), specials at 256..258, and a
    TemplateProcessing post-processor that prepends <bos> (the training input
    convention). No custom code, no trust_remote_code."""
    # GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to
    # 256+offset codepoints. ByteLevel pre-tokenizer/decoder speak this alphabet.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_char = dict(zip(bs, (chr(c) for c in cs)))
    vocab = {byte_char[b]: b for b in range(256)}
    vocab.update({"<bos>": BOS, "<eos>": EOS, "<pad>": PAD})
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": tid, "content": tok, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True}
            for tok, tid in (("<bos>", BOS), ("<eos>", EOS), ("<pad>", PAD))
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": True, "use_regex": True},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}},
                       {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "<bos>", "type_id": 0}},
                     {"Sequence": {"id": "A", "type_id": 0}},
                     {"SpecialToken": {"id": "<bos>", "type_id": 1}},
                     {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {"<bos>": {"id": "<bos>", "ids": [BOS], "tokens": ["<bos>"]}},
        },
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": []},
    }


def export_hf(model: ByteLM, out_dir: Path) -> None:
    """Write a ready-to-load stock Qwen3 or Qwen3-MoE HF directory."""
    from safetensors.torch import save_file

    cfg = model.config
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        (k if k == "lm_head.weight" else "model." + k): v.contiguous().to(torch.float32)
        for k, v in export_unfused_state_dict(model).items()
    }
    save_file(weights, str(out_dir / "model.safetensors"),
              metadata={"format": "pt"})
    is_moe = cfg.num_experts > 0
    hf_config = {
        "architectures": ["Qwen3MoeForCausalLM" if is_moe else "Qwen3ForCausalLM"],
        "model_type": "qwen3_moe" if is_moe else "qwen3",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "intermediate_size": cfg.intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "hidden_act": "silu",
        "rms_norm_eps": cfg.rms_norm_eps,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rope_theta": cfg.rope_theta,
        "rope_scaling": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "bos_token_id": BOS,
        "eos_token_id": EOS,
        "pad_token_id": PAD,
        "torch_dtype": "float32",
    }
    if is_moe:
        hf_config.update({
            "decoder_sparse_step": cfg.decoder_sparse_step,
            "moe_intermediate_size": cfg.moe_intermediate_size,
            "num_experts": cfg.num_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "norm_topk_prob": cfg.norm_topk_prob,
            "output_router_logits": False,
            "router_aux_loss_coef": cfg.router_aux_loss_coef,
            "mlp_only_layers": list(cfg.mlp_only_layers),
            "use_sliding_window": False,
            "sliding_window": None,
        })
    (out_dir / "config.json").write_text(
        json.dumps(hf_config, indent=2), encoding="utf-8"
    )
    (out_dir / "generation_config.json").write_text(json.dumps({
        "bos_token_id": BOS, "eos_token_id": EOS, "pad_token_id": PAD,
    }, indent=2), encoding="utf-8")
    (out_dir / "tokenizer.json").write_text(
        json.dumps(_byte_tokenizer_json(), ensure_ascii=False), encoding="utf-8")
    (out_dir / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
        "model_max_length": cfg.max_position_embeddings,
        "clean_up_tokenization_spaces": False,
    }, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# muP: init + per-group lrs
# --------------------------------------------------------------------------- #
def apply_mup_init(
    model: ByteLM,
    *,
    base_dim: int,
    emb_std: float,
    hidden_std: float,
    router_std: float = 0.02,
) -> dict:
    cfg = model.config
    m = cfg.hidden_size / base_dim
    std_in = hidden_std / math.sqrt(m)
    std_writer = std_in / math.sqrt(2 * cfg.num_hidden_layers)
    with torch.no_grad():
        model.embed_tokens.weight.normal_(0.0, emb_std)
        model.lm_head.weight.zero_()
        for block in model.layers:
            block.self_attn.qkv_proj.weight.normal_(0.0, std_in)
            block.self_attn.o_proj.weight.normal_(0.0, std_writer)
            if block.is_sparse:
                block.mlp.experts.gate_up_proj.normal_(0.0, std_in)
                block.mlp.experts.down_proj.normal_(0.0, std_writer)
                block.mlp.gate.weight.normal_(0.0, router_std)
            else:
                block.mlp.gate_up_proj.weight.normal_(0.0, std_in)
                block.mlp.down_proj.weight.normal_(0.0, std_writer)
    return {"width_mult": m, "hidden_init_std": std_in,
            "residual_writer_init_std": std_writer, "emb_init_std": emb_std,
            "lm_head_init": "zero", "router_init_std": router_std}


# --------------------------------------------------------------------------- #
# Muon / AdamW hybrid optimizer
# --------------------------------------------------------------------------- #
def _newtonschulz5_batched(X: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize a (B, r, c) stack of same-shape matrices (one bmm chain
    per iteration -- the step is kernel-launch-bound on GB10 otherwise).

    The staging stack is disposable, so normalize it in place and ping-pong
    explicit output buffers across iterations. Besides avoiding allocator work,
    this preserves the transpose-friendly layout of tall expert matrices.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    transpose_needed = X.shape[-2] > X.shape[-1]
    if transpose_needed:
        X = X.mT
    X.div_(X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    next_X = torch.empty_like(X)
    A = torch.empty(
        X.shape[0], X.shape[1], X.shape[1],
        device=X.device, dtype=X.dtype,
    )
    B = torch.empty_like(A)
    for _ in range(steps):
        torch.bmm(X, X.mT, out=A)
        torch.baddbmm(A, A, A, beta=b, alpha=c, out=B)
        torch.baddbmm(X, B, X, beta=a, alpha=1.0, out=next_X)
        X, next_X = next_X, X
    if transpose_needed:
        X = X.mT
    return X


class MuonAdamWHybrid(torch.optim.Optimizer):
    """Muon groups (use_muon=True; 2-D matrices, NO weight decay) + AdamW
    groups, in one optimizer so a single LRScheduler drives both (per-group
    ratios are preserved -- schedulers scale each group from its initial_lr)."""

    def __init__(self, param_groups):
        defaults = dict(lr=1e-4, weight_decay=0.0, use_muon=False, momentum=0.95,
                        nesterov=True, ns_steps=5, betas=(0.9, 0.999), eps=1e-8)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            (self._muon_group if group["use_muon"] else self._adamw_group)(group)
        return loss

    def _muon_group(self, group):
        momentum = group["momentum"]
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        grads = [p.grad.float() for p in params]
        vs = []
        for p in params:
            state = self.state[p]
            if "v" not in state:
                state["v"] = torch.zeros_like(p, dtype=torch.float32)
            vs.append(state["v"])
        # lerp evaluates the same momentum recurrences in one multi-tensor
        # pass instead of a multiply pass followed by an add pass.
        torch._foreach_lerp_(vs, grads, 1 - momentum)
        if group["nesterov"]:
            # Gradients are dead after optimizer.step (the trainer clears them
            # before the next forward), so reuse their storage for the
            # Nesterov update instead of allocating another full model-sized
            # fp32 tensor list.
            updates = grads
            torch._foreach_lerp_(updates, vs, momentum)
        else:
            updates = [v.clone() for v in vs]

        by_shape: dict[tuple[int, int, int], list] = {}
        for p, u in zip(params, updates):
            if u.ndim not in (2, 3):
                raise ValueError(f"Muon requires 2-D or 3-D parameters, got {u.shape}")
            by_shape.setdefault((u.ndim, u.shape[-2], u.shape[-1]), []).append((p, u))
        for (ndim, rows, cols), items in by_shape.items():
            batch_sizes = [1 if ndim == 2 else u.shape[0] for _, u in items]
            stacked = torch.empty(
                (sum(batch_sizes), rows, cols), device=items[0][1].device,
                dtype=torch.bfloat16,
            )
            offset = 0
            for (_, update), batch_size in zip(items, batch_sizes):
                source = update.unsqueeze(0) if ndim == 2 else update
                stacked[offset : offset + batch_size].copy_(source)
                offset += batch_size
            ortho = _newtonschulz5_batched(stacked, steps=group["ns_steps"])
            # aspect-ratio factor; also what makes Muon lr width-invariant
            lr = group["lr"] * max(1, rows / cols) ** 0.5
            if ndim == 2:
                ortho_updates = list(ortho.unbind(0))
            else:
                ortho_updates = list(ortho.split(batch_sizes, dim=0))
            torch._foreach_add_(
                [p for p, _ in items],
                ortho_updates,
                alpha=-lr,
            )

    def _adamw_group(self, group):
        beta1, beta2 = group["betas"]
        lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        grads = [p.grad for p in params]
        exp_avgs, exp_avg_sqs = [], []
        for p in params:
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"] += 1
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
        t = self.state[params[0]]["step"]
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)
        if wd != 0:
            torch._foreach_mul_(params, 1 - lr * wd)
        denom = torch._foreach_div(exp_avg_sqs, 1 - beta2**t)
        torch._foreach_sqrt_(denom)
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-(lr / (1 - beta1**t)))


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    """muP groups: Muon on 2-D body at base lr (width-invariant); AdamW on
    embedding + gains at base lr and on lm_head at lr * base/width."""
    head_mult = args.mup_base_dim / args.model_dim
    muon_params, adamw_const, adamw_head, adamw_router = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lm_head" in name:
            adamw_head.append(param)
        elif name.endswith("mlp.gate.weight"):
            adamw_router.append(param)
        elif param.ndim != 2 or "embed" in name:
            if param.ndim == 3 and ".experts." in name:
                muon_params.append(param)
            else:
                adamw_const.append(param)
        else:
            muon_params.append(param)
    router_lr = args.router_lr if args.router_lr is not None else args.adamw_lr
    optimizer = MuonAdamWHybrid([
        {"params": muon_params, "use_muon": True, "lr": args.muon_lr,
         "momentum": args.muon_momentum, "weight_decay": 0.0},
        {"params": adamw_const, "use_muon": False, "lr": args.adamw_lr,
         "weight_decay": args.weight_decay},
        {"params": adamw_head, "use_muon": False, "lr": args.adamw_lr * head_mult,
         "weight_decay": args.weight_decay},
        {"params": adamw_router, "use_muon": False, "lr": router_lr,
         "weight_decay": 0.0},
    ])
    return optimizer, {
        "mode": "mup_muon_hybrid",
        "head_lr_mult": head_mult,
        "router_lr": router_lr,
        "router_param_count": sum(p.numel() for p in adamw_router),
        "muon_param_count": sum(p.numel() for p in muon_params),
        "adamw_param_count": sum(
            p.numel() for p in adamw_const + adamw_head + adamw_router
        ),
    }


def global_load_balancing_loss(
    router_logits: tuple[torch.Tensor, ...],
    selected_experts: tuple[torch.Tensor, ...],
    valid_mask: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Qwen3 auxiliary router loss over all sparse layers and DDP ranks.

    The hard top-k counts are global constants.  The local differentiable
    proxy is scaled so DDP's gradient averaging produces the gradient of the
    true global-batch loss.  The straight-through value correction makes every
    rank report the same global scalar, including when one rank is all filler.
    """
    if not router_logits:
        zero = valid_mask.sum(dtype=torch.float32) * 0.0
        return zero, {
            "aux_loss": zero.detach(),
            "assignment_min_frac": zero.detach(),
            "assignment_max_frac": zero.detach(),
            "assignment_cv": zero.detach(),
            "router_entropy": zero.detach(),
            "unused_experts": zero.detach(),
        }
    if len(router_logits) != len(selected_experts):
        raise ValueError("router logits and selected-expert layers do not match")

    mask = valid_mask.to(dtype=torch.float32)
    soft_sums = torch.zeros(num_experts, device=valid_mask.device, dtype=torch.float32)
    hard_counts = torch.zeros_like(soft_sums)
    entropy_sum = torch.zeros((), device=valid_mask.device, dtype=torch.float32)
    valid_rows = torch.zeros((), device=valid_mask.device, dtype=torch.float32)

    for logits, indices in zip(router_logits, selected_experts):
        if logits.shape[:-1] != valid_mask.shape or logits.shape[-1] != num_experts:
            raise ValueError("router-logit shape does not match the token mask/config")
        if indices.shape != (*valid_mask.shape, top_k):
            raise ValueError("selected-expert shape does not match the token mask/config")
        probs = F.softmax(logits, dtype=torch.float32, dim=-1)
        soft_sums = soft_sums + (probs * mask.unsqueeze(-1)).sum(dim=0)
        count_weights = mask.unsqueeze(-1).expand(-1, top_k).reshape(-1)
        hard_counts = hard_counts + torch.bincount(
            indices.reshape(-1), weights=count_weights, minlength=num_experts
        )
        with torch.no_grad():
            detached_probs = probs.detach()
            entropy_sum = entropy_sum - (
                detached_probs
                * detached_probs.clamp_min(torch.finfo(detached_probs.dtype).tiny).log()
                * mask.unsqueeze(-1)
            ).sum()
        valid_rows = valid_rows + mask.sum()

    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    global_counts = hard_counts.detach().clone()
    global_soft_sums = soft_sums.detach().clone()
    global_entropy_sum = entropy_sum.detach().clone()
    global_valid_rows = valid_rows.detach().clone()
    if distributed:
        for tensor in (
            global_counts,
            global_soft_sums,
            global_entropy_sum,
            global_valid_rows,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    safe_global_rows = global_valid_rows.clamp_min(1.0)
    has_valid_rows = (global_valid_rows > 0).to(dtype=torch.float32)
    assignments_per_row = global_counts / safe_global_rows
    global_mean_probs = global_soft_sums / safe_global_rows
    global_aux = (
        has_valid_rows
        * num_experts
        * torch.dot(assignments_per_row, global_mean_probs)
    )
    local_proxy = (
        has_valid_rows
        * world_size
        * num_experts
        / safe_global_rows
        * torch.dot(assignments_per_row, soft_sums)
    )
    aux_loss = local_proxy + (global_aux - local_proxy.detach())

    assignment_frac = global_counts / (safe_global_rows * top_k)
    assignment_mean = assignment_frac.mean()
    stats = {
        "aux_loss": global_aux,
        "assignment_min_frac": assignment_frac.min(),
        "assignment_max_frac": assignment_frac.max(),
        "assignment_cv": assignment_frac.std(unbiased=False)
        / assignment_mean.clamp_min(1e-12),
        "router_entropy": global_entropy_sum / safe_global_rows,
        "unused_experts": (global_counts == 0).sum(dtype=torch.float32),
    }
    return aux_loss, stats


# --------------------------------------------------------------------------- #
# Data: compact byte stream -> per-doc pair streams -> fixed-shape batches
# --------------------------------------------------------------------------- #
def load_compact_tokenized(path: Path, *, target_tokens: int | None,
                           flush_tokens: int = 200_000_000):
    """Stream a jsonl corpus ({"text": ...} per line) into one flat int16
    token buffer + per-doc offsets (~2 bytes/token; 10B+ corpora fit RAM).
    Selection is by cumulative loss-token budget in file order."""
    chunks: list[np.ndarray] = []
    buf: list[int] = []
    offsets: list[int] = [0]
    running = 0
    loss_tokens = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            encoded = json.loads(line)["text"].encode("utf-8")
            buf.append(BOS)
            buf.extend(encoded)
            buf.append(EOS)
            n = len(encoded) + 2
            running += n
            offsets.append(running)
            loss_tokens += n - 1
            if len(buf) >= flush_tokens:
                chunks.append(np.asarray(buf, dtype=np.int16))
                buf = []
            if target_tokens is not None and loss_tokens >= target_tokens:
                break
    if buf:
        chunks.append(np.asarray(buf, dtype=np.int16))
    flat = np.concatenate(chunks) if len(chunks) > 1 else (
        chunks[0] if chunks else np.zeros(0, dtype=np.int16))
    return flat, offsets, loss_tokens, len(offsets) - 1


def build_pair_streams(flat: np.ndarray, offsets: list[int]):
    """Per-doc (input, target) pair streams: doc tokens [t0..tn] contribute
    inputs t0..t(n-1) and targets t1..tn -- every packed position has a real
    target and no target crosses a doc boundary. `bounds` are cumulative
    pair-stream offsets of doc boundaries."""
    starts = np.asarray(offsets[:-1], dtype=np.int64)
    ends = np.asarray(offsets[1:], dtype=np.int64)
    keep_in = np.ones(flat.shape[0], dtype=bool)
    keep_in[ends - 1] = False
    keep_tg = np.ones(flat.shape[0], dtype=bool)
    keep_tg[starts] = False
    bounds = np.concatenate([[0], np.cumsum(ends - starts - 1)])
    return flat[keep_in], flat[keep_tg], bounds


@dataclass(frozen=True)
class PackedBatch:
    """A consecutive, whole-document span placed in one static batch."""

    start_doc: int
    end_doc: int
    real_tokens: int


def build_whole_document_batches(bounds: np.ndarray, window: int) -> list[PackedBatch]:
    """Sequential next-fit packing without ever splitting a document.

    The returned batches describe only real pair-stream tokens. Materialization
    adds ignored PAD filler to reach ``window``. Keeping the plan as document
    spans avoids materializing a second, padded copy of a multi-billion-token
    corpus in host memory.
    """
    bounds = np.asarray(bounds, dtype=np.int64)
    if window <= 0:
        raise ValueError("window must be positive")
    if bounds.ndim != 1 or bounds.size < 2 or bounds[0] != 0:
        raise ValueError("bounds must be a 1-D cumulative array starting at zero")
    lengths = np.diff(bounds)
    if np.any(lengths <= 0):
        raise ValueError("every document must contribute at least one loss token")
    longest = int(lengths.max())
    if longest > window:
        doc = int(np.argmax(lengths))
        raise ValueError(
            f"document {doc} has {longest} loss tokens, exceeding "
            f"--tokens-per-batch ({window}); whole-document packing will not split it"
        )

    batches: list[PackedBatch] = []
    start_doc = 0
    used = 0
    for doc, length_np in enumerate(lengths):
        length = int(length_np)
        if used and used + length > window:
            batches.append(PackedBatch(start_doc, doc, used))
            start_doc = doc
            used = 0
        used += length
    if used:
        batches.append(PackedBatch(start_doc, len(lengths), used))
    return batches


def packed_batch_metadata(bounds: np.ndarray, batch: PackedBatch, window: int,
                          max_segment_length: int):
    """Build fixed-shape cu_seqlens and RoPE positions for a packed batch.

    Filler is split into one or more isolated segments so its positions and
    flash-attn ``max_seqlen`` never exceed the real-data model limit. Its loss
    is ignored, and separating it from the final document prevents PAD tokens
    from changing any real-token activation.
    """
    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be positive")
    local = np.asarray(
        bounds[batch.start_doc : batch.end_doc + 1], dtype=np.int64
    ) - int(bounds[batch.start_doc])
    if int(local[-1]) != batch.real_tokens or batch.real_tokens > window:
        raise ValueError("packed batch plan does not match document bounds")
    cu_list = local.tolist()
    remaining = window - batch.real_tokens
    while remaining:
        chunk = min(remaining, max_segment_length)
        cu_list.append(cu_list[-1] + chunk)
        remaining -= chunk
    cu = np.asarray(cu_list, dtype=np.int32)
    seg_lens = np.diff(cu)
    pos = np.arange(window, dtype=np.int64) - np.repeat(cu[:-1].astype(np.int64), seg_lens)
    return cu, pos


def materialize_packed_batch(inputs: np.ndarray, targets: np.ndarray,
                             bounds: np.ndarray, batch: PackedBatch, window: int,
                             max_segment_length: int):
    """Return static-shape arrays; only the prefix contains loss-bearing data."""
    lo = int(bounds[batch.start_doc])
    hi = int(bounds[batch.end_doc])
    if hi - lo != batch.real_tokens:
        raise ValueError("packed batch token count does not match source span")
    ids = np.full(window, PAD, dtype=np.int64)
    tgt = np.full(window, LOSS_IGNORE_INDEX, dtype=np.int64)
    ids[:batch.real_tokens] = inputs[lo:hi]
    tgt[:batch.real_tokens] = targets[lo:hi]
    cu, pos = packed_batch_metadata(
        bounds, batch, window, max_segment_length
    )
    return ids, tgt, pos, cu


def cosine_with_warmup(step, *, warmup_steps, total_steps, min_lr_ratio):
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = min(1.0, float(step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def resolve_training_schedule(
    actual_steps: int,
    *,
    lr_schedule_steps: int | None,
    warmup_frac: float,
    val_interval_frac: float,
    val_interval_steps: int | None,
) -> tuple[int, int, int]:
    """Resolve LR horizon and validation cadence for this finite data prefix.

    Ordinarily the LR horizon equals the number of actual optimizer steps.  A
    longer explicit horizon makes a short run an exact prefix of that longer
    schedule, including its warmup.  This is useful for extrapolation studies
    without loading or training on the complete data stream.
    """
    if actual_steps <= 0:
        raise ValueError("actual training steps must be positive")
    schedule_steps = actual_steps if lr_schedule_steps is None else lr_schedule_steps
    if schedule_steps < actual_steps:
        raise ValueError("--lr-schedule-steps must be at least the actual training steps")
    if warmup_frac < 0:
        raise ValueError("--warmup-frac must be non-negative")
    warmup_steps = max(10, math.ceil(warmup_frac * schedule_steps))
    if val_interval_steps is not None:
        if val_interval_steps <= 0:
            raise ValueError("--val-interval-steps must be positive")
        resolved_val_interval = val_interval_steps
    else:
        if val_interval_frac <= 0:
            raise ValueError("--val-interval-frac must be positive")
        resolved_val_interval = max(1, math.ceil(val_interval_frac * actual_steps))
    return schedule_steps, warmup_steps, resolved_val_interval


def resolve_run_steps(available_steps: int, max_train_steps: int | None) -> int:
    """Limit execution without changing the shuffled data schedule itself."""
    if available_steps <= 0:
        raise ValueError("available training steps must be positive")
    if max_train_steps is None:
        return available_steps
    if max_train_steps <= 0:
        raise ValueError("--max-train-steps must be positive")
    if max_train_steps > available_steps:
        raise ValueError("--max-train-steps exceeds the available training steps")
    return max_train_steps


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
def finish_wandb_with_timeout(wandb_run, timeout_seconds: float) -> bool:
    """Finish W&B without allowing its helper process to wedge the trainer.

    W&B finalization can wait indefinitely for wandb-core even after the core
    reports that all files were uploaded.  Run it on a daemon thread so the
    successful training process retains control of its own shutdown.
    """
    done = threading.Event()
    errors: list[BaseException] = []

    def finish() -> None:
        try:
            wandb_run.finish()
        except BaseException as exc:  # report after the thread hands control back
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=finish, name="wandb-finish", daemon=True)
    thread.start()
    if not done.wait(timeout_seconds):
        print(json.dumps({"event": "wandb_finish_timeout",
                          "timeout_seconds": timeout_seconds}),
              file=sys.__stderr__, flush=True)
        return False
    if errors:
        print(json.dumps({"event": "wandb_finish_error",
                          "error": repr(errors[0])}),
              file=sys.__stderr__, flush=True)
        return False
    return True


def wandb_train_metrics(record: dict) -> dict:
    """Build one comparable W&B history row for a training step.

    ``train/loss`` is always the unregularized language-model cross entropy.
    For MoE runs, ``train/aux_loss`` exposes the raw expert-balancing loss; the
    same value is also available under the router-namespaced
    ``router/aux_loss`` key.
    """
    metrics = {
        "train/loss": record["loss"],
        "train/tokens_per_sec": record["real_tokens_per_second"],
        "train/compute_tokens_per_sec": record["window_tokens_per_second"],
        "train/packing_utilization": record["observed_packing_utilization"],
        "train/lr": record["lr"],
        "train/peak_cuda_memory_gb": record["peak_cuda_memory_gb"],
        "train/step": record["step"],
        "train/tokens": record["global_total_tokens"],
    }
    if "router_aux_loss" in record:
        metrics.update({
            "train/aux_loss": record["router_aux_loss"],
            "router/aux_loss": record["router_aux_loss"],
            "router/assignment_min_frac": record["router_assignment_min_frac"],
            "router/assignment_max_frac": record["router_assignment_max_frac"],
            "router/assignment_cv": record["router_assignment_cv"],
            "router/entropy": record["router_entropy"],
            "router/unused_experts": record["router_unused_experts"],
        })
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="sparkgpt: DDP muP byte-level packed pretrainer.")
    p.add_argument("--train-path", default="lang_data/fineweb_1b.jsonl")
    p.add_argument("--run-name", default="sparkgpt")
    p.add_argument("--target-tokens", type=int, default=1_000_000_000,
                   help="loss-token budget (global, before rank sharding); default = 1B, "
                        "i.e. Chinchilla-optimal for the default 50M model")
    p.add_argument(
        "--max-train-steps", type=int, default=None,
        help="stop after this many optimizer steps while retaining the data schedule "
             "built from --target-tokens (default: execute the complete schedule)",
    )
    # model (defaults = 50M; see docstring for the 0.6B flag set)
    p.add_argument("--model-layers", type=int, default=16)
    p.add_argument("--model-dim", type=int, default=512)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--intermediate-size", type=int, default=1536)
    p.add_argument("--rope-theta", type=float, default=1_000_000.0)
    p.add_argument("--max-position-embeddings", type=int, default=4096)
    # optional Qwen3-MoE feed-forward layers (--num-experts 0 keeps dense mode)
    p.add_argument("--num-experts", type=int, default=0)
    p.add_argument("--num-experts-per-tok", type=int, default=2)
    p.add_argument("--moe-intermediate-size", type=int, default=768)
    p.add_argument("--decoder-sparse-step", type=int, default=1)
    p.add_argument(
        "--mlp-only-layers", default="",
        help="comma-separated zero-based layer indices that remain dense",
    )
    p.add_argument(
        "--norm-topk-prob", action=argparse.BooleanOptionalAction, default=True,
        help="renormalize the selected expert probabilities (Qwen3 checkpoints: true)",
    )
    p.add_argument("--router-aux-loss-coef", type=float, default=1e-3)
    p.add_argument("--router-init-std", type=float, default=0.02)
    # muP (base hparams defined at --mup-base-dim; verified transfer to 1024)
    p.add_argument("--mup-base-dim", type=int, default=256)
    p.add_argument("--mup-emb-std", type=float, default=0.02)
    p.add_argument("--mup-hidden-std", type=float, default=0.02)
    # optimization (4e-3 verified optimal across widths, 2026-07-01)
    p.add_argument("--muon-lr", type=float, default=4e-3)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--adamw-lr", type=float, default=5e-4)
    p.add_argument(
        "--router-lr", type=float, default=None,
        help="router AdamW learning rate (default: --adamw-lr; no weight decay)",
    )
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW groups only")
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--lr-decay-factor", type=float, default=0.1)
    p.add_argument(
        "--lr-schedule-steps", type=int, default=None,
        help="LR schedule horizon in optimizer steps (default: actual training steps); "
             "set longer than the run to train an exact prefix of that schedule",
    )
    # batching / runtime
    p.add_argument("--tokens-per-batch", type=int, default=49152,
                   help="window size per rank per step (GB10 memory optimum)")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=None, help="default: --seed")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ddp-timeout-seconds", type=int, default=300,
                   help="maximum wait for any DDP collective, including teardown")
    # validation (off unless --val-path is set; lang_data ships
    # fineweb_10m_val_fixed_seed0.jsonl for exactly this)
    p.add_argument("--val-path", default="",
                   help="held-out jsonl; rank 0 evaluates every --val-interval-frac "
                        "of training and at the end")
    p.add_argument("--val-tokens", type=int, default=2_000_000,
                   help="loss-token budget for the val set (packed like training)")
    p.add_argument("--val-interval-frac", type=float, default=0.05)
    p.add_argument(
        "--val-interval-steps", type=int, default=None,
        help="exact validation cadence in optimizer steps; overrides "
             "--val-interval-frac",
    )
    p.add_argument("--save-final", action="store_true")
    p.add_argument("--save-every", type=int, default=0,
                   help="write a RESUMABLE checkpoint (model+optimizer+scheduler+step) "
                        "every N steps; every rank saves to its own local disk (0 = off)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
                   help="resume from the latest ckpt_step*.pt in checkpoints/<run-name>/ "
                        "(same command line; data/sharding must match the fingerprint)")
    # wandb
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wandb-project", default="sparkgpt")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--wandb-tags", default="")
    p.add_argument("--wandb-finish-timeout-seconds", type=float, default=30.0,
                   help="maximum wait for wandb-core after training has completed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.attention_heads % args.kv_heads != 0:
        raise ValueError("--attention-heads must be divisible by --kv-heads")
    if args.ddp_timeout_seconds <= 0:
        raise ValueError("--ddp-timeout-seconds must be positive")
    if args.wandb_finish_timeout_seconds <= 0:
        raise ValueError("--wandb-finish-timeout-seconds must be positive")
    try:
        mlp_only_layers = tuple(
            int(item) for item in args.mlp_only_layers.split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError("--mlp-only-layers must be comma-separated integers") from exc
    if len(set(mlp_only_layers)) != len(mlp_only_layers):
        raise ValueError("--mlp-only-layers contains duplicate indices")
    if any(layer < 0 or layer >= args.model_layers for layer in mlp_only_layers):
        raise ValueError("--mlp-only-layers indices must refer to existing layers")
    if args.num_experts < 0:
        raise ValueError("--num-experts must be nonnegative")
    if args.num_experts > 0:
        if not 1 <= args.num_experts_per_tok <= args.num_experts:
            raise ValueError("--num-experts-per-tok must be between 1 and --num-experts")
        if args.moe_intermediate_size <= 0:
            raise ValueError("--moe-intermediate-size must be positive")
        if args.decoder_sparse_step <= 0:
            raise ValueError("--decoder-sparse-step must be positive")
        if args.router_aux_loss_coef < 0:
            raise ValueError("--router-aux-loss-coef must be nonnegative")
        if args.router_init_std <= 0:
            raise ValueError("--router-init-std must be positive")
        sparse_layers = [
            layer for layer in range(args.model_layers)
            if layer not in mlp_only_layers
            and (layer + 1) % args.decoder_sparse_step == 0
        ]
        if not sparse_layers:
            raise ValueError("MoE is enabled but the layer selection produces no sparse layers")
    elif mlp_only_layers:
        raise ValueError("--mlp-only-layers requires --num-experts > 0")
    data_seed = args.data_seed if args.data_seed is not None else args.seed

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    is_main = rank == 0
    if not torch.cuda.is_available():
        raise RuntimeError("sparkgpt requires CUDA (flash_attn varlen)")
    if distributed:
        torch.cuda.set_device(0)  # one GPU per Spark node
        dist.init_process_group(
            "nccl", device_id=torch.device("cuda:0"),
            timeout=timedelta(seconds=args.ddp_timeout_seconds),
        )
    device = torch.device("cuda")

    torch.manual_seed(args.seed)  # same seed on every rank -> identical init
    random.seed(data_seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- data ----
    data_start = time.perf_counter()
    flat, offsets, selected_tokens, num_docs = load_compact_tokenized(
        Path(args.train_path), target_tokens=args.target_tokens
    )
    inputs_np, targets_np, bounds = build_pair_streams(flat, offsets)
    window = args.tokens_per_batch
    max_doc_pairs = int(np.diff(bounds).max())
    train_batches = build_whole_document_batches(bounds, window)
    global_max_seqlen = max_doc_pairs  # CONSTANT (compile trap)
    if global_max_seqlen > args.max_position_embeddings:
        raise ValueError(f"longest packed segment ({global_max_seqlen}) exceeds "
                         f"--max-position-embeddings ({args.max_position_embeddings})")
    global_batch_order = list(range(len(train_batches)))
    random.Random(data_seed + 1).shuffle(global_batch_order)
    # Pad the global schedule, rather than dropping a real batch, so every rank
    # executes the same number of DDP steps and every selected document trains.
    # A -1 slot materializes as an all-PAD, zero-real-token batch; at least one
    # other rank has real tokens in that final step.
    ddp_filler_batches = (-len(global_batch_order)) % world_size
    global_batch_order.extend([-1] * ddp_filler_batches)
    scheduled_batches = len(global_batch_order)
    window_order = global_batch_order[rank::world_size]
    available_steps = len(window_order)
    steps_total = resolve_run_steps(available_steps, args.max_train_steps)
    window_order = window_order[:steps_total]
    executed_global_order = global_batch_order[:steps_total * world_size]
    training_real_tokens = sum(
        train_batches[w].real_tokens for w in executed_global_order if w >= 0
    )
    # Every rank knows the whole schedule, so the global real-token count of
    # each DDP step is a host-side constant: no per-step all_reduce, and no
    # device sync before the optimizer can be launched.
    step_global_tokens = [
        sum(train_batches[w].real_tokens
            for w in global_batch_order[i * world_size:(i + 1) * world_size]
            if w >= 0)
        for i in range(steps_total)
    ]
    training_compute_tokens = steps_total * world_size * window
    packing_utilization = training_real_tokens / training_compute_tokens
    lr_schedule_steps, warmup_steps, val_interval = resolve_training_schedule(
        steps_total,
        lr_schedule_steps=args.lr_schedule_steps,
        warmup_frac=args.warmup_frac,
        val_interval_frac=args.val_interval_frac,
        val_interval_steps=args.val_interval_steps,
    )

    # Held-out validation set (rank 0 only): whole-document packed in file order
    # so the metric is stable across runs.
    val_windows: list = []
    if args.val_path and is_main:
        vflat, voffsets, _, _ = load_compact_tokenized(
            Path(args.val_path), target_tokens=args.val_tokens
        )
        vin, vtg, vbounds = build_pair_streams(vflat, voffsets)
        val_batches = build_whole_document_batches(vbounds, window)
        val_max_seg = int(np.diff(vbounds).max())
        if val_max_seg > args.max_position_embeddings:
            raise ValueError("longest val segment exceeds --max-position-embeddings")
        global_max_seqlen = max(global_max_seqlen, val_max_seg)
        for val_batch in val_batches:
            ids, tgt, pos, cu = materialize_packed_batch(
                vin, vtg, vbounds, val_batch, window, global_max_seqlen
            )
            val_windows.append((ids, tgt, pos, cu, val_batch.real_tokens))
    if distributed:
        # Rank 0 may have widened this for validation. Keep the Python constant
        # identical on every rank so training compiles the same graph/kernel.
        max_seqlen_t = torch.tensor(global_max_seqlen, device=device, dtype=torch.int64)
        dist.broadcast(max_seqlen_t, src=0)
        global_max_seqlen = int(max_seqlen_t.item())
    data_elapsed = time.perf_counter() - data_start

    # ---- model / muP / DDP / compile ----
    config = ModelConfig(
        hidden_size=args.model_dim,
        num_hidden_layers=args.model_layers,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.attention_heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.head_dim,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=args.rope_theta,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
        decoder_sparse_step=args.decoder_sparse_step,
        mlp_only_layers=mlp_only_layers,
        norm_topk_prob=args.norm_topk_prob,
        router_aux_loss_coef=args.router_aux_loss_coef,
    )
    model = ByteLM(config).to(device)
    mup_summary = apply_mup_init(model, base_dim=args.mup_base_dim,
                                 emb_std=args.mup_emb_std, hidden_std=args.mup_hidden_std,
                                 router_std=args.router_init_std)
    raw_model = model  # for checkpointing
    if distributed:
        # RoPE caches are immutable and deterministically identical on every
        # rank, so broadcasting them before every forward is unnecessary.  It
        # is also dangerous for rank-0-only validation (see run_val below).
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[0], broadcast_buffers=False,
        )
    if args.compile:
        model = torch.compile(model, mode="default", dynamic=False)
    optimizer, optimizer_summary = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup(
            step, warmup_steps=warmup_steps, total_steps=lr_schedule_steps,
            min_lr_ratio=args.lr_decay_factor),
    )

    run_dir = Path("checkpoints") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    param_count = sum(p.numel() for p in raw_model.parameters())
    expert_param_count = sum(
        p.numel() for name, p in raw_model.named_parameters() if ".experts." in name
    )
    active_expert_param_count = (
        expert_param_count * args.num_experts_per_tok // args.num_experts
        if args.num_experts > 0 else 0
    )
    active_param_count = param_count - expert_param_count + active_expert_param_count
    config_payload = {
        "args": vars(args),
        "model_config": asdict(config),
        "parameter_count": param_count,
        "active_parameter_count": active_param_count,
        "expert_parameter_count": expert_param_count,
        "mup": mup_summary,
        "optimizer": optimizer_summary,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "selected_docs": num_docs,
        "selected_loss_tokens": selected_tokens,
        "packing_format": PACKING_FORMAT,
        "window_tokens": window,
        "packed_real_batches": len(train_batches),
        "ddp_filler_batches": ddp_filler_batches,
        "scheduled_batches": scheduled_batches,
        "available_steps_per_rank": available_steps,
        "executed_scheduled_batches": steps_total * world_size,
        "real_tokens_used": training_real_tokens,
        "padding_compute_tokens": training_compute_tokens - training_real_tokens,
        "packing_utilization": packing_utilization,
        "steps_per_rank": steps_total,
        "lr_schedule_steps": lr_schedule_steps,
        "warmup_steps": warmup_steps,
        "val_interval_steps": val_interval,
        "world_size": world_size,
        "rank": rank,
        "data_prep_seconds": data_elapsed,
    }
    if is_main:
        print(json.dumps(config_payload, indent=2), flush=True)
        (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2),
                                             encoding="utf-8")

    wandb_run = None
    if args.wandb and is_main:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity or None,
            name=args.wandb_run_name or args.run_name,
            tags=[t for t in args.wandb_tags.split(",") if t],
            config=config_payload,
        )
        wandb.define_metric("train/tokens")
        wandb.define_metric("*", step_metric="train/tokens")

    def batch_for(w: int):
        packed = train_batches[w] if w >= 0 else PackedBatch(0, 0, 0)
        ids_np, tgt_np, pos, cu = materialize_packed_batch(
            inputs_np, targets_np, bounds, packed, window, global_max_seqlen
        )
        ids = torch.from_numpy(ids_np).to(device, non_blocking=True)
        tgt = torch.from_numpy(tgt_np).to(device, non_blocking=True)
        pos_t = torch.from_numpy(pos).to(device, non_blocking=True)
        cu_t = torch.from_numpy(cu).to(device, non_blocking=True)
        torch._dynamo.mark_dynamic(cu_t, 0)  # varying segment count (compile trap)
        return ids, tgt, pos_t, cu_t, packed.real_tokens

    def forward_backward(w: int, global_tokens: int):
        ids, tgt, pos_t, cu_t, real_tokens = batch_for(w)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            model_output = model(
                ids,
                pos_t,
                cu_t,
                global_max_seqlen,
                output_router_logits=config.num_experts > 0,
            )
            if config.num_experts > 0:
                logits, router_logits, selected_experts = model_output
            else:
                logits = model_output
            loss_sum = F.cross_entropy(
                logits.float(), tgt, ignore_index=LOSS_IGNORE_INDEX, reduction="sum"
            )
        # Packed batches contain different numbers of real targets. DDP averages
        # gradients across ranks, so compensate by world_size/global_tokens to
        # make the result exactly the global token-mean gradient.
        scaled_loss = loss_sum * (world_size / global_tokens)
        router_stats = None
        if config.num_experts > 0:
            router_aux_loss, router_stats = global_load_balancing_loss(
                router_logits,
                selected_experts,
                tgt != LOSS_IGNORE_INDEX,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
            )
            scaled_loss = scaled_loss + config.router_aux_loss_coef * router_aux_loss
        scaled_loss.backward()
        return loss_sum.detach(), real_tokens, global_tokens, router_stats

    # Val windows live on the GPU for the whole run (~1.2 MB each).  Validation
    # is rank 0 only and MUST bypass the DDP wrapper: otherwise its forward-time
    # buffer broadcasts collide with rank 1's next training/teardown collective.
    # Rank 1 safely waits at its next DDP collective while rank 0 evaluates.
    val_gpu = []
    for vin_w, vtg_w, vpos, vcu, vreal in val_windows:
        ids = torch.from_numpy(vin_w).to(device)
        tgt = torch.from_numpy(vtg_w).to(device)
        pos_t = torch.from_numpy(vpos).to(device)
        cu_t = torch.from_numpy(vcu).to(device)
        torch._dynamo.mark_dynamic(cu_t, 0)
        val_gpu.append((ids, tgt, pos_t, cu_t, vreal))
    if val_gpu:
        print(json.dumps({"event": "val_setup", "val_windows": len(val_gpu),
                          "val_interval_steps": val_interval}), flush=True)

    @torch.no_grad()
    def run_val() -> float:
        raw_model.eval()
        total_loss, total_toks = 0.0, 0
        for ids, tgt, pos_t, cu_t, real_tokens in val_gpu:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = raw_model(ids, pos_t, cu_t, global_max_seqlen)
                loss = F.cross_entropy(
                    logits.float(), tgt, ignore_index=LOSS_IGNORE_INDEX, reduction="sum"
                )
            total_loss += loss.float().item()
            total_toks += real_tokens
        raw_model.train()
        return total_loss / max(1, total_toks)

    # ---- resume ----
    # The packed-batch schedule is deterministic (data_seed + fingerprint below), so a
    # checkpoint only needs the per-rank step count to rejoin the exact stream.
    # Every rank writes/reads its OWN local copy (no shared fs across Sparks);
    # model/optimizer states are identical across ranks by DDP construction.
    fingerprint = {
        "train_path": args.train_path,
        "model_config": asdict(config),
        "selected_tokens": selected_tokens,
        "window": window,
        "packing_format": PACKING_FORMAT,
        "scheduled_batches": scheduled_batches,
        "ddp_filler_batches": ddp_filler_batches,
        "available_steps": available_steps,
        "real_tokens_used": training_real_tokens,
        "data_seed": data_seed,
        "world_size": world_size,
        "steps_total": steps_total,
        "lr_schedule_steps": lr_schedule_steps,
        "warmup_steps": warmup_steps,
        "val_interval_steps": val_interval,
    }

    def save_resumable(step: int):
        torch.save({
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "fingerprint": fingerprint,
        }, run_dir / f"ckpt_step{step:07d}.pt")
        if is_main:  # ready-to-load HF twin (model only)
            export_hf(raw_model, run_dir / f"hf_step{step:07d}")

    start_step = 0
    if args.resume:
        ckpts = sorted(run_dir.glob("ckpt_step*.pt"))
        ckpts = [p for p in ckpts if not p.stem.endswith("_unfused")]
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
            if ckpt["fingerprint"] != fingerprint:
                raise ValueError(f"resume fingerprint mismatch: ckpt {ckpt['fingerprint']} "
                                 f"vs current {fingerprint}")
            raw_model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt["step"]
            if is_main:
                print(json.dumps({"event": "resume", "from": str(ckpts[-1]),
                                  "step": start_step}), flush=True)
        elif is_main:
            print(json.dumps({"event": "resume", "from": None, "step": 0}), flush=True)
    if distributed:  # every rank must have found the same step
        agree = torch.tensor([start_step, -start_step], device=device, dtype=torch.float64)
        dist.all_reduce(agree, op=dist.ReduceOp.MIN)
        if int(agree[0].item()) != -int(agree[1].item()):
            raise RuntimeError(f"ranks disagree on resume step (rank {rank}: {start_step})")
    remaining = window_order[start_step:]

    # ---- prewarm (one static shape; both ranks in lockstep) ----
    model.train()
    if remaining:
        prewarm_start = time.perf_counter()
        warm_loss_sum, _, warm_global_tokens, warm_router_stats = forward_backward(
            remaining[0], step_global_tokens[start_step]
        )
        if distributed:
            dist.all_reduce(warm_loss_sum, op=dist.ReduceOp.SUM)
        warm_loss = warm_loss_sum / warm_global_tokens
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if is_main:
            prewarm_record = {
                "event": "prewarm",
                "elapsed_seconds": time.perf_counter() - prewarm_start,
                "warm_loss": float(warm_loss.float().item()),
            }
            if warm_router_stats is not None:
                prewarm_record["router_aux_loss"] = float(
                    warm_router_stats["aux_loss"].item()
                )
            print(json.dumps(prewarm_record), flush=True)

    # ---- train ----
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    local_total_tokens = sum(
        train_batches[w].real_tokens for w in window_order[:start_step] if w >= 0
    )
    global_total_tokens_t = torch.tensor(
        local_total_tokens, device=device, dtype=torch.int64
    )
    if distributed:
        dist.all_reduce(global_total_tokens_t, op=dist.ReduceOp.SUM)
    global_total_tokens = int(global_total_tokens_t.item())
    global_compute_tokens = start_step * window * world_size
    session_start_global_tokens = global_total_tokens
    session_start_compute_tokens = global_compute_tokens
    last_log_time = started_at
    last_log_global_tokens = global_total_tokens
    last_log_compute_tokens = global_compute_tokens
    last_val_loss = None
    last_router_stats = None
    cumulative_validation_seconds = 0.0
    metrics_file = (run_dir / "metrics.jsonl").open(
        "a" if start_step else "w", encoding="utf-8") if is_main else None

    for step, w in enumerate(remaining, start=start_step + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_sum, local_step_tokens, global_step_tokens, router_stats = forward_backward(
            w, step_global_tokens[step - 1]
        )
        last_router_stats = router_stats
        optimizer.step()
        scheduler.step()
        local_total_tokens += local_step_tokens
        global_total_tokens += global_step_tokens
        global_compute_tokens += window * world_size

        if step == 1 or step % args.log_every == 0 or step == steps_total:
            torch.cuda.synchronize()
            now = time.perf_counter()
            if distributed:  # exact token-weighted loss across ranks
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            loss = loss_sum / global_step_tokens
            if is_main:
                record = {
                    "step": step,
                    "total_steps": steps_total,
                    "loss": float(loss.float().item()),
                    "total_tokens": local_total_tokens,
                    "global_total_tokens": global_total_tokens,
                    "global_compute_tokens": global_compute_tokens,
                    "elapsed_seconds": now - started_at,
                    "training_elapsed_seconds":
                        now - started_at - cumulative_validation_seconds,
                    "window_tokens_per_second":
                        (global_compute_tokens - last_log_compute_tokens)
                        / max(1e-9, now - last_log_time),
                    "real_tokens_per_second":
                        (global_total_tokens - last_log_global_tokens)
                        / max(1e-9, now - last_log_time),
                    "global_tokens_per_second":
                        (global_total_tokens - session_start_global_tokens)
                        / max(1e-9, now - started_at),
                    "observed_packing_utilization":
                        (global_total_tokens - session_start_global_tokens)
                        / max(1, global_compute_tokens - session_start_compute_tokens),
                    "lr": optimizer.param_groups[0]["lr"],
                    "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
                }
                if router_stats is not None:
                    record.update({
                        "router_aux_loss": float(router_stats["aux_loss"].item()),
                        "router_assignment_min_frac": float(
                            router_stats["assignment_min_frac"].item()
                        ),
                        "router_assignment_max_frac": float(
                            router_stats["assignment_max_frac"].item()
                        ),
                        "router_assignment_cv": float(
                            router_stats["assignment_cv"].item()
                        ),
                        "router_entropy": float(router_stats["router_entropy"].item()),
                        "router_unused_experts": int(
                            router_stats["unused_experts"].item()
                        ),
                    })
                print(json.dumps(record), flush=True)
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                if wandb_run is not None:
                    # A single call keeps LM and router metrics on the same W&B
                    # history row and therefore on the same token x-axis.
                    wandb_run.log(wandb_train_metrics(record))
            last_log_time = now
            last_log_global_tokens = global_total_tokens
            last_log_compute_tokens = global_compute_tokens

        if val_gpu and (step % val_interval == 0 or step == steps_total):
            validation_started_at = time.perf_counter()
            v_loss = run_val()
            validation_finished_at = time.perf_counter()
            validation_seconds = validation_finished_at - validation_started_at
            cumulative_validation_seconds += validation_seconds
            last_val_loss = v_loss
            elapsed_at_validation = validation_finished_at - started_at
            vrec = {"event": "validation", "step": step, "total_steps": steps_total,
                    "val_loss": v_loss,
                    "elapsed_seconds": elapsed_at_validation,
                    "training_elapsed_seconds":
                        elapsed_at_validation - cumulative_validation_seconds,
                    "validation_seconds": validation_seconds,
                    "cumulative_validation_seconds": cumulative_validation_seconds}
            print(json.dumps(vrec), flush=True)
            metrics_file.write(json.dumps(vrec) + "\n")
            metrics_file.flush()
            if wandb_run is not None:
                wandb_run.log({"val/loss": v_loss, "train/step": step,
                               "train/tokens": global_total_tokens})

        if args.save_every and step % args.save_every == 0:
            save_resumable(step)  # every rank, to its own disk
            if is_main:
                print(json.dumps({"event": "ckpt", "step": step}), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at
    finished = {
        "event": "finished",
        "elapsed_seconds": elapsed,
        "training_elapsed_seconds": elapsed - cumulative_validation_seconds,
        "validation_elapsed_seconds": cumulative_validation_seconds,
        "total_tokens": local_total_tokens,
        "global_total_tokens": global_total_tokens,
        "global_compute_tokens": global_compute_tokens,
        "global_tokens_per_second":
            (global_total_tokens - session_start_global_tokens) / max(1e-9, elapsed),
        "global_compute_tokens_per_second":
            (global_compute_tokens - session_start_compute_tokens) / max(1e-9, elapsed),
        "packing_utilization": packing_utilization,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "world_size": world_size,
    }
    if last_val_loss is not None:
        finished["final_val_loss"] = last_val_loss
    if last_router_stats is not None:
        finished.update({
            "final_router_aux_loss": float(last_router_stats["aux_loss"].item()),
            "final_router_assignment_cv": float(
                last_router_stats["assignment_cv"].item()
            ),
            "final_router_entropy": float(last_router_stats["router_entropy"].item()),
            "final_router_unused_experts": int(
                last_router_stats["unused_experts"].item()
            ),
        })

    # Tear DDP down while every rank is at the same point, before rank 0 enters
    # checkpoint export or W&B finalization.  Previously rank 1 waited in this
    # barrier while rank 0 called wandb.finish(); a wedged wandb-core therefore
    # kept both torchrun jobs (and both GPUs) alive forever.
    if distributed:
        print(json.dumps({"event": "ddp_teardown_start", "rank": rank}), flush=True)
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()
        print(json.dumps({"event": "ddp_teardown_complete", "rank": rank}), flush=True)

    if is_main:
        metrics_file.close()
        if args.save_final:
            torch.save(raw_model.state_dict(), run_dir / "model_final.pt")
            export_hf(raw_model, run_dir / "hf")
        (run_dir / "summary.json").write_text(json.dumps(finished, indent=2), encoding="utf-8")
        print(json.dumps(finished), flush=True)
        if wandb_run is not None:
            wandb_run.log({"final/tokens_per_sec": finished["global_tokens_per_second"],
                           "train/tokens": finished["global_total_tokens"]})
            if not finish_wandb_with_timeout(
                    wandb_run, args.wandb_finish_timeout_seconds):
                # All artifacts are durable and DDP is already gone.  Bypass
                # W&B's atexit hook/non-daemon threads so torchrun sees a clean
                # successful worker exit rather than another indefinite wait.
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)


if __name__ == "__main__":
    main()
