"""sparkgpt: single-file DDP muP byte-level LM pretrainer for DGX Spark (GB10).

One file, one recipe. Qwen3-shaped decoder over raw UTF-8 bytes (vocab 259:
256 bytes + BOS/EOS/PAD), trained with:

  * Varlen sequence packing -- every batch is exactly --tokens-per-batch loss
    tokens (one static shape, zero padding). Attention is flash-varlen with
    cu_seqlens at document boundaries: block-diagonal, NO inter-document
    attention, RoPE restarts per segment. Docs cut by a window boundary
    continue as fresh segments (truncated context, never cross-window).
  * muP width scaling (always on). Base hparams are defined at width
    --mup-base-dim (256) and transfer: hidden init 0.02*sqrt(base/width),
    residual writers (o/down) get an extra 1/sqrt(2L), embedding std constant,
    lm_head UNTIED + zero-init (init loss = ln 259), AdamW lm_head lr scaled
    by base/width. Muon lr needs NO width scaling (the orthogonalized update
    with the aspect-ratio factor is width-invariant). Verified 2026-07-01:
    optimal muon-lr 4e-3 at widths 256/512/1024. Keep head_dim fixed (128)
    across the width family.
  * Muon/AdamW hybrid (always). 2-D body matrices -> Muon (NO weight decay);
    embedding, lm_head, norm gains -> AdamW.
  * DDP across Spark nodes via torchrun; windows are sharded round-robin by
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
  # single node (0.6B shape runs ~12.5k tok/s on one GB10):
  python train.py --run-name <name> --train-path lang_data/fineweb_2b.jsonl \
      --target-tokens 2000000000 --save-final
  # two nodes (run on each node with its own --node-rank; rank 0 is master):
  torchrun --nnodes 2 --node-rank <0|1> --nproc-per-node 1 \
      --master-addr <node0-ip> --master-port 29500 train.py ...

Hard-won GB10 (sm_121) facts -- do not relearn these:
  * compile mode "default" only: reduce-overhead's cudagraph pools OOM unified
    memory, and max-autotune is ~6.5% SLOWER than default (triton < cuBLAS).
  * fp8 is a net loss at dim 1024 (dynamic-scaling casts are bandwidth-bound;
    273 GB/s). Revisit at dim >= 2048.
  * torch.compile traps: max_seqlen must be a CONSTANT python int and
    cu_seqlens' varying length must be mark_dynamic'd, else a silent
    recompile-limit eager fallback costs 2x throughput and +30 GB.
  * muon-lr 1e-2 (the old 20M-model default) diverges at 440M without muP.

Checkpoints: --save-final writes model_final.pt (native fused layout) plus a
READY-TO-LOAD HF directory checkpoints/<run>/hf/ (stock Qwen3ForCausalLM
config + fp32 safetensors + byte tokenizer as a plain tokenizer.json -- no
custom code, no trust_remote_code; also loads in mlx_lm as qwen3). Periodic
--save-every checkpoints get an hf_step<N>/ twin on rank 0. RoPE is standard
rotate-half throughout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return out.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


@torch._dynamo.disable
def _varlen_attention(q, k, v, cu_seqlens, max_seqlen):
    """Eager island: block-diagonal causal attention over packed segments.
    cu_seqlens has a data-dependent LENGTH (docs per window) and must stay
    outside the compiled graph to avoid per-window recompiles."""
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


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, cu_seqlens, max_seqlen)
        return h + self.mlp(self.post_attention_layernorm(h))


class ByteLM(nn.Module):
    """Qwen3-shaped byte LM, flat packed (total_tokens,) layout, fused
    qkv/gate_up, untied lm_head (muP)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers))
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

    def forward(self, input_ids, position_ids, cu_seqlens, max_seqlen):
        x = self.embed_tokens(input_ids)
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        for layer in self.layers:
            x = layer(x, cos, sin, cu_seqlens, max_seqlen)
        return self.lm_head(self.norm(x))


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
    """Write a ready-to-load HF model directory: stock Qwen3ForCausalLM config
    (the arch matches exactly; standard rotate-half RoPE) + fp32 safetensors +
    byte tokenizer. Loads with AutoModelForCausalLM / AutoTokenizer, no custom
    code. Also consumable by mlx_lm as model_type qwen3."""
    from safetensors.torch import save_file

    cfg = model.config
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        (k if k == "lm_head.weight" else "model." + k): v.contiguous().to(torch.float32)
        for k, v in export_unfused_state_dict(model).items()
    }
    save_file(weights, str(out_dir / "model.safetensors"),
              metadata={"format": "pt"})
    (out_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
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
    }, indent=2), encoding="utf-8")
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
def apply_mup_init(model: ByteLM, *, base_dim: int, emb_std: float, hidden_std: float) -> dict:
    cfg = model.config
    m = cfg.hidden_size / base_dim
    std_in = hidden_std / math.sqrt(m)
    std_writer = std_in / math.sqrt(2 * cfg.num_hidden_layers)
    with torch.no_grad():
        model.embed_tokens.weight.normal_(0.0, emb_std)
        model.lm_head.weight.zero_()
        for block in model.layers:
            block.self_attn.qkv_proj.weight.normal_(0.0, std_in)
            block.mlp.gate_up_proj.weight.normal_(0.0, std_in)
            block.self_attn.o_proj.weight.normal_(0.0, std_writer)
            block.mlp.down_proj.weight.normal_(0.0, std_writer)
    return {"width_mult": m, "hidden_init_std": std_in,
            "residual_writer_init_std": std_writer, "emb_init_std": emb_std,
            "lm_head_init": "zero"}


# --------------------------------------------------------------------------- #
# Muon / AdamW hybrid optimizer
# --------------------------------------------------------------------------- #
def _newtonschulz5_batched(X: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize a (B, r, c) stack of same-shape matrices (one bmm chain
    per iteration -- the step is kernel-launch-bound on GB10 otherwise)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    transpose_needed = X.shape[-2] > X.shape[-1]
    if transpose_needed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)
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
        torch._foreach_mul_(vs, momentum)
        torch._foreach_add_(vs, grads, alpha=1 - momentum)
        if group["nesterov"]:
            updates = torch._foreach_mul(grads, 1 - momentum)
            torch._foreach_add_(updates, vs, alpha=momentum)
        else:
            updates = [v.clone() for v in vs]

        by_shape: dict[tuple[int, int], list] = {}
        for p, u in zip(params, updates):
            by_shape.setdefault(tuple(u.shape), []).append((p, u))
        for (rows, cols), items in by_shape.items():
            # NS in bf16 (fp32 matmul is slow on GB10; NS is approximate anyway)
            stacked = torch.stack([u for _, u in items]).bfloat16()
            ortho = _newtonschulz5_batched(stacked, steps=group["ns_steps"])
            # aspect-ratio factor; also what makes Muon lr width-invariant
            lr = group["lr"] * max(1, rows / cols) ** 0.5
            torch._foreach_add_(
                [p for p, _ in items],
                [o.to(p.dtype) for (p, _), o in zip(items, ortho.unbind(0))],
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
    muon_params, adamw_const, adamw_head = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lm_head" in name:
            adamw_head.append(param)
        elif param.ndim != 2 or "embed" in name:
            adamw_const.append(param)
        else:
            muon_params.append(param)
    optimizer = MuonAdamWHybrid([
        {"params": muon_params, "use_muon": True, "lr": args.muon_lr,
         "momentum": args.muon_momentum, "weight_decay": 0.0},
        {"params": adamw_const, "use_muon": False, "lr": args.adamw_lr,
         "weight_decay": args.weight_decay},
        {"params": adamw_head, "use_muon": False, "lr": args.adamw_lr * head_mult,
         "weight_decay": args.weight_decay},
    ])
    return optimizer, {
        "mode": "mup_muon_hybrid",
        "head_lr_mult": head_mult,
        "muon_param_count": sum(p.numel() for p in muon_params),
        "adamw_param_count": sum(p.numel() for p in adamw_const + adamw_head),
    }


# --------------------------------------------------------------------------- #
# Data: compact byte stream -> per-doc pair streams -> fixed windows
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


def window_metadata(bounds: np.ndarray, w: int, window: int):
    """cu_seqlens / position_ids for window w. Segments = doc boundaries plus
    window edges; a doc cut by the edge restarts as a fresh segment."""
    lo, hi = w * window, (w + 1) * window
    i0 = np.searchsorted(bounds, lo, side="right")
    i1 = np.searchsorted(bounds, hi, side="left")
    cu = np.empty(i1 - i0 + 2, dtype=np.int32)
    cu[0] = 0
    cu[1:-1] = bounds[i0:i1] - lo
    cu[-1] = hi - lo
    seg_lens = np.diff(cu)
    pos = np.arange(window, dtype=np.int64) - np.repeat(cu[:-1].astype(np.int64), seg_lens)
    return cu, pos


def cosine_with_warmup(step, *, warmup_steps, total_steps, min_lr_ratio):
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = min(1.0, float(step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="sparkgpt: DDP muP byte-level packed pretrainer.")
    p.add_argument("--train-path", default="lang_data/fineweb_1b.jsonl")
    p.add_argument("--run-name", default="sparkgpt")
    p.add_argument("--target-tokens", type=int, default=1_000_000_000,
                   help="loss-token budget (global, before rank sharding); default = 1B, "
                        "i.e. Chinchilla-optimal for the default 50M model")
    # model (defaults = 50M; see docstring for the 0.6B flag set)
    p.add_argument("--model-layers", type=int, default=16)
    p.add_argument("--model-dim", type=int, default=512)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--intermediate-size", type=int, default=1536)
    p.add_argument("--rope-theta", type=float, default=1_000_000.0)
    p.add_argument("--max-position-embeddings", type=int, default=4096)
    # muP (base hparams defined at --mup-base-dim; verified transfer to 1024)
    p.add_argument("--mup-base-dim", type=int, default=256)
    p.add_argument("--mup-emb-std", type=float, default=0.02)
    p.add_argument("--mup-hidden-std", type=float, default=0.02)
    # optimization (4e-3 verified optimal across widths, 2026-07-01)
    p.add_argument("--muon-lr", type=float, default=4e-3)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--adamw-lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW groups only")
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--lr-decay-factor", type=float, default=0.1)
    # batching / runtime
    p.add_argument("--tokens-per-batch", type=int, default=49152,
                   help="window size per rank per step (GB10 memory optimum)")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=None, help="default: --seed")
    p.add_argument("--log-every", type=int, default=50)
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.attention_heads % args.kv_heads != 0:
        raise ValueError("--attention-heads must be divisible by --kv-heads")
    data_seed = args.data_seed if args.data_seed is not None else args.seed

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    is_main = rank == 0
    if not torch.cuda.is_available():
        raise RuntimeError("sparkgpt requires CUDA (flash_attn varlen)")
    if distributed:
        torch.cuda.set_device(0)  # one GPU per Spark node
        dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
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
    total_pairs = int(bounds[-1])
    num_windows = total_pairs // window
    max_doc_pairs = int(np.diff(bounds).max())
    global_max_seqlen = int(min(max_doc_pairs, window))  # CONSTANT (compile trap)
    if global_max_seqlen > args.max_position_embeddings:
        raise ValueError(f"longest packed segment ({global_max_seqlen}) exceeds "
                         f"--max-position-embeddings ({args.max_position_embeddings})")
    window_order = list(range(num_windows))
    random.Random(data_seed + 1).shuffle(window_order)
    if distributed:  # round-robin shard; every rank gets the same step count
        usable = num_windows - num_windows % world_size
        window_order = window_order[rank:usable:world_size]
    steps_total = len(window_order)
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
    )
    model = ByteLM(config).to(device)
    mup_summary = apply_mup_init(model, base_dim=args.mup_base_dim,
                                 emb_std=args.mup_emb_std, hidden_std=args.mup_hidden_std)
    raw_model = model  # for checkpointing
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])
    if args.compile:
        model = torch.compile(model, mode="default", dynamic=False)
    optimizer, optimizer_summary = build_optimizer(model, args)
    warmup_steps = max(10, math.ceil(args.warmup_frac * steps_total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup(
            step, warmup_steps=warmup_steps, total_steps=steps_total,
            min_lr_ratio=args.lr_decay_factor),
    )

    run_dir = Path("checkpoints") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    param_count = sum(p.numel() for p in raw_model.parameters())
    config_payload = {
        "args": vars(args),
        "model_config": asdict(config),
        "parameter_count": param_count,
        "mup": mup_summary,
        "optimizer": optimizer_summary,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "selected_docs": num_docs,
        "selected_loss_tokens": selected_tokens,
        "window_tokens": window,
        "steps_per_rank": steps_total,
        "warmup_steps": warmup_steps,
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
        lo = w * window
        cu, pos = window_metadata(bounds, w, window)
        ids = torch.from_numpy(inputs_np[lo : lo + window].astype(np.int64)).to(device, non_blocking=True)
        tgt = torch.from_numpy(targets_np[lo : lo + window].astype(np.int64)).to(device, non_blocking=True)
        pos_t = torch.from_numpy(pos).to(device, non_blocking=True)
        cu_t = torch.from_numpy(cu).to(device, non_blocking=True)
        torch._dynamo.mark_dynamic(cu_t, 0)  # varying segment count (compile trap)
        return ids, tgt, pos_t, cu_t

    def forward_backward(w: int):
        ids, tgt, pos_t, cu_t = batch_for(w)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(ids, pos_t, cu_t, global_max_seqlen)
            loss = F.cross_entropy(logits.float(), tgt)
        loss.backward()
        return loss.detach()

    # ---- resume ----
    # The window schedule is deterministic (data_seed + fingerprint below), so a
    # checkpoint only needs the per-rank step count to rejoin the exact stream.
    # Every rank writes/reads its OWN local copy (no shared fs across Sparks);
    # model/optimizer states are identical across ranks by DDP construction.
    fingerprint = {
        "train_path": args.train_path,
        "selected_tokens": selected_tokens,
        "window": window,
        "data_seed": data_seed,
        "world_size": world_size,
        "steps_total": steps_total,
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
        warm_loss = forward_backward(remaining[0])
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if is_main:
            print(json.dumps({"event": "prewarm",
                              "elapsed_seconds": time.perf_counter() - prewarm_start,
                              "warm_loss": float(warm_loss.float().item())}), flush=True)

    # ---- train ----
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    total_tokens = start_step * window  # includes pre-resume tokens
    session_start_tokens = total_tokens  # rates count this session only
    last_log_time, last_log_tokens = started_at, total_tokens
    metrics_file = (run_dir / "metrics.jsonl").open(
        "a" if start_step else "w", encoding="utf-8") if is_main else None

    for step, w in enumerate(remaining, start=start_step + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = forward_backward(w)
        optimizer.step()
        scheduler.step()
        total_tokens += window

        if step == 1 or step % args.log_every == 0 or step == steps_total:
            torch.cuda.synchronize()
            now = time.perf_counter()
            global_tokens = total_tokens * world_size  # exact: fixed window size
            if distributed:  # sync loss across ranks for a stable log signal
                loss = loss.clone()
                dist.all_reduce(loss)
                loss /= world_size
            if is_main:
                record = {
                    "step": step,
                    "total_steps": steps_total,
                    "loss": float(loss.float().item()),
                    "total_tokens": total_tokens,
                    "global_total_tokens": global_tokens,
                    "elapsed_seconds": now - started_at,
                    "window_tokens_per_second": (total_tokens - last_log_tokens) * world_size
                    / max(1e-9, now - last_log_time),
                    "global_tokens_per_second": (total_tokens - session_start_tokens)
                    * world_size / max(1e-9, now - started_at),
                    "lr": optimizer.param_groups[0]["lr"],
                    "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
                }
                print(json.dumps(record), flush=True)
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": record["loss"],
                        "train/tokens_per_sec": record["window_tokens_per_second"],
                        "train/lr": record["lr"],
                        "train/peak_cuda_memory_gb": record["peak_cuda_memory_gb"],
                        "train/step": step,
                        "train/tokens": global_tokens,
                    })
            last_log_time, last_log_tokens = now, total_tokens

        if args.save_every and step % args.save_every == 0:
            save_resumable(step)  # every rank, to its own disk
            if is_main:
                print(json.dumps({"event": "ckpt", "step": step}), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at
    finished = {
        "event": "finished",
        "elapsed_seconds": elapsed,
        "total_tokens": total_tokens,
        "global_total_tokens": total_tokens * world_size,
        "global_tokens_per_second": (total_tokens - session_start_tokens)
        * world_size / max(1e-9, elapsed),
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "world_size": world_size,
    }
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
            wandb_run.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
