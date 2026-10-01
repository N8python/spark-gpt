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
  # single node (0.6B shape runs ~15.4k tok/s on one GB10):
  python train.py --run-name <name> --train-path lang_data/fineweb_2b.jsonl \
      --target-tokens 2000000000 --save-final \
      --val-path lang_data/fineweb_10m_val_fixed_seed0.jsonl
  # two nodes (run on each node with its own --node-rank; rank 0 is master):
  torchrun --nnodes 2 --node-rank <0|1> --nproc-per-node 1 \
      --master-addr <node0-ip> --master-port 29500 train.py ...

Hard-won GB10 (sm_121) facts -- do not relearn these:
  * compile mode "default" only: reduce-overhead's cudagraph pools OOM unified
    memory. mode "max-autotune" never tried Triton GEMMs here at all: inductor's
    is_big_gpu() gate (>= 68 SMs) silently disables its GEMM templates on the
    48-SM GB10. With the gate bypassed (--autotune-gemm, default on) Triton
    beats cuBLAS on the output-heavy shapes (+4.3% dense); cuBLAS addmm with a
    matrix bias is 1.8-2.5x slower than mm on sm_121, so residual-add-after-
    GEMM is exactly where the Triton template wins most.
  * torch._grouped_mm on sm_121 is the host-synchronizing per-expert cuBLAS
    fallback; the Triton grouped GEMMs in this file (device offsets, fused
    SwiGLU epilogue, natural-layout weight gradient) are 13-31% faster per
    expert GEMM and free the CPU. Inductor's own grouped template (TMA loads)
    loses to aten on the 2d x 3d expert shapes; its prologue-fusion heuristics
    refuse the SwiGLU backward (reads > writes, fp32 math), so only the
    forward fusion is reachable through the compiler.
  * fp8 is a net loss at dim 1024 (dynamic-scaling casts are bandwidth-bound;
    273 GB/s). Revisit at dim >= 2048.
  * Embedding is not autocast by PyTorch. Cast its output to the active
    autocast dtype once or every residual add promotes back to fp32; keeping
    the 0.6B residual stream in bf16 improved throughput by ~9% on GB10.
  * torch.compile traps: max_seqlen must be a CONSTANT python int and
    cu_seqlens' varying length must be mark_dynamic'd, else a silent
    recompile-limit eager fallback costs 2x throughput and +30 GB.
  * muon-lr 1e-2 (the old 20M-model default) diverges at 440M without muP.
  * cuBLAS batched bf16 bmm/baddbmm is mis-tuned on sm_121 for the Muon
    Newton-Schulz stacks (32x32 wmma kernels, ~19-34 TFLOP/s, and baddbmm
    copies its bias matrix into the output first); the hand-written Triton
    batched GEMM in _newtonschulz5_batched runs them at 51-64 TFLOP/s with
    bit-identical results. Keep its tile configs shape-derived (never
    autotuned) so DDP ranks stay bit-identical.
  * flash-attn >= 2.7 varlen is a torch custom op: keep it INSIDE the
    compiled graph (one graph per fwd/bwd), with cu_seqlens mark_dynamic'd.
  * Inductor launches user Triton kernels with runtime Python floats typed
    fp64 (Triton's own launcher uses fp32): a float argument silently turns
    the math it touches into fp64 (~1/64 rate on GB10). Make such arguments
    tl.constexpr. At dim 512 everything but attention is bandwidth-bound, so
    the win is in fewer passes, not faster ones: the q/k norm + RoPE path was
    9 inductor kernels moving ~2x its minimum bytes (_QKNormRoPEAttention).

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

try:  # ships with every CUDA torch wheel; only used by the Muon optimizer
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only test environments
    triton = None

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

    def forward(self, x, rope, cu_seqlens, max_seqlen):
        # rope = (cos, sin, cos_table, sin_table, position_ids): the per-token
        # rows for the unfused path, the fp32 tables + positions for the fused one.
        cos, sin, cos_table, sin_table, position_ids = rope
        total = x.shape[0]
        qkv = self.qkv_proj(x)
        if _use_fused_qk_rope(qkv, self.head_dim):
            out = _QKNormRoPEAttention.apply(
                qkv, self.q_norm.weight, self.k_norm.weight, cos_table, sin_table,
                position_ids, cu_seqlens, max_seqlen, self.n_heads, self.n_kv_heads,
                self.q_norm.eps,
            )
            return self.o_proj(out.reshape(total, self.n_heads * self.head_dim))
        q, k, v = qkv.split(
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
        # fp8 MLP state (see _FP8MLPBlock); per-rank, not checkpointed
        self.register_buffer("fp8_scale", torch.ones(5), persistent=False)
        self.register_buffer("fp8_amax", torch.zeros(5), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _use_fused_swiglu(x):
            weight = self.gate_up_proj.weight.to(dtype=x.dtype).unsqueeze(0)
            offs = torch.full((1,), x.shape[0], device=x.device, dtype=torch.int32)
            return self.down_proj(_FusedSwiGLU.apply(x, weight, offs))
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# --------------------------------------------------------------------------- #
# Fused qk-norm + RoPE around flash-attn varlen (Triton)
# --------------------------------------------------------------------------- #
FUSED_QK_ROPE = True  # --no-fused-qk-rope falls back to separate norm/RoPE ops


def _use_fused_qk_rope(qkv: torch.Tensor, head_dim: int) -> bool:
    # the kernels tile each head as two power-of-two halves
    return (FUSED_QK_ROPE and triton is not None and qkv.is_cuda
            and qkv.dtype == torch.bfloat16 and head_dim >= 32
            and head_dim & (head_dim - 1) == 0)


@triton.jit
def _qk_norm_rope_fwd_kernel(
    QKV, QW, KW, COS, SIN, POS, QOUT, KOUT, RSTD,
    T, stride_qkv,
    EPS: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """q/k heads of one token block: per-head RMSNorm (fp32) then rotate-half
    RoPE, one bf16 rounding at the store (correctly rounded vs an fp64
    reference). Each head is handled as its two contiguous halves so
    rotate-half needs no permuted (unvectorizable) loads. EPS is a constexpr:
    inductor passes runtime Python floats as fp64, which would promote the
    rstd math to fp64."""
    HALF: tl.constexpr = D // 2
    rt = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    rh = tl.arange(0, HALF)
    t_mask = rt < T
    m = t_mask[:, None]
    pos = tl.load(POS + rt, mask=t_mask, other=0)
    table = pos[:, None] * D + rh[None, :]
    cos1 = tl.load(COS + table, mask=m, other=0.0)
    cos2 = tl.load(COS + table + HALF, mask=m, other=0.0)
    sin1 = tl.load(SIN + table, mask=m, other=0.0)
    sin2 = tl.load(SIN + table + HALF, mask=m, other=0.0)
    for h in tl.static_range(HQ + HKV):
        if h < HQ:
            w1 = tl.load(QW + rh)
            w2 = tl.load(QW + HALF + rh)
            out_ptrs = QOUT + rt[:, None] * (HQ * D) + h * D + rh[None, :]
        else:
            w1 = tl.load(KW + rh)
            w2 = tl.load(KW + HALF + rh)
            out_ptrs = KOUT + rt[:, None] * (HKV * D) + (h - HQ) * D + rh[None, :]
        row = QKV + rt[:, None] * stride_qkv + h * D + rh[None, :]
        x1 = tl.load(row, mask=m, other=0.0).to(tl.float32)
        x2 = tl.load(row + HALF, mask=m, other=0.0).to(tl.float32)
        rstd = tl.rsqrt((tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)) / D + EPS)
        n1 = x1 * rstd[:, None] * w1[None, :]
        n2 = x2 * rstd[:, None] * w2[None, :]
        tl.store(out_ptrs, (n1 * cos1 - n2 * sin1).to(tl.bfloat16), mask=m)
        tl.store(out_ptrs + HALF, (n2 * cos2 + n1 * sin2).to(tl.bfloat16), mask=m)
        tl.store(RSTD + rt * (HQ + HKV) + h, rstd, mask=t_mask)


@triton.jit
def _qk_norm_rope_bwd_kernel(
    DQ, DK, QKV, QW, KW, COS, SIN, POS, RSTD, DQKV, DW,
    T, stride_qkv, stride_dqkv,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """RoPE backward, then RMSNorm backward, for the q/k heads of one token
    block, written straight into the dq/dk columns of dqkv. Per-program partial
    norm-weight gradients go to DW[pid, 0 (q) | 1 (k), :] and are summed on the
    host side (deterministic, no atomics). The RoPE gradient is rounded to bf16
    before the norm backward, as the unfused graph materializes it in bf16."""
    HALF: tl.constexpr = D // 2
    pid = tl.program_id(0)
    rt = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    rh = tl.arange(0, HALF)
    t_mask = rt < T
    m = t_mask[:, None]
    pos = tl.load(POS + rt, mask=t_mask, other=0)
    table = pos[:, None] * D + rh[None, :]
    cos1 = tl.load(COS + table, mask=m, other=0.0)
    cos2 = tl.load(COS + table + HALF, mask=m, other=0.0)
    sin1 = tl.load(SIN + table, mask=m, other=0.0)
    sin2 = tl.load(SIN + table + HALF, mask=m, other=0.0)
    dwq1 = tl.zeros((HALF,), dtype=tl.float32)
    dwq2 = tl.zeros((HALF,), dtype=tl.float32)
    dwk1 = tl.zeros((HALF,), dtype=tl.float32)
    dwk2 = tl.zeros((HALF,), dtype=tl.float32)
    for h in tl.static_range(HQ + HKV):
        if h < HQ:
            w1 = tl.load(QW + rh)
            w2 = tl.load(QW + HALF + rh)
            g_row = DQ + rt[:, None] * (HQ * D) + h * D + rh[None, :]
        else:
            w1 = tl.load(KW + rh)
            w2 = tl.load(KW + HALF + rh)
            g_row = DK + rt[:, None] * (HKV * D) + (h - HQ) * D + rh[None, :]
        g1 = tl.load(g_row, mask=m, other=0.0).to(tl.float32)
        g2 = tl.load(g_row + HALF, mask=m, other=0.0).to(tl.float32)
        dn1 = (g1 * cos1 + g2 * sin2).to(tl.bfloat16).to(tl.float32)
        dn2 = (g2 * cos2 - g1 * sin1).to(tl.bfloat16).to(tl.float32)
        row = QKV + rt[:, None] * stride_qkv + h * D + rh[None, :]
        x1 = tl.load(row, mask=m, other=0.0).to(tl.float32)
        x2 = tl.load(row + HALF, mask=m, other=0.0).to(tl.float32)
        r = tl.load(RSTD + rt * (HQ + HKV) + h, mask=t_mask, other=0.0)
        dw1 = dn1 * w1[None, :]
        dw2 = dn2 * w2[None, :]
        s = tl.sum(dw1 * x1, 1) + tl.sum(dw2 * x2, 1)
        if h < HQ:
            dwq1 += tl.sum(dn1 * (x1 * r[:, None]), 0)
            dwq2 += tl.sum(dn2 * (x2 * r[:, None]), 0)
        else:
            dwk1 += tl.sum(dn1 * (x1 * r[:, None]), 0)
            dwk2 += tl.sum(dn2 * (x2 * r[:, None]), 0)
        coef = (s * -0.5 * (r * r * r) * (2.0 / D))[:, None]
        out_ptrs = DQKV + rt[:, None] * stride_dqkv + h * D + rh[None, :]
        tl.store(out_ptrs, (coef * x1 + dw1 * r[:, None]).to(tl.bfloat16), mask=m)
        tl.store(out_ptrs + HALF, (coef * x2 + dw2 * r[:, None]).to(tl.bfloat16), mask=m)
    base = DW + pid * (2 * D) + rh
    tl.store(base, dwq1)
    tl.store(base + HALF, dwq2)
    tl.store(base + D, dwk1)
    tl.store(base + D + HALF, dwk2)


QK_ROPE_BLOCK_T = 32


class _QKNormRoPEAttention(torch.autograd.Function):
    """qkv (T, (Hq + 2 Hkv) D) -> per-head q/k RMSNorm -> RoPE -> causal varlen
    flash-attn. Replaces two forward and seven backward elementwise kernels:
    pre-norm q/k are read straight out of qkv (no saved copies), the norm and
    RoPE backward run in one pass, and flash-attn writes dv directly into its
    slice of dqkv. Norm weights stay fp32, as in the compiled unfused path."""

    @staticmethod
    def forward(ctx, qkv, q_weight, k_weight, cos_table, sin_table, position_ids,
                cu_seqlens, max_seqlen, n_heads, n_kv_heads, eps):
        total = qkv.shape[0]
        head_dim = cos_table.shape[1]
        q = torch.empty(total, n_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        k = torch.empty(total, n_kv_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        rstd = torch.empty(total, n_heads + n_kv_heads, device=qkv.device, dtype=torch.float32)
        grid = (triton.cdiv(total, QK_ROPE_BLOCK_T),)
        _qk_norm_rope_fwd_kernel[grid](
            qkv, q_weight, k_weight, cos_table, sin_table, position_ids, q, k, rstd,
            total, qkv.stride(0),
            EPS=eps, HQ=n_heads, HKV=n_kv_heads, D=head_dim, BLOCK_T=QK_ROPE_BLOCK_T,
            num_warps=4,
        )
        v = qkv[:, (n_heads + n_kv_heads) * head_dim:].view(total, n_kv_heads, head_dim)
        scale = head_dim ** -0.5
        out, lse, _, rng_state = torch.ops.flash_attn._flash_attn_varlen_forward(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, 0.0, scale, True,
        )
        ctx.save_for_backward(qkv, q_weight, k_weight, cos_table, sin_table, position_ids,
                              cu_seqlens, q, k, out, lse, rstd, rng_state)
        ctx.shape = (n_heads, n_kv_heads, head_dim, max_seqlen, scale)
        return out

    @staticmethod
    def backward(ctx, dout):
        (qkv, q_weight, k_weight, cos_table, sin_table, position_ids,
         cu_seqlens, q, k, out, lse, rstd, rng_state) = ctx.saved_tensors
        n_heads, n_kv_heads, head_dim, max_seqlen, scale = ctx.shape
        total = qkv.shape[0]
        dqkv = torch.empty_like(qkv)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        v_cols = slice((n_heads + n_kv_heads) * head_dim, None)
        v = qkv[:, v_cols].view(total, n_kv_heads, head_dim)
        dv = dqkv[:, v_cols].view(total, n_kv_heads, head_dim)  # flash-attn writes dv in place
        torch.ops.flash_attn._flash_attn_varlen_backward(
            dout, q, k, v, out, lse, dq, dk, dv,
            cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, 0.0, scale, True,
            -1, -1, 0.0, None, False, rng_state,
        )
        grid = (triton.cdiv(total, QK_ROPE_BLOCK_T),)
        dw = torch.empty(grid[0], 2, head_dim, device=qkv.device, dtype=torch.float32)
        _qk_norm_rope_bwd_kernel[grid](
            dq, dk, qkv, q_weight, k_weight, cos_table, sin_table, position_ids, rstd,
            dqkv, dw, total, qkv.stride(0), dqkv.stride(0),
            HQ=n_heads, HKV=n_kv_heads, D=head_dim, BLOCK_T=QK_ROPE_BLOCK_T,
            num_warps=4,
        )
        dw = dw.sum(0)
        return (dqkv, dw[0], dw[1], None, None, None, None, None, None, None, None)


# --------------------------------------------------------------------------- #
# fp8 MLP sub-block (Triton): RMSNorm -> SwiGLU MLP -> residual, fp8 storage
# --------------------------------------------------------------------------- #
FP8_MLP = False  # --fp8-mlp: dense MLPs and MoE experts keep activations/grads in fp8 while training

# Per-layer fp8 state (MLP buffers fp8_scale / fp8_amax, one slot per tensor):
#   0 = x (normed MLP input, e4m3), 1 = gu (gate/up pre-activation, e4m3),
#   2 = h (SwiGLU output, e4m3),    3 = d_gu (its gradient, e5m2),
#   4 = dy (gradient of the sub-block output, e5m2; quantized in registers).
# Scales are delayed: each step quantizes with powers of two derived from the
# previous step's amax (one binade of headroom); the kernels record this step's
# amax of the UNquantized values with atomic max. fp8_update_scales() rolls them.
FP8_SITE_MAX = (448.0, 448.0, 448.0, 57344.0, 57344.0)
FP8_HEADROOM_BINADES = 1


def _use_fp8_mlp(block, h: torch.Tensor) -> bool:
    if not (FP8_MLP and block.training and not block.is_sparse and triton is not None
            and h.is_cuda and h.dtype == torch.bfloat16):
        return False  # checked first: sparse blocks' mlp has no down_proj
    d, inter = h.shape[1], block.mlp.down_proj.weight.shape[1]
    # the norm kernels tile a whole row (power-of-two hidden size); the GEMM grids
    # assume 128-divisible hidden and intermediate sizes
    return d % 128 == 0 and d & (d - 1) == 0 and inter % 128 == 0


@torch.no_grad()
def fp8_update_scales(model: nn.Module) -> None:
    """Roll every fp8 MLP / expert block's recorded amax into next step's scales."""
    for m in model.modules():
        if isinstance(m, (MLP, Experts)):
            fmax = m.fp8_scale.new_tensor(FP8_SITE_MAX)
            new = torch.exp2(torch.floor(torch.log2(fmax / m.fp8_amax.clamp(min=1e-30)))
                             - FP8_HEADROOM_BINADES)
            m.fp8_scale.copy_(torch.where(m.fp8_amax > 0, new, m.fp8_scale))
            m.fp8_amax.zero_()


def _fp8_quantize_weight(w: torch.Tensor):
    """Per-tensor e4m3 copy of an fp32 master weight (just-in-time pow2 scale)."""
    q = torch.exp2(torch.floor(torch.log2(448.0 / w.detach().abs().amax().clamp(min=1e-30))))
    return (w.detach() * q).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), q


@triton.jit
def _fp8_norm_quant_kernel(H, NW, X8, X8T, RSTD, SC, AMAX, T,
                           EPS: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr):
    """x8 = e4m3(rmsnorm(h) * w * scale_x), written row-major for the forward GEMM and
    transposed (token-contiguous) for the weight gradient, whose fp8 MMA wants its
    reduction dim contiguous; rstd kept for the backward."""
    rt = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    rd = tl.arange(0, D)
    m = (rt < T)[:, None]
    h = tl.load(H + rt[:, None] * D + rd[None, :], mask=m, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(h * h, 1) / D + EPS)
    n = h * rstd[:, None] * tl.load(NW + rd)[None, :]
    q = tl.load(SC)
    x8 = tl.clamp(n * q, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(X8 + rt[:, None] * D + rd[None, :], x8, mask=m)
    tl.store(X8T + rd[None, :] * T + rt[:, None], x8, mask=m)
    tl.store(RSTD + rt, rstd, mask=rt < T)
    tl.atomic_max(AMAX, tl.max(tl.abs(n)))


@triton.jit
def _fp8_swiglu_fwd_kernel(X8, W8, SC, WS, GU8, H8, AMAX, T, K, I,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                           GROUP_M: tl.constexpr):
    """gu = x8 @ W8^T (fp8 MMA, fp32 acc); g, u rounded to bf16 as in the bf16 path;
    h = silu(g) * u; gu and h stored e4m3."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(T, BLOCK_M)
    num_n = tl.cdiv(I, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gsz = tl.minimum(num_m - first_m, GROUP_M)
    m_tile = first_m + (pid % group) % gsz
    n_tile = (pid % group) // gsz
    rm = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m = (rm < T)[:, None]
    x_ptrs = X8 + rm[:, None] * K + rk[None, :]
    wg_ptrs = W8 + rn[None, :] * K + rk[:, None]
    wu_ptrs = wg_ptrs + I * K
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=m, other=0.0)
        acc_g = tl.dot(x, tl.load(wg_ptrs), acc_g)
        acc_u = tl.dot(x, tl.load(wu_ptrs), acc_u)
        x_ptrs += BLOCK_K
        wg_ptrs += BLOCK_K
        wu_ptrs += BLOCK_K
    deq = 1.0 / (tl.load(SC) * tl.load(WS))
    g = (acc_g * deq).to(tl.bfloat16).to(tl.float32)
    u = (acc_u * deq).to(tl.bfloat16).to(tl.float32)
    h = g * tl.sigmoid(g) * u
    q_gu = tl.load(SC + 1)
    q_h = tl.load(SC + 2)
    gu_ptrs = GU8 + rm[:, None] * (2 * I) + rn[None, :]
    tl.store(gu_ptrs, tl.clamp(g * q_gu, -448.0, 448.0).to(tl.float8e4nv), mask=m)
    tl.store(gu_ptrs + I, tl.clamp(u * q_gu, -448.0, 448.0).to(tl.float8e4nv), mask=m)
    tl.store(H8 + rm[:, None] * I + rn[None, :], tl.clamp(h * q_h, -448.0, 448.0).to(tl.float8e4nv), mask=m)
    tl.atomic_max(AMAX + 1, tl.maximum(tl.max(tl.abs(g)), tl.max(tl.abs(u))))
    tl.atomic_max(AMAX + 2, tl.max(tl.abs(h)))


@triton.jit
def _fp8_down_fwd_kernel(H8, WD8, RES, Y, SC, WS, T, N, K,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         GROUP_M: tl.constexpr):
    """y = res + h8 @ Wd8^T (fp8 MMA), residual add fused into the epilogue."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(T, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gsz = tl.minimum(num_m - first_m, GROUP_M)
    m_tile = first_m + (pid % group) % gsz
    n_tile = (pid % group) // gsz
    rm = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m = (rm < T)[:, None]
    a_ptrs = H8 + rm[:, None] * K + rk[None, :]
    b_ptrs = WD8 + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs, mask=m, other=0.0), tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    out = (acc / (tl.load(SC + 2) * tl.load(WS + 1))).to(tl.bfloat16).to(tl.float32)
    off = rm[:, None] * N + rn[None, :]
    res = tl.load(RES + off, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + off, (res + out).to(tl.bfloat16), mask=m)


@triton.jit
def _fp8_down_dgrad_swiglu_kernel(DY, WD, GU8, DGU8, SC, AMAX, T, K, I,
                                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                  BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """dh = dy @ Wd (bf16 MMA: at K = d an in-register e5m2 cast of dy costs more than
    the fp8 MMA saves; dh is never stored); the SwiGLU backward runs in the epilogue
    against the fp8 gu and writes d_gu once, in e5m2. Also records dy's amax for the
    fp8 down-proj weight gradient."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(T, BLOCK_M)
    num_n = tl.cdiv(I, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gsz = tl.minimum(num_m - first_m, GROUP_M)
    m_tile = first_m + (pid % group) % gsz
    n_tile = (pid % group) // gsz
    rm = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m = (rm < T)[:, None]
    a_ptrs = DY + rm[:, None] * K + rk[None, :]
    b_ptrs = WD + rk[:, None] * I + rn[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    dy_amax = 0.0
    for _ in range(0, K, BLOCK_K):
        dy = tl.load(a_ptrs, mask=m, other=0.0)
        if n_tile == 0:  # every n tile reads the same dy rows; one column records the amax
            dy_amax = tl.maximum(dy_amax, tl.max(tl.abs(dy.to(tl.float32))))
        acc = tl.dot(dy, tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * I
    if n_tile == 0:
        tl.atomic_max(AMAX + 4, dy_amax)
    dh = acc.to(tl.bfloat16).to(tl.float32)
    s_gu = 1.0 / tl.load(SC + 1)
    q = tl.load(SC + 3)
    gu_ptrs = GU8 + rm[:, None] * (2 * I) + rn[None, :]
    g = tl.load(gu_ptrs, mask=m, other=0.0).to(tl.float32) * s_gu
    u = tl.load(gu_ptrs + I, mask=m, other=0.0).to(tl.float32) * s_gu
    sig = tl.sigmoid(g)
    d_gate = dh * u * (sig * (1.0 + g * (1.0 - sig)))
    d_up = dh * g * sig
    out_ptrs = DGU8 + rm[:, None] * (2 * I) + rn[None, :]
    tl.store(out_ptrs, tl.clamp(d_gate * q, -57344.0, 57344.0).to(tl.float8e5), mask=m)
    tl.store(out_ptrs + I, tl.clamp(d_up * q, -57344.0, 57344.0).to(tl.float8e5), mask=m)
    tl.atomic_max(AMAX + 3, tl.maximum(tl.max(tl.abs(d_gate)), tl.max(tl.abs(d_up))))


@triton.jit
def _fp8_mm_kernel(A, B, OUT, SA, SB, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   GROUP_M: tl.constexpr):
    """OUT = (A @ B) / (SA * SB): fp8 operands at arbitrary strides, quantization scales
    SA/SB undone in the epilogue (gate_up dgrad d_gu @ W, and wgrad d_gu^T @ x with A
    read transposed)."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gsz = tl.minimum(num_m - first_m, GROUP_M)
    m_tile = first_m + (pid % group) % gsz
    n_tile = (pid % group) // gsz
    rm = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + rk) < K
        a = tl.load(a_ptrs, mask=(rm < M)[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    deq = 1.0 / (tl.load(SA) * tl.load(SB))
    tl.store(OUT + rm[:, None] * N + rn[None, :], (acc * deq).to(OUT.dtype.element_ty),
             mask=(rm < M)[:, None])


@triton.jit
def _fp8_norm_bwd_kernel(H, RSTD, DXN, DY, NW, DH, DNW, T,
                         D: tl.constexpr, BLOCK_T: tl.constexpr):
    """dh = dy + rmsnorm backward of dxn; per-program partial norm-weight grads."""
    pid = tl.program_id(0)
    rt = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    rd = tl.arange(0, D)
    m = (rt < T)[:, None]
    off = rt[:, None] * D + rd[None, :]
    h = tl.load(H + off, mask=m, other=0.0).to(tl.float32)
    r = tl.load(RSTD + rt, mask=rt < T, other=0.0)
    g = tl.load(DXN + off, mask=m, other=0.0).to(tl.float32)
    gw = g * tl.load(NW + rd)[None, :]
    s = tl.sum(gw * h, 1)
    dh = gw * r[:, None] - h * (s * r * r * r * (1.0 / D))[:, None]
    dh += tl.load(DY + off, mask=m, other=0.0).to(tl.float32)
    tl.store(DH + off, dh.to(tl.bfloat16), mask=m)
    tl.store(DNW + pid * D + rd, tl.sum(g * (h * r[:, None]), 0))


@triton.jit
def _fp8_wgrad_kernel(A, B, OUT, SA, SB, T, MA, NB, ROWS_PER_SPLIT,
                      A_BF16: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr):
    """OUT[split, m, n] = sum_{t in split} A[t, m] B[t, n] / (SA * SB): weight gradients
    reduced over tokens. Row tiles are loaded contiguously and transposed in registers;
    a bf16 A (dy) is quantized to e5m2 in registers with scale SA. Splits over tokens are
    summed by the caller (deterministic)."""
    pid = tl.program_id(0)
    split = tl.program_id(1)
    num_n = tl.cdiv(NB, BLOCK_N)
    rm = (pid // num_n) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = (pid % num_n) * BLOCK_N + tl.arange(0, BLOCK_N)
    rr = tl.arange(0, BLOCK_R)
    qa = tl.load(SA)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    r_end = tl.minimum((split + 1) * ROWS_PER_SPLIT, T)
    for r0 in range(split * ROWS_PER_SPLIT, r_end, BLOCK_R):
        rows = r0 + rr
        rmask = (rows < r_end)[:, None]
        a = tl.load(A + rows[:, None] * MA + rm[None, :], mask=rmask, other=0.0)
        if A_BF16:
            a = tl.clamp(a.to(tl.float32) * qa, -57344.0, 57344.0).to(tl.float8e5)
        b = tl.load(B + rows[:, None] * NB + rn[None, :], mask=rmask, other=0.0)
        acc = tl.dot(tl.trans(a), b, acc)
    tl.store(OUT + split * MA * NB + rm[:, None] * NB + rn[None, :], acc / (qa * tl.load(SB)))


def _fp8_wgrad(a, b, sa, sb, a_bf16, *, BLOCK_R=64, BLOCK_M=128, BLOCK_N=128, splits=2,
               num_warps=4, num_stages=3):
    T_, MA = a.shape
    NB = b.shape[1]
    rows = triton.cdiv(triton.cdiv(T_, splits), BLOCK_R) * BLOCK_R
    out = torch.empty(splits, MA, NB, device=a.device, dtype=torch.float32)
    _fp8_wgrad_kernel[((MA // BLOCK_M) * (NB // BLOCK_N), splits)](
        a, b, out, sa, sb, T_, MA, NB, rows, A_BF16=a_bf16,
        BLOCK_R=BLOCK_R, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps, num_stages=num_stages)
    return out.sum(0) if splits > 1 else out[0]


FP8_NORM_BLOCK_T = 16


class _FP8MLPBlock(torch.autograd.Function):
    """y = h + down(SwiGLU(gate_up(rmsnorm(h)))) with every large MLP activation and
    activation gradient stored in fp8 (x, gu, h: e4m3; d_gu: e5m2) and fp8 MMAs for
    every GEMM but the down-proj dgrad (bf16 dy; the down-proj wgrad quantizes dy to
    e5m2 in registers). Master weights stay fp32 (quantized per call); weight and
    norm-weight grads are fp32."""

    @staticmethod
    def forward(ctx, h, norm_w, w_gu, w_down, scale, amax, eps):
        T_, D = h.shape
        I = w_down.shape[1]
        dev = h.device
        x8 = torch.empty(T_, D, device=dev, dtype=torch.float8_e4m3fn)
        x8t = torch.empty(D, T_, device=dev, dtype=torch.float8_e4m3fn)
        rstd = torch.empty(T_, device=dev, dtype=torch.float32)
        _fp8_norm_quant_kernel[(triton.cdiv(T_, 32),)](
            h, norm_w, x8, x8t, rstd, scale, amax, T_, EPS=eps, D=D, BLOCK_T=32,
            num_warps=8)
        w8, q_gu = _fp8_quantize_weight(w_gu)
        wd8, q_d = _fp8_quantize_weight(w_down)
        wscale = torch.stack((q_gu, q_d)).float()
        gu8 = torch.empty(T_, 2 * I, device=dev, dtype=torch.float8_e4m3fn)
        h8 = torch.empty(T_, I, device=dev, dtype=torch.float8_e4m3fn)
        _fp8_swiglu_fwd_kernel[(triton.cdiv(T_, 128) * (I // 64),)](
            x8, w8, scale, wscale, gu8, h8, amax, T_, D, I,
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=4)
        y = torch.empty_like(h)
        _fp8_down_fwd_kernel[(triton.cdiv(T_, 128) * (D // 64),)](
            h8, wd8, h, y, scale, wscale, T_, D, I,
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, GROUP_M=8, num_warps=4, num_stages=3)
        # fp8 MMA wants K-contiguous operands: the dgrad's B is W8^T (tiny, copied here)
        ctx.save_for_backward(h, norm_w, w_down.to(torch.bfloat16), x8t, rstd,
                              w8.t().contiguous(), wscale, gu8, h8, scale, amax)
        return y

    @staticmethod
    def backward(ctx, dy):
        h, norm_w, wd, x8t, rstd, w8t, wscale, gu8, h8, scale, amax = ctx.saved_tensors
        dy = dy.contiguous()
        T_, D = h.shape
        I = wd.shape[1]
        dev = h.device
        dgu8 = torch.empty(T_, 2 * I, device=dev, dtype=torch.float8_e5m2)
        _fp8_down_dgrad_swiglu_kernel[(triton.cdiv(T_, 64) * (I // 64),)](
            dy, wd, gu8, dgu8, scale, amax, T_, D, I,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=4)
        # plain fp8 GEMM, no epilogue: cuBLAS beats the Triton kernel here (1.07 vs 1.42 ms)
        dxn = torch._scaled_mm(dgu8, w8t.t(), scale_a=1.0 / scale[3], scale_b=1.0 / wscale[0],
                               out_dtype=torch.bfloat16, use_fast_accum=False)
        dh = torch.empty_like(h)
        n_prog = triton.cdiv(T_, FP8_NORM_BLOCK_T)
        dnw = torch.empty(n_prog, D, device=dev, dtype=torch.float32)
        _fp8_norm_bwd_kernel[(n_prog,)](
            h, rstd, dxn, dy, norm_w, dh, dnw, T_, D=D, BLOCK_T=FP8_NORM_BLOCK_T, num_warps=4)
        dwd = _fp8_wgrad(dy, h8, scale[4:], scale[2:], True, BLOCK_R=32)
        dwgu = torch.empty(2 * I, D, device=dev, dtype=torch.float32)
        _fp8_mm_kernel[((2 * I) // 128 * (D // 128),)](  # d_gu^T @ x8, A read transposed
            dgu8, x8t, dwgu, scale[3:], scale[0:], 2 * I, D, T_, 1, 2 * I, 1, T_,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=3)
        return dh, dnw.sum(0), dwgu, dwd, None, None, None


# --------------------------------------------------------------------------- #
# fp8 MoE experts (Triton): grouped versions of the _FP8MLPBlock kernels
# --------------------------------------------------------------------------- #
def _use_fp8_moe(experts, x: torch.Tensor) -> bool:
    if not (FP8_MLP and experts.training and triton is not None and x.is_cuda
            and x.dtype == torch.bfloat16):
        return False
    d, inter = experts.hidden_dim, experts.intermediate_dim
    return d % 128 == 0 and d & (d - 1) == 0 and inter % 128 == 0


@triton.jit
def _expert_tile(OFFS, pid, N, E: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 GROUP_M: tl.constexpr):
    """Map a program id onto (expert, m tile, n tile) of a grouped GEMM whose rows
    are sorted by expert (OFFS = inclusive row-count prefix sums). The grid is
    sized for the worst case, so trailing programs find no tile."""
    num_n = tl.cdiv(N, BLOCK_N)
    tile = pid
    row_start = 0
    expert = 0
    found = False
    m_tile = 0
    n_tile = 0
    rows_e = 0
    for e in tl.static_range(E):
        end = tl.load(OFFS + e)
        n_rows = end - row_start
        n_m = tl.cdiv(n_rows, BLOCK_M)
        n_tiles = n_m * num_n
        if (not found) and (tile < n_tiles):
            found = True
            expert = e
            rows_e = n_rows
            group_size = GROUP_M * num_n
            first_m = (tile // group_size) * GROUP_M
            gsz = tl.minimum(n_m - first_m, GROUP_M)
            m_tile = first_m + (tile % group_size) % gsz
            n_tile = (tile % group_size) // gsz
        if not found:
            tile = tile - n_tiles
            row_start = end
    return found, expert, row_start, rows_e, m_tile, n_tile


@triton.jit
def _fp8_moe_gather_quant_kernel(X, TOK, X8, SC, AMAX, R,
                                 D: tl.constexpr, BLOCK_R: tl.constexpr):
    """x8[r] = e4m3(x[tok[r]] * scale_x): the expert-order gather fused with the
    quantization."""
    rr = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rd = tl.arange(0, D)
    rmask = rr < R
    m = rmask[:, None]
    tok = tl.load(TOK + rr, mask=rmask, other=0)
    x = tl.load(X + tok[:, None] * D + rd[None, :], mask=m, other=0.0).to(tl.float32)
    x8 = tl.clamp(x * tl.load(SC), -448.0, 448.0).to(tl.float8e4nv)
    tl.store(X8 + rr[:, None] * D + rd[None, :], x8, mask=m)
    tl.atomic_max(AMAX, tl.max(tl.abs(x)))


@triton.jit
def _fp8_moe_swiglu_fwd_kernel(X8, W8, OFFS, SC, WS, GU8, H8, AMAX, K, I,
                               E: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                               BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """Per expert e: gu = x8 @ W8[e]^T (fp8 MMA); SwiGLU epilogue; gu, h stored e4m3."""
    found, expert, row_start, rows_e, m_tile, n_tile = _expert_tile(
        OFFS, tl.program_id(0), I, E, BLOCK_M, BLOCK_N, GROUP_M)
    if not found:
        return
    lm = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m = (lm < rows_e)[:, None]
    rows = row_start + lm
    x_ptrs = X8 + rows[:, None] * K + rk[None, :]
    wg_ptrs = W8 + expert * (2 * I * K) + rn[None, :] * K + rk[:, None]
    wu_ptrs = wg_ptrs + I * K
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=m, other=0.0)
        acc_g = tl.dot(x, tl.load(wg_ptrs), acc_g)
        acc_u = tl.dot(x, tl.load(wu_ptrs), acc_u)
        x_ptrs += BLOCK_K
        wg_ptrs += BLOCK_K
        wu_ptrs += BLOCK_K
    deq = 1.0 / (tl.load(SC) * tl.load(WS + expert))
    g = (acc_g * deq).to(tl.bfloat16).to(tl.float32)
    u = (acc_u * deq).to(tl.bfloat16).to(tl.float32)
    h = g * tl.sigmoid(g) * u
    q_gu = tl.load(SC + 1)
    q_h = tl.load(SC + 2)
    gu_ptrs = GU8 + rows[:, None] * (2 * I) + rn[None, :]
    tl.store(gu_ptrs, tl.clamp(g * q_gu, -448.0, 448.0).to(tl.float8e4nv), mask=m)
    tl.store(gu_ptrs + I, tl.clamp(u * q_gu, -448.0, 448.0).to(tl.float8e4nv), mask=m)
    tl.store(H8 + rows[:, None] * I + rn[None, :], tl.clamp(h * q_h, -448.0, 448.0).to(tl.float8e4nv), mask=m)
    tl.atomic_max(AMAX + 1, tl.maximum(tl.max(tl.abs(g)), tl.max(tl.abs(u))))
    tl.atomic_max(AMAX + 2, tl.max(tl.abs(h)))


@triton.jit
def _fp8_moe_down_fwd_kernel(H8, WD8, OFFS, SC, WSD, OUT, N, K,
                             E: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                             BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """Per expert e: out = h8 @ Wd8[e]^T (fp8 MMA), bf16 rows in expert order."""
    found, expert, row_start, rows_e, m_tile, n_tile = _expert_tile(
        OFFS, tl.program_id(0), N, E, BLOCK_M, BLOCK_N, GROUP_M)
    if not found:
        return
    lm = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m = (lm < rows_e)[:, None]
    rows = row_start + lm
    a_ptrs = H8 + rows[:, None] * K + rk[None, :]
    b_ptrs = WD8 + expert * (N * K) + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs, mask=m, other=0.0), tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    acc = acc / (tl.load(SC + 2) * tl.load(WSD + expert))
    tl.store(OUT + rows[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=m)


@triton.jit
def _fp8_moe_down_dgrad_swiglu_kernel(DY, WD, GU8, DGU8, OFFS, SC, AMAX, K, I,
                                      E: tl.constexpr, BLOCK_M: tl.constexpr,
                                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                      GROUP_M: tl.constexpr):
    """Per expert e: dh = dy @ Wd[e] (bf16 MMA, never stored); SwiGLU backward in the
    epilogue against the fp8 gu; d_gu written once in e5m2. Records dy's amax."""
    found, expert, row_start, rows_e, m_tile, n_tile = _expert_tile(
        OFFS, tl.program_id(0), I, E, BLOCK_M, BLOCK_N, GROUP_M)
    if not found:
        return
    lm = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m = (lm < rows_e)[:, None]
    rows = row_start + lm
    a_ptrs = DY + rows[:, None] * K + rk[None, :]
    b_ptrs = WD + expert * (K * I) + rk[:, None] * I + rn[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    dy_amax = 0.0
    for _ in range(0, K, BLOCK_K):
        dy = tl.load(a_ptrs, mask=m, other=0.0)
        if n_tile == 0:
            dy_amax = tl.maximum(dy_amax, tl.max(tl.abs(dy.to(tl.float32))))
        acc = tl.dot(dy, tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * I
    if n_tile == 0:
        tl.atomic_max(AMAX + 4, dy_amax)
    dh = acc.to(tl.bfloat16).to(tl.float32)
    s_gu = 1.0 / tl.load(SC + 1)
    q = tl.load(SC + 3)
    gu_ptrs = GU8 + rows[:, None] * (2 * I) + rn[None, :]
    g = tl.load(gu_ptrs, mask=m, other=0.0).to(tl.float32) * s_gu
    u = tl.load(gu_ptrs + I, mask=m, other=0.0).to(tl.float32) * s_gu
    sig = tl.sigmoid(g)
    d_gate = dh * u * (sig * (1.0 + g * (1.0 - sig)))
    d_up = dh * g * sig
    out_ptrs = DGU8 + rows[:, None] * (2 * I) + rn[None, :]
    tl.store(out_ptrs, tl.clamp(d_gate * q, -57344.0, 57344.0).to(tl.float8e5), mask=m)
    tl.store(out_ptrs + I, tl.clamp(d_up * q, -57344.0, 57344.0).to(tl.float8e5), mask=m)
    tl.atomic_max(AMAX + 3, tl.maximum(tl.max(tl.abs(d_gate)), tl.max(tl.abs(d_up))))


@triton.jit
def _fp8_moe_dgrad_kernel(DGU8, W8T, OFFS, SC, WS, OUT, N, K,
                          E: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """Per expert e: dx = d_gu8 @ W8[e] (fp8 MMA); B read from W8^T (K-contiguous)."""
    found, expert, row_start, rows_e, m_tile, n_tile = _expert_tile(
        OFFS, tl.program_id(0), N, E, BLOCK_M, BLOCK_N, GROUP_M)
    if not found:
        return
    lm = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m = (lm < rows_e)[:, None]
    rows = row_start + lm
    a_ptrs = DGU8 + rows[:, None] * K + rk[None, :]
    b_ptrs = W8T + expert * (N * K) + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs, mask=m, other=0.0), tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    acc = acc / (tl.load(SC + 3) * tl.load(WS + expert))
    tl.store(OUT + rows[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=m)


@triton.jit
def _fp8_moe_wgrad_kernel(A, B, OFFS, ORDER, OUT, SA, SB, MA, NB,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """OUT[e, m, n] = sum over expert e's rows r of A[r, m] B[r, n] / (SA * SB):
    fp8 A (R, MA) read transposed, fp8 B (R, NB) row-major. A token-contiguous B
    would suit the MMA better, but its row-boundary mask then varies inside each
    contiguous group and the loads go scalar (3.3 vs 1.9 ms). Experts run in
    ORDER (largest first) so the router's imbalance does not stretch the tail."""
    pid = tl.program_id(0)
    expert = tl.load(ORDER + tl.program_id(1))
    num_n = tl.cdiv(NB, BLOCK_N)
    rm = (pid // num_n) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = (pid % num_n) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    row_start = 0
    if expert > 0:
        row_start = tl.load(OFFS + expert - 1)
    row_end = tl.load(OFFS + expert)
    qa = tl.load(SA)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # start on a BLOCK_K boundary (masking the previous expert's rows) so the loads
    # stay provably aligned
    for r0 in range((row_start // BLOCK_K) * BLOCK_K, row_end, BLOCK_K):
        kmask = ((r0 + rk) >= row_start) & ((r0 + rk) < row_end)
        a = tl.load(A + (r0 + rk)[None, :] * MA + rm[:, None], mask=kmask[None, :], other=0.0)
        b = tl.load(B + (r0 + rk)[:, None] * NB + rn[None, :], mask=kmask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(OUT + expert * (MA * NB) + rm[:, None] * NB + rn[None, :], acc / (qa * tl.load(SB)))


@triton.jit
def _expert_chunk(OFFS, cid, CH: tl.constexpr, E: tl.constexpr):
    """Map a chunk id onto (expert, row range) with every expert's rows cut into
    CH-row chunks (the grid is sized for cdiv(R, CH) + E chunks)."""
    row_start = 0
    found = False
    expert = 0
    rs = 0
    re = 0
    for e in tl.static_range(E):
        end = tl.load(OFFS + e)
        n = tl.cdiv(end - row_start, CH)
        if (not found) and (cid < n):
            found = True
            expert = e
            rs = row_start + cid * CH
            re = tl.minimum(rs + CH, end)
        if not found:
            cid = cid - n
            row_start = end
    return found, expert, rs, re


@triton.jit
def _fp8_moe_wgrad_rows_kernel(A, B, OFFS, OUT, SA, SB, MA, NB,
                               CH: tl.constexpr, E: tl.constexpr, BLOCK_R: tl.constexpr,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """OUT[e] += sum over one CH-row chunk of expert e of A[r, m] B[r, n] / (SA * SB)
    (OUT zero-filled; fp32 atomic accumulation). Splitting experts into chunks keeps
    the GPU busy when the router is unbalanced -- single layers routinely send
    35-50% of their tokens to one expert. Row tiles load contiguously; the bf16 A
    (dy) is quantized to e5m2 and transposed in registers."""
    found, expert, rs, re = _expert_chunk(OFFS, tl.program_id(1), CH, E)
    if not found:
        return
    pid = tl.program_id(0)
    num_n = tl.cdiv(NB, BLOCK_N)
    rm = (pid // num_n) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = (pid % num_n) * BLOCK_N + tl.arange(0, BLOCK_N)
    rr = tl.arange(0, BLOCK_R)
    qa = tl.load(SA)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for r0 in range((rs // BLOCK_R) * BLOCK_R, re, BLOCK_R):  # aligned start
        rows = r0 + rr
        rmask = ((rows >= rs) & (rows < re))[:, None]
        a = tl.load(A + rows[:, None] * MA + rm[None, :], mask=rmask, other=0.0)
        a = tl.clamp(a.to(tl.float32) * qa, -57344.0, 57344.0).to(tl.float8e5)
        b = tl.load(B + rows[:, None] * NB + rn[None, :], mask=rmask, other=0.0)
        acc = tl.dot(tl.trans(a), b, acc)
    tl.atomic_add(OUT + expert * (MA * NB) + rm[:, None] * NB + rn[None, :],
                  acc / (qa * tl.load(SB)), sem="relaxed")


FP8_MOE_WGRAD_CHUNK = 4096


def _fp8_quantize_experts(w: torch.Tensor):
    """Per-expert e4m3 copies of (E, out, in) fp32 master weights (pow2 scales)."""
    amax = w.detach().abs().amax(dim=(1, 2)).clamp(min=1e-30)
    q = torch.exp2(torch.floor(torch.log2(448.0 / amax)))
    return (w.detach() * q[:, None, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), q


class _FP8Experts(torch.autograd.Function):
    """Grouped expert SwiGLU MLP with fp8 storage, the MoE twin of _FP8MLPBlock:
    the expert-order gather is fused with the e4m3 quantization of x; gu, h (e4m3)
    and d_gu (e5m2) are the only stored activations; expert weights get per-expert
    e4m3 copies (fp32 masters, fp32 grads). Returns unweighted bf16 expert rows in
    expert order (the router-weighted top-k combine stays bf16)."""

    @staticmethod
    def forward(ctx, x, w_gu, w_d, token_index, inverse_permutation, offsets, top_k,
                scale, amax):
        T_, D = x.shape
        E, twoI, _ = w_gu.shape
        I = twoI // 2
        R = token_index.numel()
        dev = x.device
        x8 = torch.empty(R, D, device=dev, dtype=torch.float8_e4m3fn)
        _fp8_moe_gather_quant_kernel[(triton.cdiv(R, 32),)](
            x, token_index, x8, scale, amax, R, D=D, BLOCK_R=32, num_warps=8)
        w8, q_gu = _fp8_quantize_experts(w_gu)
        wd8, q_d = _fp8_quantize_experts(w_d)
        gu8 = torch.empty(R, twoI, device=dev, dtype=torch.float8_e4m3fn)
        h8 = torch.empty(R, I, device=dev, dtype=torch.float8_e4m3fn)
        _fp8_moe_swiglu_fwd_kernel[((triton.cdiv(R, 128) + E) * (I // 64),)](
            x8, w8, offsets, scale, q_gu, gu8, h8, amax, D, I,
            E=E, BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=4)
        out = torch.empty(R, D, device=dev, dtype=torch.bfloat16)
        _fp8_moe_down_fwd_kernel[((triton.cdiv(R, 64) + E) * (D // 64),)](
            h8, wd8, offsets, scale, q_d, out, D, I,
            E=E, BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, GROUP_M=8, num_warps=4, num_stages=3)
        ctx.save_for_backward(x8, w8.transpose(1, 2).contiguous(), q_gu,
                              w_d.to(torch.bfloat16), gu8, h8, offsets,
                              inverse_permutation, scale, amax)
        ctx.shape = (T_, top_k)
        return out

    @staticmethod
    def backward(ctx, dout):
        x8, w8t, q_gu, wd, gu8, h8, offsets, inverse_permutation, scale, amax = ctx.saved_tensors
        T_, top_k = ctx.shape
        dout = dout.contiguous()
        R, D = dout.shape
        E, _, I = wd.shape
        dev = dout.device
        dgu8 = torch.empty(R, 2 * I, device=dev, dtype=torch.float8_e5m2)
        _fp8_moe_down_dgrad_swiglu_kernel[((triton.cdiv(R, 64) + E) * (I // 64),)](
            dout, wd, gu8, dgu8, offsets, scale, amax, D, I,
            E=E, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=4)
        dx = torch.empty(R, D, device=dev, dtype=torch.bfloat16)
        _fp8_moe_dgrad_kernel[((triton.cdiv(R, 128) + E) * (D // 128),)](
            dgu8, w8t, offsets, scale, q_gu, dx, D, 2 * I,
            E=E, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=4)
        counts = torch.diff(offsets, prepend=offsets.new_zeros(1))
        order = torch.argsort(counts, descending=True).to(torch.int32)  # largest expert first
        dw_gu = torch.empty(E, 2 * I, D, device=dev, dtype=torch.float32)
        _fp8_moe_wgrad_kernel[((2 * I) // 128 * (D // 128), E)](
            dgu8, x8, offsets, order, dw_gu, scale[3:], scale[0:], 2 * I, D,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=3)
        dw_d = torch.zeros(E, D, I, device=dev, dtype=torch.float32)
        _fp8_moe_wgrad_rows_kernel[(D // 128 * (I // 128), triton.cdiv(R, FP8_MOE_WGRAD_CHUNK) + E)](
            dout, h8, offsets, dw_d, scale[4:], scale[2:], D, I, CH=FP8_MOE_WGRAD_CHUNK, E=E,
            BLOCK_R=32, BLOCK_M=128, BLOCK_N=128, num_warps=4, num_stages=3)
        dxn = _sum_topk_rows(dx, inverse_permutation, top_k)
        return dxn, dw_gu, dw_d, None, None, None, None, None, None


# --------------------------------------------------------------------------- #
# Fused SwiGLU grouped GEMM (Triton): h = silu(x W_gate^T) * (x W_up^T)
# --------------------------------------------------------------------------- #
try:  # inductor inlines user Triton kernels and expects the `triton`/`tl` names
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None


FUSED_SWIGLU = True  # --no-fused-swiglu falls back to GEMM + separate SwiGLU


def _use_fused_swiglu(x: torch.Tensor) -> bool:
    return FUSED_SWIGLU and triton is not None and x.is_cuda and x.dtype == torch.bfloat16


@triton.jit
def _swiglu_gemm_kernel(
    X, W, OFFS, GU, H,
    K, I,
    stride_xr, stride_xk,
    stride_we, stride_wn, stride_wk,
    stride_gur, stride_gun,
    stride_hr, stride_hn,
    E: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, WRITE_GU: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(I, BLOCK_N)
    # ---- locate this program's (expert, m tile, n tile) ----------------
    # tiles are laid out expert by expert; each expert has cdiv(rows_e, BLOCK_M)
    # m tiles.  The grid is sized for the worst case, so trailing programs
    # may find no tile and exit.
    tile = pid
    row_start = 0
    expert = 0
    found = False
    m_tile = 0
    n_tile = 0
    rows_e = 0
    for e in tl.static_range(E):
        end = tl.load(OFFS + e)
        n_rows = end - row_start
        n_m = tl.cdiv(n_rows, BLOCK_M)
        n_tiles = n_m * num_n
        if (not found) and (tile < n_tiles):
            found = True
            expert = e
            rows_e = n_rows
            # grouped ordering within the expert for L2 reuse of X rows
            group_size = GROUP_M * num_n
            group_id = tile // group_size
            first_m = group_id * GROUP_M
            gsz = tl.minimum(n_m - first_m, GROUP_M)
            m_tile = first_m + (tile % group_size) % gsz
            n_tile = (tile % group_size) // gsz
        if not found:
            tile = tile - n_tiles
            row_start = end
    if not found:
        return
    # ---- main loop -------------------------------------------------------
    rm = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    row_mask = rm < rows_e
    x_ptrs = X + (row_start + rm)[:, None] * stride_xr + rk[None, :] * stride_xk
    wg_ptrs = W + expert * stride_we + rn[None, :] * stride_wn + rk[:, None] * stride_wk
    wu_ptrs = W + expert * stride_we + (rn + I)[None, :] * stride_wn + rk[:, None] * stride_wk
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_mask = rn < I
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + rk) < K
        x = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        wg = tl.load(wg_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        wu = tl.load(wu_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc_g = tl.dot(x, wg, acc_g)
        acc_u = tl.dot(x, wu, acc_u)
        x_ptrs += BLOCK_K * stride_xk
        wg_ptrs += BLOCK_K * stride_wk
        wu_ptrs += BLOCK_K * stride_wk
    # ---- epilogue --------------------------------------------------------
    g16 = acc_g.to(tl.bfloat16)
    u16 = acc_u.to(tl.bfloat16)
    out_mask = row_mask[:, None] & n_mask[None, :]
    rows = row_start + rm
    if WRITE_GU:
        tl.store(GU + rows[:, None] * stride_gur + rn[None, :] * stride_gun, g16, mask=out_mask)
        tl.store(GU + rows[:, None] * stride_gur + (rn + I)[None, :] * stride_gun, u16, mask=out_mask)
    g = g16.to(tl.float32)
    u = u16.to(tl.float32)
    h = g * tl.sigmoid(g) * u
    tl.store(H + rows[:, None] * stride_hr + rn[None, :] * stride_hn, h.to(tl.bfloat16), mask=out_mask)


def _swiglu_gemm(x, w, offs, *, write_gu=True, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                GROUP_M=8, num_warps=4, num_stages=3):
    """x (R, K) bf16 sorted by group, w (E, 2I, K) bf16, offs (E,) int32 -> (gu, h)."""
    R, K = x.shape
    E, twoI, _ = w.shape
    I = twoI // 2
    gu = torch.empty(R, twoI, device=x.device, dtype=x.dtype) if write_gu else None
    h = torch.empty(R, I, device=x.device, dtype=x.dtype)
    max_tiles = (triton.cdiv(R, BLOCK_M) + E) * triton.cdiv(I, BLOCK_N)
    _swiglu_gemm_kernel[(max_tiles,)](
        x, w, offs, gu if write_gu else h, h,
        K, I,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        gu.stride(0) if write_gu else 0, gu.stride(1) if write_gu else 0,
        h.stride(0), h.stride(1),
        E=E, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        WRITE_GU=write_gu, num_warps=num_warps, num_stages=num_stages,
    )
    return gu, h




@triton.jit
def _grouped_gemm_kernel(
    X, W, OFFS, OUT, K, N,
    stride_xr, stride_xk, stride_we, stride_wn, stride_wk, stride_or, stride_on,
    E: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """OUT[r] = X[r] @ B_e^T for rows r of group e, B_e[n, k] = W[e, n, k] (any strides)."""
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    tile = pid
    row_start = 0
    expert = 0
    found = False
    m_tile = 0
    n_tile = 0
    rows_e = 0
    for e in tl.static_range(E):
        end = tl.load(OFFS + e)
        n_rows = end - row_start
        n_m = tl.cdiv(n_rows, BLOCK_M)
        n_tiles = n_m * num_n
        if (not found) and (tile < n_tiles):
            found = True
            expert = e
            rows_e = n_rows
            group_size = GROUP_M * num_n
            group_id = tile // group_size
            first_m = group_id * GROUP_M
            gsz = tl.minimum(n_m - first_m, GROUP_M)
            m_tile = first_m + (tile % group_size) % gsz
            n_tile = (tile % group_size) // gsz
        if not found:
            tile = tile - n_tiles
            row_start = end
    if not found:
        return
    rm = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    row_mask = rm < rows_e
    n_mask = rn < N
    x_ptrs = X + (row_start + rm)[:, None] * stride_xr + rk[None, :] * stride_xk
    w_ptrs = W + expert * stride_we + rn[None, :] * stride_wn + rk[:, None] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + rk) < K
        x = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    rows = row_start + rm
    tl.store(OUT + rows[:, None] * stride_or + rn[None, :] * stride_on, acc.to(tl.bfloat16),
             mask=row_mask[:, None] & n_mask[None, :])


def _grouped_gemm(x, w, offs, *, transpose_w=False):
    """x (R, K) bf16 with rows sorted by group; w (E, N, K) bf16, or (E, K, N)
    with transpose_w=True (read through strides, no copy); offs (E,) int32
    cumulative row counts -> (R, N) bf16.  Device-side offsets: no host sync.
    Tile configs are the measured best for the default MoE shapes on GB10."""
    R, K = x.shape
    E = w.shape[0]
    N = w.shape[2] if transpose_w else w.shape[1]
    se, sn, sk = (w.stride(0), w.stride(2), w.stride(1)) if transpose_w else w.stride()
    if N <= 512 and K >= 1024:
        bm, bn, bk, warps = 64, 128, 64, 8
    else:
        bm, bn, bk, warps = 64, 256, 32, 4
    out = torch.empty(R, N, device=x.device, dtype=x.dtype)
    max_tiles = (triton.cdiv(R, bm) + E) * triton.cdiv(N, bn)
    _grouped_gemm_kernel[(max_tiles,)](
        x, w, offs, out, K, N, x.stride(0), x.stride(1), se, sn, sk, out.stride(0), out.stride(1),
        E=E, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=warps, num_stages=3,
    )
    return out


@triton.jit
def _grouped_wgrad_kernel(
    A, B, OFFS, OUT, N, K,
    stride_ar, stride_an, stride_br, stride_bk, stride_oe, stride_on, stride_ok,
    E: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """OUT[e, n, k] = sum over rows r of group e of A[r, n] * B[r, k]: the
    grouped weight gradient, reading A and B in their natural row-major
    layout (no transposed copy of the (R, N) activation gradient)."""
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    num_k = tl.cdiv(K, BLOCK_K)
    expert = pid // (num_n * num_k)
    rem = pid % (num_n * num_k)
    n_tile = rem // num_k
    k_tile = rem % num_k
    row_start = 0
    if expert > 0:
        row_start = tl.load(OFFS + expert - 1)
    row_end = tl.load(OFFS + expert)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    rr = tl.arange(0, BLOCK_R)
    n_mask = rn < N
    k_mask = rk < K
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for r0 in range(row_start, row_end, BLOCK_R):
        rows = r0 + rr
        r_mask = rows < row_end
        a = tl.load(A + rows[:, None] * stride_ar + rn[None, :] * stride_an,
                    mask=r_mask[:, None] & n_mask[None, :], other=0.0)
        b = tl.load(B + rows[:, None] * stride_br + rk[None, :] * stride_bk,
                    mask=r_mask[:, None] & k_mask[None, :], other=0.0)
        acc = tl.dot(tl.trans(a), b, acc)
    tl.store(OUT + expert * stride_oe + rn[:, None] * stride_on + rk[None, :] * stride_ok,
             acc.to(tl.bfloat16), mask=n_mask[:, None] & k_mask[None, :])


def _grouped_wgrad(grad_out, x, offs):
    """grad_out (R, N), x (R, K) bf16 with rows sorted by group -> (E, N, K) bf16."""
    R, N = grad_out.shape
    K = x.shape[1]
    E = offs.numel()
    out = torch.empty(E, N, K, device=x.device, dtype=x.dtype)
    bn = bk = 128
    grid = (E * triton.cdiv(N, bn) * triton.cdiv(K, bk),)
    _grouped_wgrad_kernel[grid](
        grad_out, x, offs, out, N, K, grad_out.stride(0), grad_out.stride(1),
        x.stride(0), x.stride(1), out.stride(0), out.stride(1), out.stride(2),
        E=E, BLOCK_R=64, BLOCK_N=bn, BLOCK_K=bk, num_warps=4, num_stages=3,
    )
    return out


class _GroupedLinear(torch.autograd.Function):
    """Expert down projection: out[r] = h[r] @ W_e^T, W (E, N, K), rows sorted by
    group.  Forward and dgrad use the Triton grouped GEMM above (faster than
    the sm_121 aten fallback and free of its host sync); the weight gradient
    uses the grouped wgrad kernel, which reads the activation gradient in its
    natural layout instead of forcing a transposed copy of it."""

    @staticmethod
    def forward(ctx, h, weight, offs):
        ctx.save_for_backward(h, weight, offs)
        return _grouped_gemm(h, weight, offs)

    @staticmethod
    def backward(ctx, dout):
        h, weight, offs = ctx.saved_tensors
        dh = _grouped_gemm(dout, weight, offs, transpose_w=True)
        dw = _grouped_wgrad(dout, h, offs)
        return dh, dw, None


@triton.jit
def _swiglu_tiles(g, u, dh):
    sig = tl.sigmoid(g)
    d_gate = dh * u * (sig * (1.0 + g * (1.0 - sig)))
    d_up = dh * g * sig
    return d_gate.to(tl.bfloat16), d_up.to(tl.bfloat16)


@triton.jit
def _swiglu_dgrad_kernel(
    GU, DH, W, OFFS, OUT, I, N,
    stride_gur, stride_dhr, stride_we, stride_wk, stride_wn, stride_or,
    E: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """OUT[r, n] = sum_k d_gu[r, k] * W[e, k, n], d_gu computed from GU/DH tiles."""
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    tile = pid
    row_start = 0
    expert = 0
    found = False
    m_tile = 0
    n_tile = 0
    rows_e = 0
    for e in tl.static_range(E):
        end = tl.load(OFFS + e)
        n_rows = end - row_start
        n_m = tl.cdiv(n_rows, BLOCK_M)
        n_tiles = n_m * num_n
        if (not found) and (tile < n_tiles):
            found = True
            expert = e
            rows_e = n_rows
            group_size = GROUP_M * num_n
            group_id = tile // group_size
            first_m = group_id * GROUP_M
            gsz = tl.minimum(n_m - first_m, GROUP_M)
            m_tile = first_m + (tile % group_size) % gsz
            n_tile = (tile % group_size) // gsz
        if not found:
            tile = tile - n_tiles
            row_start = end
    if not found:
        return
    rm = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    row_mask = rm < rows_e
    n_mask = rn < N
    rows = row_start + rm
    g_ptrs = GU + rows[:, None] * stride_gur + rk[None, :]
    u_ptrs = g_ptrs + I
    dh_ptrs = DH + rows[:, None] * stride_dhr + rk[None, :]
    wg_ptrs = W + expert * stride_we + rk[:, None] * stride_wk + rn[None, :] * stride_wn
    wu_ptrs = wg_ptrs + I * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, I, BLOCK_K):
        k_mask = (k0 + rk) < I
        m = row_mask[:, None] & k_mask[None, :]
        g = tl.load(g_ptrs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptrs, mask=m, other=0.0).to(tl.float32)
        dh = tl.load(dh_ptrs, mask=m, other=0.0).to(tl.float32)
        d_gate, d_up = _swiglu_tiles(g, u, dh)
        wm = k_mask[:, None] & n_mask[None, :]
        wg = tl.load(wg_ptrs, mask=wm, other=0.0)
        wu = tl.load(wu_ptrs, mask=wm, other=0.0)
        acc = tl.dot(d_gate, wg, acc)
        acc = tl.dot(d_up, wu, acc)
        g_ptrs += BLOCK_K
        u_ptrs += BLOCK_K
        dh_ptrs += BLOCK_K
        wg_ptrs += BLOCK_K * stride_wk
        wu_ptrs += BLOCK_K * stride_wk
    tl.store(OUT + rows[:, None] * stride_or + rn[None, :], acc.to(tl.bfloat16),
             mask=row_mask[:, None] & n_mask[None, :])


@triton.jit
def _swiglu_wgrad_kernel(
    GU, DH, X, OFFS, OUT, I, K,
    stride_gur, stride_dhr, stride_xr, stride_oe, stride_on, stride_ok,
    E: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """OUT[e, n, k] and OUT[e, I + n, k] = sum_r d_gate[r, n] x[r, k], d_up[r, n] x[r, k]."""
    pid = tl.program_id(0)
    num_n = tl.cdiv(I, BLOCK_N)
    num_k = tl.cdiv(K, BLOCK_K)
    expert = pid // (num_n * num_k)
    rem = pid % (num_n * num_k)
    n_tile = rem // num_k
    k_tile = rem % num_k
    row_start = 0
    if expert > 0:
        row_start = tl.load(OFFS + expert - 1)
    row_end = tl.load(OFFS + expert)
    rn = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    rr = tl.arange(0, BLOCK_R)
    n_mask = rn < I
    k_mask = rk < K
    acc_g = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for r0 in range(row_start, row_end, BLOCK_R):
        rows = r0 + rr
        r_mask = rows < row_end
        m = r_mask[:, None] & n_mask[None, :]
        g_off = rows[:, None] * stride_gur + rn[None, :]
        g = tl.load(GU + g_off, mask=m, other=0.0).to(tl.float32)
        u = tl.load(GU + g_off + I, mask=m, other=0.0).to(tl.float32)
        dh = tl.load(DH + rows[:, None] * stride_dhr + rn[None, :], mask=m, other=0.0).to(tl.float32)
        d_gate, d_up = _swiglu_tiles(g, u, dh)
        x = tl.load(X + rows[:, None] * stride_xr + rk[None, :],
                    mask=r_mask[:, None] & k_mask[None, :], other=0.0)
        acc_g = tl.dot(tl.trans(d_gate), x, acc_g)
        acc_u = tl.dot(tl.trans(d_up), x, acc_u)
    out_mask = n_mask[:, None] & k_mask[None, :]
    base = OUT + expert * stride_oe + rn[:, None] * stride_on + rk[None, :] * stride_ok
    tl.store(base, acc_g.to(tl.bfloat16), mask=out_mask)
    tl.store(base + I * stride_on, acc_u.to(tl.bfloat16), mask=out_mask)


def _swiglu_dgrad(gu, dh, w, offs, *, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=3):
    # 128x128 tiles halve the per-tile SwiGLU-derivative recompute along N:
    # 1.08x dense / 1.19x MoE over 64x64 on GB10, bit-identical output.
    R, twoI = gu.shape
    I = twoI // 2
    E, _, N = w.shape
    out = torch.empty(R, N, device=gu.device, dtype=gu.dtype)
    grid = ((triton.cdiv(R, BLOCK_M) + E) * triton.cdiv(N, BLOCK_N),)
    _swiglu_dgrad_kernel[grid](gu, dh, w, offs, out, I, N, gu.stride(0), dh.stride(0),
                               w.stride(0), w.stride(1), w.stride(2), out.stride(0),
                               E=E, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8,
                               num_warps=num_warps, num_stages=num_stages)
    return out


def _swiglu_wgrad(gu, dh, x, offs, *, BLOCK_R=64, BLOCK_N=64, BLOCK_K=256, num_warps=8, num_stages=2):
    # K=256 tiles halve the derivative recompute along K: 1.07x dense /
    # 1.06x MoE over (32, 64, 128) on GB10, bit-identical output.
    R, twoI = gu.shape
    I = twoI // 2
    K = x.shape[1]
    E = offs.numel()
    out = torch.empty(E, twoI, K, device=gu.device, dtype=gu.dtype)
    grid = (E * triton.cdiv(I, BLOCK_N) * triton.cdiv(K, BLOCK_K),)
    _swiglu_wgrad_kernel[grid](gu, dh, x, offs, out, I, K, gu.stride(0), dh.stride(0), x.stride(0),
                               out.stride(0), out.stride(1), out.stride(2),
                               E=E, BLOCK_R=BLOCK_R, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                               num_warps=num_warps, num_stages=num_stages)
    return out



class _FusedSwiGLU(torch.autograd.Function):
    """One Triton kernel computes gate_up = x @ W_e^T per group and its SwiGLU
    in the epilogue, writing gate_up (for the backward) and h.  Saves the
    separate bandwidth-bound SwiGLU pass over the (R, 2I) activation that the
    plain GEMM-then-pointwise formulation needs; the backward is the usual
    SwiGLU derivative (fused by inductor) plus the dgrad/wgrad GEMMs.
    Rounding points match the unfused path: gate_up rounded to bf16, SwiGLU
    evaluated in fp32 from the rounded values."""

    @staticmethod
    def forward(ctx, x, weight, offs):
        gu, h = _swiglu_gemm(x, weight, offs)
        ctx.save_for_backward(x, weight, gu, offs)
        return h

    @staticmethod
    def backward(ctx, dh):
        x, weight, gu, offs = ctx.saved_tensors
        dh = dh.contiguous()
        # SwiGLU derivative computed inside both GEMM prologues: no d_gate_up
        # buffer is ever written or re-read (7.6 -> 6.0 ms per dense layer).
        dx = _swiglu_dgrad(gu, dh, weight, offs)
        dw = _swiglu_wgrad(gu, dh, x, offs)
        return dx, dw, None


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


class _GatherTopK(torch.autograd.Function):
    """rows = hidden[token_index] where every token appears exactly top_k
    times.  The backward is therefore a plain gather + sum over the top_k slots
    (deterministic, coalesced) instead of autograd's default sort-based
    index_put_(accumulate=True)."""

    @staticmethod
    def forward(ctx, hidden, token_index, inverse_permutation, top_k):
        ctx.save_for_backward(inverse_permutation)
        ctx.top_k = top_k
        return hidden[token_index]

    @staticmethod
    def backward(ctx, grad_rows):
        (inverse_permutation,) = ctx.saved_tensors
        return _sum_topk_rows(grad_rows, inverse_permutation, ctx.top_k), None, None, None


def _sum_topk_rows(rows, inverse_permutation, top_k):
    """out[t] = sum over the top_k slots of rows[inverse_permutation[t, s]].
    (Measured: this gather -> view -> sum form compiles ~1.5% faster end to
    end than top_k explicit per-slot gathers added pointwise.)"""
    return rows[inverse_permutation].view(-1, top_k, rows.shape[-1]).sum(dim=1)


class _CombineTopK(torch.autograd.Function):
    """out[t] = sum_s w[j(t,s)] * rows[j(t,s)] for the top_k sorted rows j of
    token t, using only gathers in both directions: the forward gathers the
    weighted rows back into assignment order and sums the top_k slots; the
    backward gathers grad_out by token index.  Autograd's default for the
    same scatter would be an index_put into a zero-filled buffer forward and a
    sort-based accumulation backward."""

    @staticmethod
    def forward(ctx, rows, weights, inverse_permutation, token_index, top_k):
        ctx.save_for_backward(rows, weights, token_index)
        ctx.top_k = top_k
        weighted = rows * weights.unsqueeze(-1)
        return _sum_topk_rows(weighted, inverse_permutation, top_k)

    @staticmethod
    def backward(ctx, grad_out):
        rows, weights, token_index = ctx.saved_tensors
        grad_rows_full = grad_out[token_index]
        grad_rows = grad_rows_full * weights.unsqueeze(-1)
        grad_weights = (grad_rows_full * rows).sum(dim=-1)
        return grad_rows, grad_weights, None, None, None


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
        # fp8 expert state (see _FP8Experts); per-rank, not checkpointed
        self.register_buffer("fp8_scale", torch.ones(5), persistent=False)
        self.register_buffer("fp8_amax", torch.zeros(5), persistent=False)

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
        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(
            permutation.numel(), device=permutation.device
        )
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

        if _use_fp8_moe(self, hidden_states):
            expert_output = _FP8Experts.apply(
                hidden_states, self.gate_up_proj, self.down_proj, sorted_token_indices,
                inverse_permutation, offsets, top_k, self.fp8_scale, self.fp8_amax,
            )
            return _CombineTopK.apply(
                expert_output, sorted_weights, inverse_permutation,
                sorted_token_indices, top_k,
            ).to(dtype=hidden_states.dtype)
        sorted_hidden = _GatherTopK.apply(
            hidden_states, sorted_token_indices, inverse_permutation, top_k
        )
        if _use_fused_swiglu(sorted_hidden):
            expert_hidden = _FusedSwiGLU.apply(
                sorted_hidden, self.gate_up_proj.to(dtype=sorted_hidden.dtype), offsets
            )
            expert_output = _GroupedLinear.apply(
                expert_hidden, self.down_proj.to(dtype=sorted_hidden.dtype), offsets
            )
        else:
            gate, up = _grouped_linear(
                sorted_hidden, self.gate_up_proj, offsets
            ).chunk(2, dim=-1)
            expert_hidden = F.silu(gate) * up
            expert_output = _grouped_linear(
                expert_hidden, self.down_proj, offsets
            )
        return _CombineTopK.apply(
            expert_output, sorted_weights, inverse_permutation,
            sorted_token_indices, top_k,
        ).to(dtype=hidden_states.dtype)


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

    def forward(self, x, rope, cu_seqlens, max_seqlen):
        h = x + self.self_attn(self.input_layernorm(x), rope, cu_seqlens, max_seqlen)
        if _use_fp8_mlp(self, h):
            y = _FP8MLPBlock.apply(
                h, self.post_attention_layernorm.weight, self.mlp.gate_up_proj.weight,
                self.mlp.down_proj.weight, self.mlp.fp8_scale, self.mlp.fp8_amax,
                self.post_attention_layernorm.eps,
            )
            return y, None, None
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
        rope = (cos, sin, self.cos_cached, self.sin_cached, position_ids)
        router_logits = []
        selected_experts = []
        for layer in self.layers:
            x, layer_router_logits, layer_selected_experts = layer(
                x, rope, cu_seqlens, max_seqlen
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


def enable_triton_gemm_autotune() -> None:
    """Let inductor benchmark its Triton GEMM templates against cuBLAS per shape.

    Inductor's ``is_big_gpu()`` gate (>= 68 SMs) silently disables Triton GEMM
    templates on the 48-SM GB10, so ``mode="max-autotune"`` never actually
    tried them there.  With the gate bypassed and only GEMM autotuning on (the
    compile mode stays "default": no cudagraphs, no pointwise benchmarking),
    Triton wins the output-heavy shapes on sm_121 -- gate_up forward 2.02 ms vs
    cuBLAS 2.70 ms, down-proj addmm 1.24 vs 1.74 ms -- while cuBLAS keeps the
    K=49152 weight-gradient GEMMs: +4.3% dense / +1.0% MoE throughput at the
    default shape.  Costs ~25 s of benchmarking on the first (uncached) compile;
    configs that exceed the 99 KB shared memory are skipped by inductor.
    """
    import torch._inductor.config as inductor_config
    import torch._inductor.utils as inductor_utils

    inductor_utils.is_big_gpu = lambda *args, **kwargs: True
    inductor_config.max_autotune_gemm = True
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"


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
if triton is not None:

    @triton.jit
    def _bmm_epilogue_kernel(
        A, B, C, Out, M, N, K,
        sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, sob, som, son,
        alpha, beta,
        HAS_C: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Out[b] = alpha * A[b] @ B[b] + beta * C[b]; fp32 accumulation,
        arbitrary strides, tiles must divide M/N/K (checked by the caller)."""
        pid = tl.program_id(0)
        bid = tl.program_id(1)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        a_ptrs = A + bid * sab + rm[:, None] * sam + rk[None, :] * sak
        b_ptrs = B + bid * sbb + rk[:, None] * sbk + rn[None, :] * sbn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * sak
            b_ptrs += BLOCK_K * sbk
        acc = acc * alpha
        if HAS_C:
            c = tl.load(C + bid * scb + rm[:, None] * scm + rn[None, :] * scn)
            acc = acc + beta * c.to(tl.float32)
        out_ptrs = Out + bid * sob + rm[:, None] * som + rn[None, :] * son
        tl.store(out_ptrs, acc.to(Out.dtype.element_ty))


if triton is not None:

    @triton.jit
    def _momentum_stage_kernel(G, V, U, n, w_v, w_u, BLOCK: tl.constexpr):
        """v <- lerp(v, g, w_v) in place (fp32); u <- lerp(g, v_new, w_u)
        stored in U's dtype. Uses torch.lerp's weight<0.5 formula for the
        first and its weight>=0.5 formula for the second, so the results are
        bit-identical to the two-pass torch version for momentum >= 0.5."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        g = tl.load(G + offs, mask=mask, other=0.0)
        v = tl.load(V + offs, mask=mask, other=0.0)
        v_new = v + w_v * (g - v)
        u = v_new - (v_new - g) * (1.0 - w_u)
        tl.store(V + offs, v_new, mask=mask)
        tl.store(U + offs, u.to(U.dtype.element_ty), mask=mask)


def _momentum_stage_supported(g, v, u, momentum: float) -> bool:
    return (
        triton is not None and g.is_cuda
        and g.is_contiguous() and v.is_contiguous() and u.is_contiguous()
        and g.dtype == torch.float32 and v.dtype == torch.float32
        and 1.0 - momentum < 0.5 <= momentum
    )


def _momentum_stage(g, v, u, momentum: float) -> None:
    """One fused pass: momentum buffer update + Nesterov update staged as
    bf16. Saves re-reading g and v (the step is bandwidth bound on GB10)."""
    n = g.numel()
    grid = (triton.cdiv(n, 4096),)
    _momentum_stage_kernel[grid](
        g.view(-1), v.view(-1), u.view(-1), n, 1.0 - momentum, momentum,
        BLOCK=4096, num_warps=8,
    )


def _bmm_epilogue_supported(rows: int, cols: int) -> bool:
    """The Triton path needs tiles that divide the (rows <= cols) NS operands.
    Every product in the iteration is (rows x rows) @ (rows x cols) or the
    (rows x cols) @ (cols x rows) Gram matrix, so this covers all three."""
    return triton is not None and rows % 128 == 0 and cols % 64 == 0


def _bmm_epilogue(A, B, out, *, C=None, alpha=1.0, beta=0.0):
    """out = alpha * bmm(A, B) + beta * C with a fixed, shape-derived tile
    config.  No autotuning: every DDP rank must pick the same kernel so the
    Muon updates stay bit-identical across ranks."""
    batch, m, k = A.shape
    n = B.shape[-1]
    block_n = 256 if n % 256 == 0 else 128
    block_k = 64 if k % 64 == 0 else 32
    if C is None:
        C, strides_c = out, (0, 0, 0)
    else:
        strides_c = C.stride()
    grid = ((m // 128) * (n // block_n), batch)
    _bmm_epilogue_kernel[grid](
        A, B, C, out, m, n, k,
        *A.stride(), *B.stride(), *strides_c, *out.stride(),
        float(alpha), float(beta), HAS_C=C is not out,
        BLOCK_M=128, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=8,
        num_warps=8 if block_n == 256 else 4,
        num_stages=3 if block_n == 256 else 4,
    )
    return out


def _newtonschulz5_batched(X: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize a (B, r, c) stack of same-shape matrices (one bmm chain
    per iteration -- the step is kernel-launch-bound on GB10 otherwise).

    The staging stack is disposable, so normalize it in place and ping-pong
    explicit output buffers across iterations. Besides avoiding allocator work,
    this preserves the transpose-friendly layout of tall expert matrices.

    On CUDA the three products run through a Triton batched GEMM with the
    alpha*A@B + beta*C epilogue fused in: cuBLAS's batched bf16 heuristics on
    GB10 (sm_121) pick 32x32 wmma kernels (~19 TFLOP/s) and baddbmm copies X
    into the output before accumulating; the Triton kernel reaches 50-60
    TFLOP/s with the same fp32-accumulate math.  Other devices/shapes use the
    cuBLAS chain.
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
    use_triton = X.is_cuda and _bmm_epilogue_supported(X.shape[1], X.shape[2])
    for _ in range(steps):
        if use_triton:
            _bmm_epilogue(X, X.mT, A)
            _bmm_epilogue(A, A, B, C=A, alpha=c, beta=b)
            _bmm_epilogue(B, X, next_X, C=X, alpha=1.0, beta=a)
        else:
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
        by_shape: dict[tuple[int, int, int], list] = {}
        for p, g, v in zip(params, grads, vs):
            if p.ndim not in (2, 3):
                raise ValueError(f"Muon requires 2-D or 3-D parameters, got {p.shape}")
            by_shape.setdefault((p.ndim, p.shape[-2], p.shape[-1]), []).append((p, g, v))
        for (ndim, rows, cols), items in by_shape.items():
            batch_sizes = [1 if ndim == 2 else p.shape[0] for p, _, _ in items]
            stacked = torch.empty(
                (sum(batch_sizes), rows, cols), device=items[0][0].device,
                dtype=torch.bfloat16,
            )
            offset = 0
            for (p, g, v), batch_size in zip(items, batch_sizes):
                if ndim == 2:
                    g, v = g.unsqueeze(0), v.unsqueeze(0)
                staged = stacked[offset : offset + batch_size]
                # Momentum update v <- lerp(v, g, 1-m) and the Nesterov update
                # lerp(g, v, m) written straight into the bf16 Newton-Schulz
                # staging stack (fp32 math, one rounding). On CUDA both happen
                # in one fused Triton pass: one read of g and v, one fp32 and
                # one bf16 write. The optimizer step is bandwidth bound on
                # GB10, so these passes are most of its non-GEMM time. The
                # torch path below is bit-identical and serves CPU/edge cases.
                if group["nesterov"] and _momentum_stage_supported(g, v, staged, momentum):
                    _momentum_stage(g, v, staged, momentum)
                else:
                    v.lerp_(g, 1 - momentum)
                    if group["nesterov"]:
                        torch.lerp(g, v, momentum, out=staged)
                    else:
                        staged.copy_(v)
                offset += batch_size
            ortho = _newtonschulz5_batched(stacked, steps=group["ns_steps"])
            # aspect-ratio factor; also what makes Muon lr width-invariant
            lr = group["lr"] * max(1, rows / cols) ** 0.5
            if ndim == 2:
                ortho_updates = list(ortho.unbind(0))
            else:
                ortho_updates = list(ortho.split(batch_sizes, dim=0))
            torch._foreach_add_(
                [p for p, _, _ in items],
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
    # One packed all-reduce instead of four latency-bound ones per step.
    packed = torch.cat([
        hard_counts.detach(),
        soft_sums.detach(),
        entropy_sum.detach().reshape(1),
        valid_rows.detach().reshape(1),
    ])
    if distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    global_counts = packed[:num_experts]
    global_soft_sums = packed[num_experts:2 * num_experts]
    global_entropy_sum = packed[2 * num_experts]
    global_valid_rows = packed[2 * num_experts + 1]

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
    p.add_argument("--autotune-gemm", action=argparse.BooleanOptionalAction, default=True,
                   help="benchmark inductor Triton GEMM templates against cuBLAS per "
                        "shape (see enable_triton_gemm_autotune); +4.3%% dense on GB10")
    p.add_argument("--fused-swiglu", action=argparse.BooleanOptionalAction, default=True,
                   help="Triton grouped GEMM with the SwiGLU fused into its epilogue for "
                        "the dense MLP and the MoE experts (see _FusedSwiGLU)")
    p.add_argument("--fp8-mlp", action=argparse.BooleanOptionalAction, default=False,
                   help="dense MLP sub-blocks and MoE experts store activations and "
                        "activation grads in fp8 with fp8 GEMMs while training (delayed "
                        "per-tensor scaling; see _FP8MLPBlock / _FP8Experts). Changes "
                        "numerics; validation stays bf16")
    p.add_argument("--fused-qk-rope", action=argparse.BooleanOptionalAction, default=True,
                   help="one Triton pass each way for q/k RMSNorm + RoPE around "
                        "flash-attn (see _QKNormRoPEAttention)")
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
    global FUSED_SWIGLU, FUSED_QK_ROPE, FP8_MLP
    FUSED_SWIGLU = args.fused_swiglu
    FUSED_QK_ROPE = args.fused_qk_rope
    FP8_MLP = args.fp8_mlp

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
        if args.autotune_gemm:
            enable_triton_gemm_autotune()
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
        if FP8_MLP:  # the prewarm step's amax seeds the first real step's fp8 scales
            fp8_update_scales(raw_model)
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
        if FP8_MLP:
            fp8_update_scales(raw_model)
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
