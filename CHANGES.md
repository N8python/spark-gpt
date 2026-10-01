# Changes

## 2026-10-01 — `--fp8-mlp`: fp8-storage dense MLP (opt-in)

At dim 512 the MLP kernels are bandwidth-bound on their activations, not
their weights (a weight is ~3 MB; `gu`, `h` and their gradients are
75-450 MB per layer), so fp8 pays only if the *activations* and *activation
gradients* live in fp8 end to end. `_FP8MLPBlock` replaces the whole
post-attention sub-block (RMSNorm -> SwiGLU MLP -> residual):

- forward: RMSNorm writes `x` straight to e4m3 (row-major for the GEMM, plus
  a token-contiguous copy for the weight gradient); the fused SwiGLU GEMM
  (fp8 MMA) writes `gu` and `h` in e4m3; the down projection (fp8 MMA) adds
  the residual in its epilogue.
- backward: the down-proj dgrad runs the SwiGLU backward in its epilogue and
  writes `d_gu` once, in e5m2 (`dh` is never stored; with fp8 storage this is
  cheaper than the bf16 design's recompute-in-every-consumer); gate_up dgrad is
  a cuBLAS fp8 GEMM (`_scaled_mm`), gate_up and down wgrad are Triton fp8
  GEMMs (the down wgrad quantizes `dy` to e5m2 in registers); one kernel does
  the RMSNorm backward + residual gradient.
- scaling: delayed per-tensor power-of-two scales with one binade of
  headroom; kernels record each tensor's amax with atomics and the trainer
  rolls them into the next step's scales after the prewarm step and every
  optimizer step. Master weights stay fp32 (an e4m3 copy per call); weight and
  norm-weight gradients are fp32. Validation and export use the bf16 path.

fp8 MMA on GB10 wants both operands contiguous along the reduction: the
gate_up dgrad takes a transposed copy of the (tiny) fp8 weight, the wgrad a
transposed copy of fp8 `x` written by the norm kernel. Block-scaled MXFP8
(`tl.dot_scaled`) is slower than bf16 on sm_121 in Triton 3.7 (emulated), so
the scaling is per tensor.

Throughput, same protocol as below (5M tokens, 104 steps, seed 0, steady
window over the second half, fresh compile per run), one run per tree on each
GB10 (alice, bob):

| | `b9f2292` (bf16) | `--fp8-mlp` | |
| --- | ---: | ---: | ---: |
| dense 50M | 106.4k-106.7k (461 ms/step) | **121.2k (405 ms/step)** | +13.7% |
| dense 0.6B | 15.3k-15.5k | **17.4k-17.7k** | +14% |
| dense 50M, 2 nodes | 204k aggregate | **232k aggregate** (1.91x) | +13.7% |
| peak alloc 50M / 0.6B | 13.96 / 63.2 GiB | 10.26 / 49.8 GiB | |

The MoE default is unaffected (its MLPs are all sparse; `--mlp-only-layers`
dense layers do take the fp8 path).

Quality, same 4,100-step parity protocol as the 2026-09-30 entry
(held-out loss; `b9f2292` bf16 vs this tree):

| step | 410 | 820 | 1230 | 1640 | 2050 | 2460 | 2870 | 3280 | 3690 | 4100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 1.3692 | 1.0989 | 1.0297 | 0.9878 | 0.9573 | 0.9332 | 0.9137 | 0.8986 | 0.8876 | 0.8810 |
| fp8 fake-quant (just-in-time scales) | 1.3604 | 1.0966 | 1.0266 | 0.9855 | 0.9551 | 0.9315 | 0.9120 | 0.8971 | 0.8862 | 0.8799 |
| `--fp8-mlp` | 1.3396 | 1.0897 | 1.0234 | 0.9824 | 0.9523 | 0.9286 | 0.9087 | 0.8933 | 0.8824 | 0.8762 |

`--fp8-mlp` is ahead at every checkpoint. Not a precision effect (the real
kernels are more precise than the fake-quant emulation in several places):
a saturation count shows the forward tensors never clip, while early-training
gradient spikes in `d_gu` and `dy` do (steps 1-199: ~0.3-0.8 clipped
layer-tensors per step, up to 4.7x over range; rare after step 200) — delayed
scaling acts as elementwise gradient clipping where training is spikiest.
`FP8_HEADROOM_BINADES` trades that off. The flag stays opt-in because it
changes numerics; per-element error vs fp32 is 5-8% on every MLP output and
gradient.

## 2026-09-30 — fused q/k norm + RoPE, SwiGLU backward tiles

Kernel-level trace of the compiled 50M step (489 ms, no idle gaps): GEMMs
261 ms (the 147 ms of fused-SwiGLU GEMMs included), flash-attn 105 ms,
q/k norm + RoPE 56 ms, main RMSNorm 42 ms, optimizer 20 ms. At dim 512
everything outside attention is bandwidth-bound — the non-SwiGLU GEMMs
already run at ~190-237 GB/s — so the remaining wins are passes removed,
not faster kernels.

1. **`--fused-qk-rope` (default on), `_QKNormRoPEAttention`.** The q/k
   RMSNorm + RoPE path was 2 forward and 7 backward inductor kernels moving
   ~575 MB/layer in the backward against a ~275 MB minimum: two
   weight-gradient reductions re-read the RoPE gradient, the forward saved
   contiguous copies of pre-norm q/k that already live in `qkv`, and one
   kernel only copied dv into the dqkv buffer. Now one Triton kernel each
   way: the forward reads q/k straight out of `qkv` and writes rotated q/k +
   a per-head rstd; flash-attn writes dv directly into its slice of dqkv;
   the backward does RoPE + norm backward for every q/k head of a token block
   in one pass with deterministic per-program norm-weight partial sums.
   Attention block fwd+bwd 15.2 -> 13.3 ms/layer. Numerics: the forward is
   correctly rounded against an fp64 reference (the compiled unfused path:
   70%), and every gradient (dq, dk, dv, both norm weights) is closer to fp64
   than the unfused path's.
2. **SwiGLU backward tiles.** dgrad 64x64x32/4 warps -> 128x128x32/8 warps,
   wgrad (32, 64, 128)/4 warps -> (64, 64, 256)/8 warps: halves the per-tile
   SwiGLU-derivative recompute. Bit-identical output; dgrad 1.08x dense /
   1.19x MoE, wgrad 1.07x / 1.06x. The forward's 64x64x32 was already the
   best of 49 configs.

Found along the way: inductor launches user Triton kernels with runtime
Python floats typed **fp64** (Triton's own launcher uses fp32), which made
the first fused forward 2.9x slower in-graph than standalone; float kernel
arguments must be `tl.constexpr`.

Same protocol as below (5M tokens, 104 steps, seed 0, steady window over the
second half, fresh compile per run), one run per tree on each GB10:

| | dense 50M | MoE 8x2 | 0.6B | peak alloc |
| --- | ---: | ---: | ---: | --- |
| `cbb60d6` | 99.6k (493 ms/step) | 74.8k-74.9k (656 ms) | 14.6k-14.8k | unchanged |
| + SwiGLU tiles only | 100.8k-100.9k (+1.2%) | | | |
| 2 nodes (alice + bob), dense | | | | 192k -> **204k aggregate** (1.91x) |
| + fused q/k norm + RoPE | **106.7k-106.8k (+7.2%, 460 ms)** | **80.1k-80.3k (+7.1%, 612 ms)** | **15.3k-15.5k (+4.5%)** | 13.96 / 18.10 / 63.2 GiB |

Matched dense run, `cbb60d6` (bob) vs this tree (alice): the 1B-token
schedule truncated at 4,100 steps (~200M tokens, `--max-train-steps 4100`),
held-out `fineweb_10m_val_fixed_seed0` every 410 steps. This tree is ahead or
tied at every checkpoint and trains the 4,100 steps 7.1% faster
(1,884 s vs 2,027 s):

| step | 410 | 820 | 1230 | 1640 | 2050 | 2460 | 2870 | 3280 | 3690 | 4100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cbb60d6` val | 1.3744 | 1.1010 | 1.0305 | 0.9878 | 0.9581 | 0.9342 | 0.9144 | 0.8993 | 0.8883 | 0.8817 |
| this tree | 1.3692 | 1.0989 | 1.0297 | 0.9878 | 0.9573 | 0.9332 | 0.9137 | 0.8986 | 0.8876 | 0.8810 |

Measured and not adopted: fp8 GEMMs at dim 512 (`_scaled_mm` is 1.25-2x
faster per GEMM, saving at most ~74 ms/step, but operand quantization costs
~143 ms/step even with delayed scaling, and fp8 dgrad would re-materialize
`d_gate_up`); a custom flash-attn build (its main backward kernel already
runs at ~55 TFLOP/s; the recoverable ~3.5% is in side passes — the GQA
dk/dv sum and fp32 dq conversion — that need a patched FA2 extension).

## 2026-08-23 — GB10 throughput optimization (dense 50M and MoE 8x2)

Two parallel optimization sessions on top of `40096c0`, merged here. The
training recipe is unchanged: whole-document no-split packing, data schedule,
resume fingerprint, checkpoint layout and HF export are all as before; every
change is at the kernel / overhead level, and the defaults still train the
same model (verified with matched 1B-token runs below).

Protocol for single-node numbers: one GB10, `fineweb_1b.jsonl`, seed 0,
`--target-tokens 5000000` (104 steps of 49,152 tokens), steady window tok/s
over the second half of the run, compile mode `default`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

| | dense 50M | MoE 8x2 | peak alloc |
| --- | ---: | ---: | --- |
| `40096c0` | 83.7k | 59.3k | 14.08 / 18.13 GiB |
| + attention / optimizer changes (A1-A5) | 88.4k (+5.6%) | 65.2k (+10.0%) | unchanged |
| + MoE dispatch, GEMM autotune, fused SwiGLU (B1-B5) | **96.8k-99.3k (+16-19%)** | **72.4k-74.5k (+22-25%)** | 13.9-14.05 / 18.1 GiB |
| 2 nodes (alice + bob), dense | ~140k before | **192k aggregate**, rank checkpoints bit-identical | |

Dense numbers vary by about ±1% between compiles (GEMM autotuning picks can
differ from one benchmarking pass to the next), so they are quoted as ranges.

### Matched 1B-token runs (base vs this tree, same seed/data/flags, validation on `fineweb_10m_val_fixed_seed0` every 5%)

| path | base final val (nats/byte) | optimized final val | base wall / real tok/s | optimized wall / real tok/s |
| --- | ---: | ---: | ---: | ---: |
| MoE 8x2 | 0.7443 | 0.7434 | 4.91 h / 56.6k | 3.92 h / 70.9k |
| dense 50M | 0.7737 | 0.7717 | 3.48 h / 79.8k | 2.96 h / 94.0k |

Validation loss tracks within 0.001-0.004 nats at every 5% checkpoint on both
paths (the optimized tree is marginally ahead throughout).

### Session A: attention, forward/backward, optimizer

1. **flash-attn varlen traced inside the compiled graph.** flash-attn 2.8
   registers its varlen kernels as torch custom ops, so the per-layer
   `@torch._dynamo.disable` island is gone: one graph per forward/backward
   (1 unique graph across varying document counts; `cu_seqlens` stays
   `mark_dynamic`). +2.4% dense.
2. **Host-side per-step global token counts.** Every rank knows the whole
   packed schedule, so the global real-token count of each DDP step is a
   constant: no per-step all_reduce and no `.item()` sync between backward
   and the optimizer. +0.5%.
3. **Triton batched GEMM for Muon's Newton-Schulz.** cuBLAS's batched bf16
   heuristics on sm_121 pick 32x32 wmma kernels for the NS products
   (19-34 TFLOP/s) and `baddbmm` copies its bias matrix into the output
   first; a small Triton kernel computing `alpha*A@B + beta*C` with fp32
   accumulation runs them at 51-64 TFLOP/s, bit-identical to cuBLAS. Tile
   configs are derived from the shape (never autotuned) so DDP ranks stay
   bit-identical; cuBLAS fallback off-CUDA. Optimizer step 33 -> 22 ms
   (dense), 133 -> 73 ms (MoE).
4. **Nesterov update staged straight into the bf16 NS buffer** and
5. **fused momentum + Nesterov Triton pass** (bit-identical to the two-pass
   `torch.lerp` version; torch path kept as fallback). Optimizer step
   22 -> 17.8 ms (dense), 73 -> 61 ms (MoE).

Not adopted (measured): cuDNN attention (jagged-NT SDPA cannot train in
torch 2.12; the THD `aten::_cudnn_attention_forward` call hangs); torch's
built-in FA2 (same speed); SYRK-style symmetric NS products (3-5%, not
worth a second kernel); cuBLASLt / fp16 / layout changes for the NS bmm.
A custom sm_121 build of flash-attn 2.8.3 with 8 forward / 11 backward
`kernel_traits` variants showed the stock backward config is already the
best that fits GB10's 99 KB shared memory (the 4.7x bwd/fwd ratio is
structural); the forward could gain 9% (~0.3% of a step) by taking FA2's
sm8x 64x64 path on cc 12.x — left as an upstream suggestion.

### Session B: MoE dispatch, GEMM selection, fused SwiGLU

1. **Gather-only expert dispatch** (`_GatherTopK` / `_CombineTopK`): the
   sort-based `index_put_(accumulate=True)` backward and the scatter combine
   become gathers by the inverse permutation. +3.6% MoE.
2. **One packed all-reduce** in `global_load_balancing_loss` (was four).
3. **`--autotune-gemm` (default on).** Inductor's `is_big_gpu()` gate
   (>= 68 SMs) silently disabled Triton GEMM templates on the 48-SM GB10, so
   `mode="max-autotune"` never tried them (hence the old "max-autotune is
   slower" note). With the gate bypassed and only GEMM autotuning enabled,
   Triton wins the output-heavy shapes (gate_up 2.02 vs 2.70 ms, down-proj
   `addmm` 1.24 vs 1.74 ms) and cuBLAS keeps the K=49152 weight gradients.
   +4.3% dense. First uncached compile +20-30 s. Use `--no-autotune-gemm` on
   heterogeneous nodes.
4. **`--fused-swiglu` (default on).** Triton grouped GEMM with the SwiGLU
   epilogue for the dense MLP (E=1) and the experts (E=8, stock weight
   layout), Triton grouped dgrad / down-proj / wgrad kernels with device-side
   offsets (`torch._grouped_mm` on sm_121 is a host-synchronizing per-expert
   cuBLAS loop), and
5. **SwiGLU derivative evaluated inside the dgrad/wgrad GEMM prologues**, so
   the `d_gate_up` buffer and its ~755 MB/layer pass disappear. Rounding
   points match the unfused path. +11-12% on both paths over (3).

Dead ends (don't repeat): coordinate-descent tuning (no change), custom
RMSNorm / qk-norm+RoPE kernels (inductor's are at bandwidth), hand-written
dispatch gather/scatter kernels, FlexAttention with document block masks
(slower than FA2), inductor's own Triton grouped_mm template (TMA, loses to
aten on these shapes), padding the lm_head to 264/320.

### Measured GB10 (sm_121) facts worth keeping

- bf16 cuBLAS GEMM 55-77 TFLOP/s at the default shapes; ~220 GB/s device
  bandwidth; at dim 512 every elementwise pass is bandwidth-bound.
- cuBLAS batched bf16 `bmm`/`baddbmm` is mis-tuned (see A3); cuBLAS `addmm`
  with a full-matrix bias is 1.8-2.5x slower than `mm` + add.
- Inductor refuses Triton GEMM templates below 68 SMs unless `is_big_gpu()`
  is bypassed (B3).
- `F.grouped_mm` is the host-synchronizing fallback (DtoH + stream sync per
  call).
- With `expandable_segments:True` the first ~10-20 steps are slow while the
  allocator grows; benchmark harnesses need >= 20 warm-up steps.
- flash-attn 2.8.3 varlen custom ops trace under `torch.compile`; FA2 fwd
  1.2 ms / bwd 5.4-5.7 ms per layer at the default window.

Where the time goes now (dense, ~500 ms/step): GEMMs ~190 ms, flash-attn
~95 ms (backward 75), SwiGLU/activation traffic ~60 ms, norms/RoPE ~50 ms,
optimizer 18 ms. The remaining headroom without changing the recipe is
mostly the FA2 backward kernel and bf16 activation traffic the recipe
requires.
