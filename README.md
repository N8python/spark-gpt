# sparkgpt

(Written by Claude Fable 5, with assistance from N8Programs)

A **single-file** byte-level LM pretrainer, tuned for the NVIDIA DGX Spark
(GB10) but happy on any modern CUDA GPU. Everything lives in
[train.py](train.py): the model, the optimizer, the data pipeline, distributed
training, and checkpoint export. No framework, no config system, no second file.

- **Byte-level**: raw UTF-8 bytes, vocab 259 (256 bytes + BOS/EOS/PAD). No
  tokenizer training, no OOV, fully multilingual by construction.
- **Qwen3-shaped model**: GQA, qk-norm, SwiGLU, rotate-half RoPE — numerically
  identical to HF `Qwen3ForCausalLM`, so checkpoints export as stock HF models.
- **Optional Qwen3-MoE layers**: bias-free fp32-softmax top-k routing, packed
  SwiGLU experts, BF16 grouped GEMM, selected-probability renormalization, and
  a globally aggregated load-balancing loss. MoE checkpoints export as stock
  `Qwen3MoeForCausalLM`; dense mode remains the default.
- **muP width scaling** (always on): tune hyperparameters on a tiny model,
  transfer them to any width unchanged. The default `muon-lr 4e-3` was tuned at
  width 256 and verified optimal at 512 and 1024.
- **Muon/AdamW hybrid optimizer**: Muon (Newton–Schulz orthogonalized momentum)
  on 2-D body and per-expert 3-D matrices, AdamW on embeddings/head/gains and
  routers (with no router weight decay).
- **Whole-document varlen packing**: documents are never cut across training
  batches. Every batch still has exactly `--tokens-per-batch` positions — one
  static shape and one `torch.compile` graph — with unused tail positions filled
  by PAD tokens whose targets are ignored. Attention is block-diagonal via
  flash-attn varlen: **no cross-document attention**, and RoPE restarts per
  document. The config and metrics report real-token utilization separately
  from fixed-shape compute throughput.
- **DDP** via `torchrun`, with crash-safe **resume** (optimizer + scheduler +
  data-stream state; every rank checkpoints to its own local disk, so no shared
  filesystem is needed).
- **Ready-to-load HF export**: checkpoints include a directory that loads with
  `AutoModelForCausalLM` / `AutoTokenizer` directly — no `trust_remote_code`
  (the byte tokenizer is a plain `tokenizer.json`: byte-level BPE with an empty
  merge table). Also loads in `mlx_lm` as `model_type: qwen3`.

## Installation

Python ≥ 3.10 and a CUDA GPU. Order matters — flash-attn compiles against
torch, so torch must be installed first:

```bash
# 1. PyTorch with CUDA — pick the index for your CUDA version, see pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 2. flash-attn (pip builds it against the torch you just installed; on an
#    unusual arch like GB10/sm_121 this compiles from source — takes a while)
pip install flash-attn --no-build-isolation

# 3. everything else (wandb/transformers/datasets are optional; see the file)
pip install -r requirements.txt
```

Tested with Python 3.12, torch 2.12.0+cu130, flash-attn 2.8.3, numpy 2.4,
safetensors 0.8, transformers 5.11 on NVIDIA GB10 (DGX Spark). Sanity check:

```bash
python -c "import torch, flash_attn; print(torch.cuda.get_device_name(0), flash_attn.__version__)"
python -m unittest discover -s tests -v
```

## Data

One JSONL file, one document per line: `{"text": "..."}`.

Each document must fit within both `--tokens-per-batch` and
`--max-position-embeddings` after byte tokenization. The trainer raises a clear
error for an oversized document instead of silently splitting its context.

Ready-made example data (the FineWeb slices the defaults point at, including
`fineweb_1b.jsonl`) is available at
[N8Programs/lang_data](https://huggingface.co/datasets/N8Programs/lang_data):

```bash
hf download N8Programs/lang_data --repo-type dataset --local-dir lang_data
```

Or build your own slice — e.g. 1B tokens of FineWeb:

```python
import json
from datasets import load_dataset

budget = 1_000_000_000
with open("lang_data/fineweb_1b.jsonl", "w") as f:
    for row in load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                            split="train", streaming=True):
        f.write(json.dumps({"text": row["text"]}) + "\n")
        budget -= len(row["text"].encode()) + 1
        if budget <= 0:
            break
```

## Train

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # recommended on unified-memory GPUs

# Chinchilla-optimal 50M model on 1B tokens (the defaults):
python train.py --run-name my_run --save-final

# two nodes (run on each node with its own --node-rank; rank 0 is master):
torchrun --nnodes 2 --node-rank <0|1> --nproc-per-node 1 \
    --master-addr <node0-ip> --master-port 29500 \
    train.py --run-name my_run --save-final
```

Defaults are a 50M model (16 layers / 512 dim / 4Q+2KV heads / head_dim 128)
on 1B tokens — about 2.7h on one GB10 (~105k tok/s), ~1.4h on two (~204k
aggregate, ~1.91× single-node). The Qwen3-0.6B shape (440M non-embedding params) is
`--model-layers 28 --model-dim 1024 --attention-heads 16 --kv-heads 8
--intermediate-size 3072` — same hyperparameters, muP transfers them.

Enable compute-matched top-2 MoE feed-forward layers on the default 50M shape:

```bash
python train.py --run-name moe_8x2 --save-final \
    --num-experts 8 --num-experts-per-tok 2 --moe-intermediate-size 768
```

The dense FFN activates width 1536; top-2 experts of width 768 activate the
same aggregate FFN width per token. This model has about 164M total parameters
and 50.7M active parameters. `--decoder-sparse-step N` makes every Nth layer
sparse, while `--mlp-only-layers 0,5` keeps listed zero-based layers dense.
`--no-norm-topk-prob` is available for experiments, but released Qwen3-MoE
checkpoints renormalize selected probabilities.

The Qwen3 auxiliary statistic is computed over all sparse layers and real
(non-filler) tokens. Hard counts are synchronized across DDP ranks, and the
differentiable local proxy is scaled so DDP averaging yields the global-batch
gradient. W&B logs language-model CE as `train/loss` for both dense and MoE
runs, and the raw expert-balancing term as `train/aux_loss` for MoE runs;
metrics also include assignment min/max/CV, router entropy, and unused experts.
Dense muP transfer has not been established across expert count, top-k, router
initialization, or auxiliary-loss coefficient; tune those before treating a
large MoE run as canonical.

**Hyperparameter tuning:** sweep at `--model-dim 256` (minutes per run), keep
`head_dim` 128 / `kv-heads = heads/2` / `intermediate = 3*dim`, and the optimum
transfers to any width in the family.

## Checkpoints, resume, export

- `--save-every N` writes resumable checkpoints (model + optimizer + scheduler
  + step) every N steps, each rank to its own disk; rerun the same command with
  `--resume` after a crash and training rejoins the exact data stream (a
  fingerprint guards against mismatched data/sharding).
- `--val-path lang_data/fineweb_10m_val_fixed_seed0.jsonl` (the val slice ships
  with the example data) evaluates a fixed held-out set every 5% of training
  and at the end — `val/loss` in wandb, `final_val_loss` in `summary.json`.
- `--save-final` writes `model_final.pt` (native fused layout) and
  `checkpoints/<run>/hf/` — the ready-to-load HF directory (periodic saves get
  `hf_step<N>/` twins on rank 0). Dense exports load as `Qwen3ForCausalLM` and
  MoE exports load as `Qwen3MoeForCausalLM`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("checkpoints/my_run/hf")
tok = AutoTokenizer.from_pretrained("checkpoints/my_run/hf")
```

## Performance notes (DGX Spark / GB10)

Hard-won facts, also documented in the train.py docstring: keep
`torch.compile` on mode `default` (`reduce-overhead`'s CUDA-graph pools OOM
unified memory; `max-autotune` never actually tried Triton GEMMs on the 48-SM
GB10 because of inductor's 68-SM gate — `--autotune-gemm`, on by default,
bypasses that gate and lets inductor pick Triton or cuBLAS per GEMM shape),
fp8 matmuls are a net loss below ~2048 hidden dim (dynamic-scaling casts are
bandwidth-bound), and always set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The dense MLP and the
MoE experts run on hand-written Triton grouped GEMMs with the SwiGLU fused
into the epilogue (`--fused-swiglu`, on by default; `torch._grouped_mm` is a
host-synchronizing per-expert cuBLAS loop on sm_121), and the q/k RMSNorm +
RoPE around flash-attn is one Triton pass each way (`--fused-qk-rope`, on by
default). GEMM autotuning
benchmarks on every rank; identical GB10s pick identical kernels (2-node
checkpoints verified bit-identical), but on heterogeneous nodes use
`--no-autotune-gemm`. At the 0.6B shape the
trainer sustains ~15.3-15.5k tok/s on a single GB10 at the default 49,152-token
window (~28.2k aggregate on two before the 2026-09-30 changes; torch
2.12.0+cu130, flash-attn 2.8.3.post1). That is fixed-shape
compute-token throughput; real loss-token throughput is reported separately
and equals compute throughput times observed packing utilization.

Single-GB10 comparison on the same checksum-pinned FineWeb stream (`seed=0`,
5,001,341 real tokens, 104 steps, 49,152 tokens/step, compile `default`,
steady window tok/s over the second half of the run), before and after the
2026-08-23 and 2026-09-30 kernel work (see [CHANGES.md](CHANGES.md)):

| Model | Total params | Active params | Window tok/s @ `40096c0` | @ `cbb60d6` (08-23) | Window tok/s now | Peak allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense 16L/512d/MLP1536 | 50,617,856 | 50,617,856 | 83.7k | 96.8k-99.6k | **106.7k-106.8k** | 14.0 GiB |
| MoE 8 experts/top-2/I768 | 163,929,600 | 50,683,392 | 59.3k | 72.4k-74.9k | **80.1k-80.3k** | 18.1 GiB |
| Dense 0.6B 28L/1024d/MLP3072 | 440,997,888 | 440,997,888 | 13.8k-14.0k | 14.6k-14.8k | **15.3k-15.5k** | 63.2 GiB |

Dense numbers vary by about ±1% between compiles (GEMM autotune picks), so
they are quoted as ranges (the 0.6B and 2026-09-30 ranges are one fresh
compile on each of two GB10s). The 0.6B shape gains less (~11% over
`40096c0`, vs ~28% dense), most likely because GEMMs take a larger share of
the step at width 1024, while most of the kernel work removed overhead and
bandwidth-bound passes. The compute-matched MoE delivers ~75% of dense
throughput (was 69%). Matched 1B-token runs of both defaults (base vs this
tree, same seed, held-out validation every 5%) end within 0.002 nats/byte of
each other — MoE 0.7443 vs 0.7434, dense 0.7737 vs 0.7717 — at 20% (MoE) and
15% (dense) less wall time; the optimizations do not change what is trained.
The 2026-09-30 changes were checked the same way on a 4,100-step (~200M
token) prefix of the dense schedule: held-out loss ahead or tied at all ten
checkpoints, 0.8810 vs 0.8817 at the end.

### Opt-in: `--fp8-mlp`

Dense MLP sub-blocks can keep their activations and activation gradients in
fp8 (e4m3 / e5m2, delayed per-tensor scaling, fp32 master weights, bf16
validation): 50M 106.7k -> 121.2k tok/s (+13.7%), 0.6B 15.4k -> 17.5k (+14%),
2 nodes 232k aggregate, and ~25% less peak memory; MoE experts likewise
(8x2 default 80.2k -> 88.0k, +9.6%). It changes numerics, so it is off by
default; on a matched full 1B-token run it ended ahead of bf16 on held-out loss
(0.7706 vs 0.7723) in 12% less training time, and the MoE default was ahead at
every checkpoint of a 4,100-step run. Details in [CHANGES.md](CHANGES.md).

## License

MIT
