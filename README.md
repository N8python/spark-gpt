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
on 1B tokens — about 4h on one GB10, 2h on two (~140k tok/s, ~1.95×
single-node). The Qwen3-0.6B shape (440M non-embedding params) is
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
gradient. Metrics include CE and optimization losses, auxiliary loss,
assignment min/max/CV, router entropy, and unused experts. Dense muP transfer
has not been established across expert count, top-k, router initialization, or
auxiliary-loss coefficient; tune those before treating a large MoE run as
canonical.

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
unified memory; `max-autotune` is ~6.5% slower than `default` on sm_121), fp8
matmuls are a net loss below ~2048 hidden dim (dynamic-scaling casts are
bandwidth-bound), and always set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. At the 0.6B shape the
trainer sustains ~14.0k tok/s on a single GB10 at the default 49,152-token
window (torch 2.12.0+cu130, flash-attn 2.8.3.post1). That is fixed-shape
compute-token throughput; real loss-token throughput is reported separately
and equals compute throughput times observed packing utilization.

Controlled 2026-08-21 single-GB10 comparison on the same checksum-pinned
FineWeb stream (`seed=0`, 5,001,341 real tokens, 104 post-prewarm steps,
49,152 tokens/step, compile `default`):

| Model | Total params | Active params | Real tok/s | Window tok/s | Peak allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense 16L/512d/MLP1536 | 50,617,856 | 50,617,856 | 81,606 | 83,408 | 14.08 GiB |
| MoE 8 experts/top-2/I768 | 163,929,600 | 50,683,392 | 56,530 | 57,779 | 18.14 GiB |

The compute-matched MoE delivered 69.27% of dense throughput (30.73% slower)
and used 4.06 GiB more peak allocation. This isolates training systems cost;
the short run is not a model-quality comparison.

## License

MIT
