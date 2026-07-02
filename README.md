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
- **muP width scaling** (always on): tune hyperparameters on a tiny model,
  transfer them to any width unchanged. The default `muon-lr 4e-3` was tuned at
  width 256 and verified optimal at 512 and 1024.
- **Muon/AdamW hybrid optimizer**: Muon (Newton–Schulz orthogonalized momentum)
  on the 2-D body matrices, AdamW on embeddings/head/gains.
- **Varlen sequence packing**: every batch is exactly `--tokens-per-batch` loss
  tokens — one static shape, zero padding, one `torch.compile` graph. Attention
  is block-diagonal via flash-attn varlen: **no cross-document attention**,
  RoPE restarts per document.
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
```

## Data

One JSONL file, one document per line: `{"text": "..."}`.

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
  `hf_step<N>/` twins on rank 0):

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
trainer sustains ~12.5k tok/s (~41% MFU) on a single GB10.

## License

MIT
