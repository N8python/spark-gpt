# sparkgpt

(Written by Claude Fable 5, with assistance from N8Programs)

An LM pretrainer -- byte-level by default, or on a trained superword BPE --
tuned for the NVIDIA DGX Spark (GB10) but happy on any modern CUDA GPU. The training side lives in [train.py](train.py): the
model, the optimizer, the packing and batch pipeline, distributed training, and
checkpoint export. [tokenize_data.py](tokenize_data.py) owns the tokenizer and
writes the memory-mapped document caches training reads. No framework, no
config system.

- **Byte-level by default**: raw UTF-8 bytes, vocab 259 (256 bytes +
  BOS/EOS/PAD). No tokenizer training, no OOV, fully multilingual by
  construction.
- **Or a superword BPE**: `tokenize_data.py` trains a byte-level BPE whose
  merges may span words, with one rule -- every digit is its own token -- and
  [tokenizers/fineweb10bt-superword-32k](tokenizers/fineweb10bt-superword-32k)
  ships a 32k one trained on FineWeb. The tokenizer comes from the data cache:
  `train.py` sizes the vocabulary and special ids from it and exports it with
  the HF checkpoint.
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

**Tokenize once, then train from the cache** (recommended):

```bash
python tokenize_data.py build lang_data/fineweb_1b.jsonl lang_data/fineweb_1b.bytes
python tokenize_data.py build lang_data/fineweb_10m_val_fixed_seed0.jsonl \
    lang_data/fineweb_10m_val_fixed_seed0.bytes
python train.py --train-path lang_data/fineweb_1b.bytes \
    --val-path lang_data/fineweb_10m_val_fixed_seed0.bytes ...
```

The cache is `uint16` token shards (documents never split across shards, ~1B
tokens each by default) plus per-document offsets and a `manifest.json` that
records the tokenizer and the source file's checksum. Training memory-maps the
shards and reads only the documents of each batch, so RAM no longer scales
with the corpus: a 1B-token run loads in 0.1 s and ~1 GB of process memory
(vs ~19 s and 4.5 GB encoding the jsonl). It produces exactly the batches of
the jsonl path. `--train-path` / `--val-path` still accept a jsonl file
directly; that path encodes into RAM at startup. In multi-node runs every node
reads its own copy of the cache, and training refuses to start if the ranks'
caches differ. Tokenizing takes ~5 s per GB of text on a GB10.

**Ready-made BPE caches.** The train and validation caches that the
from-scratch commands below build (FineWeb `sample-10BT`, the shipped
tokenizer, `--max-doc-tokens 4095`, 10% validation split; 6.95B training and
770M validation tokens, 15.5 GB) are on the Hub, ready to train on:

```bash
hf download N8Programs/lang_data --repo-type dataset --local-dir lang_data \
    --include "fineweb10bt.tok32k/*" --include "fineweb10bt.tok32k.val/*"
python train.py --train-path lang_data/fineweb10bt.tok32k \
    --val-path lang_data/fineweb10bt.tok32k.val ...
```

**BPE on FineWeb, from scratch.** The `sample-10BT` subset of
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (15 parquet
files, 30.6 GB) is a uniform random sample of the full dataset:

```bash
hf download HuggingFaceFW/fineweb --repo-type dataset --include "sample/10BT/*" --local-dir fineweb
# optional: train your own tokenizer (14 min, ~65 GB RAM for a 1 GB sample)
python tokenize_data.py train-tokenizer fineweb/sample/10BT my_tok --vocab-size 32768 \
    --sample-frac 0.025 --val-frac 0.1
python tokenize_data.py build fineweb/sample/10BT fw.tok32k \
    --tokenizer tokenizers/fineweb10bt-superword-32k \
    --max-doc-tokens 4095 --shuffle-seed 0 --val-frac 0.1 --val-out fw.tok32k.val
python train.py --train-path fw.tok32k --val-path fw.tok32k.val ...
```

The validation split is a seeded hash of each document's id (10%), decided
before tokenization, so the tokenizer never trains on validation text.
Documents longer than `--max-doc-tokens` are dropped, never split (4095 leaves
room for BOS in 4096 positions: 1.1% of FineWeb's documents, 15% of its
tokens), and each side is written in its own seeded permutation, so any prefix
(`--target-tokens`, `--val-tokens`) is a uniform sample. The shipped tokenizer
gives 5.1 bytes/token on held-out FineWeb; the cache above holds 6.95B
training tokens and builds in ~16 min. Validation reports `val_bpb` (bits per
UTF-8 byte), which is comparable across tokenizers. On identical documents
(<= 4,095 bytes, both caches built with `--max-doc-bytes 4095`) and a matched
~50-minute budget, the 50M model reached 1.273 bpb with this tokenizer vs
1.304 byte-level, in 11% less time (see [CHANGES.md](CHANGES.md)).

Ready-made example data (the FineWeb slices the defaults point at, including
`fineweb_1b.jsonl`) is available at
[N8Programs/lang_data](https://huggingface.co/datasets/N8Programs/lang_data):

```bash
hf download N8Programs/lang_data --repo-type dataset --local-dir lang_data --include "*.jsonl"
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
on 1B tokens — about 2.3h on one GB10 (~121k tok/s), ~1.3h on two (~218k
aggregate, ~1.86× single-node). The Qwen3-0.6B shape (440M non-embedding params) is
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
  fingerprint guards against mismatched data/sharding; with a cache it
  includes the cache's content digest).
- `--val-path lang_data/fineweb_10m_val_fixed_seed0.jsonl` (the val slice ships
  with the example data) evaluates a fixed held-out set every 5% of training
  and at the end — `val/loss` and `val/bpb` (bits per UTF-8 byte, comparable
  across tokenizers) in wandb, `final_val_loss` / `final_val_bpb` in
  `summary.json`.
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
RoPE is one Triton pass forward (`--fused-qk-rope`, on by default). Attention
itself is Triton (`--triton-attention`, on by default): varlen causal GQA whose
backward is deterministic, needs no per-query-head dK/dV buffers or fp32 dQ
accumulator, and applies the RoPE + q/k-norm backward in its epilogues;
`--no-triton-attention` falls back to flash-attn. Each attention block's
input RMSNorm is folded into its qkv GEMM in both directions
(`--fused-norm-qkv`, on by default), and each dense MLP's into its SwiGLU
GEMM (`--fused-norm-mlp`, on by default), and the LM head and cross-entropy
run as one chunked pass that never materializes the logits (`--fused-ce`, on
by default). GEMM autotuning
benchmarks on every rank; identical GB10s pick identical kernels (2-node
checkpoints verified bit-identical), but on heterogeneous nodes use
`--no-autotune-gemm`. At the 0.6B shape the
trainer sustains ~17.5k tok/s on a single GB10 at the default 49,152-token
window (~28.2k aggregate on two before the 2026-09-30 changes; torch
2.12.0+cu130, flash-attn 2.8.3.post1). That is fixed-shape
compute-token throughput; real loss-token throughput is reported separately
and equals compute throughput times observed packing utilization.

Single-GB10 comparison on the same checksum-pinned FineWeb stream (`seed=0`,
5,001,341 real tokens, 104 steps, 49,152 tokens/step, compile `default`,
steady window tok/s over the second half of the run), before and after the
2026-08-23, 2026-09-30 and 2026-10-01 kernel work (see [CHANGES.md](CHANGES.md)):

| Model | Total params | Active params | Window tok/s @ `40096c0` | @ `cbb60d6` (08-23) | Window tok/s now | Peak allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense 16L/512d/MLP1536 | 50,617,856 | 50,617,856 | 83.7k | 96.8k-99.6k | **122.6k-123.0k** | 12.5 GiB |
| MoE 8 experts/top-2/I768 | 163,929,600 | 50,683,392 | 59.3k | 72.4k-74.9k | **87.7k** | 17.4 GiB |
| Dense 0.6B 28L/1024d/MLP3072 | 440,997,888 | 440,997,888 | 13.8k-14.0k | 14.6k-14.8k | **17.4k-17.6k** | 57.3 GiB |

Dense numbers vary by about ±1% between compiles (GEMM autotune picks), so
they are quoted as ranges (the 0.6B range is one fresh compile on each of
two GB10s, as are the current dense numbers; the current MoE number is a
single run). The 0.6B shape gains less (~26% over `40096c0`, vs ~46% dense), most likely because GEMMs take a larger
share of the step at width 1024, while most of the kernel work removed
overhead and bandwidth-bound passes. The compute-matched MoE delivers ~73% of
dense throughput (was 69%). Matched 1B-token runs of both defaults (base vs this
tree, same seed, held-out validation every 5%) end within 0.002 nats/byte of
each other — MoE 0.7443 vs 0.7434, dense 0.7737 vs 0.7717 — at 20% (MoE) and
15% (dense) less wall time; the optimizations do not change what is trained.
The 2026-09-30 changes were checked the same way on a 4,100-step (~200M
token) prefix of the dense schedule: held-out loss ahead or tied at all ten
checkpoints, 0.8810 vs 0.8817 at the end; the 2026-10-01 Triton attention
ended at 0.8791 vs 0.8810 on the same protocol, the fused input norm at
0.8809 / 0.8771 vs 0.8791 / 0.8775, and the fused MLP norm at 0.8753 / 0.8765
(two runs each).

### Opt-in: `--fp8-mlp`

Dense MLP sub-blocks can keep their activations and activation gradients in
fp8 (e4m3 / e5m2, delayed per-tensor scaling, fp32 master weights, bf16
validation): 50M 122.8k -> 140.4k tok/s (+14%), 0.6B 17.5k -> 19.6k (+12%),
and ~25% less peak memory; MoE experts likewise (8x2 default 87.7k -> 95.3k,
+9%). It changes numerics, so it is off by
default; on a matched full 1B-token run it ended ahead of bf16 on held-out loss
(0.7706 vs 0.7723) in 12% less training time, and the MoE default was ahead at
every checkpoint of a 4,100-step run. Details in [CHANGES.md](CHANGES.md).

## License

MIT
