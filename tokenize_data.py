"""Tokenizers and tokenized document caches for sparkgpt.

Train a byte-level BPE on a uniform random sample of a corpus, then tokenize
the corpus once into memory-mappable shards that train.py reads its batches
from (--train-path / --val-path), instead of re-encoding the corpus and holding
it in RAM on every run:

    python tokenize_data.py train-tokenizer fineweb/sample/10BT tok32k --vocab-size 32768 \\
        --sample-frac 0.022 --val-frac 0.1
    python tokenize_data.py build fineweb/sample/10BT fineweb.tok32k --tokenizer tok32k \\
        --max-doc-tokens 4095 --shuffle-seed 0 --val-frac 0.1 --val-out fineweb.tok32k.val
    python tokenize_data.py build lang_data/fineweb_1b.jsonl lang_data/fineweb_1b.bytes  # byte-level

A source is a jsonl file ({"text": ...} per line) or a directory of parquet
files with a "text" column (e.g. a FineWeb sample), read in sorted file order.

Cache layout (format sparkgpt-doc-cache-v1):
  manifest.json            tokenizer, source checksums, filter/order/split, shard list, totals
  tokenizer.json           the BPE tokenizer (BPE caches only)
  shard_00000.tokens.u16   every document as [BOS] tokens [EOS], concatenated (uint16)
  shard_00000.offsets.i64  (docs + 1) start offsets of each document in the shard
  shard_00000.bytes.i32    UTF-8 byte length of each document's text (for bits per byte)

Documents are never split across shards. Only offsets and byte lengths are read
into RAM; token shards are memory-mapped.

Train/validation split: a document is in validation iff a seeded hash of its
key (the parquet "id" column, else its text) falls in the bottom --val-frac of
the hash range. Membership is decided before tokenization, so the tokenizer can
be trained on the training side only (train-tokenizer --val-frac), and it is the
same for every tokenizer. build then drops over-long documents on both sides
and writes each side in its own seeded permutation, so every prefix of either
cache (what --target-tokens / --val-tokens select) is a uniform sample.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Byte-level tokenizer: ids 0-255 are UTF-8 bytes, then three specials.
BOS, EOS, PAD = 256, 257, 258
VOCAB_SIZE = 259
BYTE_TOKENIZER = {"type": "bytes-v1", "vocab_size": VOCAB_SIZE, "bos": BOS, "eos": EOS, "pad": PAD}

CACHE_FORMAT = "sparkgpt-doc-cache-v1"
MANIFEST = "manifest.json"
TOKENIZER_FILE = "tokenizer.json"
TOKEN_DTYPE = np.uint16  # room for vocabularies up to 65,535
OFFSET_DTYPE = np.int64
BYTES_DTYPE = np.int32
# Appended after training as the last three ids. If a learned token already
# spelled one of these (a corpus full of literal "<|eos|>"), the special would
# reuse its id; train_tokenizer refuses that instead of silently aliasing.
SPECIAL_TOKENS = ("<|bos|>", "<|eos|>", "<|pad|>")
# Superword BPE: no word-level pre-tokenization, so merges may span spaces,
# words and punctuation. The one rule: every decimal digit is its own token
# (never merged with anything), so numbers are always spelled digit by digit.
DIGIT_PATTERN = r"\p{Nd}"


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def source_files(src: Path) -> list[Path]:
    src = Path(src)
    if src.is_dir():
        files = sorted(src.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"{src}: no .parquet files")
        return files
    return [src]


def iter_text_batches(src: Path, batch_docs: int = 8192):
    """(texts, split keys) batches in source order. The key is a document's
    parquet "id" when there is one, else its text."""
    for path in source_files(src):
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            cols = ["text", "id"] if "id" in pf.schema_arrow.names else ["text"]
            for batch in pf.iter_batches(batch_size=batch_docs, columns=cols):
                texts = batch.column(0).to_pylist()
                yield texts, (batch.column(1).to_pylist() if len(cols) == 2 else texts)
        else:
            texts = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    texts.append(json.loads(line)["text"])
                    if len(texts) == batch_docs:
                        yield texts, texts
                        texts = []
            if texts:
                yield texts, texts


def is_val(keys, val_frac: float, split_seed: int) -> np.ndarray:
    """Validation membership: a seeded 64-bit BLAKE2b hash of each key, compared
    against val_frac of the hash range. Stable across runs, files and tokenizers."""
    if not val_frac:
        return np.zeros(len(keys), dtype=bool)
    salt = split_seed.to_bytes(8, "little", signed=True)
    cut = int(val_frac * 2 ** 64)
    return np.fromiter((int.from_bytes(hashlib.blake2b(k.encode("utf-8"), digest_size=8,
                                                       salt=salt).digest(), "little") < cut
                        for k in keys), dtype=bool, count=len(keys))


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 24):
            sha.update(chunk)
    return sha.hexdigest()


def describe_source(src: Path) -> dict:
    src = Path(src)
    root = src if src.is_dir() else src.parent
    return {"path": str(src), "files": [
        {"name": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": file_sha256(p)}
        for p in source_files(src)]}


# --------------------------------------------------------------------------- #
# Tokenizers
# --------------------------------------------------------------------------- #
class ByteEncoder:
    spec = BYTE_TOKENIZER

    def encode_batch(self, texts):
        out = []
        for text in texts:
            raw = text.encode("utf-8")
            out.append((np.frombuffer(raw, dtype=np.uint8).astype(TOKEN_DTYPE), len(raw)))
        return out


class BPEEncoder:
    """A trained tokenizer.json. Special-token strings that occur in the text are
    encoded as ordinary text, never as the special ids."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer
        path = Path(path)
        self.path = path / TOKENIZER_FILE if path.is_dir() else path
        self.tokenizer = Tokenizer.from_file(str(self.path))
        self.tokenizer.encode_special_tokens = True
        ids = [self.tokenizer.token_to_id(t) for t in SPECIAL_TOKENS]
        vocab = self.tokenizer.get_vocab_size(with_added_tokens=True)
        if ids != list(range(vocab - len(SPECIAL_TOKENS), vocab)):
            raise ValueError(f"{self.path}: special tokens {SPECIAL_TOKENS} must be the last ids, got {ids}")
        if vocab > np.iinfo(TOKEN_DTYPE).max + 1:
            raise ValueError(f"vocab {vocab} does not fit {TOKEN_DTYPE.__name__}")
        self.spec = {"type": "bpe-v1", "vocab_size": vocab, "bos": ids[0], "eos": ids[1],
                     "pad": ids[2], "sha256": file_sha256(self.path)}

    def encode_batch(self, texts):
        encs = self.tokenizer.encode_batch_fast(texts, add_special_tokens=False)
        return [(np.asarray(e.ids, dtype=TOKEN_DTYPE), len(t.encode("utf-8")))
                for e, t in zip(encs, texts)]


def load_encoder(tokenizer: str):
    return ByteEncoder() if tokenizer == "bytes" else BPEEncoder(Path(tokenizer))


def train_tokenizer(src: Path, out_dir: Path, *, vocab_size: int = 32768,
                    sample_frac: float = 0.05, seed: int = 0, val_frac: float = 0.0,
                    split_seed: int = 0, log=print) -> dict:
    """Superword byte-level BPE trained on a uniform random sample of src's
    documents (each kept independently with probability sample_frac, seeded).
    Text is split only around single decimal digits; everything else may merge
    across word boundaries. Documents in the validation split (val_frac,
    split_seed; see is_val) are never sampled. The three special tokens take the
    last ids, after vocab_size - 3 learned tokens."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    stats = {"docs": 0, "val_docs_skipped": 0, "sampled_docs": 0, "sampled_bytes": 0}

    def sample():
        for texts, keys in iter_text_batches(src):
            stats["docs"] += len(texts)
            val = is_val(keys, val_frac, split_seed)
            stats["val_docs_skipped"] += int(val.sum())
            for text, keep in zip(texts, (rng.random(len(texts)) < sample_frac) & ~val):
                if keep:
                    stats["sampled_docs"] += 1
                    stats["sampled_bytes"] += len(text)
                    yield text

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(DIGIT_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),  # bytes -> symbols only
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size - len(SPECIAL_TOKENS), min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    start = time.perf_counter()
    tok.train_from_iterator(sample(), trainer)
    learned = tok.get_vocab_size(with_added_tokens=True)
    if learned != vocab_size - len(SPECIAL_TOKENS):
        raise ValueError(f"learned {learned} tokens, wanted {vocab_size - len(SPECIAL_TOKENS)} "
                         "(sample too small for this many merges?)")
    tok.add_special_tokens(list(SPECIAL_TOKENS))
    ids = [tok.token_to_id(t) for t in SPECIAL_TOKENS]
    if ids != list(range(learned, vocab_size)):
        raise ValueError(f"special tokens got ids {ids}, expected the last {len(SPECIAL_TOKENS)}")
    tok.save(str(out_dir / TOKENIZER_FILE))
    info = {"vocab_size": vocab_size, "sample_frac": sample_frac, "seed": seed,
            "split": {"val_frac": val_frac, "split_seed": split_seed}, **stats,
            "seconds": round(time.perf_counter() - start, 1)}
    log(f"trained {vocab_size}-token BPE on {stats['sampled_docs']:,} of {stats['docs']:,} docs "
        f"({stats['sampled_bytes'] / 1e9:.2f} GB) in {info['seconds']:.0f}s -> {out_dir}")
    return info


# --------------------------------------------------------------------------- #
# Reading caches
# --------------------------------------------------------------------------- #
class DocumentTokens:
    """Whole documents ([BOS] tokens [EOS] each) in one or more shards, either
    in-RAM arrays or memory-mapped cache files.

    Document d contributes len_d - 1 (input, target) pairs: inputs t0..t(n-1),
    targets t1..tn, so no target crosses a document boundary. pair_bounds are
    the cumulative pair counts at document boundaries (the packing plan's
    coordinate system). doc_bytes, when known, are each document's UTF-8 byte
    length (for bits-per-byte metrics)."""

    def __init__(self, shards, shard_offsets, shard_bytes=None):
        if len(shards) != len(shard_offsets) or not shards:
            raise ValueError("need one offsets array per token shard")
        self.shards = list(shards)
        self.offsets = [np.asarray(o, dtype=OFFSET_DTYPE) for o in shard_offsets]
        for tokens, offs in zip(self.shards, self.offsets):
            if offs.ndim != 1 or offs.size < 1 or offs[0] != 0 or int(offs[-1]) != len(tokens):
                raise ValueError("shard offsets must start at 0 and end at the shard length")
        self.shard_bytes = None
        if shard_bytes is not None:
            self.shard_bytes = [np.asarray(b, dtype=np.int64) for b in shard_bytes]
            if [b.size for b in self.shard_bytes] != [o.size - 1 for o in self.offsets]:
                raise ValueError("need one byte length per document")
        counts = [o.size - 1 for o in self.offsets]
        self.first_doc = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        lengths = np.concatenate([np.diff(o) for o in self.offsets])
        if np.any(lengths < 2):
            raise ValueError("every document needs at least [BOS] and [EOS]")
        self.pair_bounds = np.concatenate([[0], np.cumsum(lengths - 1)]).astype(np.int64)

    @property
    def num_docs(self) -> int:
        return int(self.first_doc[-1])

    @property
    def num_pairs(self) -> int:
        return int(self.pair_bounds[-1])

    @property
    def num_bytes(self) -> int | None:
        return None if self.shard_bytes is None else int(sum(b.sum() for b in self.shard_bytes))

    def prefix(self, num_docs: int) -> "DocumentTokens":
        """The first num_docs documents (file order)."""
        if not 0 < num_docs <= self.num_docs:
            raise ValueError(f"prefix of {num_docs} docs out of range (have {self.num_docs})")
        shards, offsets, nbytes = [], [], []
        for i, (tokens, offs, first) in enumerate(zip(self.shards, self.offsets, self.first_doc[:-1])):
            take = min(num_docs - int(first), offs.size - 1)
            if take <= 0:
                break
            shards.append(tokens[: int(offs[take])])
            offsets.append(offs[: take + 1])
            if self.shard_bytes is not None:
                nbytes.append(self.shard_bytes[i][:take])
        return DocumentTokens(shards, offsets, nbytes if self.shard_bytes is not None else None)

    def select(self, target_pairs: int | None) -> "DocumentTokens":
        """Documents in file order until the cumulative loss-token (pair) count
        reaches target_pairs, including the document that crosses it."""
        if target_pairs is None or target_pairs >= self.num_pairs:
            return self
        return self.prefix(int(np.searchsorted(self.pair_bounds[1:], target_pairs, side="left")) + 1)

    def pairs(self, start_doc: int, end_doc: int):
        """(inputs, targets) of documents [start_doc, end_doc), concatenated."""
        inputs, targets = [], []
        shard = int(np.searchsorted(self.first_doc, start_doc, side="right")) - 1
        doc = start_doc
        while doc < end_doc:
            first = int(self.first_doc[shard])
            stop = min(end_doc, int(self.first_doc[shard + 1]))
            offs = self.offsets[shard][doc - first: stop - first + 1]
            tokens = np.asarray(self.shards[shard][int(offs[0]): int(offs[-1])])
            local = offs - offs[0]
            keep_in = np.ones(tokens.size, dtype=bool)
            keep_in[local[1:] - 1] = False   # each document's last token has no target
            keep_tg = np.ones(tokens.size, dtype=bool)
            keep_tg[local[:-1]] = False      # each document's first token is never a target
            inputs.append(tokens[keep_in])
            targets.append(tokens[keep_tg])
            doc = stop
            shard += 1
        if not inputs:
            return np.zeros(0, dtype=TOKEN_DTYPE), np.zeros(0, dtype=TOKEN_DTYPE)
        if len(inputs) == 1:
            return inputs[0], targets[0]
        return np.concatenate(inputs), np.concatenate(targets)

    def document(self, doc: int) -> np.ndarray:
        shard = int(np.searchsorted(self.first_doc, doc, side="right")) - 1
        offs = self.offsets[shard]
        local = doc - int(self.first_doc[shard])
        return np.asarray(self.shards[shard][int(offs[local]): int(offs[local + 1])])


def is_document_cache(path: Path) -> bool:
    return (Path(path) / MANIFEST).is_file()


def load_document_cache(path: Path):
    """(DocumentTokens over memory-mapped shards, manifest dict)."""
    path = Path(path)
    manifest = json.loads((path / MANIFEST).read_text())
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError(f"{path}: unsupported cache format {manifest.get('format')!r}")
    shards, offsets, nbytes = [], [], []
    for entry in manifest["shards"]:
        tokens = np.memmap(path / entry["tokens"], dtype=TOKEN_DTYPE, mode="r",
                           shape=(entry["tokens_count"],))
        offs = np.fromfile(path / entry["offsets"], dtype=OFFSET_DTYPE)
        if offs.size != entry["docs"] + 1:
            raise ValueError(f"{path}: {entry['offsets']} has {offs.size - 1} docs, "
                             f"manifest says {entry['docs']}")
        shards.append(tokens)
        offsets.append(offs)
        nbytes.append(np.fromfile(path / entry["bytes"], dtype=BYTES_DTYPE))
    docs = DocumentTokens(shards, offsets, nbytes)
    if docs.num_docs != manifest["docs"] or docs.num_pairs != manifest["loss_tokens"]:
        raise ValueError(f"{path}: shard contents do not match the manifest totals")
    return docs, manifest


def manifest_digest(manifest: dict) -> str:
    """Identity of a cache's contents, independent of where the source lived and
    how the cache was sharded."""
    source = [(f["bytes"], f["sha256"]) for f in manifest["source"]["files"]]
    key = {k: manifest[k] for k in ("format", "tokenizer", "selection", "docs", "tokens")}
    key["source"] = source
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Writing caches
# --------------------------------------------------------------------------- #
class _ShardWriter:
    def __init__(self, out_dir: Path, index: int):
        self.name = f"shard_{index:05d}"
        self.paths = {k: out_dir / f"{self.name}.{ext}" for k, ext in
                      (("tokens", "tokens.u16"), ("offsets", "offsets.i64"), ("bytes", "bytes.i32"))}
        self.handle = self.paths["tokens"].open("wb")
        self.offsets = [0]
        self.nbytes: list[int] = []
        self.pending: list[np.ndarray] = []
        self.pending_count = 0

    @property
    def count(self) -> int:
        return self.offsets[-1]

    def add(self, tokens: np.ndarray, nbytes: int):
        self.pending.append(tokens)
        self.pending_count += tokens.size
        self.offsets.append(self.offsets[-1] + tokens.size)
        self.nbytes.append(nbytes)
        if self.pending_count >= 1 << 24:
            self._flush()

    def _flush(self):
        if self.pending:
            np.concatenate(self.pending).tofile(self.handle)
            self.pending, self.pending_count = [], 0

    def close(self) -> dict:
        self._flush()
        self.handle.close()
        np.asarray(self.offsets, dtype=OFFSET_DTYPE).tofile(self.paths["offsets"])
        np.asarray(self.nbytes, dtype=BYTES_DTYPE).tofile(self.paths["bytes"])
        return {**{k: p.name for k, p in self.paths.items()},
                "docs": len(self.offsets) - 1, "tokens_count": self.count,
                "text_bytes": int(sum(self.nbytes))}


class _CacheWriter:
    """Documents in, shards + manifest out (manifest written last, so a partial
    cache is never mistaken for one)."""

    def __init__(self, out_dir: Path, shard_tokens: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if (self.out_dir / MANIFEST).exists():
            raise FileExistsError(f"{self.out_dir} already holds a cache; remove it first")
        self.shard_tokens = shard_tokens
        self.shards: list[dict] = []
        self.writer = _ShardWriter(self.out_dir, 0)

    def add(self, doc: np.ndarray, nbytes: int):
        if self.writer.count and self.writer.count + doc.size > self.shard_tokens:
            self.shards.append(self.writer.close())
            self.writer = _ShardWriter(self.out_dir, len(self.shards))
        self.writer.add(doc, nbytes)

    def close(self, header: dict) -> dict:
        self.shards.append(self.writer.close())
        docs = sum(s["docs"] for s in self.shards)
        tokens = sum(s["tokens_count"] for s in self.shards)
        manifest = {"format": CACHE_FORMAT, **header, "docs": docs, "tokens": tokens,
                    "loss_tokens": tokens - docs,
                    "text_bytes": sum(s["text_bytes"] for s in self.shards),
                    "shards": self.shards}
        if header["tokenizer"]["type"] != "bytes-v1":
            shutil.copyfile(header["tokenizer_path"], self.out_dir / TOKENIZER_FILE)
        manifest.pop("tokenizer_path", None)
        (self.out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


def build_cache(src: Path, out_dir: Path, *, tokenizer: str = "bytes",
                shard_tokens: int = 1_000_000_000, max_docs: int | None = None,
                max_doc_tokens: int | None = None, max_doc_bytes: int | None = None,
                shuffle_seed: int | None = None, val_frac: float = 0.0, split_seed: int = 0,
                val_out: Path | None = None, log=print) -> dict:
    """Tokenize src into a cache at out_dir (and a validation cache at val_out).

    Each document goes to validation iff is_val(key, val_frac, split_seed);
    train a tokenizer with the same val_frac / split_seed so its sample excludes
    this validation set. Documents
    whose text is longer than max_doc_tokens tokens or max_doc_bytes UTF-8 bytes
    are dropped (never split) on both sides; a byte limit selects the same
    documents for every tokenizer, so caches built with it (and the same split
    and shuffle seed) hold identical documents in identical order. With shuffle_seed each side is written in its own seeded
    permutation; without it, in source order."""
    src, out_dir = Path(src), Path(out_dir)
    if val_frac and (val_out is None or shuffle_seed is None):
        raise ValueError("a validation split needs val_out and shuffle_seed")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    start = time.perf_counter()
    encoder = load_encoder(tokenizer)
    spec = encoder.spec
    bos, eos = spec["bos"], spec["eos"]
    source = describe_source(src)
    stats = {"read_docs": 0, "val_docs": 0, "dropped_docs": 0, "dropped_text_tokens": 0,
             "dropped_text_bytes": 0}

    def documents():
        """(document tokens, text bytes, is validation) for every kept document."""
        for texts, keys in iter_text_batches(src):
            if max_docs is not None:
                keep = max(0, max_docs - stats["read_docs"])
                texts, keys = texts[:keep], keys[:keep]
                if not texts:
                    return
            stats["read_docs"] += len(texts)
            val = is_val(keys, val_frac, split_seed)
            if max_doc_bytes is not None:  # free to check before tokenizing: skip the encode
                nb = [len(t.encode("utf-8")) for t in texts]
                over = [b > max_doc_bytes for b in nb]
                stats["dropped_docs"] += sum(over)
                stats["dropped_text_bytes"] += sum(b for b, o in zip(nb, over) if o)
                texts = [t for t, o in zip(texts, over) if not o]
                val = val[~np.asarray(over, dtype=bool)]
            for (ids, nbytes), v in zip(encoder.encode_batch(texts), val):
                if max_doc_tokens is not None and ids.size > max_doc_tokens:
                    stats["dropped_docs"] += 1
                    stats["dropped_text_tokens"] += int(ids.size)
                    stats["dropped_text_bytes"] += nbytes
                    continue
                stats["val_docs"] += int(v)
                doc = np.empty(ids.size + 2, dtype=TOKEN_DTYPE)
                doc[0], doc[-1] = bos, eos
                doc[1:-1] = ids
                yield doc, nbytes, bool(v)
            if stats["read_docs"] % 1_000_000 < len(texts):
                log(f"  {stats['read_docs']:,} docs read, {time.perf_counter() - start:.0f}s")

    def header(part):
        return {"tokenizer": spec, "tokenizer_path": getattr(encoder, "path", None),
                "source": source,
                "selection": {"max_docs": max_docs, "max_doc_tokens": max_doc_tokens,
                              "max_doc_bytes": max_doc_bytes,
                              "shuffle_seed": shuffle_seed, "val_frac": val_frac,
                              "split_seed": split_seed, "part": part},
                "filter_stats": stats}

    parts = [("train", out_dir)] + ([("val", Path(val_out))] if val_frac else [])
    if shuffle_seed is None:
        writers = {part: _CacheWriter(path, shard_tokens) for part, path in parts}
        for doc, nbytes, v in documents():
            writers["val" if v else "train"].add(doc, nbytes)
        manifests = {part: w.close(header(part)) for part, w in writers.items()}
    else:
        # pass 1: kept documents in source order into scratch caches; pass 2: copy
        # each side out in its own seeded permutation (scratch reads hit the page cache)
        scratch = {part: path.parent / f".{path.name}.unshuffled" for part, path in parts}
        for path in scratch.values():
            if path.exists():
                shutil.rmtree(path)
        writers = {part: _CacheWriter(path, 1 << 62) for part, path in scratch.items()}
        for doc, nbytes, v in documents():
            writers["val" if v else "train"].add(doc, nbytes)
        for w in writers.values():
            w.close(header("scratch"))
        manifests = {}
        for i, (part, path) in enumerate(parts):
            docs, _ = load_document_cache(scratch[part])
            order = np.random.default_rng([shuffle_seed, i]).permutation(docs.num_docs)
            log(f"  shuffling {docs.num_docs:,} {part} docs, {time.perf_counter() - start:.0f}s")
            nbytes = docs.shard_bytes[0]
            w = _CacheWriter(path, shard_tokens)
            for d in order:
                w.add(docs.document(int(d)), int(nbytes[d]))
            manifests[part] = w.close(header(part))
            shutil.rmtree(scratch[part])
    for part, path in parts:
        m = manifests[part]
        log(f"wrote {path}: {m['docs']:,} docs, {m['tokens']:,} tokens in {len(m['shards'])} shard(s)")
    log(f"dropped {stats['dropped_docs']:,} docs over {max_doc_tokens} tokens / {max_doc_bytes} bytes "
        f"({stats['dropped_text_tokens']:,} tokens); {time.perf_counter() - start:.0f}s")
    return manifests["train"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train-tokenizer", help="train a byte-level BPE on a uniform sample")
    t.add_argument("source", help="jsonl file or directory of parquet files")
    t.add_argument("output", help="directory for tokenizer.json")
    t.add_argument("--vocab-size", type=int, default=32768, help="total, including 3 specials")
    t.add_argument("--sample-frac", type=float, default=0.05,
                   help="probability each document is in the training sample")
    t.add_argument("--seed", type=int, default=0, help="sampling seed")
    t.add_argument("--val-frac", type=float, default=0.0,
                   help="never sample the validation split (must match build's)")
    t.add_argument("--split-seed", type=int, default=0)
    b = sub.add_parser("build", help="tokenize a corpus into a document cache")
    b.add_argument("source", help="jsonl file or directory of parquet files")
    b.add_argument("output", help="cache directory to create")
    b.add_argument("--tokenizer", default="bytes",
                   help="'bytes' or a trained tokenizer (directory or tokenizer.json)")
    b.add_argument("--max-doc-tokens", type=int, default=None,
                   help="drop documents whose text has more tokens than this")
    b.add_argument("--max-doc-bytes", type=int, default=None,
                   help="drop documents whose text has more UTF-8 bytes than this "
                        "(the same documents for every tokenizer)")
    b.add_argument("--shuffle-seed", type=int, default=None,
                   help="write documents in a seeded global permutation")
    b.add_argument("--val-frac", type=float, default=0.0,
                   help="fraction of documents (by seeded hash of their id) held out as validation")
    b.add_argument("--split-seed", type=int, default=0)
    b.add_argument("--val-out", default=None, help="validation cache directory")
    b.add_argument("--shard-tokens", type=int, default=1_000_000_000,
                   help="start a new shard before exceeding this many tokens (docs are never split)")
    b.add_argument("--max-docs", type=int, default=None, help="read only the first N source docs")
    args = p.parse_args(argv)
    if args.command == "train-tokenizer":
        train_tokenizer(Path(args.source), Path(args.output), vocab_size=args.vocab_size,
                        sample_frac=args.sample_frac, seed=args.seed, val_frac=args.val_frac,
                        split_seed=args.split_seed)
    else:
        build_cache(Path(args.source), Path(args.output), tokenizer=args.tokenizer,
                    shard_tokens=args.shard_tokens, max_docs=args.max_docs,
                    max_doc_tokens=args.max_doc_tokens, max_doc_bytes=args.max_doc_bytes,
                    shuffle_seed=args.shuffle_seed,
                    val_frac=args.val_frac, split_seed=args.split_seed,
                    val_out=Path(args.val_out) if args.val_out else None)


if __name__ == "__main__":
    sys.exit(main())
