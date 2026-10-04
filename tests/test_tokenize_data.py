import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tokenize_data as td
import train



def reference_pairs(flat, offsets):
    """The trainer's original pair-stream construction (pre-cache)."""
    starts = np.asarray(offsets[:-1], dtype=np.int64)
    ends = np.asarray(offsets[1:], dtype=np.int64)
    keep_in = np.ones(flat.shape[0], dtype=bool)
    keep_in[ends - 1] = False
    keep_tg = np.ones(flat.shape[0], dtype=bool)
    keep_tg[starts] = False
    bounds = np.concatenate([[0], np.cumsum(ends - starts - 1)])
    return flat[keep_in], flat[keep_tg], bounds


class DocumentCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        words = ["spark", "gpt", "naïve", "über", "日本語", "😀", "token", "\n", "a"]
        self.fixture = Path(self.tmp.name) / "corpus.jsonl"
        with self.fixture.open("w", encoding="utf-8") as f:
            for _ in range(300):
                text = " ".join(rng.choice(words, size=int(rng.integers(1, 60))))
                f.write(json.dumps({"text": text}) + "\n")
        self.dir = Path(self.tmp.name) / "cache"
        # small shards: packed batches must cross shard boundaries
        self.manifest = td.build_cache(self.fixture, self.dir, shard_tokens=2_000, log=lambda *_: None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cache_matches_jsonl_encoding_and_batches(self):
        self.assertGreater(len(self.manifest["shards"]), 2)
        flat, offsets, loss_tokens, num_docs = train.load_compact_tokenized(self.fixture, target_tokens=None)
        inputs, targets, bounds = reference_pairs(flat, offsets)
        docs, digest, tok = train.load_documents(self.dir, target_tokens=None)
        self.assertEqual(tok, {"spec": td.BYTE_TOKENIZER, "file": None})
        self.assertIsNotNone(digest)
        self.assertEqual((docs.num_docs, docs.num_pairs), (num_docs, loss_tokens))
        np.testing.assert_array_equal(docs.pair_bounds, bounds)
        for window in (512, 4096):
            batches = train.build_whole_document_batches(docs.pair_bounds, window)
            for batch in batches:
                ids, tgt, pos, cu = train.materialize_packed_batch(docs, batch, window, window)
                lo, hi = int(bounds[batch.start_doc]), int(bounds[batch.end_doc])
                np.testing.assert_array_equal(ids[: hi - lo], inputs[lo:hi])
                np.testing.assert_array_equal(tgt[: hi - lo], targets[lo:hi])

    def test_byte_lengths_are_recorded(self):
        docs, _, _ = train.load_documents(self.dir, target_tokens=None)
        texts = [json.loads(line)["text"] for line in self.fixture.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(docs.num_bytes, sum(len(t.encode("utf-8")) for t in texts))

    def test_selection_matches_jsonl_budget(self):
        full, _, _ = train.load_documents(self.dir, target_tokens=None)
        for budget in (0, 1, 777, int(full.pair_bounds[5]), full.num_pairs - 1, full.num_pairs, 10 ** 12):
            _, _, loss_tokens, num_docs = train.load_compact_tokenized(self.fixture, target_tokens=budget)
            docs, _, _ = train.load_documents(self.dir, target_tokens=budget)
            self.assertEqual((docs.num_docs, docs.num_pairs), (num_docs, loss_tokens), budget)

    def test_manifest_identity_and_overwrite_guard(self):
        source = json.loads((self.dir / td.MANIFEST).read_text())["source"]
        self.assertEqual(source["files"][0]["bytes"], self.fixture.stat().st_size)
        _, manifest = td.load_document_cache(self.dir)
        other = Path(self.tmp.name) / "other"
        td.build_cache(self.fixture, other, shard_tokens=10 ** 9, log=lambda *_: None)
        _, manifest2 = td.load_document_cache(other)
        self.assertEqual(td.manifest_digest(manifest), td.manifest_digest(manifest2))  # layout-independent
        third = Path(self.tmp.name) / "third"
        td.build_cache(self.fixture, third, max_docs=3, log=lambda *_: None)
        _, manifest3 = td.load_document_cache(third)
        self.assertNotEqual(td.manifest_digest(manifest), td.manifest_digest(manifest3))
        with self.assertRaises(FileExistsError):
            td.build_cache(self.fixture, self.dir, log=lambda *_: None)



class BPECacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rng = np.random.default_rng(1)
        words = ["the", "spark", "model", "naïve", "über", "日本語", "😀", "token", "2026", "1234567",
                 "can't", "\n\n", "  ", "a", "of the model", "of the model"]
        cls.texts = [" ".join(rng.choice(words, size=int(rng.integers(1, 120)))) for _ in range(400)]
        cls.texts += ["", "x"]  # degenerate documents
        cls.texts += ["a literal <|eos|> and <|bos|> in web text"]  # must encode as plain text
        cls.fixture = root / "corpus.jsonl"
        with cls.fixture.open("w", encoding="utf-8") as f:
            for text in cls.texts:
                f.write(json.dumps({"text": text}) + "\n")
        cls.tok = root / "tok"
        cls.info = td.train_tokenizer(cls.fixture, cls.tok, vocab_size=500, sample_frac=1.0,
                                      val_frac=0.1, log=lambda *_: None)
        cls.enc = td.BPEEncoder(cls.tok)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tokenizer_roundtrips_and_never_emits_specials_from_text(self):
        spec = self.enc.spec
        self.assertEqual((spec["vocab_size"], spec["bos"], spec["eos"], spec["pad"]), (500, 497, 498, 499))
        for text, (ids, nbytes) in zip(self.texts, self.enc.encode_batch(self.texts)):
            self.assertEqual(self.enc.tokenizer.decode(ids.tolist(), skip_special_tokens=False), text)
            self.assertEqual(nbytes, len(text.encode("utf-8")))
            self.assertFalse(np.isin(ids, [spec["bos"], spec["eos"], spec["pad"]]).any())

    def test_digits_are_single_tokens_and_merges_span_words(self):
        vocab = self.enc.tokenizer.get_vocab()
        decoded = {tid: self.enc.tokenizer.decode([tid]) for tid in vocab.values()}
        for tid, text in decoded.items():
            if any(ch.isdigit() for ch in text):
                self.assertEqual(len(text), 1, f"token {tid} {text!r} merges a digit")
        self.assertTrue(any(" " in t.strip() for t in decoded.values()), "no superword tokens learned")
        self.assertIn("of the model", "".join(decoded[i] for i in self.enc.encode_batch(["of the model"])[0][0]))
        self.assertLessEqual(len(self.enc.encode_batch(["of the model"])[0][0]), 2)
        ids, _ = self.enc.encode_batch(["call 1234567 or 89"])[0]
        digit_tokens = [decoded[i] for i in ids.tolist() if decoded[i].isdigit()]
        self.assertEqual(digit_tokens, list("123456789"))

    def test_hf_export_ships_the_bpe_tokenizer_with_bos(self):
        from transformers import AutoTokenizer
        config = train.ModelConfig(vocab_size=self.enc.spec["vocab_size"], hidden_size=128,
                                   num_hidden_layers=1, intermediate_size=256,
                                   num_attention_heads=1, num_key_value_heads=1, head_dim=128)
        out = Path(self.tmp.name) / "hf"
        train.export_hf(train.ByteLM(config), out, {"spec": self.enc.spec,
                                                    "file": self.tok / td.TOKENIZER_FILE})
        hf = AutoTokenizer.from_pretrained(out)
        spec = self.enc.spec
        self.assertEqual((hf.bos_token_id, hf.eos_token_id, hf.pad_token_id),
                         (spec["bos"], spec["eos"], spec["pad"]))
        hf_config = json.loads((out / "config.json").read_text())
        self.assertEqual((hf_config["vocab_size"], hf_config["eos_token_id"]), (spec["vocab_size"], spec["eos"]))
        for text in self.texts[:50]:
            ours, _ = self.enc.encode_batch([text])[0]
            if any(name in text for name in td.SPECIAL_TOKENS):
                continue  # HF matches special strings in text at inference; training data never does
            self.assertEqual(hf(text)["input_ids"], [spec["bos"]] + ours.tolist())

    def test_tokenizer_sample_excludes_validation_docs(self):
        val = td.is_val(self.texts, 0.1, 0)
        self.assertGreater(val.sum(), 0)
        self.assertEqual(self.info["val_docs_skipped"], int(val.sum()))
        self.assertEqual(self.info["sampled_docs"], len(self.texts) - int(val.sum()))  # sample_frac 1

    def test_shuffled_build_filters_splits_and_covers_kept_docs(self):
        root = Path(self.tmp.name)
        cap = 40
        man = td.build_cache(self.fixture, root / "train", tokenizer=str(self.tok), max_doc_tokens=cap,
                             shuffle_seed=7, val_frac=0.1, val_out=root / "val", shard_tokens=500,
                             log=lambda *_: None)
        encoded = [ids for ids, _ in self.enc.encode_batch(self.texts)]
        val = td.is_val(self.texts, 0.1, 0)
        keep = np.array([ids.size <= cap for ids in encoded])
        want = {part: sorted(tuple(ids) for ids, k, v in zip(encoded, keep, val) if k and v == is_v)
                for part, is_v in (("train", False), ("val", True))}
        self.assertEqual(man["filter_stats"]["dropped_docs"], int((~keep).sum()))
        self.assertGreater(man["filter_stats"]["dropped_docs"], 0)
        self.assertGreater(len(man["shards"]), 1)
        spec = self.enc.spec
        for part in ("train", "val"):
            docs, _ = td.load_document_cache(root / part)
            got = []
            for d in range(docs.num_docs):
                doc = docs.document(d)
                self.assertEqual((doc[0], doc[-1]), (spec["bos"], spec["eos"]))
                self.assertLessEqual(doc.size - 2, cap)
                got.append(tuple(doc[1:-1].tolist()))
            self.assertEqual(sorted(got), want[part], part)  # exactly the kept docs of this side
            if part == "train":
                self.assertNotEqual(got, [tuple(ids) for ids, k, v in zip(encoded, keep, val) if k and not v])
        self.assertTrue((root / "train" / td.TOKENIZER_FILE).is_file())
        # same seeds -> same caches
        man2 = td.build_cache(self.fixture, root / "train2", tokenizer=str(self.tok), max_doc_tokens=cap,
                              shuffle_seed=7, val_frac=0.1, val_out=root / "val2", log=lambda *_: None)
        self.assertEqual(td.manifest_digest(man), td.manifest_digest(man2))
        a, _ = td.load_document_cache(root / "train")
        b, _ = td.load_document_cache(root / "train2")
        np.testing.assert_array_equal(a.pairs(0, a.num_docs)[0], b.pairs(0, b.num_docs)[0])
        # a cache's own copy of tokenizer.json is a valid --tokenizer
        man3 = td.build_cache(self.fixture, root / "train3", tokenizer=str(root / "train" / td.TOKENIZER_FILE),
                              max_doc_tokens=cap, shuffle_seed=7, val_frac=0.1, val_out=root / "val3",
                              log=lambda *_: None)
        self.assertEqual(td.manifest_digest(man), td.manifest_digest(man3))

    def test_byte_limit_gives_identical_documents_across_tokenizers(self):
        root = Path(self.tmp.name)
        kw = dict(max_doc_bytes=200, shuffle_seed=3, val_frac=0.1, log=lambda *_: None)
        td.build_cache(self.fixture, root / "mb_bpe", tokenizer=str(self.tok), val_out=root / "mb_bpe_val", **kw)
        td.build_cache(self.fixture, root / "mb_bytes", val_out=root / "mb_bytes_val", **kw)
        for a_name, b_name in (("mb_bpe", "mb_bytes"), ("mb_bpe_val", "mb_bytes_val")):
            a, _ = td.load_document_cache(root / a_name)
            b, _ = td.load_document_cache(root / b_name)
            self.assertEqual(a.num_docs, b.num_docs)
            for d in range(a.num_docs):
                text_a = self.enc.tokenizer.decode(a.document(d)[1:-1].tolist())
                text_b = bytes(b.document(d)[1:-1].astype("uint8").tolist()).decode("utf-8")
                self.assertEqual(text_a, text_b)
                self.assertLessEqual(len(text_b.encode("utf-8")), 200)

    def test_split_is_a_stable_hash(self):
        keys = [f"<urn:uuid:{i:08d}>" for i in range(20000)]
        v = td.is_val(keys, 0.1, 0)
        self.assertAlmostEqual(v.mean(), 0.1, delta=0.01)
        np.testing.assert_array_equal(v, td.is_val(keys, 0.1, 0))
        self.assertGreater((v != td.is_val(keys, 0.1, 1)).sum(), 0)  # seed matters
        self.assertTrue(np.all(td.is_val(keys, 0.05, 0) <= v))  # nested in val_frac

if __name__ == "__main__":
    unittest.main()
