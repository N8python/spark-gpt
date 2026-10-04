import unittest

import numpy as np

from train import (
    LOSS_IGNORE_INDEX,
    PAD,
    DocumentTokens,
    PackedBatch,
    build_whole_document_batches,
    materialize_packed_batch,
    packed_batch_metadata,
    resolve_run_steps,
    resolve_training_schedule,
)


class WholeDocumentPackingTest(unittest.TestCase):
    def test_next_fit_never_splits_and_covers_source_once(self):
        bounds = np.asarray([0, 6, 10, 17, 20], dtype=np.int64)
        batches = build_whole_document_batches(bounds, window=10)

        self.assertEqual(
            [(b.start_doc, b.end_doc, b.real_tokens) for b in batches],
            [(0, 2, 10), (2, 4, 10)],
        )
        covered = []
        for batch in batches:
            covered.extend(range(int(bounds[batch.start_doc]),
                                 int(bounds[batch.end_doc])))
        self.assertEqual(covered, list(range(20)))

    def test_materialization_masks_only_tail_filler(self):
        docs = DocumentTokens([np.arange(8, dtype=np.uint16)], [[0, 8]])  # one 8-token doc
        bounds = docs.pair_bounds
        batch = build_whole_document_batches(bounds, window=10)[0]

        ids, tgt, pos, cu = materialize_packed_batch(
            docs, batch, window=10, max_segment_length=7
        )

        np.testing.assert_array_equal(ids[:7], np.arange(7))
        np.testing.assert_array_equal(tgt[:7], np.arange(1, 8))
        np.testing.assert_array_equal(ids[7:], np.full(3, PAD))
        np.testing.assert_array_equal(tgt[7:], np.full(3, LOSS_IGNORE_INDEX))
        np.testing.assert_array_equal(cu, np.asarray([0, 7, 10], dtype=np.int32))
        np.testing.assert_array_equal(pos, np.asarray([0, 1, 2, 3, 4, 5, 6, 0, 1, 2]))

    def test_filler_is_chunked_to_position_limit(self):
        bounds = np.asarray([0, 2], dtype=np.int64)
        batch = build_whole_document_batches(bounds, window=10)[0]
        cu, pos = packed_batch_metadata(
            bounds, batch, window=10, max_segment_length=2
        )

        np.testing.assert_array_equal(cu, np.asarray([0, 2, 4, 6, 8, 10]))
        np.testing.assert_array_equal(pos, np.tile(np.asarray([0, 1]), 5))
        self.assertLessEqual(int(np.diff(cu).max()), 2)

    def test_ddp_filler_batch_is_fully_masked(self):
        docs = DocumentTokens([np.arange(3, dtype=np.uint16)], [[0, 3]])
        ids, tgt, pos, cu = materialize_packed_batch(
            docs, PackedBatch(0, 0, 0), window=6, max_segment_length=2,
        )

        np.testing.assert_array_equal(ids, np.full(6, PAD))
        np.testing.assert_array_equal(tgt, np.full(6, LOSS_IGNORE_INDEX))
        np.testing.assert_array_equal(cu, np.asarray([0, 2, 4, 6]))
        np.testing.assert_array_equal(pos, np.tile(np.asarray([0, 1]), 3))

    def test_full_batch_has_no_filler_segment(self):
        bounds = np.asarray([0, 6, 10], dtype=np.int64)
        batch = build_whole_document_batches(bounds, window=10)[0]
        cu, pos = packed_batch_metadata(
            bounds, batch, window=10, max_segment_length=6
        )

        np.testing.assert_array_equal(cu, np.asarray([0, 6, 10]))
        np.testing.assert_array_equal(pos, np.asarray([0, 1, 2, 3, 4, 5, 0, 1, 2, 3]))

    def test_oversized_document_is_rejected_instead_of_split(self):
        with self.assertRaisesRegex(ValueError, "whole-document packing will not split"):
            build_whole_document_batches(np.asarray([0, 11]), window=10)


class TrainingScheduleTest(unittest.TestCase):
    def test_step_limit_preserves_available_schedule_by_default(self):
        self.assertEqual(resolve_run_steps(20644, None), 20644)

    def test_step_limit_can_select_exact_prefix(self):
        self.assertEqual(resolve_run_steps(20644, 2065), 2065)

    def test_step_limit_rejects_unavailable_steps(self):
        with self.assertRaisesRegex(ValueError, "exceeds the available"):
            resolve_run_steps(20644, 20645)

    def test_defaults_follow_actual_run_length(self):
        schedule, warmup, val_interval = resolve_training_schedule(
            2067,
            lr_schedule_steps=None,
            warmup_frac=0.02,
            val_interval_frac=0.05,
            val_interval_steps=None,
        )
        self.assertEqual((schedule, warmup, val_interval), (2067, 42, 104))

    def test_short_run_can_be_exact_prefix_of_1b_schedule(self):
        schedule, warmup, val_interval = resolve_training_schedule(
            2067,
            lr_schedule_steps=20644,
            warmup_frac=0.02,
            val_interval_frac=0.05,
            val_interval_steps=413,
        )
        self.assertEqual((schedule, warmup, val_interval), (20644, 413, 413))

    def test_rejects_schedule_shorter_than_actual_run(self):
        with self.assertRaisesRegex(ValueError, "at least the actual"):
            resolve_training_schedule(
                2067,
                lr_schedule_steps=2000,
                warmup_frac=0.02,
                val_interval_frac=0.05,
                val_interval_steps=None,
            )

    def test_rejects_nonpositive_explicit_val_interval(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            resolve_training_schedule(
                2067,
                lr_schedule_steps=20644,
                warmup_frac=0.02,
                val_interval_frac=0.05,
                val_interval_steps=0,
            )


if __name__ == "__main__":
    unittest.main()
