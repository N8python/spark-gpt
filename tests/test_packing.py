import unittest

import numpy as np

from train import (
    LOSS_IGNORE_INDEX,
    PAD,
    PackedBatch,
    build_whole_document_batches,
    materialize_packed_batch,
    packed_batch_metadata,
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
        bounds = np.asarray([0, 7], dtype=np.int64)
        inputs = np.arange(7, dtype=np.int16)
        targets = np.arange(10, 17, dtype=np.int16)
        batch = build_whole_document_batches(bounds, window=10)[0]

        ids, tgt, pos, cu = materialize_packed_batch(
            inputs, targets, bounds, batch, window=10, max_segment_length=7
        )

        np.testing.assert_array_equal(ids[:7], inputs)
        np.testing.assert_array_equal(tgt[:7], targets)
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
        bounds = np.asarray([0, 2], dtype=np.int64)
        ids, tgt, pos, cu = materialize_packed_batch(
            np.arange(2), np.arange(2), bounds, PackedBatch(0, 0, 0),
            window=6, max_segment_length=2,
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


if __name__ == "__main__":
    unittest.main()
