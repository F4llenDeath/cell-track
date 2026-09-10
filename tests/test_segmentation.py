from __future__ import annotations

import unittest

import numpy as np

from cell_track.segmentation import (
    CellposeConfig,
    build_eval_kwargs,
    normalize_frame,
    run_cellpose,
)


class FakeCellposeModel:
    def __init__(self) -> None:
        self.kwargs = None

    def eval(self, frames, **kwargs):
        self.kwargs = kwargs
        masks = []
        for frame in frames:
            mask = np.zeros_like(frame, dtype=np.uint16)
            mask[frame > 0.5] = 1
            masks.append(mask)
        return masks, None, None


class SegmentationTests(unittest.TestCase):
    def test_normalize_constant_frame(self) -> None:
        result = normalize_frame(np.ones((8, 8), dtype=np.float32))
        np.testing.assert_array_equal(result, np.zeros((8, 8), dtype=np.float32))

    def test_eval_kwargs_match_notebook_defaults(self) -> None:
        kwargs = build_eval_kwargs(CellposeConfig())
        self.assertEqual(kwargs["flow_threshold"], 0.4)
        self.assertEqual(kwargs["cellprob_threshold"], 0.0)
        self.assertFalse(kwargs["normalize"])

    def test_run_cellpose_with_provided_model(self) -> None:
        stack = np.zeros((3, 16, 16), dtype=np.float32)
        stack[:, 4:9, 5:10] = 10
        model = FakeCellposeModel()
        result = run_cellpose(
            stack,
            CellposeConfig(batch_size=2),
            model=model,
            resolved_model="fake",
            resolved_device="cpu",
            show_progress=False,
        )
        self.assertEqual(result.labels.shape, stack.shape)
        np.testing.assert_array_equal(result.cell_counts, np.ones(3, dtype=np.int64))
        self.assertEqual(model.kwargs["batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
