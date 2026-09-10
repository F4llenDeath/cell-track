from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import shift as ndi_shift

from cell_track.basicpy import _finalize, percentile_normalize, register_translation_stack


class BasicPyUtilityTests(unittest.TestCase):
    def test_percentile_normalize_constant_frame(self) -> None:
        result = percentile_normalize(np.ones((8, 8), dtype=np.float32))
        np.testing.assert_array_equal(result, np.zeros((8, 8), dtype=np.float32))

    def test_registration_recovers_translation(self) -> None:
        reference = np.zeros((48, 48), dtype=np.float32)
        reference[12:22, 15:27] = 3
        reference[31:37, 8:14] = 1
        moved = ndi_shift(reference, shift=(2, -3), order=0, mode="nearest")
        stack = np.stack([reference, moved])
        aligned, shifts = register_translation_stack(
            stack,
            upsample_factor=1,
            show_progress=False,
        )
        self.assertEqual(aligned.shape, stack.shape)
        expected_relative = shifts[1] - shifts[0]
        np.testing.assert_allclose(expected_relative, (-2, 3), atol=0.5)

    def test_finalize_rejects_non_finite_output(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _finalize(np.asarray([[[np.nan]]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
