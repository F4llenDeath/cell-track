from __future__ import annotations

import unittest

import numpy as np

from cell_track.ecc_rpca import EccRpcaConfig, rpca_ialm, run_ecc_rpca, to01


class EccRpcaTests(unittest.TestCase):
    def test_to01_handles_constant_image(self) -> None:
        np.testing.assert_array_equal(to01(np.ones((4, 4))), np.zeros((4, 4)))

    def test_rpca_reconstructs_input(self) -> None:
        low_rank = np.outer(
            np.arange(1, 9, dtype=np.float32),
            np.asarray([1, 2, 3, 4], dtype=np.float32),
        )
        sparse = np.zeros_like(low_rank)
        sparse[1, 1] = 30
        sparse[6, 3] = -20
        matrix = low_rank + sparse
        result = rpca_ialm(
            matrix,
            1 / np.sqrt(max(matrix.shape)),
            tolerance=1e-6,
            max_iterations=300,
            verbose=False,
        )
        relative_error = np.linalg.norm(
            matrix - result.low_rank - result.sparse
        ) / np.linalg.norm(matrix)
        self.assertLess(relative_error, 1e-5)

    def test_run_ecc_rpca_preserves_shape(self) -> None:
        yy, xx = np.mgrid[:24, :24]
        background = (xx + yy).astype(np.float32)
        stack = np.stack([background.copy() for _ in range(4)])
        for t in range(stack.shape[0]):
            stack[t, 5 + t : 8 + t, 10:13] += 30
        result = run_ecc_rpca(
            stack,
            EccRpcaConfig(
                motion="translation",
                ecc_max_iterations=20,
                lambda_multiplier=1.0,
                rpca_max_iterations=20,
                rpca_tolerance=1e-5,
            ),
            show_progress=False,
        )
        self.assertEqual(result.foreground.shape, stack.shape)
        self.assertEqual(result.warps.shape, (4, 2, 3))
        self.assertTrue(np.isfinite(result.foreground).all())


if __name__ == "__main__":
    unittest.main()
