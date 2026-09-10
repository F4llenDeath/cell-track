from __future__ import annotations

import tempfile
import unittest

import numpy as np
import pandas as pd

from cell_track.qc import generate_qc


class QcTests(unittest.TestCase):
    def test_generate_qc_outputs(self) -> None:
        foreground = np.zeros((3, 16, 16), dtype=np.float32)
        labels = np.zeros((3, 16, 16), dtype=np.uint16)
        tracked = np.zeros_like(labels)
        rows = []
        for t in range(3):
            foreground[t, 4:8, 5 + t : 9 + t] = 1
            labels[t, 4:8, 5 + t : 9 + t] = 1
            tracked[t, 4:8, 5 + t : 9 + t] = 7
            rows.append({"track_id": 7, "t": t, "y": 5.5, "x": 6.5 + t})
        tracks = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_qc(foreground, labels, tracks, tracked, directory)
            self.assertEqual(
                set(paths),
                {
                    "cells_per_frame",
                    "segmentation_overlays",
                    "track_durations",
                    "tracking_overlays",
                    "summary",
                },
            )
            self.assertTrue(all(path.exists() for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
