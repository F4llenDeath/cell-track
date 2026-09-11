from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cell_track.tracking import UltrackConfig, build_main_config, run_ultrack


class TrackingTests(unittest.TestCase):
    def test_build_config_maps_notebook_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = build_main_config(UltrackConfig(), directory)
            self.assertEqual(config.segmentation_config.min_area, 50)
            self.assertEqual(config.segmentation_config.max_area, 20_000)
            self.assertEqual(config.data_config.n_workers, 4)
            self.assertEqual(config.segmentation_config.n_workers, 4)
            self.assertEqual(config.linking_config.n_workers, 4)
            self.assertEqual(config.linking_config.max_distance, 80)
            self.assertEqual(config.tracking_config.division_weight, -0.1)
            self.assertEqual(config.tracking_config.solution_gap, 0.0)

    def test_tracks_simple_translation(self) -> None:
        labels = np.zeros((3, 32, 32), dtype=np.uint16)
        for t in range(3):
            labels[t, 10:15, 10 + t : 15 + t] = 1
        with tempfile.TemporaryDirectory() as directory:
            result = run_ultrack(
                labels,
                UltrackConfig(
                    contour_sigma=1.0,
                    min_area=2,
                    max_area=1_000,
                    max_distance=10,
                    n_workers=1,
                ),
                working_dir=Path(directory),
            )
            self.assertEqual(result.tracked_labels.shape, labels.shape)
            self.assertEqual(result.tracks["track_id"].nunique(), 1)
            self.assertEqual(result.tracks["t"].nunique(), 3)
            self.assertEqual(len(result.database_artifacts), 1)
            self.assertTrue(result.database_artifacts[0][1].exists())

    def test_empty_labels_return_valid_empty_result(self) -> None:
        labels = np.zeros((4, 12, 10), dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            result = run_ultrack(
                labels,
                UltrackConfig(n_workers=1),
                working_dir=directory,
            )
            self.assertEqual(result.tracked_labels.shape, labels.shape)
            self.assertEqual(len(result.tracks), 0)
            self.assertEqual(
                list(result.tracks.columns),
                [
                    "track_id",
                    "t",
                    "y",
                    "x",
                    "id",
                    "parent_track_id",
                    "parent_id",
                ],
            )
            self.assertTrue(result.metadata["empty_input"])

    def test_empty_gaps_split_segments_and_preserve_time_axis(self) -> None:
        labels = np.zeros((5, 32, 32), dtype=np.uint16)
        labels[0, 10:15, 10:15] = 1
        labels[1, 10:15, 11:16] = 1
        labels[3, 20:24, 20:24] = 1
        with tempfile.TemporaryDirectory() as directory:
            result = run_ultrack(
                labels,
                UltrackConfig(
                    contour_sigma=1.0,
                    min_area=2,
                    max_area=1_000,
                    max_distance=10,
                    n_workers=1,
                ),
                working_dir=directory,
            )
            self.assertEqual(result.tracked_labels.shape, labels.shape)
            self.assertEqual(sorted(result.tracks["t"].unique().tolist()), [0, 1, 3])
            self.assertEqual(result.tracks["track_id"].nunique(), 2)
            self.assertEqual(int(result.tracked_labels[2].max()), 0)
            self.assertEqual(int(result.tracked_labels[4].max()), 0)
            self.assertEqual(result.metadata["empty_frames"], [2, 4])


if __name__ == "__main__":
    unittest.main()
