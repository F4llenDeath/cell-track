from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cell_track.common import label_dtype, validate_stack, write_json


class CommonTests(unittest.TestCase):
    def test_validate_stack_accepts_numeric_tyx(self) -> None:
        stack = np.zeros((2, 3, 4), dtype=np.float32)
        self.assertIs(validate_stack(stack), stack)

    def test_validate_stack_rejects_2d_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape \\(T, Y, X\\)"):
            validate_stack(np.zeros((3, 4)))

    def test_label_dtype_does_not_truncate(self) -> None:
        self.assertEqual(label_dtype(65_535), np.dtype(np.uint16))
        self.assertEqual(label_dtype(65_536), np.dtype(np.int32))

    def test_write_json_is_stable_and_creates_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "metadata.json"
            write_json(path, {"b": 2, "a": 1})
            self.assertEqual(path.read_text(), '{\n  "a": 1,\n  "b": 2\n}\n')


if __name__ == "__main__":
    unittest.main()
