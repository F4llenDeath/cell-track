from __future__ import annotations

import unittest

from cell_track.cli.basicpy import build_parser as basicpy_parser
from cell_track.cli.cellpose import build_parser as cellpose_parser
from cell_track.cli.ecc_rpca import build_parser as ecc_rpca_parser
from cell_track.cli.qc import build_parser as qc_parser
from cell_track.cli.ultrack import build_parser as ultrack_parser


class CliParserTests(unittest.TestCase):
    def test_all_clis_expose_help(self) -> None:
        for parser_factory in (
            basicpy_parser,
            ecc_rpca_parser,
            cellpose_parser,
            ultrack_parser,
            qc_parser,
        ):
            help_text = parser_factory().format_help()
            self.assertIn("--help", help_text)

    def test_qc_accepts_comma_normalized_frame_values(self) -> None:
        args = qc_parser().parse_args(
            [
                "--foreground",
                "foreground.tif",
                "--cellpose-labels",
                "labels.tif",
                "--tracks",
                "tracks.csv",
                "--tracked-labels",
                "tracked.tif",
                "--output-dir",
                "qc",
                "--frames",
                "0",
                "22",
                "43",
            ]
        )
        self.assertEqual(args.frames, [0, 22, 43])


if __name__ == "__main__":
    unittest.main()
