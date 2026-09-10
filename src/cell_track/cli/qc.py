from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate headless QC plots for a completed sample")
    parser.add_argument("--foreground", required=True, type=Path)
    parser.add_argument("--cellpose-labels", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--tracked-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, nargs="*")
    return parser


def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    import tifffile

    from cell_track.qc import generate_qc

    args = build_parser().parse_args(argv)
    paths = generate_qc(
        tifffile.imread(args.foreground),
        tifffile.imread(args.cellpose_labels),
        pd.read_csv(args.tracks),
        tifffile.imread(args.tracked_labels),
        args.output_dir,
        frames=args.frames or None,
    )
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
