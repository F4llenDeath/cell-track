from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BaSiCPy preprocessing on a (T,Y,X) TIFF stack")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--align-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upsample-factor", type=int, default=10)
    parser.add_argument("--get-darkfield", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-aligned-stack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    import tifffile

    from cell_track.basicpy import BasicPyConfig, run_basicpy, save_basicpy_result

    args = build_parser().parse_args(argv)
    stack = tifffile.imread(args.input)
    config = BasicPyConfig(
        align_frames=args.align_frames,
        upsample_factor=args.upsample_factor,
        get_darkfield=args.get_darkfield,
        autotune=args.autotune,
        save_aligned_stack=args.save_aligned_stack,
    )
    result = run_basicpy(stack, config, show_progress=args.progress)
    paths = save_basicpy_result(
        result,
        args.output_dir,
        prefix=args.prefix or args.input.stem,
        input_stack=stack,
    )
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
