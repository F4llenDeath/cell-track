from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment a foreground TIFF stack with Cellpose-SAM")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="cpsam_v2")
    parser.add_argument("--fallback-model", default="cpsam")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--diameter", type=float)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--niter", type=int)
    parser.add_argument("--bsize", type=int)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lower-quantile", type=float, default=0.001)
    parser.add_argument("--upper-quantile", type=float, default=0.9999)
    parser.add_argument("--save-normalized-stack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    import tifffile

    from cell_track.segmentation import CellposeConfig, run_cellpose, save_cellpose_result

    args = build_parser().parse_args(argv)
    stack = tifffile.imread(args.input)
    config = CellposeConfig(
        model=args.model,
        fallback_model=args.fallback_model,
        device=args.device,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        batch_size=args.batch_size,
        niter=args.niter,
        bsize=args.bsize,
        gamma=args.gamma,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        save_normalized_stack=args.save_normalized_stack,
    )
    result = run_cellpose(stack, config, show_progress=args.progress)
    paths = save_cellpose_result(result, args.output_dir)
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
