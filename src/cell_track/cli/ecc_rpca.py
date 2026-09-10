from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ECC alignment and robust PCA")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--motion", choices=("translation", "euclidean", "affine"), default="euclidean")
    parser.add_argument("--ecc-max-iterations", type=int, default=200)
    parser.add_argument("--ecc-epsilon", type=float, default=1e-7)
    parser.add_argument("--gaussian-filter-size", type=int, default=5)
    parser.add_argument("--background-percentile", type=float, default=90.0)
    parser.add_argument("--lambda-multiplier", type=float, default=2.0)
    parser.add_argument("--rpca-max-iterations", type=int, default=80)
    parser.add_argument("--rpca-tolerance", type=float, default=1e-7)
    parser.add_argument("--save-aligned-stack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-ecc-residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    import tifffile

    from cell_track.ecc_rpca import EccRpcaConfig, run_ecc_rpca, save_ecc_rpca_result

    args = build_parser().parse_args(argv)
    stack = tifffile.imread(args.input)
    config = EccRpcaConfig(
        motion=args.motion,
        ecc_max_iterations=args.ecc_max_iterations,
        ecc_epsilon=args.ecc_epsilon,
        gaussian_filter_size=args.gaussian_filter_size,
        background_percentile=args.background_percentile,
        lambda_multiplier=args.lambda_multiplier,
        rpca_max_iterations=args.rpca_max_iterations,
        rpca_tolerance=args.rpca_tolerance,
        save_aligned_stack=args.save_aligned_stack,
    )
    result = run_ecc_rpca(stack, config, show_progress=args.progress)
    paths = save_ecc_rpca_result(
        result,
        args.output_dir,
        prefix=args.prefix or args.input.stem.removesuffix("_basicpy_timelapse"),
        save_ecc_residual=args.save_ecc_residual,
    )
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
