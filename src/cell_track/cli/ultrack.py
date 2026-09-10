from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track Cellpose labels with Ultrack")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--working-dir", required=True, type=Path)
    parser.add_argument("--contour-sigma", type=float, default=4.0)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--max-area", type=int, default=20_000)
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--appear-weight", type=float, default=-1.0)
    parser.add_argument("--disappear-weight", type=float, default=-1.0)
    parser.add_argument("--division-weight", type=float, default=-0.1)
    parser.add_argument("--power", type=int, default=4)
    parser.add_argument("--bias", type=float, default=-0.001)
    parser.add_argument("--solution-gap", type=float, default=0.0)
    parser.add_argument("--time-limit", type=int, default=36_000)
    parser.add_argument("--solver-name", default="")
    parser.add_argument("--save-database", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    import tifffile

    from cell_track.tracking import UltrackConfig, run_ultrack, save_ultrack_result

    args = build_parser().parse_args(argv)
    labels = tifffile.imread(args.input)
    config = UltrackConfig(
        contour_sigma=args.contour_sigma,
        min_area=args.min_area,
        max_area=args.max_area,
        max_distance=args.max_distance,
        n_workers=args.n_workers,
        appear_weight=args.appear_weight,
        disappear_weight=args.disappear_weight,
        division_weight=args.division_weight,
        power=args.power,
        bias=args.bias,
        solution_gap=args.solution_gap,
        time_limit=args.time_limit,
        solver_name=args.solver_name,
        save_database=args.save_database,
    )
    result = run_ultrack(labels, config, working_dir=args.working_dir)
    paths = save_ultrack_result(result, args.output_dir, save_database=args.save_database)
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
