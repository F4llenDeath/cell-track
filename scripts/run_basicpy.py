#!/usr/bin/env python3
"""Run BaSiCPy illumination/background correction for the cell-track project.

Why this is a separate script/env:
    BaSiCPy 2.0 pins NumPy/SciPy versions incompatible with the `ultrack`
    notebook environment (installing it there downgrades NumPy to 1.26 and
    breaks numba/ultrack). Run this from the separate `basicpy` conda env
    instead, and write a corrected TIFF that `main.ipynb` (running in the
    `ultrack` env) can simply load.

Recipe (informed by BaSiCPy's official example notebooks):
  - `timelapse_nanog.ipynb`      -> is_timelapse=True corrects per-frame
                                    intensity/contrast drift over time.
  - `timelapse_brightfield.ipynb`-> same modality family as ours
                                    (transmitted/scattered light).
  - `WSI_brain.ipynb`            -> get_darkfield=True models a persistent
                                    ADDITIVE pattern that appears in every
                                    image regardless of content -- in their
                                    case a sensor artifact repeated across
                                    mosaic tiles; in ours, the patterned
                                    hydrogel substrate repeated across time.

None of BaSiCPy's examples handles frame-to-frame misalignment (vibration),
so we do that ourselves first with sub-pixel phase cross-correlation
registration against the temporal median, which BOTH fixes the vibration
you observed AND sharpens BaSiC's darkfield estimate (no longer averaging
a wobbling pattern across frames).

Pipeline:
  1. [optional] Register frames (sub-pixel translation) against the
     temporal median -> aligned stack.
  2. BaSiC(get_darkfield=True)  (WSI-brain style: persistent additive
     pattern across every frame).
  3. basic.autotune(aligned, is_timelapse=True)  (Nanog/brightfield style:
     auto-search smoothness/regularization hyperparameters, accounting for
     temporal drift).
  4. basic.fit(aligned)
  5. Two transform outputs are saved for side-by-side comparison in napari:
       - is_timelapse=True  (also removes per-frame contrast drift)
       - is_timelapse=False (flatfield/darkfield only, no drift removal)
     Your data most resembles the Nanog case (you reported contrast drift),
     so try the `is_timelapse=True` output first, but compare both.

Example:
    mamba run -n basicpy python scripts/run_basicpy.py \\
        --input images/stacked.tif \\
        --align
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from basicpy import BaSiC
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation
from tqdm import tqdm


def percentile_normalize(frame: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    """Normalize one frame to [0, 1] using robust percentiles (registration only)."""
    lo, hi = np.percentile(frame, (p_low, p_high))
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.float32)
    out = (frame.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


def register_translation_stack(
    stack: np.ndarray,
    upsample_factor: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel translational registration using phase cross correlation.

    Registers every frame against the temporal median (robust to moving
    cells, since the substrate/background dominates the median).

    Returns:
        aligned_stack: float32 stack with same shape as input.
        shifts: (T, 2) array of (dy, dx) shifts applied to each frame.
    """
    reference = np.median(stack, axis=0)
    reference_norm = percentile_normalize(reference)

    aligned = np.empty(stack.shape, dtype=np.float32)
    shifts = np.zeros((stack.shape[0], 2), dtype=np.float32)

    for t, frame in enumerate(tqdm(stack, desc="register")):
        moving_norm = percentile_normalize(frame)
        est_shift, _error, _phase = phase_cross_correlation(
            reference_norm,
            moving_norm,
            upsample_factor=upsample_factor,
            normalization="phase",
        )
        shifts[t] = est_shift[:2]
        aligned[t] = ndi_shift(
            frame.astype(np.float32),
            shift=est_shift[:2],
            order=1,
            mode="nearest",
            prefilter=False,
        )

    return aligned, shifts


def save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(array, dtype=np.float32), photometric="minisblack")


def save_baseline_plot(path: Path, baseline: np.ndarray) -> None:
    """Best-effort plot; never let a missing/broken matplotlib fail the pipeline."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  (skipping baseline plot, matplotlib unavailable: {exc})")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.asarray(baseline).reshape(-1))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Baseline")
    ax.set_title("BaSiC per-frame baseline (contrast drift)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=Path("images/stacked.tif"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("images"), help="directory to write corrected stacks into"
    )
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=Path("images/basicpy_diagnostics"),
        help="directory to write flatfield/darkfield/baseline/alignment diagnostics into",
    )
    parser.add_argument("--align", action="store_true", help="register frames before BaSiC fitting/correction")
    parser.add_argument("--upsample-factor", type=int, default=10, help="sub-pixel registration precision")
    parser.add_argument("--no-darkfield", action="store_true", help="disable additive darkfield/pattern estimation")
    parser.add_argument(
        "--skip-autotune",
        action="store_true",
        help="skip basic.autotune() and use BaSiC defaults directly (faster, less accurate)",
    )
    args = parser.parse_args()


    stack = tifffile.imread(args.input)
    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D (T, Y, X) stack, got shape {stack.shape}")

    print(f"Loaded {args.input}: shape={stack.shape}, dtype={stack.dtype}, min={stack.min()}, max={stack.max()}")

    shifts: np.ndarray | None = None
    if args.align:
        print("Registering frames (phase cross-correlation vs. temporal median)...")
        work_stack, shifts = register_translation_stack(stack, upsample_factor=args.upsample_factor)
        save_image(args.diagnostics_dir / "aligned_stack.tif", work_stack)
        np.savetxt(
            args.diagnostics_dir / "alignment_shifts_yx.csv",
            shifts,
            delimiter=",",
            header="dy,dx",
            comments="",
        )
        print(
            f"  max |dy|={np.abs(shifts[:, 0]).max():.2f}px, "
            f"max |dx|={np.abs(shifts[:, 1]).max():.2f}px"
        )
    else:
        work_stack = stack.astype(np.float32)

    # get_darkfield=True: WSI-brain style -- models a persistent additive
    # pattern present in every frame (their sensor artifact; our hydrogel).
    basic = BaSiC(get_darkfield=not args.no_darkfield)

    if not args.skip_autotune:
        print("Autotuning BaSiC hyperparameters (is_timelapse=True)...")
        basic.autotune(work_stack, is_timelapse=True)
    else:
        print("Skipping autotune, using BaSiC defaults.")


    print("Fitting BaSiC (flatfield/darkfield)...")
    basic.fit(work_stack)

    print("Applying correction (transform, is_timelapse=True)...")
    corrected_tl = basic.transform(work_stack, is_timelapse=True)
    baseline = None
    if isinstance(corrected_tl, tuple):
        corrected_tl_stack, baseline = corrected_tl
    else:
        corrected_tl_stack = corrected_tl

    print("Applying correction (transform, is_timelapse=False), for comparison...")
    corrected_flat = basic.transform(work_stack, is_timelapse=False)
    corrected_flat_stack = corrected_flat[0] if isinstance(corrected_flat, tuple) else corrected_flat

    def _finalize(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        arr -= np.nanmin(arr)
        return arr

    corrected_tl_stack = _finalize(corrected_tl_stack)
    corrected_flat_stack = _finalize(corrected_flat_stack)

    out_timelapse = args.output_dir / "stacked_basicpy_timelapse.tif"
    out_flat = args.output_dir / "stacked_basicpy_flatfield_only.tif"
    save_image(out_timelapse, corrected_tl_stack)
    save_image(out_flat, corrected_flat_stack)

    save_image(args.diagnostics_dir / "flatfield.tif", np.asarray(basic.flatfield, dtype=np.float32))
    if getattr(basic, "darkfield", None) is not None:
        save_image(args.diagnostics_dir / "darkfield.tif", np.asarray(basic.darkfield, dtype=np.float32))
    if baseline is not None:
        np.savetxt(args.diagnostics_dir / "timelapse_baseline.csv", np.asarray(baseline).reshape(-1), delimiter=",")
        save_baseline_plot(args.diagnostics_dir / "timelapse_baseline.png", baseline)
    elif getattr(basic, "baseline", None) is not None:
        base_arr = np.asarray(basic.baseline).reshape(-1)
        np.savetxt(args.diagnostics_dir / "timelapse_baseline.csv", base_arr, delimiter=",")
        save_baseline_plot(args.diagnostics_dir / "timelapse_baseline.png", base_arr)

    metadata = {
        "input": str(args.input),
        "output_timelapse": str(out_timelapse),
        "output_flatfield_only": str(out_flat),
        "shape": list(stack.shape),
        "align": args.align,
        "upsample_factor": args.upsample_factor if args.align else None,
        "get_darkfield": not args.no_darkfield,
        "autotuned": not args.skip_autotune,
        "output_timelapse_min": float(np.nanmin(corrected_tl_stack)),

        "output_timelapse_max": float(np.nanmax(corrected_tl_stack)),
        "output_flatfield_only_min": float(np.nanmin(corrected_flat_stack)),
        "output_flatfield_only_max": float(np.nanmax(corrected_flat_stack)),
    }
    if shifts is not None:
        metadata["max_abs_shift_dy"] = float(np.abs(shifts[:, 0]).max())
        metadata["max_abs_shift_dx"] = float(np.abs(shifts[:, 1]).max())

    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (args.diagnostics_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Wrote corrected stack (timelapse-corrected):  {out_timelapse}")
    print(f"Wrote corrected stack (flatfield/darkfield only): {out_flat}")
    print(f"Wrote diagnostics: {args.diagnostics_dir}")
    print("Compare both outputs in napari -- pick whichever removes background drift")
    print("without introducing artifacts around moving cells.")


if __name__ == "__main__":
    main()
