from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import runtime_metadata, validate_stack, write_json


@dataclass(frozen=True)
class BasicPyConfig:
    """Parameters matching the original ``basicpy.ipynb`` workflow."""

    align_frames: bool = True
    upsample_factor: int = 10
    get_darkfield: bool = True
    autotune: bool = True
    save_aligned_stack: bool = False


@dataclass
class BasicPyResult:
    corrected_timelapse: np.ndarray
    corrected_flatfield_only: np.ndarray
    flatfield: np.ndarray
    darkfield: np.ndarray | None
    baseline: np.ndarray | None
    aligned_stack: np.ndarray | None
    shifts_yx: np.ndarray | None
    metadata: dict[str, Any]


def percentile_normalize(
    frame: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.8,
) -> np.ndarray:
    """Normalize one frame to ``[0, 1]`` using robust percentiles."""
    lo, hi = np.percentile(frame, (p_low, p_high))
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.float32)
    output = (frame.astype(np.float32) - lo) / (hi - lo)
    return np.clip(output, 0, 1).astype(np.float32)


def register_translation_stack(
    stack: np.ndarray,
    upsample_factor: int = 10,
    *,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Register frames to the temporal median by phase cross-correlation."""
    from scipy.ndimage import shift as ndi_shift
    from skimage.registration import phase_cross_correlation
    from tqdm import tqdm

    stack = validate_stack(stack)
    if upsample_factor < 1:
        raise ValueError("upsample_factor must be at least 1")

    reference = np.median(stack, axis=0)
    reference_norm = percentile_normalize(reference)
    aligned = np.empty(stack.shape, dtype=np.float32)
    shifts = np.zeros((stack.shape[0], 2), dtype=np.float32)

    iterator = tqdm(stack, desc="register", disable=not show_progress)
    for t, frame in enumerate(iterator):
        moving_norm = percentile_normalize(frame)
        estimated_shift, _error, _phase = phase_cross_correlation(
            reference_norm,
            moving_norm,
            upsample_factor=upsample_factor,
            normalization="phase",
        )
        shifts[t] = estimated_shift[:2]
        aligned[t] = ndi_shift(
            frame.astype(np.float32),
            shift=estimated_shift[:2],
            order=1,
            mode="nearest",
            prefilter=False,
        )
    return aligned, shifts


def _finalize(array: np.ndarray) -> np.ndarray:
    output = np.asarray(array, dtype=np.float32).copy()
    if not np.isfinite(output).all():
        finite_count = int(np.isfinite(output).sum())
        raise RuntimeError(
            "BaSiCPy returned non-finite corrected values "
            f"({finite_count}/{output.size} values are finite). "
            "Check that the movie has enough frames and intensity variation for fitting."
        )
    output -= np.nanmin(output)
    return output


def run_basicpy(
    stack: np.ndarray,
    config: BasicPyConfig = BasicPyConfig(),
    *,
    show_progress: bool = True,
) -> BasicPyResult:
    """Run the BaSiCPy stage and return arrays for CLI or notebook use."""
    from basicpy import BaSiC

    stack = validate_stack(stack)
    if config.align_frames:
        work_stack, shifts = register_translation_stack(
            stack,
            upsample_factor=config.upsample_factor,
            show_progress=show_progress,
        )
        aligned_stack = work_stack if config.save_aligned_stack else None
    else:
        work_stack = stack.astype(np.float32)
        shifts = None
        aligned_stack = None

    model = BaSiC(get_darkfield=config.get_darkfield)
    if config.autotune:
        model.autotune(work_stack, is_timelapse=True)
    model.fit(work_stack)

    transformed = model.transform(work_stack, is_timelapse=True)
    if isinstance(transformed, tuple):
        corrected_timelapse, baseline = transformed
    else:
        corrected_timelapse, baseline = transformed, None

    transformed_flat = model.transform(work_stack, is_timelapse=False)
    corrected_flat = transformed_flat[0] if isinstance(transformed_flat, tuple) else transformed_flat

    corrected_timelapse = _finalize(corrected_timelapse)
    corrected_flat = _finalize(corrected_flat)
    if baseline is None and getattr(model, "baseline", None) is not None:
        baseline = np.asarray(model.baseline).reshape(-1)
    elif baseline is not None:
        baseline = np.asarray(baseline).reshape(-1)

    darkfield_value = getattr(model, "darkfield", None)
    darkfield = None if darkfield_value is None else np.asarray(darkfield_value, dtype=np.float32)

    metadata: dict[str, Any] = {
        "stage": "basicpy",
        "shape": list(stack.shape),
        "input_dtype": str(stack.dtype),
        "parameters": asdict(config),
        "output_timelapse_min": float(np.nanmin(corrected_timelapse)),
        "output_timelapse_max": float(np.nanmax(corrected_timelapse)),
        "output_flatfield_only_min": float(np.nanmin(corrected_flat)),
        "output_flatfield_only_max": float(np.nanmax(corrected_flat)),
        "runtime": runtime_metadata(
            ["BaSiCPy", "numpy", "scipy", "scikit-image", "tifffile"]
        ),
    }
    if shifts is not None:
        metadata["max_abs_shift_dy"] = float(np.abs(shifts[:, 0]).max())
        metadata["max_abs_shift_dx"] = float(np.abs(shifts[:, 1]).max())

    return BasicPyResult(
        corrected_timelapse=corrected_timelapse,
        corrected_flatfield_only=corrected_flat,
        flatfield=np.asarray(model.flatfield, dtype=np.float32),
        darkfield=darkfield,
        baseline=baseline,
        aligned_stack=aligned_stack,
        shifts_yx=shifts,
        metadata=metadata,
    )


def save_basicpy_result(
    result: BasicPyResult,
    output_dir: str | Path,
    *,
    prefix: str,
    input_stack: np.ndarray | None = None,
) -> dict[str, Path]:
    """Save the BaSiCPy output contract used by Nextflow and notebooks."""
    import tifffile

    output_dir = Path(output_dir)
    diagnostics_dir = output_dir / "basicpy_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "corrected_timelapse": output_dir / f"{prefix}_basicpy_timelapse.tif",
        "corrected_flatfield_only": output_dir / f"{prefix}_basicpy_flatfield_only.tif",
        "flatfield": diagnostics_dir / "flatfield.tif",
        "metadata": diagnostics_dir / "metadata.json",
    }
    for key, array in (
        ("corrected_timelapse", result.corrected_timelapse),
        ("corrected_flatfield_only", result.corrected_flatfield_only),
        ("flatfield", result.flatfield),
    ):
        tifffile.imwrite(
            paths[key],
            np.asarray(array, dtype=np.float32),
            photometric="minisblack",
        )

    if result.darkfield is not None:
        paths["darkfield"] = diagnostics_dir / "darkfield.tif"
        tifffile.imwrite(
            paths["darkfield"],
            np.asarray(result.darkfield, dtype=np.float32),
            photometric="minisblack",
        )
    if result.baseline is not None:
        paths["baseline"] = diagnostics_dir / "timelapse_baseline.csv"
        np.savetxt(paths["baseline"], result.baseline, delimiter=",")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        paths["baseline_plot"] = diagnostics_dir / "timelapse_baseline.png"
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(result.baseline)
        ax.set(xlabel="Frame", ylabel="Baseline", title="BaSiC per-frame baseline")
        fig.tight_layout()
        fig.savefig(paths["baseline_plot"], dpi=150)
        plt.close(fig)
    if result.shifts_yx is not None:
        paths["shifts"] = diagnostics_dir / "alignment_shifts_yx.csv"
        np.savetxt(
            paths["shifts"],
            result.shifts_yx,
            delimiter=",",
            header="dy,dx",
            comments="",
        )
    if result.aligned_stack is not None:
        paths["aligned_stack"] = diagnostics_dir / "aligned_stack.tif"
        tifffile.imwrite(
            paths["aligned_stack"],
            np.asarray(result.aligned_stack, dtype=np.float32),
            photometric="minisblack",
        )

    if input_stack is not None:
        input_stack = validate_stack(input_stack, name="input stack")
        if input_stack.shape != result.corrected_timelapse.shape:
            raise ValueError(
                "input_stack must match the corrected output shape: "
                f"{input_stack.shape} != {result.corrected_timelapse.shape}"
            )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        paths["comparison"] = diagnostics_dir / "basicpy_comparison.png"
        frames = sorted(
            {
                0,
                input_stack.shape[0] // 4,
                input_stack.shape[0] // 2,
                input_stack.shape[0] - 1,
            }
        )
        fig, axes = plt.subplots(len(frames), 3, figsize=(13, 4.5 * len(frames)))
        axes = np.atleast_2d(axes)
        for row, t in enumerate(frames):
            panels = (
                ("raw", input_stack[t]),
                ("BaSiCPy timelapse", result.corrected_timelapse[t]),
                ("BaSiCPy flatfield only", result.corrected_flatfield_only[t]),
            )
            for ax, (title, array) in zip(axes[row], panels):
                lo, hi = np.percentile(array, (1, 99.5))
                ax.imshow(array, cmap="gray", vmin=lo, vmax=hi)
                ax.set_title(f"t={t}: {title}", fontsize=10)
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(paths["comparison"], dpi=150)
        plt.close(fig)

    metadata = dict(result.metadata)
    metadata["outputs"] = {key: str(path.name) for key, path in paths.items() if key != "metadata"}
    write_json(paths["metadata"], metadata)
    return paths
