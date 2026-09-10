from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import runtime_metadata, validate_stack, write_json


@dataclass(frozen=True)
class EccRpcaConfig:
    """Parameters matching the ECC + RPCA sections of ``main.ipynb``."""

    motion: str = "euclidean"
    ecc_max_iterations: int = 200
    ecc_epsilon: float = 1e-7
    gaussian_filter_size: int = 5
    background_percentile: float = 90.0
    lambda_multiplier: float = 2.0
    rpca_max_iterations: int = 80
    rpca_tolerance: float = 1e-7
    save_aligned_stack: bool = False


@dataclass
class RpcaSolution:
    low_rank: np.ndarray
    sparse: np.ndarray
    iterations: int
    error: float
    rank: int
    converged: bool


@dataclass
class EccRpcaResult:
    foreground: np.ndarray
    ecc_pattern_subtracted: np.ndarray
    aligned_stack: np.ndarray | None
    warps: np.ndarray
    gains_ab: np.ndarray
    low_rank_aligned: np.ndarray
    sparse_aligned: np.ndarray
    metadata: dict[str, Any]


def to01(array: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    """Robustly normalize an image to ``[0, 1]`` for ECC registration."""
    lo, hi = np.percentile(array, (p_low, p_high))
    output = (array.astype(np.float32) - lo) / max(float(hi - lo), 1e-6)
    return np.clip(output, 0, 1).astype(np.float32)


def _motion_code(motion: str) -> int:
    import cv2

    values = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    try:
        return values[motion.lower()]
    except KeyError as error:
        raise ValueError(f"motion must be one of {sorted(values)}, got {motion!r}") from error


def estimate_ecc_warps(
    stack: np.ndarray,
    config: EccRpcaConfig = EccRpcaConfig(),
    *,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate notebook-compatible ECC warps and pattern gain/bias values."""
    import cv2
    from tqdm import tqdm

    stack = validate_stack(stack).astype(np.float32, copy=False)
    if not 0 < config.background_percentile < 100:
        raise ValueError("background_percentile must be between 0 and 100")

    template = np.median(stack, axis=0).astype(np.float32)
    template01 = to01(template)
    height, width = template.shape
    motion_code = _motion_code(config.motion)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.ecc_max_iterations,
        config.ecc_epsilon,
    )

    residual = np.empty_like(stack)
    warps = np.zeros((stack.shape[0], 2, 3), dtype=np.float32)
    gains = np.zeros((stack.shape[0], 2), dtype=np.float32)
    iterator = tqdm(stack, desc="ECC align + gain/bias", disable=not show_progress)

    for t, frame in enumerate(iterator):
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _cc, warp = cv2.findTransformECC(
                to01(frame),
                template01,
                warp,
                motion_code,
                criteria,
                None,
                config.gaussian_filter_size,
            )
        except cv2.error:
            warp = np.eye(2, 3, dtype=np.float32)

        warped_template = cv2.warpAffine(
            template,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        background_mask = frame < np.percentile(frame, config.background_percentile)
        design = np.stack(
            [
                warped_template[background_mask],
                np.ones(int(background_mask.sum()), dtype=np.float32),
            ],
            axis=1,
        )
        (gain, bias), *_ = np.linalg.lstsq(design, frame[background_mask], rcond=None)
        residual[t] = frame - (gain * warped_template + bias)
        warps[t] = warp
        gains[t] = (gain, bias)

    return np.clip(residual, 0, None).astype(np.float32), warps, gains, template


def rpca_ialm(
    matrix: np.ndarray,
    lam: float,
    *,
    tolerance: float = 1e-7,
    max_iterations: int = 100,
    verbose: bool = True,
) -> RpcaSolution:
    """Inexact augmented Lagrange multiplier Robust PCA solver."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError(f"Expected a non-empty 2D matrix, got {matrix.shape}")
    if lam <= 0:
        raise ValueError("lam must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    norm_two = np.linalg.norm(matrix, 2)
    norm_inf = np.linalg.norm(matrix, np.inf)
    norm_fro = np.linalg.norm(matrix, "fro")
    if norm_fro == 0:
        zeros = np.zeros_like(matrix)
        return RpcaSolution(zeros, zeros.copy(), 0, 0.0, 0, True)

    dual = matrix / max(norm_two, norm_inf / lam)
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    low_rank = np.zeros_like(matrix)
    sparse = np.zeros_like(matrix)
    error = float("inf")
    rank = 0

    for iteration in range(max_iterations):
        left, singular_values, right = np.linalg.svd(
            matrix - sparse + dual / mu,
            full_matrices=False,
        )
        thresholded = np.maximum(singular_values - 1.0 / mu, 0)
        rank = int(np.sum(thresholded > 0))
        low_rank = (left[:, : len(thresholded)] * thresholded) @ right

        temporary = matrix - low_rank + dual / mu
        sparse = np.sign(temporary) * np.maximum(np.abs(temporary) - lam / mu, 0)
        residual = matrix - low_rank - sparse
        dual = dual + mu * residual
        mu = min(mu * rho, mu_bar)
        error = float(np.linalg.norm(residual, "fro") / norm_fro)

        if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
            print(f"RPCA iteration {iteration:3d}: error={error:.6g}, rank={rank}")
        if error < tolerance:
            return RpcaSolution(low_rank, sparse, iteration + 1, error, rank, True)

    return RpcaSolution(low_rank, sparse, max_iterations, error, rank, False)


def run_ecc_rpca(
    stack: np.ndarray,
    config: EccRpcaConfig = EccRpcaConfig(),
    *,
    show_progress: bool = True,
) -> EccRpcaResult:
    """Run ECC alignment and RPCA, returning foreground in raw coordinates."""
    import cv2
    from tqdm import tqdm

    stack = validate_stack(stack).astype(np.float32, copy=False)
    ecc_residual, warps, gains, _template = estimate_ecc_warps(
        stack,
        config,
        show_progress=show_progress,
    )
    timepoints, height, width = stack.shape
    aligned = np.empty_like(stack)
    iterator = enumerate(tqdm(stack, desc="RPCA: aligning frames", disable=not show_progress))
    for t, frame in iterator:
        aligned[t] = cv2.warpAffine(
            frame,
            warps[t],
            (width, height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )

    matrix = aligned.reshape(timepoints, height * width).T
    lam = config.lambda_multiplier / np.sqrt(max(matrix.shape))
    started = time.perf_counter()
    solution = rpca_ialm(
        matrix,
        lam,
        tolerance=config.rpca_tolerance,
        max_iterations=config.rpca_max_iterations,
        verbose=show_progress,
    )
    elapsed = time.perf_counter() - started
    low_rank = solution.low_rank.T.reshape(timepoints, height, width)
    sparse = solution.sparse.T.reshape(timepoints, height, width)

    foreground = np.empty_like(sparse)
    for t in range(timepoints):
        foreground[t] = cv2.warpAffine(
            sparse[t],
            warps[t],
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    metadata = {
        "stage": "ecc_rpca",
        "shape": list(stack.shape),
        "parameters": asdict(config),
        "lambda": float(lam),
        "rpca": {
            "iterations": solution.iterations,
            "error": solution.error,
            "rank": solution.rank,
            "converged": solution.converged,
            "elapsed_seconds": elapsed,
        },
        "max_abs_shift_dx": float(np.abs(warps[:, 0, 2]).max()),
        "max_abs_shift_dy": float(np.abs(warps[:, 1, 2]).max()),
        "gain_range": [float(gains[:, 0].min()), float(gains[:, 0].max())],
        "foreground_min": float(foreground.min()),
        "foreground_max": float(foreground.max()),
        "runtime": runtime_metadata(["numpy", "opencv-python-headless", "tifffile"]),
    }
    return EccRpcaResult(
        foreground=foreground,
        ecc_pattern_subtracted=ecc_residual,
        aligned_stack=aligned if config.save_aligned_stack else None,
        warps=warps,
        gains_ab=gains,
        low_rank_aligned=low_rank,
        sparse_aligned=sparse,
        metadata=metadata,
    )


def save_ecc_rpca_result(
    result: EccRpcaResult,
    output_dir: str | Path,
    *,
    prefix: str,
    save_ecc_residual: bool = False,
) -> dict[str, Path]:
    """Save the ECC/RPCA output contract."""
    import tifffile

    output_dir = Path(output_dir)
    diagnostics_dir = output_dir / "ecc_rpca_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "foreground": output_dir / f"{prefix}_rpca_corrected.tif",
        "warps": diagnostics_dir / "ecc_warps.csv",
        "warp_shifts": diagnostics_dir / "ecc_warps_dx_dy.csv",
        "gains": diagnostics_dir / "ecc_gains_a_b.csv",
        "warp_gain_plot": diagnostics_dir / "ecc_warp_gain_over_time.png",
        "metadata": diagnostics_dir / "metadata.json",
    }
    tifffile.imwrite(
        paths["foreground"],
        np.asarray(result.foreground, dtype=np.float32),
        photometric="minisblack",
    )
    np.savetxt(
        paths["warps"],
        result.warps.reshape(result.warps.shape[0], 6),
        delimiter=",",
        header="m00,m01,m02,m10,m11,m12",
        comments="",
    )
    np.savetxt(
        paths["warp_shifts"],
        np.column_stack((result.warps[:, 0, 2], result.warps[:, 1, 2])),
        delimiter=",",
        header="dx,dy",
        comments="",
    )
    np.savetxt(
        paths["gains"],
        result.gains_ab,
        delimiter=",",
        header="gain,bias",
        comments="",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timepoints = np.arange(result.warps.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(timepoints, result.warps[:, 0, 2], label="dx")
    axes[0].plot(timepoints, result.warps[:, 1, 2], label="dy")
    axes[0].set_ylabel("shift (pixels)")
    axes[0].legend()
    axes[1].plot(timepoints, result.gains_ab[:, 0], label="gain")
    axes[1].plot(timepoints, result.gains_ab[:, 1], label="bias")
    axes[1].set(xlabel="frame", ylabel="fit value")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(paths["warp_gain_plot"], dpi=150)
    plt.close(fig)
    if result.aligned_stack is not None:
        paths["aligned_stack"] = diagnostics_dir / "aligned_stack.tif"
        tifffile.imwrite(
            paths["aligned_stack"],
            np.asarray(result.aligned_stack, dtype=np.float32),
            photometric="minisblack",
        )
    if save_ecc_residual:
        paths["ecc_residual"] = output_dir / f"{prefix}_basicpy_ecc_corrected.tif"
        tifffile.imwrite(
            paths["ecc_residual"],
            np.asarray(result.ecc_pattern_subtracted, dtype=np.float32),
            photometric="minisblack",
        )

    metadata = dict(result.metadata)
    metadata["outputs"] = {key: str(path.name) for key, path in paths.items() if key != "metadata"}
    write_json(paths["metadata"], metadata)
    return paths
