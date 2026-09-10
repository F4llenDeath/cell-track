from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import label_dtype, runtime_metadata, validate_stack, write_json


@dataclass(frozen=True)
class CellposeConfig:
    """Cellpose-SAM parameters matching ``main.ipynb`` defaults."""

    model: str = "cpsam_v2"
    fallback_model: str = "cpsam"
    device: str = "auto"
    diameter: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    batch_size: int = 4
    niter: int | None = None
    bsize: int | None = None
    gamma: float = 1.0
    lower_quantile: float = 0.001
    upper_quantile: float = 0.9999
    save_normalized_stack: bool = False


@dataclass
class CellposeResult:
    labels: np.ndarray
    normalized_stack: np.ndarray | None
    cell_counts: np.ndarray
    requested_model: str
    resolved_model: str
    device: str
    metadata: dict[str, Any]


def normalize_frame(
    frame: np.ndarray,
    *,
    gamma: float = 1.0,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.9999,
) -> np.ndarray:
    """Match the frame-wise normalization used by Ultrack's helper."""
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Expected 0 <= lower_quantile < upper_quantile <= 1")
    normalized = np.asarray(frame, dtype=np.float32).copy()
    normalized -= np.quantile(normalized, lower_quantile)
    scale = float(np.quantile(normalized, upper_quantile))
    if scale <= 0 or not np.isfinite(scale):
        return np.zeros_like(normalized, dtype=np.float32)
    normalized /= scale
    normalized = np.clip(normalized, 0, 1)
    if gamma != 1.0:
        normalized = np.power(normalized, gamma)
    return np.asarray(normalized, dtype=np.float32)


def normalize_stack(
    stack: np.ndarray,
    config: CellposeConfig = CellposeConfig(),
    *,
    show_progress: bool = True,
) -> np.ndarray:
    """Normalize each time point independently for Cellpose."""
    from tqdm import tqdm

    stack = validate_stack(stack)
    iterator = tqdm(stack, desc="normalize", disable=not show_progress)
    return np.stack(
        [
            normalize_frame(
                frame,
                gamma=config.gamma,
                lower_quantile=config.lower_quantile,
                upper_quantile=config.upper_quantile,
            )
            for frame in iterator
        ]
    ).astype(np.float32)


def resolve_torch_device(requested: str = "auto") -> Any:
    """Resolve ``auto``, ``cuda``, ``mps``, or ``cpu`` to a torch device."""
    import torch

    requested = requested.lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, mps, cpu")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but is not available")
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "mps":
        raise RuntimeError("MPS was requested but is not available")
    return torch.device("cpu")


def build_eval_kwargs(config: CellposeConfig, *, resolved_model: str | None = None) -> dict[str, Any]:
    """Build one Cellpose eval configuration for test and full-stack runs."""
    kwargs: dict[str, Any] = {
        "diameter": config.diameter,
        "flow_threshold": config.flow_threshold,
        "cellprob_threshold": config.cellprob_threshold,
        "batch_size": config.batch_size,
        "normalize": False,
    }
    if config.niter is not None:
        kwargs["niter"] = config.niter
    model_name = resolved_model or config.model
    if config.bsize is not None and model_name.startswith("cpdino"):
        kwargs["bsize"] = config.bsize
    return kwargs


def load_cellpose_model(config: CellposeConfig) -> tuple[Any, str, str]:
    """Load the requested model with the notebook's graceful cpsam fallback."""
    from cellpose import models

    device = resolve_torch_device(config.device)
    requested = config.model
    try:
        model = models.CellposeModel(
            gpu=device.type != "cpu",
            pretrained_model=requested,
            device=device,
        )
    except Exception:
        if requested == config.fallback_model:
            raise
        model = models.CellposeModel(
            gpu=device.type != "cpu",
            pretrained_model=config.fallback_model,
            device=device,
        )

    resolved = Path(str(model.pretrained_model)).name
    return model, resolved, str(model.device)


def run_cellpose(
    stack: np.ndarray,
    config: CellposeConfig = CellposeConfig(),
    *,
    model: Any | None = None,
    resolved_model: str | None = None,
    resolved_device: str | None = None,
    show_progress: bool = True,
) -> CellposeResult:
    """Normalize and segment a ``(T,Y,X)`` foreground stack."""
    stack = validate_stack(stack)
    normalized = normalize_stack(stack, config, show_progress=show_progress)
    if model is None:
        model, resolved_model, resolved_device = load_cellpose_model(config)
    else:
        resolved_model = resolved_model or config.model
        resolved_device = resolved_device or "provided-model"

    masks, _flows, _styles = model.eval(
        [normalized[t] for t in range(normalized.shape[0])],
        **build_eval_kwargs(config, resolved_model=resolved_model),
    )
    labels = np.stack(masks)
    labels = labels.astype(label_dtype(int(labels.max()) if labels.size else 0), copy=False)
    cell_counts = np.asarray([int(frame.max()) for frame in labels], dtype=np.int64)
    metadata = {
        "stage": "cellpose",
        "shape": list(labels.shape),
        "parameters": asdict(config),
        "requested_model": config.model,
        "resolved_model": resolved_model,
        "device": resolved_device,
        "cell_count_min": int(cell_counts.min()) if cell_counts.size else 0,
        "cell_count_max": int(cell_counts.max()) if cell_counts.size else 0,
        "runtime": runtime_metadata(
            ["cellpose", "torch", "torchvision", "numpy", "tifffile"]
        ),
    }
    return CellposeResult(
        labels=labels,
        normalized_stack=normalized if config.save_normalized_stack else None,
        cell_counts=cell_counts,
        requested_model=config.model,
        resolved_model=resolved_model,
        device=resolved_device,
        metadata=metadata,
    )


def save_cellpose_result(
    result: CellposeResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save Cellpose labels, counts, and provenance."""
    import tifffile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "labels": output_dir / "cellpose_labels.tif",
        "cell_counts": output_dir / "cell_counts.csv",
        "metadata": output_dir / "cellpose_metadata.json",
    }
    tifffile.imwrite(paths["labels"], result.labels, photometric="minisblack")
    np.savetxt(
        paths["cell_counts"],
        np.column_stack((np.arange(len(result.cell_counts)), result.cell_counts)),
        fmt="%d",
        delimiter=",",
        header="t,cell_count",
        comments="",
    )
    if result.normalized_stack is not None:
        paths["normalized"] = output_dir / "normalized_foreground.tif"
        tifffile.imwrite(
            paths["normalized"],
            np.asarray(result.normalized_stack, dtype=np.float32),
            photometric="minisblack",
        )
    metadata = dict(result.metadata)
    metadata["outputs"] = {key: path.name for key, path in paths.items() if key != "metadata"}
    write_json(paths["metadata"], metadata)
    return paths
