from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import runtime_metadata, validate_stack, write_json


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def generate_qc(
    foreground: np.ndarray,
    cellpose_labels: np.ndarray,
    tracks: Any,
    tracked_labels: np.ndarray,
    output_dir: str | Path,
    *,
    frames: list[int] | None = None,
) -> dict[str, Path]:
    """Generate headless segmentation/tracking QC plots and a JSON summary."""
    foreground = validate_stack(foreground, name="foreground stack")
    cellpose_labels = validate_stack(cellpose_labels, name="Cellpose labels")
    tracked_labels = validate_stack(tracked_labels, name="tracked labels")
    if foreground.shape != cellpose_labels.shape or foreground.shape != tracked_labels.shape:
        raise ValueError(
            "Foreground, Cellpose labels, and tracked labels must have identical shapes: "
            f"{foreground.shape}, {cellpose_labels.shape}, {tracked_labels.shape}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    timepoints = foreground.shape[0]
    if frames is None:
        frames = sorted({0, timepoints // 2, timepoints - 1})
    frames = [int(frame) for frame in frames if 0 <= int(frame) < timepoints]
    if not frames:
        raise ValueError("At least one QC frame must be within the movie time range")

    paths: dict[str, Path] = {}
    cell_counts = np.asarray([int(frame.max()) for frame in cellpose_labels])
    paths["cells_per_frame"] = output_dir / "cells_per_frame.png"
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(timepoints), cell_counts, marker="o", markersize=3, linewidth=1)
    ax.set(xlabel="frame", ylabel="number of masks", title="Cells detected per frame")
    fig.tight_layout()
    fig.savefig(paths["cells_per_frame"], dpi=150)
    plt.close(fig)

    paths["segmentation_overlays"] = output_dir / "segmentation_overlays.png"
    fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 5), squeeze=False)
    for ax, t in zip(axes[0], frames):
        lo, hi = np.percentile(foreground[t], (1, 99.5))
        ax.imshow(foreground[t], cmap="gray", vmin=lo, vmax=hi)
        ax.imshow(np.ma.masked_equal(cellpose_labels[t], 0), cmap="turbo", alpha=0.35)
        ax.set_title(f"t={t}: {cell_counts[t]} masks")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(paths["segmentation_overlays"], dpi=150)
    plt.close(fig)

    track_durations = (
        tracks.groupby("track_id")["t"].nunique().to_numpy(dtype=np.int64)
        if len(tracks)
        else np.asarray([], dtype=np.int64)
    )
    paths["track_durations"] = output_dir / "track_durations.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    if track_durations.size:
        ax.hist(track_durations, bins=min(30, max(1, int(track_durations.max()))))
    ax.set(xlabel="frames", ylabel="tracks", title="Track duration distribution")
    fig.tight_layout()
    fig.savefig(paths["track_durations"], dpi=150)
    plt.close(fig)

    paths["tracking_overlays"] = output_dir / "tracking_overlays.png"
    fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 5), squeeze=False)
    for ax, t in zip(axes[0], frames):
        lo, hi = np.percentile(foreground[t], (1, 99.5))
        ax.imshow(foreground[t], cmap="gray", vmin=lo, vmax=hi)
        ax.imshow(np.ma.masked_equal(tracked_labels[t], 0), cmap="turbo", alpha=0.4)
        active = int((tracks["t"] == t).sum()) if len(tracks) else 0
        ax.set_title(f"t={t}: {active} tracked objects")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(paths["tracking_overlays"], dpi=150)
    plt.close(fig)

    paths["summary"] = output_dir / "summary.json"
    summary = {
        "shape": list(foreground.shape),
        "qc_frames": frames,
        "cell_count_min": int(cell_counts.min()),
        "cell_count_max": int(cell_counts.max()),
        "track_rows": int(len(tracks)),
        "unique_tracks": int(tracks["track_id"].nunique()) if len(tracks) else 0,
        "track_duration_median": float(np.median(track_durations)) if track_durations.size else None,
        "runtime": runtime_metadata(["matplotlib", "numpy", "pandas"]),
        "outputs": {key: path.name for key, path in paths.items() if key != "summary"},
    }
    write_json(paths["summary"], summary)
    return paths
