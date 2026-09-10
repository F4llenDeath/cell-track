from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import label_dtype, runtime_metadata, validate_stack, write_json


@dataclass(frozen=True)
class UltrackConfig:
    """Ultrack parameters matching the original notebook defaults."""

    contour_sigma: float = 4.0
    min_area: int = 50
    max_area: int = 20_000
    max_distance: float = 80.0
    n_workers: int = 8
    appear_weight: float = -1.0
    disappear_weight: float = -1.0
    division_weight: float = -0.1
    power: int = 4
    bias: float = -0.001
    solution_gap: float = 0.0
    time_limit: int = 36_000
    solver_name: str = ""
    save_database: bool = False


@dataclass
class UltrackResult:
    tracks: Any
    graph: dict[int, list[int]]
    tracked_labels: np.ndarray
    area_values: np.ndarray
    database_artifacts: list[tuple[str, Path, Path]]
    metadata: dict[str, Any]


def build_main_config(config: UltrackConfig, working_dir: str | Path) -> Any:
    """Create an installed-version-compatible Ultrack ``MainConfig``."""
    from ultrack.config import MainConfig

    main_config = MainConfig()
    main_config.data_config.working_dir = Path(working_dir)
    main_config.data_config.n_workers = config.n_workers
    main_config.segmentation_config.min_area = config.min_area
    main_config.segmentation_config.max_area = config.max_area
    main_config.segmentation_config.n_workers = config.n_workers
    main_config.linking_config.max_distance = config.max_distance
    main_config.linking_config.n_workers = config.n_workers
    main_config.tracking_config.appear_weight = config.appear_weight
    main_config.tracking_config.disappear_weight = config.disappear_weight
    main_config.tracking_config.division_weight = config.division_weight
    main_config.tracking_config.power = config.power
    main_config.tracking_config.bias = config.bias
    main_config.tracking_config.solution_gap = config.solution_gap
    main_config.tracking_config.time_limit = config.time_limit
    main_config.tracking_config.solver_name = config.solver_name
    return main_config


def run_ultrack(
    labels: np.ndarray,
    config: UltrackConfig = UltrackConfig(),
    *,
    working_dir: str | Path,
) -> UltrackResult:
    """Track labels, safely handling empty gaps in Ultrack 0.7.x inputs.

    Ultrack 0.7.x cannot link through an empty frame and cannot export a
    one-frame solution. Contiguous runs with at least two eligible frames are
    therefore tracked independently and merged with globally unique IDs;
    isolated eligible frames are exported as one-frame tracks.
    """
    import pandas as pd
    from ultrack import to_tracks_layer, track, tracks_to_zarr
    from ultrack.utils import estimate_parameters_from_labels, labels_to_contours

    labels = validate_stack(labels, name="label stack")
    if np.any(labels < 0):
        raise ValueError("Label IDs must be non-negative")
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "track_id",
        "t",
        "y",
        "x",
        "id",
        "parent_track_id",
        "parent_id",
    ]
    area_values = (
        np.asarray(
            estimate_parameters_from_labels(labels, is_timelapse=True)["area"],
            dtype=np.float64,
        )
        if np.any(labels)
        else np.asarray([], dtype=np.float64)
    )

    eligible = np.zeros(labels.shape[0], dtype=bool)
    for t, frame in enumerate(labels):
        values, counts = np.unique(frame, return_counts=True)
        object_counts = counts[values != 0]
        eligible[t] = bool(
            np.any(
                (object_counts >= config.min_area)
                & (object_counts <= config.max_area)
            )
        )

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for t, is_eligible in enumerate(np.append(eligible, False)):
        if is_eligible and start is None:
            start = t
        elif not is_eligible and start is not None:
            runs.append((start, t - 1))
            start = None

    track_tables: list[Any] = []
    tracked_labels = np.zeros(labels.shape, dtype=np.int32)
    graph: dict[int, list[int]] = {}
    database_artifacts: list[tuple[str, Path, Path]] = []
    segment_metadata: list[dict[str, Any]] = []
    next_track_id = 1
    next_node_id = 1

    for start, end in runs:
        segment_name = f"segment_{start:06d}_{end:06d}"
        if start == end:
            rows: list[dict[str, int | float]] = []
            frame = labels[start]
            for source_label in np.unique(frame):
                if source_label == 0:
                    continue
                positions = np.argwhere(frame == source_label)
                area = positions.shape[0]
                if not config.min_area <= area <= config.max_area:
                    continue
                track_id = next_track_id
                node_id = next_node_id
                next_track_id += 1
                next_node_id += 1
                tracked_labels[start][frame == source_label] = track_id
                rows.append(
                    {
                        "track_id": track_id,
                        "t": start,
                        "y": float(positions[:, 0].mean()),
                        "x": float(positions[:, 1].mean()),
                        "id": node_id,
                        "parent_track_id": -1,
                        "parent_id": -1,
                    }
                )
            if rows:
                track_tables.append(pd.DataFrame(rows, columns=columns))
            segment_metadata.append(
                {
                    "start": start,
                    "end": end,
                    "mode": "single_frame_fallback",
                    "tracks": len(rows),
                }
            )
            continue

        segment_dir = working_dir / segment_name
        segment_labels = labels[start : end + 1]
        main_config = build_main_config(config, segment_dir)
        foreground, contours = labels_to_contours(
            segment_labels,
            sigma=config.contour_sigma,
        )
        track(
            config=main_config,
            foreground=foreground,
            contours=contours,
            overwrite=True,
        )
        local_tracks, local_graph = to_tracks_layer(main_config)
        local_tracked = np.asarray(
            tracks_to_zarr(
                main_config,
                local_tracks,
                store_or_path=segment_dir / "tracked_labels.zarr",
                overwrite=True,
            )
        )

        track_map = {
            old: new
            for old, new in zip(
                sorted(int(value) for value in local_tracks["track_id"].unique()),
                range(next_track_id, next_track_id + local_tracks["track_id"].nunique()),
            )
        }
        next_track_id += len(track_map)
        all_node_ids = set(int(value) for value in local_tracks["id"] if int(value) >= 0)
        all_node_ids.update(
            int(value) for value in local_tracks["parent_id"] if int(value) >= 0
        )
        node_map = {
            old: new
            for old, new in zip(
                sorted(all_node_ids),
                range(next_node_id, next_node_id + len(all_node_ids)),
            )
        }
        next_node_id += len(node_map)

        merged_tracks = local_tracks.copy()
        merged_tracks["track_id"] = merged_tracks["track_id"].map(track_map).astype(int)
        merged_tracks["parent_track_id"] = merged_tracks["parent_track_id"].map(
            lambda value: -1 if int(value) < 0 else track_map[int(value)]
        )
        merged_tracks["id"] = merged_tracks["id"].map(node_map).astype(int)
        merged_tracks["parent_id"] = merged_tracks["parent_id"].map(
            lambda value: -1 if int(value) < 0 else node_map[int(value)]
        )
        merged_tracks["t"] = merged_tracks["t"].astype(int) + start
        track_tables.append(merged_tracks[columns])

        for old_track, new_track in track_map.items():
            tracked_labels[start : end + 1][local_tracked == old_track] = new_track
        for parent, children in local_graph.items():
            graph[track_map[int(parent)]] = [track_map[int(child)] for child in children]

        database_path = segment_dir / main_config.data_config.database_file_name
        metadata_path = Path(main_config.data_config.metadata_path)
        if database_path.exists() and metadata_path.exists():
            database_artifacts.append((segment_name, database_path, metadata_path))
        segment_metadata.append(
            {
                "start": start,
                "end": end,
                "mode": "ultrack",
                "tracks": len(track_map),
            }
        )

    tracks = (
        pd.concat(track_tables, ignore_index=True).sort_values(["track_id", "t"])
        if track_tables
        else pd.DataFrame(columns=columns)
    )
    tracked_labels = tracked_labels.astype(
        label_dtype(int(tracked_labels.max()) if tracked_labels.size else 0),
        copy=False,
    )
    track_count = int(tracks["track_id"].nunique()) if len(tracks) else 0
    metadata = {
        "stage": "ultrack",
        "shape": list(labels.shape),
        "parameters": asdict(config),
        "track_rows": int(len(tracks)),
        "unique_tracks": track_count,
        "time_min": int(tracks["t"].min()) if len(tracks) else None,
        "time_max": int(tracks["t"].max()) if len(tracks) else None,
        "empty_input": not bool(np.any(eligible)),
        "empty_frames": np.flatnonzero(~eligible).astype(int).tolist(),
        "segments": segment_metadata,
        "area_quantiles": {
            str(q): float(np.quantile(area_values, q)) if area_values.size else None
            for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
        "runtime": runtime_metadata(
            ["ultrack", "pandas", "numpy", "tifffile", "zarr", "mip"]
        ),
    }
    return UltrackResult(
        tracks=tracks,
        graph=graph,
        tracked_labels=tracked_labels,
        area_values=area_values,
        database_artifacts=database_artifacts,
        metadata=metadata,
    )


def save_ultrack_result(
    result: UltrackResult,
    output_dir: str | Path,
    *,
    save_database: bool = False,
) -> dict[str, Path]:
    """Save final tracking outputs and optional Ultrack internals."""
    import tifffile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "tracks": output_dir / "tracks_df.csv",
        "tracked_labels": output_dir / "tracked_labels.tif",
        "areas": output_dir / "cell_areas.csv",
        "metadata": output_dir / "tracking_metadata.json",
    }
    result.tracks.to_csv(paths["tracks"], index=False)
    tifffile.imwrite(paths["tracked_labels"], result.tracked_labels, photometric="minisblack")
    np.savetxt(
        paths["areas"],
        result.area_values,
        delimiter=",",
        header="area",
        comments="",
    )
    if save_database:
        paths["database_bundle"] = output_dir / "ultrack_databases"
        paths["database_bundle"].mkdir(parents=True, exist_ok=True)
        for segment_name, database_path, metadata_path in result.database_artifacts:
            segment_dir = paths["database_bundle"] / segment_name
            segment_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(database_path, segment_dir / "data.db")
            shutil.copy2(metadata_path, segment_dir / "metadata.toml")
        write_json(
            paths["database_bundle"] / "manifest.json",
            {
                "segments": [
                    {
                        "name": segment_name,
                        "database": f"{segment_name}/data.db",
                        "metadata": f"{segment_name}/metadata.toml",
                    }
                    for segment_name, _database_path, _metadata_path in result.database_artifacts
                ]
            },
        )

    metadata = dict(result.metadata)
    metadata["outputs"] = {key: path.name for key, path in paths.items() if key != "metadata"}
    write_json(paths["metadata"], metadata)
    return paths
