from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def validate_stack(stack: np.ndarray, *, name: str = "image stack") -> np.ndarray:
    """Validate and return a non-empty ``(T, Y, X)`` image stack."""
    array = np.asarray(stack)
    if array.ndim != 3:
        raise ValueError(f"Expected {name} to have shape (T, Y, X), got {array.shape}")
    if any(size == 0 for size in array.shape):
        raise ValueError(f"Expected {name} to be non-empty, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Expected {name} to be numeric, got {array.dtype}")
    return array


def label_dtype(max_label: int) -> np.dtype[Any]:
    """Choose a compact dtype without truncating a non-negative label ID."""
    if max_label < 0:
        raise ValueError("Label IDs must be non-negative")
    return np.dtype(np.uint16 if max_label <= np.iinfo(np.uint16).max else np.int32)


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Return installed package versions without importing heavy packages."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_metadata(packages: Iterable[str] = ()) -> dict[str, Any]:
    """Return lightweight runtime provenance suitable for JSON metadata."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(packages),
    }


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    """Write deterministic, human-readable JSON and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return output_path
