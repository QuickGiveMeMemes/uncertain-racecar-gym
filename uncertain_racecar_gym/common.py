from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def package_asset_path(relative_path: str) -> Path:
    return Path(str(files("uncertain_racecar_gym").joinpath("assets").joinpath(relative_path)))


def resolve_resource_path(path: str | Path, base_path: str | Path | None = None) -> Path:
    candidate = Path(path)
    if isinstance(path, str) and path.startswith("package://"):
        return package_asset_path(path.replace("package://", "", 1))
    if candidate.is_absolute():
        return candidate
    if base_path is None:
        return candidate.resolve()
    return (Path(base_path).parent / candidate).resolve()


def softmax_sample_weights(distances: np.ndarray) -> np.ndarray:
    if distances.size == 0:
        return distances
    scale = float(np.mean(distances)) + 1e-6
    logits = -distances / scale
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    return weights / np.sum(weights)
