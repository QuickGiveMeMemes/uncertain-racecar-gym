from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from uncertain_racecar_gym.common import ensure_dir
from uncertain_racecar_gym.dataset import load_records


@dataclass(slots=True)
class TrackBuildArtifacts:
    csv_path: Path
    scenario_path: Path | None
    report_path: Path | None
    metadata_path: Path
    estimated_width: float
    num_bins: int
    source_count: int


def _circular_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="wrap")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _fill_missing_circular(values: np.ndarray) -> np.ndarray:
    filled = values.astype(float).copy()
    valid = np.isfinite(filled)
    if valid.all():
        return filled
    indices = np.arange(len(filled))
    valid_indices = indices[valid]
    if len(valid_indices) == 0:
        raise ValueError("Unable to reconstruct track centerline: no valid bins found.")
    extended_indices = np.concatenate([valid_indices - len(filled), valid_indices, valid_indices + len(filled)])
    extended_values = np.concatenate([filled[valid], filled[valid], filled[valid]])
    interpolated = np.interp(indices, extended_indices, extended_values)
    return interpolated


def _prepare_progress_frame(inputs: Iterable[str | Path]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    frames = []
    manifest = []
    for input_path in inputs:
        loaded = load_records(input_path)
        frame = loaded.frame.copy()
        if "NormalizedSplinePosition" in frame:
            progress = frame["NormalizedSplinePosition"].astype(float)
        elif "progress" in frame:
            progress = frame["progress"].astype(float)
        else:
            continue
        if "world_position_x" in frame:
            x = frame["world_position_x"].astype(float)
            y = frame["world_position_y"].astype(float)
        elif "x" in frame:
            x = frame["x"].astype(float)
            y = frame["y"].astype(float)
        else:
            continue
        sample = pd.DataFrame(
            {
                "progress": np.mod(progress.to_numpy(dtype=float), 1.0),
                "x": x.to_numpy(dtype=float),
                "y": y.to_numpy(dtype=float),
                "source": Path(input_path).name,
            }
        )
        sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
        if sample.empty:
            continue
        frames.append(sample)
        manifest.append(
            {
                "input_path": str(Path(input_path)),
                "source_format": loaded.source_format,
                "rows": int(len(sample)),
            }
        )
    if not frames:
        raise ValueError("No usable progress-position records were found in the input files.")
    return pd.concat(frames, ignore_index=True), manifest


def derive_centerline_from_progress(
    inputs: Iterable[str | Path],
    output_csv: str | Path,
    num_bins: int = 2000,
    smoothing_window: int = 31,
) -> tuple[Path, pd.DataFrame, dict]:
    frame, manifest = _prepare_progress_frame(inputs)

    bin_index = np.floor(frame["progress"].to_numpy() * num_bins).astype(int) % num_bins
    grouped_x = np.full(num_bins, np.nan, dtype=float)
    grouped_y = np.full(num_bins, np.nan, dtype=float)
    counts = np.bincount(bin_index, minlength=num_bins)
    for index in range(num_bins):
        mask = bin_index == index
        if mask.any():
            grouped_x[index] = float(frame.loc[mask, "x"].mean())
            grouped_y[index] = float(frame.loc[mask, "y"].mean())

    grouped_x = _fill_missing_circular(grouped_x)
    grouped_y = _fill_missing_circular(grouped_y)
    grouped_x = _circular_moving_average(grouped_x, smoothing_window)
    grouped_y = _circular_moving_average(grouped_y, smoothing_window)

    progress = (np.arange(num_bins, dtype=float) + 0.5) / float(num_bins)
    centerline = pd.DataFrame({"progress": progress, "x": grouped_x, "y": grouped_y})
    csv_path = Path(output_csv)
    ensure_dir(csv_path.parent)
    centerline.to_csv(csv_path, index=False)

    derivatives = np.roll(centerline[["x", "y"]].to_numpy(), -1, axis=0) - np.roll(centerline[["x", "y"]].to_numpy(), 1, axis=0)
    tangents = derivatives / np.maximum(np.linalg.norm(derivatives, axis=1, keepdims=True), 1e-9)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    centerline_lookup = centerline[["x", "y"]].to_numpy()[bin_index]
    normal_lookup = normals[bin_index]
    offsets = np.sum((frame[["x", "y"]].to_numpy() - centerline_lookup) * normal_lookup, axis=1)
    abs_offsets = np.abs(offsets)
    offset_q95 = float(np.percentile(abs_offsets, 95.0))
    offset_q99 = float(np.percentile(abs_offsets, 99.0))
    offset_q995 = float(np.percentile(abs_offsets, 99.5))
    estimated_width = float(np.clip(2.0 * offset_q95 + 1.5, 8.0, 25.0))

    metadata = {
        "source_count": len(manifest),
        "num_bins": int(num_bins),
        "smoothing_window": int(smoothing_window),
        "estimated_width": estimated_width,
        "offset_abs_q95": offset_q95,
        "offset_abs_q99": offset_q99,
        "offset_abs_q995": offset_q995,
        "counts_min": int(counts.min()),
        "counts_max": int(counts.max()),
        "counts_mean": float(counts.mean()),
        "manifest": manifest,
    }
    metadata_path = csv_path.with_name(f"{csv_path.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path, centerline, metadata


def write_scenario_yaml(
    output_path: str | Path,
    scenario_name: str,
    track_csv: str | Path,
    width: float,
    progress_bins: int = 48,
) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    payload = {
        "name": scenario_name,
        "track": {
            "csv": str(Path(track_csv).resolve()),
            "width": float(width),
            "progress_bins": int(progress_bins),
            "closed": True,
        },
        "vehicle": {
            "wheelbase": 3.05,
            "lf": 1.45,
            "lr": 1.60,
            "mass": 720.0,
            "inertia_z": 900.0,
            "cornering_stiffness_front": 90000.0,
            "cornering_stiffness_rear": 98000.0,
            "max_steer_rad": 0.32,
            "max_accel": 12.0,
            "max_brake": 18.0,
            "drag_coefficient": 0.85,
            "wheel_radius": 0.33,
            "chassis_size": [3.2, 1.4, 0.32],
        },
        "simulation": {
            "dt": 0.05,
            "max_steps": 2000,
            "lookahead_points": 6,
            "lookahead_spacing_m": 10.0,
        },
        "uncertainty": {
            "history_length": 5,
            "neighbor_count": 48,
            "block_length": 25,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def write_track_report(
    progress_frame: pd.DataFrame,
    centerline: pd.DataFrame,
    metadata: dict,
    report_dir: str | Path,
) -> Path:
    report_dir = ensure_dir(report_dir)
    plot_dir = ensure_dir(report_dir / "plots")

    sample = progress_frame.sample(n=min(40000, len(progress_frame)), random_state=7) if len(progress_frame) > 40000 else progress_frame

    fig, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(sample["x"], sample["y"], s=1, alpha=0.10, color="#94a3b8", label="raw samples")
    axis.plot(centerline["x"], centerline["y"], color="#dc2626", linewidth=2.0, label="derived centerline")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Barcelona track reconstruction")
    axis.legend(loc="upper right")
    fig.tight_layout()
    overlay_path = plot_dir / "track_overlay.png"
    fig.savefig(overlay_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(sample["progress"], bins=40, color="#2563eb", alpha=0.85)
    axes[0].set_title("Progress coverage")
    axes[0].set_xlabel("Normalized progress")
    axes[0].set_ylabel("Count")
    diffs = np.linalg.norm(np.diff(np.vstack([centerline[["x", "y"]].to_numpy(), centerline[["x", "y"]].to_numpy()[0]]), axis=0), axis=1)
    axes[1].plot(diffs, color="#0f766e")
    axes[1].set_title("Centerline point spacing")
    axes[1].set_xlabel("Centerline index")
    axes[1].set_ylabel("Spacing [m]")
    fig.tight_layout()
    diagnostics_path = plot_dir / "track_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=180)
    plt.close(fig)

    approx_bin = np.floor(progress_frame["progress"].to_numpy() * len(centerline)).astype(int) % len(centerline)
    derivatives = np.roll(centerline[["x", "y"]].to_numpy(), -1, axis=0) - np.roll(centerline[["x", "y"]].to_numpy(), 1, axis=0)
    tangents = derivatives / np.maximum(np.linalg.norm(derivatives, axis=1, keepdims=True), 1e-9)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    center_lookup = centerline[["x", "y"]].to_numpy()[approx_bin]
    normal_lookup = normals[approx_bin]
    offsets = np.sum((progress_frame[["x", "y"]].to_numpy() - center_lookup) * normal_lookup, axis=1)

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.hist(offsets, bins=80, color="#2563eb", alpha=0.85)
    axis.set_title("Lateral offset distribution around derived centerline")
    axis.set_xlabel("Signed offset [m]")
    axis.set_ylabel("Count")
    fig.tight_layout()
    offset_path = plot_dir / "offset_distribution.png"
    fig.savefig(offset_path, dpi=180)
    plt.close(fig)

    report_path = report_dir / "track_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Track Reconstruction Report",
                "",
                "## Summary",
                "",
                f"- Source files used: `{metadata['source_count']}`",
                f"- Centerline bins: `{metadata['num_bins']}`",
                f"- Smoothing window: `{metadata['smoothing_window']}`",
                f"- Recommended track width: `{metadata['estimated_width']:.2f} m`",
                f"- |offset| 95th percentile: `{metadata['offset_abs_q95']:.2f} m`",
                f"- |offset| 99th percentile: `{metadata['offset_abs_q99']:.2f} m`",
                f"- |offset| 99.5th percentile: `{metadata['offset_abs_q995']:.2f} m`",
                f"- Bin count range: `{metadata['counts_min']}` to `{metadata['counts_max']}`",
                f"- Mean samples per bin: `{metadata['counts_mean']:.2f}`",
                "",
                "## Plots",
                "",
                f"![Track Overlay](plots/{overlay_path.name})",
                "",
                f"![Track Diagnostics](plots/{diagnostics_path.name})",
                "",
                f"![Offset Distribution](plots/{offset_path.name})",
            ]
        ),
        encoding="utf-8",
    )
    return report_path


def build_track_from_dataset(
    inputs: Iterable[str | Path],
    output_csv: str | Path,
    scenario_output: str | Path | None = None,
    report_dir: str | Path | None = None,
    scenario_name: str = "ks_barcelona_layout_gp_dallara_f317",
    width: float | None = None,
    num_bins: int = 2000,
    smoothing_window: int = 31,
) -> TrackBuildArtifacts:
    csv_path, centerline, metadata = derive_centerline_from_progress(
        inputs=inputs,
        output_csv=output_csv,
        num_bins=num_bins,
        smoothing_window=smoothing_window,
    )
    progress_frame, _ = _prepare_progress_frame(inputs)
    scenario_path = None
    if scenario_output is not None:
        scenario_path = write_scenario_yaml(
            output_path=scenario_output,
            scenario_name=scenario_name,
            track_csv=csv_path,
            width=width if width is not None else metadata["estimated_width"],
        )
    report_path = None
    if report_dir is not None:
        report_path = write_track_report(progress_frame, centerline, metadata, report_dir)
    return TrackBuildArtifacts(
        csv_path=csv_path,
        scenario_path=scenario_path,
        report_path=report_path,
        metadata_path=csv_path.with_name(f"{csv_path.stem}_metadata.json"),
        estimated_width=float(width if width is not None else metadata["estimated_width"]),
        num_bins=num_bins,
        source_count=metadata["source_count"],
    )
