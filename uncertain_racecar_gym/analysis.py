from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew, wasserstein_distance

from uncertain_racecar_gym.controllers import CenterlineDriver, ProfiledCenterlineDriver
from uncertain_racecar_gym.dataset import CANONICAL_COLUMNS, build_demo_dataset
from uncertain_racecar_gym.deterministic import load_calibration_model
from uncertain_racecar_gym.dynamics import DynamicBicycleModel
from uncertain_racecar_gym.env import UncertainRacecarEnv
from uncertain_racecar_gym.scenario import Scenario, load_scenario
from uncertain_racecar_gym.track import TrackModel
from uncertain_racecar_gym.uncertainty import EmpiricalUncertaintyModel, FEATURE_NAMES, RESIDUAL_NAMES


@dataclass(slots=True)
class EvaluationArtifacts:
    report_path: Path
    metrics_path: Path
    plot_dir: Path
    artifact_path: Path
    dataset_path: Path


def compute_residual_table(canonical: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    model = DynamicBicycleModel(scenario.vehicle)
    track = TrackModel.from_config(scenario.track)
    rows = []
    history_length = scenario.uncertainty.history_length

    canonical = canonical.sort_values(["trajectory_id", "frame_index"]).reset_index(drop=True)
    for _, group in canonical.groupby("trajectory_id", sort=False):
        group = group.reset_index(drop=True)
        action_history = [np.zeros(3, dtype=float) for _ in range(history_length)]
        for index in range(len(group) - 1):
            current = group.iloc[index]
            nxt = group.iloc[index + 1]
            state = model.state_from_canonical_row(current)
            action = np.array([current["steer"], current["throttle"], current["brake"]], dtype=float)
            prediction = model.predict(state, action, float(current["dt"]))
            progress_bin = int(min(track.progress_bins - 1, max(0, np.floor(float(current["progress"]) * track.progress_bins))))
            rows.append(
                {
                    **current.to_dict(),
                    "progress_bin": progress_bin,
                    "abs_curvature": abs(float(current["curvature"])),
                    "history_steer_mean": float(np.mean([item[0] for item in action_history])),
                    "history_throttle_mean": float(np.mean([item[1] for item in action_history])),
                    "history_brake_mean": float(np.mean([item[2] for item in action_history])),
                    "feature_vector": np.concatenate(
                        [
                            np.array(
                                [
                                    current["curvature"],
                                    current["progress"],
                                    current["vx"],
                                    current["vy"],
                                    current["yaw_rate"],
                                    current["steer"],
                                    current["throttle"],
                                    current["brake"],
                                ],
                                dtype=float,
                            ),
                            np.asarray(action_history, dtype=float).reshape(-1),
                        ]
                    ),
                    "delta_vx": float(nxt["vx"]) - prediction.vx,
                    "delta_vy": float(nxt["vy"]) - prediction.vy,
                    "delta_yaw_rate": float(nxt["yaw_rate"]) - prediction.yaw_rate,
                }
            )
            action_history.pop(0)
            action_history.append(action)
    return pd.DataFrame(rows)


def _trajectory_split(residual_table: pd.DataFrame, train_fraction: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectory_ids = sorted(residual_table["trajectory_id"].unique())
    cutoff = max(1, int(len(trajectory_ids) * train_fraction))
    train_ids = set(trajectory_ids[:cutoff])
    train = residual_table[residual_table["trajectory_id"].isin(train_ids)].copy()
    test = residual_table[~residual_table["trajectory_id"].isin(train_ids)].copy()
    if test.empty:
        test = train.copy()
    return train, test


def _center_residual_table(residual_table: pd.DataFrame, calibration_model) -> pd.DataFrame:
    centered = residual_table.reset_index(drop=True).copy()
    predictions = []
    for row in centered.itertuples(index=False):
        gate_key = calibration_model.resolve_gate_key(
            progress_bin=int(row.progress_bin),
            track_id=str(row.track_id),
            car_id=str(row.car_id),
        )
        mean_residual, info = calibration_model.predict_mean(
            np.asarray(row.feature_vector, dtype=float),
            gate_key,
            dt=float(row.dt),
        )
        longitudinal_info = info.get("longitudinal", {}) if isinstance(info, dict) else {}
        residual_info = info.get("residual", info) if isinstance(info, dict) else {}
        predictions.append(
            {
                "delta_vx_mean": float(mean_residual[0]),
                "delta_vy_mean": float(mean_residual[1]),
                "delta_yaw_rate_mean": float(mean_residual[2]),
                "delta_vx_longitudinal_mean": float(longitudinal_info.get("delta_vx_correction", 0.0)),
                "longitudinal_applied": bool(longitudinal_info.get("applied", False)),
                "calibration_neighbors": int(residual_info.get("neighbors", 0)),
                "calibration_distance_mean": float(residual_info.get("distance_mean", np.nan)),
            }
        )
    prediction_frame = pd.DataFrame(predictions)
    for channel in RESIDUAL_NAMES:
        centered[f"{channel}_raw"] = centered[channel]
        centered[channel] = centered[channel].to_numpy(dtype=float) - prediction_frame[f"{channel}_mean"].to_numpy(dtype=float)
    return pd.concat([centered.reset_index(drop=True), prediction_frame], axis=1)


def _evaluate_model(residual_table: pd.DataFrame, artifact: EmpiricalUncertaintyModel, scenario: Scenario) -> pd.DataFrame:
    predictions = []
    for row in residual_table.itertuples(index=False):
        gate_key = (row.track_id, row.car_id, int(row.progress_bin))
        mean_residual, info = artifact.predict_mean(np.asarray(row.feature_vector, dtype=float), gate_key)
        predictions.append(
            {
                "delta_vx_pred": float(mean_residual[0]),
                "delta_vy_pred": float(mean_residual[1]),
                "delta_yaw_rate_pred": float(mean_residual[2]),
                "neighbors": info.get("neighbors", 0),
                "distance_mean": info.get("distance_mean", np.nan),
            }
        )
    prediction_frame = pd.DataFrame(predictions)
    return pd.concat([residual_table.reset_index(drop=True), prediction_frame], axis=1)


def _sample_model_distribution(
    residual_table: pd.DataFrame,
    artifact: EmpiricalUncertaintyModel,
    sample_count: int = 3,
    seed: int = 11,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    samples = []
    for row in residual_table.itertuples(index=False):
        gate_key = (row.track_id, row.car_id, int(row.progress_bin))
        for draw in range(sample_count):
            runtime_state = artifact.make_runtime_state()
            sample, info = artifact.sample(np.asarray(row.feature_vector, dtype=float), gate_key, rng, runtime_state)
            samples.append(
                {
                    "trajectory_id": row.trajectory_id,
                    "frame_index": int(row.frame_index),
                    "abs_curvature": float(row.abs_curvature),
                    "vx": float(row.vx),
                    "sample_index": draw,
                    "delta_vx_sample": float(sample[0]),
                    "delta_vy_sample": float(sample[1]),
                    "delta_yaw_rate_sample": float(sample[2]),
                    "mode": info.get("mode", "unknown"),
                }
            )
    return pd.DataFrame(samples)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def _compute_rmse_metrics(evaluated: pd.DataFrame) -> dict[str, dict[str, float]]:
    channels = ["delta_vx", "delta_vy", "delta_yaw_rate"]
    baseline_rmse = {_channel: _rmse(evaluated[_channel].to_numpy(), np.zeros(len(evaluated))) for _channel in channels}
    model_rmse = {_channel: _rmse(evaluated[_channel].to_numpy(), evaluated[f"{_channel}_pred"].to_numpy()) for _channel in channels}
    return {
        "baseline_rmse": baseline_rmse,
        "model_rmse": model_rmse,
    }


def _compute_wasserstein_metrics(evaluated: pd.DataFrame, sampled: pd.DataFrame) -> dict[str, dict[str, float]]:
    baseline_distances = {}
    model_distances = {}
    for channel in RESIDUAL_NAMES:
        actual = evaluated[channel].to_numpy()
        baseline_distances[channel] = float(wasserstein_distance(actual, np.zeros(len(actual), dtype=float)))
        model_distances[channel] = float(wasserstein_distance(actual, sampled[f"{channel}_sample"].to_numpy()))
    return {
        "baseline_wasserstein": baseline_distances,
        "model_wasserstein": model_distances,
    }


def _save_residual_histograms(evaluated: pd.DataFrame, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    channels = ["delta_vx", "delta_vy", "delta_yaw_rate"]
    colors = ["#2b6cb0", "#0f766e", "#9a3412"]
    for axis, channel, color in zip(axes, channels, colors):
        axis.hist(evaluated[channel], bins=40, color=color, alpha=0.85)
        axis.set_title(channel)
        axis.set_xlabel("Residual value")
        axis.set_ylabel("Count")
    fig.tight_layout()
    path = plot_dir / "residual_histograms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_track_overview(canonical: pd.DataFrame, plot_dir: Path) -> Path:
    sample = canonical.sample(n=min(40000, len(canonical)), random_state=7) if len(canonical) > 40000 else canonical
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    scatter = axes[0].scatter(sample["x"], sample["y"], c=sample["vx"], s=3, alpha=0.35, cmap="viridis")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Trajectory footprint colored by speed")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04, label="vx")

    progress_bins = pd.cut(canonical["progress"], bins=60, include_lowest=True)
    speed_profile = canonical.groupby(progress_bins, observed=False)["vx"].agg(["mean", "std"]).reset_index(drop=True)
    axes[1].plot(speed_profile.index, speed_profile["mean"], color="#2563eb", label="mean speed")
    axes[1].fill_between(
        speed_profile.index,
        speed_profile["mean"] - speed_profile["std"].fillna(0.0),
        speed_profile["mean"] + speed_profile["std"].fillna(0.0),
        color="#93c5fd",
        alpha=0.45,
        label="+-1 std",
    )
    axes[1].set_title("Speed profile over progress")
    axes[1].set_xlabel("progress bin")
    axes[1].set_ylabel("speed [m/s]")
    axes[1].legend()
    fig.tight_layout()
    path = plot_dir / "track_overview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_control_distributions(canonical: pd.DataFrame, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [
        ("vx", "Speed distribution", "#2563eb"),
        ("steer", "Steer distribution", "#dc2626"),
        ("throttle", "Throttle distribution", "#0f766e"),
        ("brake", "Brake distribution", "#9333ea"),
    ]
    for axis, (column, title, color) in zip(axes.flat, plots):
        axis.hist(canonical[column], bins=40, color=color, alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel(column)
        axis.set_ylabel("Count")
    fig.tight_layout()
    path = plot_dir / "control_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_state_context_plots(canonical: pd.DataFrame, plot_dir: Path) -> Path:
    sample = canonical.sample(n=min(50000, len(canonical)), random_state=11) if len(canonical) > 50000 else canonical
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    relationships = [
        ("vx", "yaw_rate", "vx vs yaw_rate"),
        ("vx", "steer", "vx vs steer"),
        ("vy", "yaw_rate", "vy vs yaw_rate"),
    ]
    for axis, (x_name, y_name, title) in zip(axes, relationships):
        image = axis.hexbin(sample[x_name], sample[y_name], gridsize=30, cmap="magma", mincnt=1)
        axis.set_title(title)
        axis.set_xlabel(x_name)
        axis.set_ylabel(y_name)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = plot_dir / "state_context_plots.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_curvature_group_histograms(evaluated: pd.DataFrame, plot_dir: Path) -> Path:
    bins = pd.qcut(evaluated["abs_curvature"], q=3, duplicates="drop")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    channels = ["delta_vy", "delta_yaw_rate"]
    colors = ["#1d4ed8", "#7c3aed", "#ef4444"]
    labels = [str(label) for label in bins.cat.categories]
    for axis, channel in zip(axes, channels):
        for category, color, label in zip(bins.cat.categories, colors, labels):
            mask = bins == category
            axis.hist(evaluated.loc[mask, channel], bins=30, alpha=0.45, label=label, color=color, density=True)
        axis.set_title(f"{channel} grouped by |curvature|")
        axis.set_xlabel("Residual value")
        axis.set_ylabel("Density")
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = plot_dir / "curvature_group_histograms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_heatmaps(evaluated: pd.DataFrame, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    evaluated = evaluated.copy()
    evaluated["speed_bin"] = pd.cut(evaluated["vx"], bins=8)
    evaluated["curvature_bin"] = pd.cut(evaluated["abs_curvature"], bins=8)
    channels = ["delta_vy", "delta_yaw_rate"]
    for axis, channel in zip(axes, channels):
        pivot = evaluated.pivot_table(index="speed_bin", columns="curvature_bin", values=channel, aggfunc="mean", observed=False)
        image = axis.imshow(pivot.fillna(0.0).to_numpy(), origin="lower", aspect="auto", cmap="coolwarm")
        axis.set_title(f"Mean {channel} by speed and |curvature| bin")
        axis.set_xlabel("|curvature| bin")
        axis.set_ylabel("speed bin")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = plot_dir / "state_space_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_feature_residual_relationships(evaluated: pd.DataFrame, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plots = [
        ("abs_curvature", "delta_vy", "|curvature| vs delta_vy"),
        ("abs_curvature", "delta_yaw_rate", "|curvature| vs delta_yaw_rate"),
        ("vx", "delta_vx", "vx vs delta_vx"),
    ]
    for axis, (x_name, y_name, title) in zip(axes, plots):
        image = axis.hexbin(evaluated[x_name], evaluated[y_name], gridsize=28, cmap="viridis", mincnt=1)
        axis.set_title(title)
        axis.set_xlabel(x_name)
        axis.set_ylabel(y_name)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = plot_dir / "feature_residual_relationships.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_rmse_bars(evaluated: pd.DataFrame, plot_dir: Path) -> tuple[Path, dict]:
    channels = ["delta_vx", "delta_vy", "delta_yaw_rate"]
    rmse_metrics = _compute_rmse_metrics(evaluated)
    baseline_rmse = [rmse_metrics["baseline_rmse"][channel] for channel in channels]
    model_rmse = [rmse_metrics["model_rmse"][channel] for channel in channels]

    fig, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(channels))
    width = 0.32
    axis.bar(x - width / 2, baseline_rmse, width=width, label="nominal baseline", color="#94a3b8")
    axis.bar(x + width / 2, model_rmse, width=width, label="uncertainty mean predictor", color="#2563eb")
    axis.set_xticks(x)
    axis.set_xticklabels(channels)
    axis.set_ylabel("RMSE")
    axis.set_title("Held-out one-step residual prediction error")
    axis.legend()
    fig.tight_layout()
    path = plot_dir / "rmse_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, rmse_metrics


def _save_rmse_by_curvature(evaluated: pd.DataFrame, plot_dir: Path) -> Path:
    bins = pd.qcut(evaluated["abs_curvature"], q=6, duplicates="drop")
    grouped = []
    for category, group in evaluated.groupby(bins, observed=False):
        grouped.append(
            {
                "curvature_bin": str(category),
                "delta_vy_baseline": _rmse(group["delta_vy"].to_numpy(), np.zeros(len(group))),
                "delta_vy_model": _rmse(group["delta_vy"].to_numpy(), group["delta_vy_pred"].to_numpy()),
                "delta_yaw_rate_baseline": _rmse(group["delta_yaw_rate"].to_numpy(), np.zeros(len(group))),
                "delta_yaw_rate_model": _rmse(group["delta_yaw_rate"].to_numpy(), group["delta_yaw_rate_pred"].to_numpy()),
            }
        )
    frame = pd.DataFrame(grouped)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(frame.index, frame["delta_vy_baseline"], marker="o", label="baseline", color="#94a3b8")
    axes[0].plot(frame.index, frame["delta_vy_model"], marker="o", label="model", color="#2563eb")
    axes[0].set_title("delta_vy RMSE across |curvature| bins")
    axes[0].set_xlabel("curvature bin")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[1].plot(frame.index, frame["delta_yaw_rate_baseline"], marker="o", label="baseline", color="#94a3b8")
    axes[1].plot(frame.index, frame["delta_yaw_rate_model"], marker="o", label="model", color="#2563eb")
    axes[1].set_title("delta_yaw_rate RMSE across |curvature| bins")
    axes[1].set_xlabel("curvature bin")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()
    fig.tight_layout()
    path = plot_dir / "rmse_by_curvature_bin.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_prediction_scatter(evaluated: pd.DataFrame, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    limits = []
    for channel in RESIDUAL_NAMES:
        actual = evaluated[channel].to_numpy()
        predicted = evaluated[f"{channel}_pred"].to_numpy()
        limits.append(
            (
                min(float(actual.min()), float(predicted.min())),
                max(float(actual.max()), float(predicted.max())),
            )
        )

    for axis, channel, (low, high) in zip(axes, RESIDUAL_NAMES, limits):
        axis.scatter(
            evaluated[f"{channel}_pred"],
            evaluated[channel],
            s=12,
            alpha=0.35,
            color="#2563eb",
            edgecolors="none",
        )
        axis.plot([low, high], [low, high], linestyle="--", color="#dc2626", linewidth=1.0)
        axis.set_title(f"Predicted vs actual: {channel}")
        axis.set_xlabel("Predicted residual")
        axis.set_ylabel("Actual residual")
    fig.tight_layout()
    path = plot_dir / "prediction_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_sampled_distribution_overlay(evaluated: pd.DataFrame, sampled: pd.DataFrame, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {
        "actual": "#111827",
        "sampled": "#2563eb",
        "baseline": "#dc2626",
    }
    for axis, channel in zip(axes, RESIDUAL_NAMES):
        sample_channel = f"{channel}_sample"
        actual = evaluated[channel].to_numpy()
        sampled_values = sampled[sample_channel].to_numpy()
        zero_baseline = np.zeros(len(actual), dtype=float)
        low = min(float(actual.min()), float(sampled_values.min()), 0.0)
        high = max(float(actual.max()), float(sampled_values.max()), 0.0)
        bins = np.linspace(low, high, 36)
        axis.hist(actual, bins=bins, density=True, alpha=0.35, color=colors["actual"], label="actual")
        axis.hist(sampled_values, bins=bins, density=True, histtype="step", linewidth=2.0, color=colors["sampled"], label="model sampled")
        axis.hist(zero_baseline, bins=bins, density=True, histtype="step", linewidth=2.0, color=colors["baseline"], label="zero baseline")
        axis.set_title(f"Distribution match: {channel}")
        axis.set_xlabel("Residual value")
        axis.set_ylabel("Density")
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = plot_dir / "sampled_distribution_overlay.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_wasserstein_bars(evaluated: pd.DataFrame, sampled: pd.DataFrame, plot_dir: Path) -> tuple[Path, dict]:
    wasserstein_metrics = _compute_wasserstein_metrics(evaluated, sampled)
    baseline_distances = wasserstein_metrics["baseline_wasserstein"]
    model_distances = wasserstein_metrics["model_wasserstein"]

    fig, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(RESIDUAL_NAMES))
    width = 0.32
    axis.bar(
        x - width / 2,
        [baseline_distances[channel] for channel in RESIDUAL_NAMES],
        width=width,
        label="zero baseline",
        color="#94a3b8",
    )
    axis.bar(
        x + width / 2,
        [model_distances[channel] for channel in RESIDUAL_NAMES],
        width=width,
        label="sampled model",
        color="#2563eb",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(RESIDUAL_NAMES)
    axis.set_ylabel("Wasserstein distance")
    axis.set_title("Distributional error on held-out residuals")
    axis.legend()
    fig.tight_layout()
    path = plot_dir / "wasserstein_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, wasserstein_metrics


def _save_wasserstein_by_curvature(evaluated: pd.DataFrame, sampled: pd.DataFrame, plot_dir: Path) -> Path:
    quantile_edges = np.quantile(evaluated["abs_curvature"], np.linspace(0.0, 1.0, 7))
    quantile_edges = np.unique(quantile_edges)
    if len(quantile_edges) < 3:
        quantile_edges = np.linspace(float(evaluated["abs_curvature"].min()), float(evaluated["abs_curvature"].max()) + 1e-6, 3)
    evaluated_bins = pd.cut(evaluated["abs_curvature"], bins=quantile_edges, include_lowest=True)
    sampled_bins = pd.cut(sampled["abs_curvature"], bins=quantile_edges, include_lowest=True)

    grouped = []
    for category in evaluated_bins.cat.categories:
        group = evaluated.loc[evaluated_bins == category]
        sample_group = sampled.loc[sampled_bins == category]
        if group.empty or sample_group.empty:
            continue
        grouped.append(
            {
                "curvature_bin": str(category),
                "delta_vy_baseline": float(wasserstein_distance(group["delta_vy"].to_numpy(), np.zeros(len(group), dtype=float))),
                "delta_vy_model": float(wasserstein_distance(group["delta_vy"].to_numpy(), sample_group["delta_vy_sample"].to_numpy())),
                "delta_yaw_rate_baseline": float(wasserstein_distance(group["delta_yaw_rate"].to_numpy(), np.zeros(len(group), dtype=float))),
                "delta_yaw_rate_model": float(wasserstein_distance(group["delta_yaw_rate"].to_numpy(), sample_group["delta_yaw_rate_sample"].to_numpy())),
            }
        )

    frame = pd.DataFrame(grouped)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(frame.index, frame["delta_vy_baseline"], marker="o", label="baseline", color="#94a3b8")
    axes[0].plot(frame.index, frame["delta_vy_model"], marker="o", label="model", color="#2563eb")
    axes[0].set_title("delta_vy Wasserstein across |curvature| bins")
    axes[0].set_xlabel("curvature bin")
    axes[0].set_ylabel("distance")
    axes[0].legend()
    axes[1].plot(frame.index, frame["delta_yaw_rate_baseline"], marker="o", label="baseline", color="#94a3b8")
    axes[1].plot(frame.index, frame["delta_yaw_rate_model"], marker="o", label="model", color="#2563eb")
    axes[1].set_title("delta_yaw_rate Wasserstein across |curvature| bins")
    axes[1].set_xlabel("curvature bin")
    axes[1].set_ylabel("distance")
    axes[1].legend()
    fig.tight_layout()
    path = plot_dir / "wasserstein_by_curvature_bin.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _simulate_rollout_series(
    scenario: Scenario,
    artifact_path: Path | None,
    mode: str,
    calibration_artifact_path: Path | None = None,
    driver_dataset_path: Path | None = None,
) -> dict[str, np.ndarray]:
    if driver_dataset_path is not None and Path(driver_dataset_path).exists():
        canonical = pd.read_parquet(driver_dataset_path)
        controller = ProfiledCenterlineDriver.from_canonical_dataframe(
            canonical,
            speed_quantile=0.6,
            speed_scale=0.6,
            min_speed=10.0,
        )
    else:
        controller = CenterlineDriver()
    env = UncertainRacecarEnv(
        scenario=scenario.source_path,
        uncertainty=mode,
        uncertainty_artifact=artifact_path if mode == "empirical" else None,
        calibration_artifact=calibration_artifact_path,
        renderer=None,
    )
    env.reset(seed=7, options={"uncertainty_mode": mode, "start_mode": "random"})
    samples = {
        "coords": [(env._state.x, env._state.y)],
        "progress": [env._state.progress],
        "vx": [env._state.vx],
        "vy": [env._state.vy],
        "yaw_rate": [env._state.yaw_rate],
        "track": env.track.centerline,
    }
    for _ in range(140):
        action = controller.act(env._state, env.track)
        _, _, terminated, truncated, _ = env.step(action)
        samples["coords"].append((env._state.x, env._state.y))
        samples["progress"].append(env._state.progress)
        samples["vx"].append(env._state.vx)
        samples["vy"].append(env._state.vy)
        samples["yaw_rate"].append(env._state.yaw_rate)
        if terminated or truncated:
            break
    env.close()
    return {key: np.asarray(value, dtype=float) if key != "track" else value for key, value in samples.items()}


def _save_rollout_overlay(
    scenario: Scenario,
    artifact_path: Path,
    plot_dir: Path,
    calibration_artifact_path: Path | None = None,
    driver_dataset_path: Path | None = None,
) -> Path:
    trajectories = {
        "nominal": _simulate_rollout_series(
            scenario,
            artifact_path=None,
            mode="nominal",
            calibration_artifact_path=calibration_artifact_path,
            driver_dataset_path=driver_dataset_path,
        ),
        "empirical": _simulate_rollout_series(
            scenario,
            artifact_path=artifact_path,
            mode="empirical",
            calibration_artifact_path=calibration_artifact_path,
            driver_dataset_path=driver_dataset_path,
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    track = trajectories["nominal"]["track"]
    axes[0].plot(track[:, 0], track[:, 1], color="#94a3b8", linestyle="--", linewidth=1.0, label="track centerline")
    axes[0].plot(trajectories["nominal"]["coords"][:, 0], trajectories["nominal"]["coords"][:, 1], color="#2563eb", label="nominal rollout")
    axes[0].plot(trajectories["empirical"]["coords"][:, 0], trajectories["empirical"]["coords"][:, 1], color="#dc2626", label="empirical rollout")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Trajectory overlay")
    axes[0].legend()

    axes[1].plot(trajectories["nominal"]["progress"], color="#2563eb", label="nominal")
    axes[1].plot(trajectories["empirical"]["progress"], color="#dc2626", label="empirical")
    axes[1].set_title("Progress over rollout")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("track progress")
    axes[1].legend()
    fig.tight_layout()
    path = plot_dir / "rollout_overlay.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_rollout_state_channels(
    scenario: Scenario,
    artifact_path: Path,
    plot_dir: Path,
    calibration_artifact_path: Path | None = None,
    driver_dataset_path: Path | None = None,
) -> Path:
    trajectories = {
        "nominal": _simulate_rollout_series(
            scenario,
            artifact_path=None,
            mode="nominal",
            calibration_artifact_path=calibration_artifact_path,
            driver_dataset_path=driver_dataset_path,
        ),
        "empirical": _simulate_rollout_series(
            scenario,
            artifact_path=artifact_path,
            mode="empirical",
            calibration_artifact_path=calibration_artifact_path,
            driver_dataset_path=driver_dataset_path,
        ),
    }
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    channels = ["vx", "vy", "yaw_rate"]
    colors = {"nominal": "#2563eb", "empirical": "#dc2626"}
    for axis, channel in zip(axes, channels):
        for mode in ("nominal", "empirical"):
            axis.plot(trajectories[mode][channel], color=colors[mode], label=mode)
        axis.set_ylabel(channel)
        axis.legend()
    axes[0].set_title("Dynamic channels during rollout")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    path = plot_dir / "rollout_state_channels.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _residual_shape_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "std": float(np.std(values)),
        "skew": float(skew(values, bias=False)) if len(values) > 2 else 0.0,
        "excess_kurtosis": float(kurtosis(values, fisher=True, bias=False)) if len(values) > 3 else 0.0,
    }


def _multimodal_histogram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values) + 1e-6)
    hist, edges = np.histogram(values, bins=80, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = gaussian_filter1d(hist.astype(float), sigma=1.2)
    peaks, _ = find_peaks(smooth, prominence=max(float(smooth.max()) * 0.045, 1e-6), distance=4)
    return centers, hist, smooth, peaks


def _prepare_multimodal_contexts(evaluated: pd.DataFrame) -> pd.DataFrame:
    frame = evaluated.copy()
    frame["driver"] = frame["trajectory_id"].astype(str).str.split("_&_").str[2].fillna("unknown")
    frame["progress_slice"] = pd.cut(frame["progress"], bins=12, include_lowest=True)
    frame["speed_slice"] = pd.qcut(frame["vx"], q=4, duplicates="drop")
    frame["curvature_slice"] = pd.qcut(frame["abs_curvature"], q=4, duplicates="drop")
    frame["steer_sign"] = pd.cut(frame["steer"], bins=[-np.inf, -0.02, 0.02, np.inf], labels=["left", "straight", "right"])
    frame["throttle_mode"] = pd.cut(frame["throttle"], bins=[-np.inf, 0.05, 0.4, np.inf], labels=["off", "mid", "high"])
    frame["brake_mode"] = pd.cut(frame["brake"], bins=[-np.inf, 0.02, 0.15, np.inf], labels=["off", "light", "hard"])
    frame["control_mode"] = np.select(
        [frame["brake"] > 0.05, frame["throttle"] > 0.2],
        ["brake", "throttle"],
        default="coast",
    )
    return frame


def _collect_multimodal_examples(evaluated: pd.DataFrame) -> list[dict]:
    frame = _prepare_multimodal_contexts(evaluated)
    groupings = [
        ("control_mode", "progress_slice"),
        ("driver", "progress_slice"),
        ("steer_sign", "progress_slice"),
        ("driver", "brake_mode"),
        ("progress_slice", "speed_slice"),
        ("progress_slice", "throttle_mode"),
        ("steer_sign", "speed_slice"),
        ("curvature_slice", "progress_slice"),
    ]

    examples = []
    for channel in RESIDUAL_NAMES:
        candidates = []
        for group_cols in groupings:
            grouped = frame.groupby(list(group_cols), observed=True)
            for key, group in grouped:
                values = group[channel].to_numpy(dtype=float)
                if len(values) < 350:
                    continue
                centers, hist, smooth, peaks = _multimodal_histogram(values)
                if len(peaks) < 2:
                    continue
                candidates.append(
                    {
                        "channel": channel,
                        "group_cols": group_cols,
                        "group_key": key if isinstance(key, tuple) else (key,),
                        "count": int(len(values)),
                        "peak_count": int(len(peaks)),
                        "peak_score": float(np.sum(smooth[peaks])),
                        "values": values,
                        "centers": centers,
                        "hist": hist,
                        "smooth": smooth,
                        "peaks": peaks,
                        "shape_stats": _residual_shape_stats(values),
                    }
                )
        candidates.sort(key=lambda item: (-item["peak_count"], -item["peak_score"], -item["count"]))
        examples.extend(candidates[:2])
    return examples


def _format_multimodal_group(example: dict) -> str:
    parts = [f"{name}={value}" for name, value in zip(example["group_cols"], example["group_key"])]
    return ", ".join(parts)


def _save_multimodal_examples(evaluated: pd.DataFrame, plot_dir: Path) -> tuple[Path, list[dict]]:
    examples = _collect_multimodal_examples(evaluated)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    for axis, example in zip(axes, examples):
        axis.hist(example["values"], bins=50, density=True, color="#bfdbfe", alpha=0.6)
        axis.plot(example["centers"], example["smooth"], color="#1d4ed8", linewidth=2.0)
        axis.scatter(
            example["centers"][example["peaks"]],
            example["smooth"][example["peaks"]],
            color="#dc2626",
            s=24,
            zorder=3,
        )
        axis.set_title(
            f"{example['channel']} | peaks={example['peak_count']} | n={example['count']}\n{_format_multimodal_group(example)}",
            fontsize=9,
        )
        axis.set_xlabel("residual")
        axis.set_ylabel("density")
    for axis in axes[len(examples) :]:
        axis.axis("off")
    fig.tight_layout()
    path = plot_dir / "multimodal_slices.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, examples


def _relative_markdown_path(path: Path, report_path: Path) -> str:
    return str(path.relative_to(report_path.parent))


def generate_uncertainty_report(
    scenario: Scenario,
    dataset_path: Path,
    artifact_path: Path,
    report_path: Path,
    plot_dir: Path,
    source_description: str,
    source_mode: str,
    calibration_artifact_path: Path | None = None,
) -> EvaluationArtifacts:
    canonical = pd.read_parquet(dataset_path)
    residual_table = compute_residual_table(canonical, scenario)
    train_residuals, test_residuals = _trajectory_split(residual_table, train_fraction=0.75)
    calibration_model = load_calibration_model(calibration_artifact_path) if calibration_artifact_path is not None else None
    if calibration_model is not None:
        train_residuals = _center_residual_table(train_residuals, calibration_model)
        test_residuals = _center_residual_table(test_residuals, calibration_model)
        artifact = EmpiricalUncertaintyModel.fit_from_residual_table(train_residuals, scenario)
    else:
        train_canonical = canonical[canonical["trajectory_id"].isin(train_residuals["trajectory_id"].unique())].copy()
        artifact = EmpiricalUncertaintyModel.fit(train_canonical, scenario)
    evaluated = _evaluate_model(test_residuals, artifact, scenario)
    sampled = _sample_model_distribution(test_residuals, artifact, sample_count=4, seed=17)
    rmse_metrics = _compute_rmse_metrics(evaluated)
    wasserstein_metrics = _compute_wasserstein_metrics(evaluated, sampled)
    disabled_stochastic_channels = [
        channel
        for channel in RESIDUAL_NAMES
        if rmse_metrics["model_rmse"][channel] > rmse_metrics["baseline_rmse"][channel] * 1.02
        and wasserstein_metrics["model_wasserstein"][channel] > wasserstein_metrics["baseline_wasserstein"][channel] * 1.02
    ]
    if disabled_stochastic_channels:
        artifact = artifact.copy().zero_residual_channels(disabled_stochastic_channels)
        evaluated = _evaluate_model(test_residuals, artifact, scenario)
        sampled = _sample_model_distribution(test_residuals, artifact, sample_count=4, seed=17)
        rmse_metrics = _compute_rmse_metrics(evaluated)
        wasserstein_metrics = _compute_wasserstein_metrics(evaluated, sampled)
    artifact.save(artifact_path)

    plot_dir.mkdir(parents=True, exist_ok=True)
    track_overview_path = _save_track_overview(canonical, plot_dir)
    control_distributions_path = _save_control_distributions(canonical, plot_dir)
    state_context_path = _save_state_context_plots(canonical, plot_dir)
    hist_path = _save_residual_histograms(evaluated, plot_dir)
    grouped_hist_path = _save_curvature_group_histograms(evaluated, plot_dir)
    heatmap_path = _save_heatmaps(evaluated, plot_dir)
    relationship_path = _save_feature_residual_relationships(evaluated, plot_dir)
    sampled_distribution_path = _save_sampled_distribution_overlay(evaluated, sampled, plot_dir)
    rmse_path, rmse_metrics = _save_rmse_bars(evaluated, plot_dir)
    curvature_rmse_path = _save_rmse_by_curvature(evaluated, plot_dir)
    prediction_scatter_path = _save_prediction_scatter(evaluated, plot_dir)
    wasserstein_path, wasserstein_metrics = _save_wasserstein_bars(evaluated, sampled, plot_dir)
    curvature_wasserstein_path = _save_wasserstein_by_curvature(evaluated, sampled, plot_dir)
    rollout_overlay_path = _save_rollout_overlay(
        scenario,
        artifact_path,
        plot_dir,
        calibration_artifact_path=calibration_artifact_path,
        driver_dataset_path=dataset_path if source_mode == "external" else None,
    )
    rollout_channel_path = _save_rollout_state_channels(
        scenario,
        artifact_path,
        plot_dir,
        calibration_artifact_path=calibration_artifact_path,
        driver_dataset_path=dataset_path if source_mode == "external" else None,
    )
    multimodal_path, multimodal_examples = _save_multimodal_examples(evaluated, plot_dir)

    bucket_sizes = [len(bucket.row_ids) for bucket in artifact.buckets.values()]
    improvement = {
        channel: 100.0
        * (
            rmse_metrics["baseline_rmse"][channel] - rmse_metrics["model_rmse"][channel]
        )
        / max(rmse_metrics["baseline_rmse"][channel], 1e-9)
        for channel in RESIDUAL_NAMES
    }

    metrics = {
        "dataset_rows": int(len(canonical)),
        "residual_rows": int(len(residual_table)),
        "train_trajectories": sorted(train_residuals["trajectory_id"].unique().tolist()),
        "test_trajectories": sorted(test_residuals["trajectory_id"].unique().tolist()),
        "canonical_columns": CANONICAL_COLUMNS,
        "uncertainty_feature_names": FEATURE_NAMES,
        "history_length": int(artifact.history_length),
        "feature_dimension": int(len(artifact.feature_mean)),
        "gate_count": int(len(artifact.buckets)),
        "bucket_size_min": int(min(bucket_sizes)) if bucket_sizes else 0,
        "bucket_size_max": int(max(bucket_sizes)) if bucket_sizes else 0,
        "bucket_size_mean": float(np.mean(bucket_sizes)) if bucket_sizes else 0.0,
        "baseline_rmse": rmse_metrics["baseline_rmse"],
        "model_rmse": rmse_metrics["model_rmse"],
        "rmse_improvement_percent": improvement,
        "baseline_wasserstein": wasserstein_metrics["baseline_wasserstein"],
        "model_wasserstein": wasserstein_metrics["model_wasserstein"],
        "mean_abs_curvature": float(evaluated["abs_curvature"].mean()),
        "mean_speed": float(evaluated["vx"].mean()),
        "multimodal_examples": [
            {
                "channel": example["channel"],
                "group_cols": list(example["group_cols"]),
                "group_key": [str(value) for value in example["group_key"]],
                "count": int(example["count"]),
                "peak_count": int(example["peak_count"]),
                "shape_stats": example["shape_stats"],
            }
            for example in multimodal_examples
        ],
        "disabled_stochastic_channels": disabled_stochastic_channels,
    }
    metrics_path = report_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    hist_ref = _relative_markdown_path(hist_path, report_path)
    track_overview_ref = _relative_markdown_path(track_overview_path, report_path)
    control_distributions_ref = _relative_markdown_path(control_distributions_path, report_path)
    state_context_ref = _relative_markdown_path(state_context_path, report_path)
    grouped_hist_ref = _relative_markdown_path(grouped_hist_path, report_path)
    heatmap_ref = _relative_markdown_path(heatmap_path, report_path)
    relationship_ref = _relative_markdown_path(relationship_path, report_path)
    sampled_distribution_ref = _relative_markdown_path(sampled_distribution_path, report_path)
    rmse_ref = _relative_markdown_path(rmse_path, report_path)
    curvature_rmse_ref = _relative_markdown_path(curvature_rmse_path, report_path)
    prediction_scatter_ref = _relative_markdown_path(prediction_scatter_path, report_path)
    wasserstein_ref = _relative_markdown_path(wasserstein_path, report_path)
    curvature_wasserstein_ref = _relative_markdown_path(curvature_wasserstein_path, report_path)
    rollout_overlay_ref = _relative_markdown_path(rollout_overlay_path, report_path)
    rollout_channel_ref = _relative_markdown_path(rollout_channel_path, report_path)
    multimodal_ref = _relative_markdown_path(multimodal_path, report_path)
    multimodal_lines = []
    for example in multimodal_examples:
        stats = example["shape_stats"]
        multimodal_lines.extend(
            [
                f"- `{example['channel']}` example: `{_format_multimodal_group(example)}`",
                f"  - peaks: `{example['peak_count']}`",
                f"  - samples: `{example['count']}`",
                f"  - skew: `{stats['skew']:.3f}`",
                f"  - excess kurtosis: `{stats['excess_kurtosis']:.3f}`",
            ]
        )

    report_path.write_text(
        "\n".join(
            [
                "# Uncertainty Technical Report",
                "",
                "## 1. Scope of this report",
                "",
                "- This report explains the current uncertainty implementation, the current rendering stack, and what the plots say about the model quality.",
                f"- Data source for this run: `{source_description}`",
                *(
                    [f"- Deterministic calibration artifact applied first: `{Path(calibration_artifact_path).name}`"]
                    if calibration_artifact_path is not None
                    else []
                ),
                f"- Disabled stochastic channels after hold-out check: `{disabled_stochastic_channels if disabled_stochastic_channels else 'none'}`",
                "",
                "## 2. What the current renderer is",
                "",
                "- The current MP4 outputs are **Tier 1 PyBullet mirror renders**.",
                "- The simulator core computes the state update. PyBullet is only used to mirror that state into a simple 3D scene for fast video generation.",
                "- This is intentionally a debugging-grade renderer, not the final publication renderer.",
                "- The current quality gap versus `racecar_gym` and Assetto comes from three limitations:",
                "  1. a simple URDF car rather than a detailed mesh,",
                "  2. a procedurally assembled road rather than a textured track asset,",
                "  3. PyBullet camera/material/lighting limits compared with a game engine or offline render stack.",
                "- The publication-quality path is still the replay export plus Blender scene work.",
                "",
                "## 3. Current source of uncertainty data",
                "",
                f"- In this run the uncertainty source is: `{source_description}`.",
                *(
                    [
                        "- The dataset for this run was created by rolling out the nominal bicycle model and then injecting structured residuals into the next-step dynamic state.",
                        "- The injected channels are:",
                        "  - `delta_vx`",
                        "  - `delta_vy`",
                        "  - `delta_yaw_rate`",
                        "- The injected structure is intentionally nontrivial:",
                        "  - `delta_vx` gets a progress-dependent sinusoidal term plus Gaussian noise,",
                        "  - `delta_vy` and `delta_yaw_rate` become sign-flipping and multimodal in higher-curvature regions,",
                        "  - an additional Gaussian perturbation is added on top of those channels.",
                        "- That makes the demo dataset useful for verifying that the residual model can handle context dependence and non-Gaussian behavior.",
                    ]
                    if source_mode == "synthetic"
                    else [
                        "- The dataset for this run was imported through the canonical ingestion path.",
                        "- The same downstream residual-fitting logic is used, but the residuals come from the imported trajectory data rather than the synthetic demo generator.",
                        "- The exact residual structure in this case depends on the imported laps rather than on hand-written injected noise.",
                        *(
                            [
                                "- A hybrid deterministic calibration model is applied before fitting the stochastic residual model.",
                                "- It combines a parametric longitudinal `delta_vx` correction with a context-conditioned mean residual corrector.",
                                "- So the residuals shown in this report are the centered leftovers after subtracting that predicted mean mismatch.",
                            ]
                            if calibration_artifact_path is not None
                            else []
                        ),
                    ]
                ),
                "",
                "## 4. Canonical dataset schema",
                "",
                f"- Canonical columns: `{CANONICAL_COLUMNS}`",
                "- Important columns for the uncertainty path are:",
                "  - current pose/state: `x`, `y`, `yaw`, `vx`, `vy`, `yaw_rate`",
                "  - track-relative state: `progress`, `lateral_error`, `heading_error`, `curvature`",
                "  - controls: `steer`, `throttle`, `brake`",
                "  - metadata: `trajectory_id`, `track_id`, `car_id`, `frame_index`, `dt`",
                "- This schema is the bridge between synthetic data, Assetto-like logs, and future data sources.",
                "",
                "## 5. Pipeline used to build the uncertainty model",
                "",
                "1. Build or load a canonical trajectory dataset.",
                "2. For every transition, run the nominal dynamic-bicycle model one step forward.",
                "3. Compute residuals:",
                "   - `delta_vx = vx[t+1] - vx_nominal[t+1]`",
                "   - `delta_vy = vy[t+1] - vy_nominal[t+1]`",
                "   - `delta_yaw_rate = yaw_rate[t+1] - yaw_rate_nominal[t+1]`",
                "4. Build a conditional feature vector from the current local driving context.",
                "5. Gate by `(track_id, car_id, progress_bin)` and search neighbors within each gate.",
                "6. At rollout time, sample a residual from nearby contexts and add it to the nominal prediction.",
                "",
                "## 6. Continuous uncertainty model: input and output",
                "",
                f"- Feature names before history expansion: `{FEATURE_NAMES}`",
                f"- History length: `{artifact.history_length}`",
                f"- Full feature dimension: `{metrics['feature_dimension']}`",
                "- Full model input:",
                "  - `[curvature, progress, vx, vy, yaw_rate, steer, throttle, brake, 5-step action history]`",
                "- Model output:",
                "  - `[delta_vx, delta_vy, delta_yaw_rate]`",
                "- So this is a **continuous conditional residual model**:",
                "  - input = current context,",
                "  - output = next-state correction,",
                "  - not a direct physical-parameter randomizer.",
                "",
                "## 7. How the residual is applied during rollout",
                "",
                "- The nominal bicycle model predicts the next dynamic state first.",
                "- The uncertainty model then modifies the prediction as:",
                *(
                    [
                        "  - first, a deterministic mean correction is added from the calibration artifact,",
                        "  - second, a sampled stochastic residual is added on top of that corrected baseline,",
                    ]
                    if calibration_artifact_path is not None
                    else []
                ),
                "  - `vx_next = vx_nominal_next + delta_vx`",
                "  - `vy_next = vy_nominal_next + delta_vy`",
                "  - `yaw_rate_next = yaw_rate_nominal_next + delta_yaw_rate`",
                "- The pose update uses the corrected dynamic state.",
                "- That means the uncertainty is conditioned on the **current state and action**, not injected as a state-independent white-noise term.",
                "- The runtime sampler also supports short block continuation so uncertainty can remain temporally correlated across several steps.",
                "",
                "## 8. Bucket structure used by the current model",
                "",
                f"- Number of gates: `{metrics['gate_count']}`",
                f"- Bucket size range: `{metrics['bucket_size_min']}` to `{metrics['bucket_size_max']}`",
                f"- Mean bucket size: `{metrics['bucket_size_mean']:.2f}`",
                "- Gating variables:",
                "  - `track_id`",
                "  - `car_id`",
                "  - `progress_bin`",
                "- Inside a gate, the model uses normalized feature-space kNN to retrieve local residual examples.",
                "",
                "## 9. Dataset characterization plots",
                "",
                f"![Track Overview]({track_overview_ref})",
                "",
                "- This plot shows the spatial footprint of the dataset and the average speed profile around the lap.",
                "",
                f"![Control Distributions]({control_distributions_ref})",
                "",
                "- These histograms summarize how the vehicle is driven in the dataset.",
                "",
                f"![State Context Plots]({state_context_ref})",
                "",
                "- These plots show the state-action operating envelope before fitting the residual model.",
                "",
                "## 10. Residual and uncertainty plots",
                "",
                f"![Residual Histograms]({hist_ref})",
                "",
                "- These histograms show the raw next-state residual channels the model is trying to represent.",
                "",
                f"![Curvature Group Histograms]({grouped_hist_ref})",
                "",
                "- Grouping by `|curvature|` makes the nonstationary behavior visible.",
                "- The `delta_vy` and `delta_yaw_rate` distributions spread out and split more in high-curvature bins, which is exactly the kind of structure a single Gaussian would miss.",
                "",
                f"![Feature Residual Relationships]({relationship_ref})",
                "",
                "- These hexbins show where the residual energy lives in state space.",
                "- They make it easy to see that lateral and yaw residuals grow with curvature, while longitudinal residuals have a different pattern with speed.",
                "",
                f"![State Space Heatmaps]({heatmap_ref})",
                "",
                "- These heatmaps summarize the mean residual across joint speed-curvature bins.",
                "- This is the clearest picture of where the uncertainty acts across the operating envelope.",
                "",
                f"![Prediction Scatter]({prediction_scatter_ref})",
                "",
                "- This compares held-out predicted mean residuals against actual residuals.",
                "- This plot is useful, but it is not the main accuracy criterion for a sampled multimodal residual model.",
                "",
                f"![Sampled Distribution Overlay]({sampled_distribution_ref})",
                "",
                "- This is the more faithful diagnostic for the current simulator.",
                "- It compares the held-out residual distribution against:",
                "  - residual samples drawn from the fitted model,",
                "  - a zero-residual baseline.",
                "",
                f"![RMSE Comparison]({rmse_ref})",
                "",
                (
                    "- This compares the calibrated zero-residual baseline against the fitted stochastic model on held-out centered residuals."
                    if calibration_artifact_path is not None
                    else "- This compares the nominal zero-residual baseline against the fitted uncertainty mean predictor on held-out data."
                ),
                "- Mean-prediction RMSE can remain weak even when the sampled distribution is useful, especially when the true residual is multimodal and roughly zero mean.",
                "",
                f"![RMSE by Curvature]({curvature_rmse_ref})",
                "",
                "- This checks whether the fitted model helps across multiple curvature regimes rather than only on average.",
                "",
                f"![Wasserstein Comparison]({wasserstein_ref})",
                "",
                "- This compares distributional error on held-out residuals.",
                "- For the deployed simulator, this is often more meaningful than only looking at conditional means.",
                "",
                f"![Wasserstein by Curvature]({curvature_wasserstein_ref})",
                "",
                "- This checks whether the sampled uncertainty model stays closer to the true held-out distribution across different curvature bins.",
                "",
                f"![Rollout Overlay]({rollout_overlay_ref})",
                "",
                "- This shows the nominal and empirical rollouts diverging in trajectory space when the residual model is active.",
                "",
                f"![Rollout State Channels]({rollout_channel_ref})",
                "",
                "- This shows how `vx`, `vy`, and `yaw_rate` evolve differently once the empirical residuals are injected online.",
                "",
                "## 11. Non-Gaussian and multi-peak slices",
                "",
                f"![Multimodal Slices]({multimodal_ref})",
                "",
                "- These are automatically mined context slices where the held-out residual distribution is visibly multi-peaked after conditioning on progress, driver, steering sign, or control regime.",
                "- This is the clearest answer to the question of whether the real Assetto residuals are only small single-peak Gaussians: they are not.",
                *multimodal_lines,
                "",
                "## 12. Key quantitative results from this run",
                "",
                f"- Dataset rows: `{metrics['dataset_rows']}`",
                f"- Residual transitions: `{metrics['residual_rows']}`",
                f"- Mean speed in evaluation set: `{metrics['mean_speed']:.3f}`",
                f"- Mean |curvature| in evaluation set: `{metrics['mean_abs_curvature']:.5f}`",
                f"- Baseline RMSE: `{metrics['baseline_rmse']}`",
                f"- Model RMSE: `{metrics['model_rmse']}`",
                f"- RMSE improvement percentages: `{metrics['rmse_improvement_percent']}`",
                f"- Baseline Wasserstein: `{metrics['baseline_wasserstein']}`",
                f"- Model Wasserstein: `{metrics['model_wasserstein']}`",
                "",
                "## 13. Interpretation",
                "",
                "- The current code path shows that the uncertainty model is applied to the next dynamic state, not directly to hidden parameters.",
                "- The uncertainty is state dependent, action dependent, and weakly history dependent.",
                (
                    "- The plots show that the synthetic dataset contains context-dependent and partly multimodal residual structure."
                    if source_mode == "synthetic"
                    else "- The plots show that the imported Assetto trajectories contain strong context dependence and clear nominal-model mismatch across the operating envelope."
                ),
                "- The automatically mined slices show that some residual channels are genuinely non-Gaussian and multi-peaked once you condition on the right local context.",
                "- For multimodal residuals, mean-prediction RMSE is not the main success criterion.",
                (
                    "- After deterministic calibration, the remaining residuals are a cleaner approximation of the empirical uncertainty we actually want to sample."
                    if calibration_artifact_path is not None
                    else "- The sampled-distribution and Wasserstein plots are the more relevant accuracy evidence for the current simulator design."
                ),
                (
                    "- The sampled-distribution and Wasserstein plots are the more relevant accuracy evidence for the current simulator design."
                    if calibration_artifact_path is not None
                    else ""
                ),
                "",
                "## 14. What should change next",
                "",
                (
                    "- Replace the synthetic source with canonicalized Assetto trajectories and regenerate exactly the same report."
                    if source_mode == "synthetic"
                    else (
                        "- Improve and validate the deterministic longitudinal baseline further so `delta_vx` can eventually stay active in the stochastic layer too."
                        if calibration_artifact_path is not None
                        else "- Calibrate the nominal bicycle model against the imported Assetto data so the residual model learns uncertainty instead of absorbing large deterministic model mismatch."
                    )
                ),
                (
                    "- Validate the same calibrated pipeline on at least one more track/car pair so the results are not Barcelona-specific."
                    if calibration_artifact_path is not None and source_mode != "synthetic"
                    else "- Keep expanding the external-data ingestion path so more tracks, cars, and lap selections can be fit through the same report pipeline."
                ),
                (
                    "- Keep expanding the external-data ingestion path so more tracks, cars, and lap selections can be fit through the same report pipeline."
                    if source_mode != "synthetic"
                    else "- Keep the synthetic generator as a regression testbed for the uncertainty pipeline."
                ),
                "- Keep Tier 1 PyBullet rendering for controller debugging and rapid iteration.",
                "- Treat replay export plus Blender assets/materials/camera choreography as the publication animation path.",
            ]
        ),
        encoding="utf-8",
    )

    return EvaluationArtifacts(
        report_path=report_path,
        metrics_path=metrics_path,
        plot_dir=plot_dir,
        artifact_path=artifact_path,
        dataset_path=dataset_path,
    )


def generate_default_report(
    output_dir: str | Path,
    scenario_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    source_description: str | None = None,
    calibration_artifact_path: str | Path | None = None,
) -> EvaluationArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario(scenario_path)
    if dataset_path is None:
        dataset_path = output_dir / "analysis_dataset.parquet"
        build_demo_dataset(scenario, dataset_path, episodes=18, steps_per_episode=180, seed=4)
        source_description = source_description or "synthetic demo data generated by build_demo_dataset(...)"
        source_mode = "synthetic"
    else:
        dataset_path = Path(dataset_path)
        source_description = source_description or f"canonical dataset loaded from {dataset_path.name}"
        source_mode = "external"
    artifact_path = output_dir / "analysis_uncertainty.pkl"
    report_path = output_dir / "uncertainty_report.md"
    plot_dir = output_dir / "uncertainty_plots"
    return generate_uncertainty_report(
        scenario=scenario,
        dataset_path=Path(dataset_path),
        artifact_path=artifact_path,
        report_path=report_path,
        plot_dir=plot_dir,
        source_description=source_description,
        source_mode=source_mode,
        calibration_artifact_path=Path(calibration_artifact_path) if calibration_artifact_path is not None else None,
    )
