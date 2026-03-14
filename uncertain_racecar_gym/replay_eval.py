from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncertain_racecar_gym.analysis import _trajectory_split, compute_residual_table
from uncertain_racecar_gym.common import ensure_dir
from uncertain_racecar_gym.env import UncertainRacecarEnv
from uncertain_racecar_gym.scenario import Scenario, load_scenario


@dataclass(slots=True)
class ReplayEvaluationArtifacts:
    report_path: Path
    metrics_path: Path
    plot_dir: Path


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "trajectory"


def _trajectory_series(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("frame_index").reset_index(drop=True)
    return ordered[["frame_index", "x", "y", "progress", "vx", "vy", "yaw_rate"]].copy()


def _simulate_from_actions(
    scenario: Scenario,
    trajectory: pd.DataFrame,
    mode: str,
    seed: int,
    uncertainty_artifact: str | Path | None = None,
    calibration_artifact: str | Path | None = None,
) -> pd.DataFrame:
    env = UncertainRacecarEnv(
        scenario=scenario.source_path,
        uncertainty=mode,
        uncertainty_artifact=uncertainty_artifact,
        calibration_artifact=calibration_artifact,
        apply_mean_correction=calibration_artifact is not None,
        renderer=None,
    )
    env.reset(seed=seed, options={"uncertainty_mode": mode, "start_mode": "grid"})
    first = trajectory.iloc[0]
    env._state = env.dynamics.state_from_canonical_row(first)
    env._history.clear()
    env._previous_feature_state = None
    env.runtime_track_id = str(first["track_id"])
    env.runtime_car_id = str(first["car_id"])

    rows = [
        {
            "frame_index": int(first["frame_index"]),
            "x": float(env._state.x),
            "y": float(env._state.y),
            "progress": float(env._state.progress),
            "vx": float(env._state.vx),
            "vy": float(env._state.vy),
            "yaw_rate": float(env._state.yaw_rate),
        }
    ]
    for row in trajectory.iloc[:-1].itertuples(index=False):
        action = np.array([row.steer, row.throttle, row.brake], dtype=float)
        _, _, terminated, truncated, _ = env.step(action)
        rows.append(
            {
                "frame_index": int(getattr(env._state, "step_count", len(rows))),
                "x": float(env._state.x),
                "y": float(env._state.y),
                "progress": float(env._state.progress),
                "vx": float(env._state.vx),
                "vy": float(env._state.vy),
                "yaw_rate": float(env._state.yaw_rate),
            }
        )
        if terminated or truncated:
            break
    env.close()
    return pd.DataFrame(rows)


def _align_series(actual: pd.DataFrame, simulated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    length = min(len(actual), len(simulated))
    return actual.iloc[:length].reset_index(drop=True), simulated.iloc[:length].reset_index(drop=True)


def _progress_delta(actual: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    delta = np.asarray(simulated, dtype=float) - np.asarray(actual, dtype=float)
    delta = np.where(delta > 0.5, delta - 1.0, delta)
    delta = np.where(delta < -0.5, delta + 1.0, delta)
    return delta


def _metric_summary(actual: pd.DataFrame, simulated: pd.DataFrame) -> dict[str, float]:
    actual_aligned, simulated_aligned = _align_series(actual, simulated)
    position_error = np.linalg.norm(
        simulated_aligned[["x", "y"]].to_numpy(dtype=float) - actual_aligned[["x", "y"]].to_numpy(dtype=float),
        axis=1,
    )
    progress_error = _progress_delta(
        actual_aligned["progress"].to_numpy(dtype=float),
        simulated_aligned["progress"].to_numpy(dtype=float),
    )
    return {
        "rows": int(len(actual_aligned)),
        "position_rmse": float(np.sqrt(np.mean(np.square(position_error)))),
        "position_mae": float(np.mean(np.abs(position_error))),
        "final_position_error": float(position_error[-1]) if len(position_error) else 0.0,
        "progress_rmse": float(np.sqrt(np.mean(np.square(progress_error)))),
        "vx_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        simulated_aligned["vx"].to_numpy(dtype=float) - actual_aligned["vx"].to_numpy(dtype=float)
                    )
                )
            )
        ),
        "vy_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        simulated_aligned["vy"].to_numpy(dtype=float) - actual_aligned["vy"].to_numpy(dtype=float)
                    )
                )
            )
        ),
        "yaw_rate_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        simulated_aligned["yaw_rate"].to_numpy(dtype=float)
                        - actual_aligned["yaw_rate"].to_numpy(dtype=float)
                    )
                )
            )
        ),
    }


def _position_error_series(actual: pd.DataFrame, simulated: pd.DataFrame) -> np.ndarray:
    actual_aligned, simulated_aligned = _align_series(actual, simulated)
    return np.linalg.norm(
        simulated_aligned[["x", "y"]].to_numpy(dtype=float) - actual_aligned[["x", "y"]].to_numpy(dtype=float),
        axis=1,
    )


def _save_overlay_plot(
    actual: pd.DataFrame,
    nominal: pd.DataFrame,
    calibrated: pd.DataFrame,
    empirical: pd.DataFrame,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(actual["x"], actual["y"], color="#111827", linewidth=2.0, label="actual")
    axes[0].plot(nominal["x"], nominal["y"], color="#2563eb", linewidth=1.6, label="nominal")
    axes[0].plot(calibrated["x"], calibrated["y"], color="#16a34a", linewidth=1.6, label="calibrated")
    axes[0].plot(empirical["x"], empirical["y"], color="#dc2626", linewidth=1.6, label="empirical")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Trajectory overlay on recorded action replay")
    axes[0].legend()

    actual_pos, nominal_pos = _align_series(actual, nominal)
    _, calibrated_pos = _align_series(actual, calibrated)
    _, empirical_pos = _align_series(actual, empirical)
    axes[1].plot(_position_error_series(actual_pos, nominal_pos), color="#2563eb", label="nominal")
    axes[1].plot(_position_error_series(actual_pos, calibrated_pos), color="#16a34a", label="calibrated")
    axes[1].plot(_position_error_series(actual_pos, empirical_pos), color="#dc2626", label="empirical")
    axes[1].set_title("Position error vs actual")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("position error [m]")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_channel_plot(
    actual: pd.DataFrame,
    nominal: pd.DataFrame,
    calibrated: pd.DataFrame,
    empirical_runs: list[pd.DataFrame],
    output_path: Path,
) -> Path:
    channels = ["vx", "vy", "yaw_rate"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = {"actual": "#111827", "nominal": "#2563eb", "calibrated": "#16a34a", "empirical": "#dc2626"}
    for axis, channel in zip(axes, channels):
        actual_aligned, nominal_aligned = _align_series(actual, nominal)
        _, calibrated_aligned = _align_series(actual, calibrated)
        empirical_aligned = [_align_series(actual, run)[1] for run in empirical_runs]
        min_length = min(len(frame) for frame in [actual_aligned, nominal_aligned, calibrated_aligned, *empirical_aligned])
        actual_values = actual_aligned[channel].to_numpy(dtype=float)[:min_length]
        nominal_values = nominal_aligned[channel].to_numpy(dtype=float)[:min_length]
        calibrated_values = calibrated_aligned[channel].to_numpy(dtype=float)[:min_length]
        empirical_matrix = np.vstack([frame[channel].to_numpy(dtype=float)[:min_length] for frame in empirical_aligned])
        empirical_mean = empirical_matrix.mean(axis=0)
        empirical_std = empirical_matrix.std(axis=0)

        axis.plot(actual_values, color=colors["actual"], linewidth=2.0, label="actual")
        axis.plot(nominal_values, color=colors["nominal"], linewidth=1.4, label="nominal")
        axis.plot(calibrated_values, color=colors["calibrated"], linewidth=1.4, label="calibrated")
        axis.plot(empirical_mean, color=colors["empirical"], linewidth=1.6, label="empirical mean")
        axis.fill_between(
            np.arange(min_length),
            empirical_mean - empirical_std,
            empirical_mean + empirical_std,
            color="#fca5a5",
            alpha=0.3,
            label="empirical +-1 std" if channel == channels[0] else None,
        )
        axis.set_ylabel(channel)
        axis.legend(loc="upper right")
    axes[0].set_title("Recorded-action replay channels")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_aggregate_plot(metrics: pd.DataFrame, output_path: Path) -> Path:
    grouped = metrics.groupby("mode", observed=True)["position_rmse"].agg(["mean", "std"]).reindex(
        ["nominal", "calibrated", "empirical"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(grouped.index, grouped["mean"], yerr=grouped["std"].fillna(0.0), color=["#2563eb", "#16a34a", "#dc2626"])
    axes[0].set_title("Replay position RMSE")
    axes[0].set_ylabel("RMSE [m]")

    pivot = metrics.pivot(index="trajectory_id", columns="mode", values="final_position_error").reindex(columns=["nominal", "calibrated", "empirical"])
    pivot.plot(kind="bar", ax=axes[1], color=["#2563eb", "#16a34a", "#dc2626"])
    axes[1].set_title("Final position error by trajectory")
    axes[1].set_ylabel("error [m]")
    axes[1].set_xlabel("trajectory")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_replay_evaluation(
    dataset_path: str | Path,
    scenario_path: str | Path | None,
    output_dir: str | Path,
    calibration_artifact_path: str | Path | None,
    uncertainty_artifact_path: str | Path | None,
    trajectory_limit: int = 2,
    empirical_seeds: tuple[int, ...] = (11, 17, 23, 29),
) -> ReplayEvaluationArtifacts:
    scenario = load_scenario(scenario_path)
    output_dir = ensure_dir(output_dir)
    plot_dir = ensure_dir(Path(output_dir) / "plots")
    canonical = pd.read_parquet(dataset_path)
    residual_table = compute_residual_table(canonical, scenario)
    _, test_residuals = _trajectory_split(residual_table, train_fraction=0.75)
    holdout_ids = set(test_residuals["trajectory_id"].unique().tolist())
    candidate_lengths = (
        canonical.loc[canonical["trajectory_id"].isin(holdout_ids)]
        .groupby("trajectory_id", observed=True)
        .size()
        .sort_values(ascending=False)
    )
    selected_ids = candidate_lengths.head(trajectory_limit).index.tolist()
    if not selected_ids:
        selected_ids = (
            canonical.groupby("trajectory_id", observed=True).size().sort_values(ascending=False).head(trajectory_limit).index.tolist()
        )

    metric_rows: list[dict[str, float | str | int]] = []
    plot_paths: list[dict[str, str]] = []
    for trajectory_id in selected_ids:
        trajectory = canonical.loc[canonical["trajectory_id"] == trajectory_id].sort_values("frame_index").reset_index(drop=True)
        actual = _trajectory_series(trajectory)
        nominal = _simulate_from_actions(
            scenario,
            trajectory,
            mode="nominal",
            seed=0,
        )
        calibrated = _simulate_from_actions(
            scenario,
            trajectory,
            mode="nominal",
            seed=0,
            calibration_artifact=calibration_artifact_path,
        )
        empirical_runs = [
            _simulate_from_actions(
                scenario,
                trajectory,
                mode="empirical",
                seed=seed,
                calibration_artifact=calibration_artifact_path,
                uncertainty_artifact=uncertainty_artifact_path,
            )
            for seed in empirical_seeds
        ]
        empirical_metrics = [_metric_summary(actual, run) for run in empirical_runs]
        empirical = empirical_runs[int(np.argmin([item["position_rmse"] for item in empirical_metrics]))]

        mode_summaries = {
            "nominal": _metric_summary(actual, nominal),
            "calibrated": _metric_summary(actual, calibrated),
            "empirical": {
                key: float(np.mean([summary[key] for summary in empirical_metrics]))
                for key in empirical_metrics[0]
            },
        }
        empirical_std = {
            key: float(np.std([summary[key] for summary in empirical_metrics]))
            for key in empirical_metrics[0]
        }
        for mode, summary in mode_summaries.items():
            metric_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "mode": mode,
                    **summary,
                    **(
                        {f"{key}_std": empirical_std[key] for key in empirical_std}
                        if mode == "empirical"
                        else {}
                    ),
                }
            )

        slug = _slugify(trajectory_id)
        overlay_path = _save_overlay_plot(
            actual=actual,
            nominal=nominal,
            calibrated=calibrated,
            empirical=empirical,
            output_path=plot_dir / f"{slug}_overlay.png",
        )
        channel_path = _save_channel_plot(
            actual=actual,
            nominal=nominal,
            calibrated=calibrated,
            empirical_runs=empirical_runs,
            output_path=plot_dir / f"{slug}_channels.png",
        )
        plot_paths.append(
            {
                "trajectory_id": trajectory_id,
                "overlay": str(overlay_path),
                "channels": str(channel_path),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    aggregate_plot_path = _save_aggregate_plot(metrics, plot_dir / "aggregate_replay_metrics.png")
    metrics_path = Path(output_dir) / "replay_eval_metrics.json"
    metrics_payload = {
        "dataset_path": str(Path(dataset_path)),
        "scenario": scenario.name,
        "selected_trajectories": selected_ids,
        "empirical_seeds": list(empirical_seeds),
        "metrics": json.loads(metrics.to_json(orient="records")),
        "plots": plot_paths,
        "aggregate_plot": str(aggregate_plot_path),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    report_path = Path(output_dir) / "replay_eval_report.md"
    lines = [
        "# Replay Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Dataset: `{Path(dataset_path).name}`",
        f"- Scenario: `{scenario.name}`",
        f"- Trajectories evaluated: `{selected_ids}`",
        f"- Empirical seeds: `{list(empirical_seeds)}`",
        "- This evaluation replays the recorded real Assetto action sequence and compares actual data against simulated nominal, calibrated, and empirical rollouts.",
        "",
        "## Aggregate metrics",
        "",
        f"![Aggregate Replay Metrics]({aggregate_plot_path.relative_to(report_path.parent)})",
        "",
        *[
            f"- `{mode}` mean position RMSE: `{group['position_rmse'].mean():.3f}` m"
            for mode, group in metrics.groupby('mode', observed=True)
        ],
    ]
    for plot_info in plot_paths:
        trajectory_id = plot_info["trajectory_id"]
        subset = metrics.loc[metrics["trajectory_id"] == trajectory_id].copy()
        lines.extend(
            [
                "",
                f"## Trajectory `{trajectory_id}`",
                "",
                *[
                    f"- `{row.mode}`: position RMSE `{row.position_rmse:.3f}` m, final position error `{row.final_position_error:.3f}` m, "
                    f"`vx` RMSE `{row.vx_rmse:.3f}`, `vy` RMSE `{row.vy_rmse:.3f}`, `yaw_rate` RMSE `{row.yaw_rate_rmse:.3f}`"
                    + (
                        f", position RMSE std `{getattr(row, 'position_rmse_std', 0.0):.3f}`"
                        if row.mode == "empirical"
                        else ""
                    )
                    for row in subset.itertuples(index=False)
                ],
                "",
                f"![Replay Overlay]({Path(plot_info['overlay']).relative_to(report_path.parent)})",
                "",
                f"![Replay Channels]({Path(plot_info['channels']).relative_to(report_path.parent)})",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return ReplayEvaluationArtifacts(
        report_path=report_path,
        metrics_path=metrics_path,
        plot_dir=plot_dir,
    )
