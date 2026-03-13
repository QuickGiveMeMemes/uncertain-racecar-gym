from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from uncertain_racecar_gym.analysis import _center_residual_table, _trajectory_split, compute_residual_table, generate_default_report
from uncertain_racecar_gym.common import ensure_dir
from uncertain_racecar_gym.deterministic import HybridCalibrationModel, fit_longitudinal_correction, longitudinal_training_mask
from uncertain_racecar_gym.scenario import load_scenario
from uncertain_racecar_gym.uncertainty import RESIDUAL_NAMES, EmpiricalUncertaintyModel


@dataclass(slots=True)
class NominalCalibrationArtifacts:
    calibration_artifact_path: Path
    calibration_report_path: Path
    calibration_metrics_path: Path
    stochastic_report_path: Path
    stochastic_artifact_path: Path
    plot_dir: Path


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _residual_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        channel: {
            "rmse": _rmse(frame[channel].to_numpy(dtype=float)),
            "mean_abs": float(np.mean(np.abs(frame[channel].to_numpy(dtype=float)))),
            "wasserstein_to_zero": float(
                wasserstein_distance(frame[channel].to_numpy(dtype=float), np.zeros(len(frame), dtype=float))
            ),
        }
        for channel in RESIDUAL_NAMES
    }


def _save_raw_vs_centered_histograms(raw_test: pd.DataFrame, centered_test: pd.DataFrame, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    colors = {"raw": "#94a3b8", "centered": "#2563eb"}
    for row_index, channel in enumerate(RESIDUAL_NAMES):
        axes[row_index, 0].hist(raw_test[channel], bins=60, color=colors["raw"], alpha=0.85)
        axes[row_index, 0].set_title(f"Raw {channel}")
        axes[row_index, 0].set_xlabel("residual")
        axes[row_index, 0].set_ylabel("count")
        axes[row_index, 1].hist(centered_test[channel], bins=60, color=colors["centered"], alpha=0.85)
        axes[row_index, 1].set_title(f"Centered {channel}")
        axes[row_index, 1].set_xlabel("residual")
        axes[row_index, 1].set_ylabel("count")
    fig.tight_layout()
    path = plot_dir / "raw_vs_centered_histograms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_progress_bias_reduction(raw_test: pd.DataFrame, centered_test: pd.DataFrame, plot_dir: Path) -> Path:
    raw_frame = raw_test.copy()
    centered_frame = centered_test.copy()
    raw_frame["progress_bin"] = pd.cut(raw_frame["progress"], bins=30, include_lowest=True)
    centered_frame["progress_bin"] = pd.cut(centered_frame["progress"], bins=30, include_lowest=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, channel in zip(axes, RESIDUAL_NAMES):
        raw_group = raw_frame.groupby("progress_bin", observed=False)[channel].agg(["mean", "std"]).reset_index(drop=True)
        centered_group = centered_frame.groupby("progress_bin", observed=False)[channel].agg(["mean", "std"]).reset_index(drop=True)
        axis.plot(raw_group.index, raw_group["mean"], color="#94a3b8", label="raw mean")
        axis.plot(centered_group.index, centered_group["mean"], color="#2563eb", label="centered mean")
        axis.fill_between(
            raw_group.index,
            raw_group["mean"] - raw_group["std"].fillna(0.0),
            raw_group["mean"] + raw_group["std"].fillna(0.0),
            color="#cbd5e1",
            alpha=0.35,
        )
        axis.fill_between(
            centered_group.index,
            centered_group["mean"] - centered_group["std"].fillna(0.0),
            centered_group["mean"] + centered_group["std"].fillna(0.0),
            color="#93c5fd",
            alpha=0.30,
        )
        axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.4)
        axis.set_ylabel(channel)
        axis.legend(loc="upper right")
    axes[0].set_title("Progress-wise residual mean and spread before/after nominal calibration")
    axes[-1].set_xlabel("progress bin")
    fig.tight_layout()
    path = plot_dir / "progress_bias_reduction.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_calibration_scatter(centered_test: pd.DataFrame, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, channel in zip(axes, RESIDUAL_NAMES):
        axis.hexbin(
            centered_test[f"{channel}_mean"],
            centered_test[f"{channel}_raw"],
            gridsize=32,
            cmap="magma",
            mincnt=1,
        )
        low = min(centered_test[f"{channel}_mean"].min(), centered_test[f"{channel}_raw"].min())
        high = max(centered_test[f"{channel}_mean"].max(), centered_test[f"{channel}_raw"].max())
        axis.plot([low, high], [low, high], linestyle="--", color="#e5e7eb", linewidth=1.0)
        axis.set_title(f"Predicted mean vs raw {channel}")
        axis.set_xlabel("predicted mean correction")
        axis.set_ylabel("raw residual")
    fig.tight_layout()
    path = plot_dir / "calibration_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_longitudinal_focus(
    raw_test: pd.DataFrame,
    longitudinal_test: pd.DataFrame,
    scenario,
    plot_dir: Path,
    min_vx: float,
    max_abs_delta_vx: float,
) -> Path:
    raw_test = raw_test.reset_index(drop=True).copy()
    longitudinal_test = longitudinal_test.reset_index(drop=True).copy()
    stable_raw_mask = longitudinal_training_mask(raw_test, scenario, min_vx=min_vx, max_abs_delta_vx=max_abs_delta_vx).to_numpy(dtype=bool)
    stable_centered_mask = longitudinal_training_mask(longitudinal_test, scenario, min_vx=min_vx, max_abs_delta_vx=max_abs_delta_vx).to_numpy(dtype=bool)
    stable_raw = raw_test.loc[stable_raw_mask].copy()
    stable_centered = longitudinal_test.loc[stable_centered_mask].copy()

    stable_raw["progress_bin"] = pd.cut(stable_raw["progress"], bins=40, include_lowest=True)
    stable_centered["progress_bin"] = pd.cut(stable_centered["progress"], bins=40, include_lowest=True)
    raw_group = stable_raw.groupby("progress_bin", observed=False)["delta_vx"].mean().reset_index(drop=True)
    centered_group = stable_centered.groupby("progress_bin", observed=False)["delta_vx"].mean().reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].hist(stable_raw["delta_vx"], bins=60, color="#94a3b8", alpha=0.82, label="raw")
    axes[0].hist(stable_centered["delta_vx"], bins=60, color="#2563eb", alpha=0.65, label="after parametric vx correction")
    axes[0].set_title("Stable driving delta_vx before/after longitudinal correction")
    axes[0].set_xlabel("delta_vx")
    axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].plot(raw_group.index, raw_group.to_numpy(dtype=float), color="#94a3b8", label="raw mean")
    axes[1].plot(centered_group.index, centered_group.to_numpy(dtype=float), color="#2563eb", label="corrected mean")
    axes[1].axhline(0.0, color="#111827", linewidth=0.8, alpha=0.4)
    axes[1].set_title("Stable driving delta_vx mean over progress")
    axes[1].set_xlabel("progress bin")
    axes[1].set_ylabel("mean delta_vx")
    axes[1].legend()
    fig.tight_layout()
    path = plot_dir / "longitudinal_focus.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def generate_nominal_calibration_package(
    dataset_path: str | Path,
    scenario_path: str | Path | None,
    output_dir: str | Path,
    source_description: str | None = None,
) -> NominalCalibrationArtifacts:
    output_dir = ensure_dir(output_dir)
    plot_dir = ensure_dir(Path(output_dir) / "calibration_plots")
    scenario = load_scenario(scenario_path)
    canonical = pd.read_parquet(dataset_path)
    residual_table = compute_residual_table(canonical, scenario)
    train_residuals, test_residuals = _trajectory_split(residual_table, train_fraction=0.75)

    longitudinal_model, longitudinal_summary, longitudinal_train, longitudinal_test = fit_longitudinal_correction(
        train_residuals,
        test_residuals,
        scenario,
    )
    residual_mean_model = EmpiricalUncertaintyModel.fit_from_residual_table(longitudinal_train, scenario)
    initial_centered_test = _center_residual_table(longitudinal_test, residual_mean_model)
    raw_metrics = _residual_metrics(test_residuals)
    longitudinal_metrics = _residual_metrics(longitudinal_test)
    initial_centered_metrics = _residual_metrics(initial_centered_test)
    disabled_channels = [
        channel
        for channel in RESIDUAL_NAMES
        if initial_centered_metrics[channel]["mean_abs"] >= longitudinal_metrics[channel]["mean_abs"]
    ]
    if disabled_channels:
        residual_mean_model = residual_mean_model.copy().zero_residual_channels(disabled_channels)
    calibration_artifact_path = Path(output_dir) / "nominal_calibration.pkl"
    calibration_model = HybridCalibrationModel(
        longitudinal_model=longitudinal_model if longitudinal_summary.get("active", False) else None,
        residual_model=residual_mean_model,
    )
    calibration_model.save(calibration_artifact_path)

    centered_train = _center_residual_table(train_residuals, calibration_model)
    centered_test = _center_residual_table(test_residuals, calibration_model)

    centered_metrics = _residual_metrics(centered_test)
    metrics = {
        "source_description": source_description or f"canonical dataset loaded from {Path(dataset_path).name}",
        "dataset_rows": int(len(canonical)),
        "train_rows": int(len(train_residuals)),
        "test_rows": int(len(test_residuals)),
        "longitudinal_summary": longitudinal_summary,
        "disabled_channels": disabled_channels,
        "raw_metrics": raw_metrics,
        "longitudinal_metrics": longitudinal_metrics,
        "centered_metrics": centered_metrics,
        "improvement_percent": {
            channel: {
                key: 100.0 * (raw_metrics[channel][key] - centered_metrics[channel][key]) / max(raw_metrics[channel][key], 1e-9)
                for key in raw_metrics[channel]
            }
            for channel in RESIDUAL_NAMES
        },
    }
    metrics_path = Path(output_dir) / "calibration_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    hist_path = _save_raw_vs_centered_histograms(test_residuals, centered_test, plot_dir)
    bias_path = _save_progress_bias_reduction(test_residuals, centered_test, plot_dir)
    scatter_path = _save_calibration_scatter(centered_test, plot_dir)
    longitudinal_focus_path = _save_longitudinal_focus(
        test_residuals,
        longitudinal_test,
        scenario,
        plot_dir,
        min_vx=float(longitudinal_summary.get("runtime_regime", {}).get("min_vx", 2.0)),
        max_abs_delta_vx=float(max(1.0, test_residuals["delta_vx"].abs().quantile(0.995))),
    )

    stochastic_report_dir = ensure_dir(Path(output_dir) / "stochastic_report")
    stochastic_artifacts = generate_default_report(
        output_dir=stochastic_report_dir,
        scenario_path=scenario.source_path,
        dataset_path=dataset_path,
        source_description=(
            source_description or f"canonical dataset loaded from {Path(dataset_path).name}"
        ),
        calibration_artifact_path=calibration_artifact_path,
    )

    report_path = Path(output_dir) / "calibration_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Nominal Calibration Report",
                "",
                "## Summary",
                "",
                f"- Data source: `{metrics['source_description']}`",
                "- This calibration artifact is a hybrid deterministic model.",
                "- Stage 1 is a parametric longitudinal `delta_vx` correction targeted at the forward-driving regime.",
                "- Stage 2 is a context-conditioned kNN mean residual model for any structured bias still left after stage 1.",
                "- The combined artifact is applied before the stochastic uncertainty sampler so the remaining residual is closer to uncertainty than to first-order bias.",
                f"- Calibration artifact: `{calibration_artifact_path.name}`",
                f"- Stochastic uncertainty artifact: `{stochastic_artifacts.artifact_path.name}`",
                f"- Disabled residual-mean channels after hold-out check: `{disabled_channels if disabled_channels else 'none'}`",
                f"- Longitudinal training rows selected: `{longitudinal_summary.get('selected_rows', 0)}` / `{longitudinal_summary.get('train_rows', 0)}`",
                f"- Stable-driving hold-out rows: `{longitudinal_summary.get('stable_test_rows', 0)}`",
                "",
                "## Test-set metric change after deterministic calibration",
                "",
                *[
                    f"- `{channel}`: "
                    f"RMSE `{raw_metrics[channel]['rmse']:.3f} -> {centered_metrics[channel]['rmse']:.3f}`, "
                    f"mean |residual| `{raw_metrics[channel]['mean_abs']:.3f} -> {centered_metrics[channel]['mean_abs']:.3f}`, "
                    f"Wasserstein-to-zero `{raw_metrics[channel]['wasserstein_to_zero']:.3f} -> {centered_metrics[channel]['wasserstein_to_zero']:.3f}`"
                    for channel in RESIDUAL_NAMES
                ],
                "",
                "## Longitudinal stage-1 details",
                "",
                f"- All-data `delta_vx` RMSE: `{longitudinal_summary.get('all_test_metrics', {}).get('rmse_raw', float('nan')):.3f} -> {longitudinal_summary.get('all_test_metrics', {}).get('rmse_centered', float('nan')):.3f}`",
                f"- All-data `delta_vx` mean |residual|: `{longitudinal_summary.get('all_test_metrics', {}).get('mean_abs_raw', float('nan')):.3f} -> {longitudinal_summary.get('all_test_metrics', {}).get('mean_abs_centered', float('nan')):.3f}`",
                f"- Stable-driving `delta_vx` RMSE: `{longitudinal_summary.get('stable_test_metrics', {}).get('rmse_raw', float('nan')):.3f} -> {longitudinal_summary.get('stable_test_metrics', {}).get('rmse_centered', float('nan')):.3f}`",
                f"- Stable-driving `delta_vx` mean |residual|: `{longitudinal_summary.get('stable_test_metrics', {}).get('mean_abs_raw', float('nan')):.3f} -> {longitudinal_summary.get('stable_test_metrics', {}).get('mean_abs_centered', float('nan')):.3f}`",
                f"- Runtime regime: `{longitudinal_summary.get('runtime_regime', {})}`",
                f"- Largest learned longitudinal features: `{longitudinal_summary.get('top_coefficients', [])}`",
                "",
                "## Plots",
                "",
                f"![Raw vs Centered Histograms]({hist_path.relative_to(report_path.parent)})",
                "",
                f"![Progress Bias Reduction]({bias_path.relative_to(report_path.parent)})",
                "",
                f"![Calibration Scatter]({scatter_path.relative_to(report_path.parent)})",
                "",
                f"![Longitudinal Focus]({longitudinal_focus_path.relative_to(report_path.parent)})",
                "",
                "## Follow-on stochastic report",
                "",
                f"- [uncertainty_report.md]({stochastic_artifacts.report_path.relative_to(report_path.parent)})",
                f"- [uncertainty_report.json]({stochastic_artifacts.metrics_path.relative_to(report_path.parent)})",
                f"- [analysis_uncertainty.pkl]({stochastic_artifacts.artifact_path.relative_to(report_path.parent)})",
            ]
        ),
        encoding="utf-8",
    )

    return NominalCalibrationArtifacts(
        calibration_artifact_path=calibration_artifact_path,
        calibration_report_path=report_path,
        calibration_metrics_path=metrics_path,
        stochastic_report_path=stochastic_artifacts.report_path,
        stochastic_artifact_path=stochastic_artifacts.artifact_path,
        plot_dir=plot_dir,
    )
