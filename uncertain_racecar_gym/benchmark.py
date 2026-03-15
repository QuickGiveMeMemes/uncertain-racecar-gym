from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from uncertain_racecar_gym.common import ensure_dir, resolve_resource_path
from uncertain_racecar_gym.controllers import (
    BenchmarkController,
    CenterlineDriver,
    ControllerStepContext,
    DriverControllerAdapter,
    PPOCheckpointController,
    ProfiledCenterlineDriver,
    load_python_controller,
)
from uncertain_racecar_gym.env import UncertainRacecarEnv
from uncertain_racecar_gym.mppi_jax import JaxMPPIConfig, JaxMPPIController
from uncertain_racecar_gym.rendering import PyBulletMirrorRenderer, write_video
from uncertain_racecar_gym.scenario import Scenario, load_scenario
from uncertain_racecar_gym.smooth_mppi_jax import JaxSmoothMPPIConfig, JaxSmoothMPPIController
from uncertain_racecar_gym.track import TrackModel


DEFAULT_GAUSSIAN_STD = (0.7, 0.45, 0.30, 0.08)
DEFAULT_MODES = ("nominal", "gaussian", "empirical")


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    description: str
    start_mode: str = "grid"
    initial_progress: float | None = None
    initial_lateral_error: float = 0.0
    initial_heading_error: float = 0.0
    initial_speed: float | None = None
    max_steps: int = 200
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class BenchmarkSuite:
    name: str
    scenario: str
    cases: tuple[BenchmarkCase, ...]
    modes: tuple[str, ...] = DEFAULT_MODES
    gaussian_std: tuple[float, float, float, float] = DEFAULT_GAUSSIAN_STD
    notes: str | None = None


@dataclass(slots=True)
class BenchmarkArtifacts:
    summary_path: Path
    aggregate_csv_path: Path
    episode_csv_path: Path
    suite_path: Path | None
    package_dir: Path | None
    plot_paths: dict[str, str]
    video_paths: list[str]


def _case_to_dict(case: BenchmarkCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "description": case.description,
        "start_mode": case.start_mode,
        "initial_progress": case.initial_progress,
        "initial_lateral_error": case.initial_lateral_error,
        "initial_heading_error": case.initial_heading_error,
        "initial_speed": case.initial_speed,
        "max_steps": case.max_steps,
        "tags": list(case.tags),
    }


def suite_to_dict(suite: BenchmarkSuite) -> dict[str, Any]:
    return {
        "name": suite.name,
        "scenario": suite.scenario,
        "modes": list(suite.modes),
        "gaussian_std": list(suite.gaussian_std),
        "notes": suite.notes,
        "cases": [_case_to_dict(case) for case in suite.cases],
    }


def write_benchmark_suite(suite: BenchmarkSuite, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(yaml.safe_dump(suite_to_dict(suite), sort_keys=False), encoding="utf-8")
    return output_path


def load_benchmark_suite(path: str | Path) -> BenchmarkSuite:
    suite_path = resolve_resource_path(path)
    raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    cases = []
    for case in raw["cases"]:
        cases.append(
            BenchmarkCase(
                case_id=str(case["case_id"]),
                description=str(case["description"]),
                start_mode=str(case.get("start_mode", "grid")),
                initial_progress=None if case.get("initial_progress") is None else float(case["initial_progress"]),
                initial_lateral_error=float(case.get("initial_lateral_error", 0.0)),
                initial_heading_error=float(case.get("initial_heading_error", 0.0)),
                initial_speed=None if case.get("initial_speed") is None else float(case["initial_speed"]),
                max_steps=int(case.get("max_steps", 200)),
                tags=tuple(case.get("tags", [])),
            )
        )
    return BenchmarkSuite(
        name=str(raw["name"]),
        scenario=str(raw["scenario"]),
        cases=tuple(cases),
        modes=tuple(raw.get("modes", list(DEFAULT_MODES))),
        gaussian_std=tuple(float(x) for x in raw.get("gaussian_std", list(DEFAULT_GAUSSIAN_STD))),
        notes=raw.get("notes"),
    )


def _circular_distance(index_a: int, index_b: int, count: int) -> int:
    delta = abs(index_a - index_b)
    return min(delta, count - delta)


def _pick_distinct_indices(scores: Sequence[tuple[float, int]], count: int, min_gap: int) -> list[int]:
    selected: list[int] = []
    for _, index in sorted(scores, key=lambda item: item[0], reverse=True):
        if all(_circular_distance(index, chosen, count) >= min_gap for chosen in selected):
            selected.append(index)
    return selected


def _choose_distinct_index(
    candidates: Sequence[int],
    *,
    count: int,
    blocked: Sequence[int],
    min_gap: int,
    fallback: int,
) -> int:
    for candidate in candidates:
        if all(_circular_distance(candidate, other, count) >= min_gap for other in blocked):
            return int(candidate)
    return int(fallback)


def _build_track_features(track: TrackModel, num_samples: int = 2048) -> pd.DataFrame:
    progress = np.linspace(0.0, 1.0, num_samples, endpoint=False, dtype=float)
    curvature = np.asarray([track.sample(float(p)).curvature for p in progress], dtype=float)
    ds = float(track.length) / float(num_samples)
    abs_curvature = np.abs(curvature)
    curvature_grad = np.gradient(abs_curvature, ds)
    return pd.DataFrame(
        {
            "index": np.arange(num_samples, dtype=int),
            "progress": progress,
            "curvature": curvature,
            "abs_curvature": abs_curvature,
            "curvature_grad": curvature_grad,
        }
    )


def build_default_stress_suite(
    scenario_path: str | Path,
    *,
    suite_name: str | None = None,
    gaussian_std: Sequence[float] = DEFAULT_GAUSSIAN_STD,
) -> BenchmarkSuite:
    scenario = load_scenario(scenario_path)
    scenario_reference = str(scenario_path)
    track = TrackModel.from_config(scenario.track)
    features = _build_track_features(track)
    count = len(features)
    min_gap = max(64, count // 10)

    peak_scores = [
        (float(row.abs_curvature), int(row.index))
        for row in features.itertuples(index=False)
        if row.abs_curvature >= float(features["abs_curvature"].quantile(0.85))
    ]
    peak_indices = _pick_distinct_indices(peak_scores, count=count, min_gap=min_gap)
    corner_entry_peak = peak_indices[0] if peak_indices else int(features["abs_curvature"].idxmax())
    corner_exit_peak = _choose_distinct_index(
        peak_indices[1:],
        count=count,
        blocked=(corner_entry_peak,),
        min_gap=max(48, min_gap // 2),
        fallback=corner_entry_peak,
    )

    grad_scores = [
        (float(max(row.curvature_grad, 0.0)), int(row.index))
        for row in features.itertuples(index=False)
        if row.curvature_grad > float(features["curvature_grad"].quantile(0.95))
    ]
    turn_in_indices = _pick_distinct_indices(grad_scores, count=count, min_gap=min_gap)
    turn_in_index = _choose_distinct_index(
        turn_in_indices,
        count=count,
        blocked=(corner_entry_peak,),
        min_gap=max(48, min_gap // 2),
        fallback=int(features["curvature_grad"].idxmax()),
    )

    sign_change_scores: list[tuple[float, int]] = []
    window = max(8, count // 60)
    curvature_values = features["curvature"].to_numpy(dtype=float)
    for index in range(count):
        left = np.mean(curvature_values[(index - window) % count : index % count] if (index - window) >= 0 else np.concatenate([curvature_values[index - window :], curvature_values[:index]]))
        right_slice_end = min(index + window, count)
        if right_slice_end <= count:
            right_values = curvature_values[index:right_slice_end]
        else:
            right_values = np.concatenate([curvature_values[index:], curvature_values[: right_slice_end - count]])
        right = np.mean(right_values)
        if left * right < 0.0:
            sign_change_scores.append((abs(left) + abs(right), index))
    chicane_indices = _pick_distinct_indices(sign_change_scores, count=count, min_gap=min_gap)
    chicane_index = _choose_distinct_index(
        chicane_indices,
        count=count,
        blocked=(corner_entry_peak, corner_exit_peak),
        min_gap=max(48, min_gap // 2),
        fallback=corner_exit_peak,
    )

    def shift_progress(index: int, distance_m: float) -> float:
        progress_shift = distance_m / max(track.length, 1e-6)
        return float((features.iloc[index]["progress"] + progress_shift) % 1.0)

    cases = (
        BenchmarkCase(
            case_id="full_lap",
            description="Full lap from the nominal grid start.",
            start_mode="grid",
            max_steps=int(scenario.simulation.max_steps),
            tags=("lap", "baseline"),
        ),
        BenchmarkCase(
            case_id="late_brake_corner_entry",
            description="High-speed approach into a major corner entry where late braking is safety-critical.",
            start_mode="grid",
            initial_progress=shift_progress(corner_entry_peak, -55.0),
            initial_speed=24.0,
            max_steps=260,
            tags=("corner_entry", "braking", "safety"),
        ),
        BenchmarkCase(
            case_id="high_speed_turn_in",
            description="Straight-to-turn transition near the steepest increase in track curvature.",
            start_mode="grid",
            initial_progress=shift_progress(turn_in_index, -30.0),
            initial_speed=26.0,
            max_steps=220,
            tags=("turn_in", "high_speed"),
        ),
        BenchmarkCase(
            case_id="corner_exit_under_throttle",
            description="Corner exit with early throttle application after the main curvature peak.",
            start_mode="grid",
            initial_progress=shift_progress(corner_exit_peak, 18.0),
            initial_speed=13.5,
            max_steps=220,
            tags=("corner_exit", "throttle"),
        ),
        BenchmarkCase(
            case_id="chicane_transition",
            description="Sign-changing curvature transition similar to a chicane or quick left-right sequence.",
            start_mode="grid",
            initial_progress=shift_progress(chicane_index, -28.0),
            initial_speed=20.0,
            max_steps=240,
            tags=("chicane", "transition"),
        ),
    )
    return BenchmarkSuite(
        name=suite_name or f"{scenario.name}_benchmark_suite",
        scenario=scenario_reference,
        cases=cases,
        modes=DEFAULT_MODES,
        gaussian_std=tuple(float(x) for x in gaussian_std),
        notes="Auto-generated stress suite from track curvature and curvature-gradient heuristics.",
    )


def _build_controller(
    *,
    scenario: str,
    controller_kind: str,
    checkpoint: str | Path | None = None,
    controller_spec: str | None = None,
    controller_kwargs_json: str | None = None,
    driver_dataset: str | None = None,
    speed_profile_quantile: float = 0.65,
    speed_profile_scale: float = 0.55,
    target_speed: float = 14.0,
    min_speed: float = 8.0,
) -> BenchmarkController:
    if controller_kind == "ppo_checkpoint":
        if checkpoint is None:
            raise ValueError("controller_kind='ppo_checkpoint' requires --checkpoint.")
        return PPOCheckpointController(checkpoint)
    if controller_kind == "centerline":
        return DriverControllerAdapter(CenterlineDriver(target_speed=target_speed, min_speed=min_speed), name="centerline")
    if controller_kind == "profiled_centerline":
        if driver_dataset is None:
            raise ValueError("controller_kind='profiled_centerline' requires --driver-dataset.")
        canonical = pd.read_parquet(driver_dataset)
        driver = ProfiledCenterlineDriver.from_canonical_dataframe(
            canonical,
            speed_quantile=speed_profile_quantile,
            speed_scale=speed_profile_scale,
            min_speed=min_speed,
        )
        return DriverControllerAdapter(driver, name="profiled_centerline")
    if controller_kind == "python":
        if controller_spec is None:
            raise ValueError("controller_kind='python' requires --controller-spec.")
        kwargs = json.loads(controller_kwargs_json) if controller_kwargs_json else {}
        return load_python_controller(controller_spec, init_kwargs=kwargs)
    if controller_kind == "mppi_jax":
        kwargs = json.loads(controller_kwargs_json) if controller_kwargs_json else {}
        kwargs.setdefault("scenario", scenario)
        kwargs.setdefault("driver_dataset", driver_dataset)
        kwargs.setdefault("speed_profile_quantile", speed_profile_quantile)
        kwargs.setdefault("speed_profile_scale", speed_profile_scale)
        kwargs.setdefault("target_speed", target_speed)
        kwargs.setdefault("min_speed", min_speed)
        return JaxMPPIController(scenario=kwargs.pop("scenario"), config=JaxMPPIConfig(**kwargs))
    if controller_kind == "smooth_mppi_jax":
        kwargs = json.loads(controller_kwargs_json) if controller_kwargs_json else {}
        kwargs.setdefault("scenario", scenario)
        kwargs.setdefault("driver_dataset", driver_dataset)
        kwargs.setdefault("speed_profile_quantile", speed_profile_quantile)
        kwargs.setdefault("speed_profile_scale", speed_profile_scale)
        kwargs.setdefault("target_speed", target_speed)
        kwargs.setdefault("min_speed", min_speed)
        return JaxSmoothMPPIController(scenario=kwargs.pop("scenario"), config=JaxSmoothMPPIConfig(**kwargs))
    raise ValueError(f"Unsupported controller kind: {controller_kind}")


def _make_env(
    *,
    suite: BenchmarkSuite,
    mode: str,
    uncertainty_artifact: str | Path | None,
    calibration_artifact: str | Path | None,
    output_dir: Path,
    gaussian_std: Sequence[float],
) -> UncertainRacecarEnv:
    uncertainty_mode = None if mode == "nominal" else mode
    if mode == "empirical" and uncertainty_artifact is None:
        raise ValueError("Empirical benchmark mode requires --uncertainty-artifact.")
    use_calibration = mode == "empirical" and calibration_artifact is not None
    return UncertainRacecarEnv(
        scenario=suite.scenario,
        uncertainty=uncertainty_mode,
        uncertainty_artifact=uncertainty_artifact,
        calibration_artifact=calibration_artifact if use_calibration else None,
        apply_mean_correction=use_calibration,
        gaussian_noise_std=gaussian_std,
        renderer=None,
        render_mode=None,
        output_dir=output_dir,
    )


def _progress_delta(initial_progress: float, state_progress: float, lap_count: int) -> float:
    return float(lap_count + state_progress - initial_progress)


def _run_episode(
    *,
    env: UncertainRacecarEnv,
    controller: BenchmarkController,
    case: BenchmarkCase,
    mode: str,
    seed: int,
    episode_index: int,
    capture_rollout: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    options = {
        "start_mode": case.start_mode,
        "uncertainty_mode": None if mode == "nominal" else mode,
    }
    if case.initial_progress is not None:
        options["initial_progress"] = float(case.initial_progress)
    if case.initial_speed is not None:
        options["initial_speed"] = float(case.initial_speed)
    if case.initial_lateral_error != 0.0:
        options["initial_lateral_error"] = float(case.initial_lateral_error)
    if case.initial_heading_error != 0.0:
        options["initial_heading_error"] = float(case.initial_heading_error)

    observation, info = env.reset(seed=seed, options=options)
    controller.reset(seed=seed, case_id=case.case_id)
    start_progress = float(env._state.progress if env._state is not None else 0.0)
    prev_x = float(env._state.x if env._state is not None else 0.0)
    prev_y = float(env._state.y if env._state is not None else 0.0)
    total_reward = 0.0
    traveled_distance_m = 0.0
    max_abs_lateral_error = 0.0
    max_abs_heading_error = 0.0
    min_safety_margin = float("inf")
    failure_step = None
    rollout_rows: list[dict[str, Any]] = []

    for step_index in range(case.max_steps):
        context = ControllerStepContext(
            observation=np.asarray(observation, dtype=np.float32),
            step_index=step_index,
            episode_index=episode_index,
            case_id=case.case_id,
            mode=None if mode == "nominal" else mode,
            env=env,
            info=info,
        )
        action = np.asarray(controller.act(observation, context=context), dtype=np.float32)
        planner_debug = None
        debug_getter = getattr(controller, "get_render_debug", None)
        if callable(debug_getter):
            raw_debug = debug_getter()
            if raw_debug is not None:
                planner_debug = {
                    "candidate_xy": np.asarray(
                        raw_debug.get("candidate_xy", np.zeros((0, 0, 2), dtype=np.float32)),
                        dtype=np.float32,
                    ).copy(),
                    "final_xy": np.asarray(
                        raw_debug.get("final_xy", np.zeros((0, 2), dtype=np.float32)),
                        dtype=np.float32,
                    ).copy(),
                }
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        assert env._state is not None
        current_x = float(env._state.x)
        current_y = float(env._state.y)
        traveled_distance_m += math.hypot(current_x - prev_x, current_y - prev_y)
        prev_x = current_x
        prev_y = current_y
        max_abs_lateral_error = max(max_abs_lateral_error, abs(float(env._state.lateral_error)))
        max_abs_heading_error = max(max_abs_heading_error, abs(float(env._state.heading_error)))
        min_safety_margin = min(min_safety_margin, float(env.track.width * 0.5 - abs(env._state.lateral_error)))
        if capture_rollout:
            rollout_rows.append(
                {
                    "step": step_index,
                    "x": float(env._state.x),
                    "y": float(env._state.y),
                    "yaw": float(env._state.yaw),
                    "progress": float(env._state.progress),
                    "lap_count": int(env._state.lap_count),
                    "lateral_error": float(env._state.lateral_error),
                    "heading_error": float(env._state.heading_error),
                    "speed": float(env._state.vx),
                    "steering_angle": float(env._state.steer * env.scenario.vehicle.max_steer_rad),
                    "wheel_rotation": float(env._state.wheel_rotation),
                    "planner_debug": planner_debug,
                }
            )
        if terminated:
            failure_step = step_index + 1
            break
        if truncated:
            break

    assert env._state is not None
    progress_delta = _progress_delta(start_progress, float(env._state.progress), int(env._state.lap_count))
    outcome = "off_track" if info.get("lap_count") is not None and failure_step is not None else ("horizon" if env._state.step_count >= case.max_steps else "finished")
    row = {
        "controller": getattr(controller, "name", type(controller).__name__),
        "case_id": case.case_id,
        "description": case.description,
        "mode": mode,
        "seed": seed,
        "episode_index": episode_index,
        "steps": int(env._state.step_count),
        "terminated": bool(failure_step is not None),
        "failure_step": int(failure_step or case.max_steps),
        "outcome": outcome,
        "total_reward": float(total_reward),
        "progress_delta": float(progress_delta),
        "traveled_distance_m": float(traveled_distance_m),
        "final_progress": float(env._state.progress + env._state.lap_count),
        "lap_count": int(env._state.lap_count),
        "offtrack": float(failure_step is not None),
        "max_abs_lateral_error": float(max_abs_lateral_error),
        "max_abs_heading_error": float(max_abs_heading_error),
        "min_safety_margin": float(min_safety_margin),
        "initial_progress": float(start_progress),
        "initial_speed": float(case.initial_speed if case.initial_speed is not None else 8.0),
        "tags": ",".join(case.tags),
    }
    return row, rollout_rows


def _aggregate_episode_rows(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["case_id", "mode"], observed=False)
    aggregate = grouped.agg(
        episodes=("seed", "size"),
        mean_reward=("total_reward", "mean"),
        mean_progress_delta=("progress_delta", "mean"),
        mean_traveled_distance_m=("traveled_distance_m", "mean"),
        std_progress_delta=("progress_delta", "std"),
        offtrack_rate=("offtrack", "mean"),
        mean_failure_step=("failure_step", "mean"),
        mean_max_abs_lateral_error=("max_abs_lateral_error", "mean"),
        mean_max_abs_heading_error=("max_abs_heading_error", "mean"),
        mean_min_safety_margin=("min_safety_margin", "mean"),
    ).reset_index()
    aggregate["std_progress_delta"] = aggregate["std_progress_delta"].fillna(0.0)
    return aggregate


def _save_benchmark_plots(aggregate: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plot_dir = ensure_dir(output_dir / "plots")
    modes = list(dict.fromkeys(aggregate["mode"].tolist()))
    cases = list(dict.fromkeys(aggregate["case_id"].tolist()))

    def pivot(value: str) -> pd.DataFrame:
        return aggregate.pivot(index="case_id", columns="mode", values=value).reindex(index=cases, columns=modes)

    progress = pivot("mean_progress_delta")
    offtrack = pivot("offtrack_rate")
    failure_step = pivot("mean_failure_step")
    safety = pivot("mean_min_safety_margin")

    paths: dict[str, str] = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    progress.plot(kind="bar", ax=axes[0, 0], colormap="viridis")
    axes[0, 0].set_title("Mean Progress Delta")
    axes[0, 0].set_ylabel("Progress + laps")

    offtrack.plot(kind="bar", ax=axes[0, 1], colormap="plasma")
    axes[0, 1].set_title("Off-track Rate")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_ylim(0.0, 1.0)

    failure_step.plot(kind="bar", ax=axes[1, 0], colormap="magma")
    axes[1, 0].set_title("Mean Failure Step")
    axes[1, 0].set_ylabel("Step")

    safety.plot(kind="bar", ax=axes[1, 1], colormap="cividis")
    axes[1, 1].set_title("Mean Minimum Safety Margin")
    axes[1, 1].set_ylabel("Meters")

    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(alpha=0.2)
    plot_path = plot_dir / "benchmark_dashboard.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    paths["benchmark_dashboard"] = plot_path.as_posix()

    for metric, title in [
        ("mean_max_abs_lateral_error", "Mean Max Lateral Error"),
        ("mean_max_abs_heading_error", "Mean Max Heading Error"),
    ]:
        fig, axis = plt.subplots(figsize=(10, 4))
        pivot(metric).plot(kind="bar", ax=axis, colormap="Accent")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.tick_params(axis="x", rotation=25)
        metric_path = plot_dir / f"{metric}.png"
        fig.tight_layout()
        fig.savefig(metric_path, dpi=180)
        plt.close(fig)
        paths[metric] = metric_path.as_posix()
    return paths


def _render_rollout_video(
    *,
    scenario: Scenario,
    track: TrackModel,
    rollout_rows: list[dict[str, Any]],
    output_path: Path,
    render_mode: str,
    width: int,
    height: int,
    stride: int,
) -> Path | None:
    if not rollout_rows:
        return None
    renderer = PyBulletMirrorRenderer(scenario, track, render_mode, width=width, height=height)
    frames = []
    for step_index, row in enumerate(rollout_rows[:: max(int(stride), 1)]):
        frames.append(
            renderer.render(
                {
                    "x": row["x"],
                    "y": row["y"],
                    "yaw": row["yaw"],
                    "steering_angle": row["steering_angle"],
                    "wheel_rotation": row["wheel_rotation"],
                    "progress": row["progress"],
                    "frame_index": step_index,
                    "speed": row["speed"],
                },
                planner_debug=row.get("planner_debug"),
            )
        )
    renderer.close()
    fps = max(1, round(1.0 / scenario.simulation.dt / max(int(stride), 1)))
    write_video(frames, output_path, fps=fps)
    return output_path


def _write_summary_markdown(
    *,
    output_path: Path,
    suite: BenchmarkSuite,
    controller_name: str,
    aggregate: pd.DataFrame,
    episode_rows: pd.DataFrame,
    plot_paths: dict[str, str],
    video_paths: list[str],
    suite_path: Path | None,
    package_dir: Path | None,
) -> Path:
    lines = [
        f"# Benchmark Summary: {suite.name}",
        "",
        f"- Controller: `{controller_name}`",
        f"- Scenario: `{suite.scenario}`",
        f"- Modes: `{list(suite.modes)}`",
        f"- Gaussian std: `{list(suite.gaussian_std)}`",
        f"- Episodes: `{len(episode_rows)}`",
        f"- Cases: `{len(suite.cases)}`",
    ]
    if suite_path is not None:
        lines.append(f"- Suite YAML: `{suite_path}`")
    if package_dir is not None:
        lines.append(f"- Baseline package: `{package_dir}`")
    if suite.notes:
        lines.extend(["", "## Notes", "", suite.notes])
    if plot_paths:
        lines.extend(["", "## Dashboard", ""])
        for name, path in plot_paths.items():
            lines.append(f"![{name}]({Path(path).resolve().as_posix()})")
            lines.append("")
    lines.extend(["## Aggregate Metrics", ""])
    for case in suite.cases:
        lines.extend([f"### {case.case_id}", "", f"- {case.description}", ""])
        subset = aggregate.loc[aggregate["case_id"] == case.case_id].sort_values("mode")
        for row in subset.itertuples(index=False):
            lines.extend(
                [
                    f"- `{row.mode}`: progress `{row.mean_progress_delta:.4f} +/- {row.std_progress_delta:.4f}`, "
                    f"distance `{row.mean_traveled_distance_m:.1f} m`, off-track `{row.offtrack_rate:.3f}`, "
                    f"failure step `{row.mean_failure_step:.1f}`, min safety margin `{row.mean_min_safety_margin:.3f} m`",
                ]
            )
        lines.append("")
    if video_paths:
        lines.extend(["## Representative Videos", ""])
        for path in video_paths:
            lines.append(f"- `{Path(path).name}`")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def package_baseline_bundle(
    *,
    suite: BenchmarkSuite,
    suite_path: Path | None,
    controller_kind: str,
    controller_name: str,
    checkpoint: str | Path | None,
    uncertainty_artifact: str | Path | None,
    calibration_artifact: str | Path | None,
    output_dir: str | Path,
) -> Path:
    package_dir = ensure_dir(output_dir)
    scenario = load_scenario(suite.scenario)
    raw_scenario = yaml.safe_load(Path(scenario.source_path).read_text(encoding="utf-8"))

    track_source = resolve_resource_path(raw_scenario["track"]["csv"], scenario.source_path)
    track_dir = ensure_dir(package_dir / "tracks")
    copied_track_path = track_dir / Path(track_source).name
    shutil.copy2(track_source, copied_track_path)

    raw_scenario["track"]["csv"] = f"../tracks/{copied_track_path.name}"
    scenario_dir = ensure_dir(package_dir / "scenario")
    scenario_path = scenario_dir / Path(scenario.source_path).name
    scenario_path.write_text(yaml.safe_dump(raw_scenario, sort_keys=False), encoding="utf-8")

    bundled_suite = BenchmarkSuite(
        name=suite.name,
        scenario=f"scenario/{scenario_path.name}",
        cases=suite.cases,
        modes=suite.modes,
        gaussian_std=suite.gaussian_std,
        notes=suite.notes,
    )
    bundled_suite_path = write_benchmark_suite(bundled_suite, package_dir / "benchmark_suite.yaml")

    copied_controller = None
    if checkpoint is not None:
        controller_dir = ensure_dir(package_dir / "controller")
        copied_controller = controller_dir / Path(checkpoint).name
        shutil.copy2(checkpoint, copied_controller)

    copied_uncertainty = None
    if uncertainty_artifact is not None:
        uncertainty_dir = ensure_dir(package_dir / "uncertainty")
        copied_uncertainty = uncertainty_dir / Path(uncertainty_artifact).name
        shutil.copy2(uncertainty_artifact, copied_uncertainty)

    copied_calibration = None
    if calibration_artifact is not None:
        uncertainty_dir = ensure_dir(package_dir / "uncertainty")
        copied_calibration = uncertainty_dir / Path(calibration_artifact).name
        shutil.copy2(calibration_artifact, copied_calibration)

    manifest = {
        "suite_name": suite.name,
        "controller_kind": controller_kind,
        "controller_name": controller_name,
        "scenario": bundled_suite.scenario,
        "suite_yaml": bundled_suite_path.name,
        "checkpoint": None if copied_controller is None else copied_controller.relative_to(package_dir).as_posix(),
        "uncertainty_artifact": None if copied_uncertainty is None else copied_uncertainty.relative_to(package_dir).as_posix(),
        "calibration_artifact": None if copied_calibration is None else copied_calibration.relative_to(package_dir).as_posix(),
        "source_suite": None if suite_path is None else str(suite_path),
        "recommended_command": "uv run --extra jax --extra rl uncertain-racecar-benchmark "
        f"--suite {bundled_suite_path.name} --controller-kind {controller_kind}"
        + (f" --checkpoint {copied_controller.relative_to(package_dir).as_posix()}" if copied_controller is not None else "")
        + (f" --uncertainty-artifact {copied_uncertainty.relative_to(package_dir).as_posix()}" if copied_uncertainty is not None else "")
        + (f" --calibration-artifact {copied_calibration.relative_to(package_dir).as_posix()}" if copied_calibration is not None else "")
        + " --output-dir rerun",
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (package_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Baseline Package: {suite.name}",
                "",
                f"- Controller kind: `{controller_kind}`",
                f"- Controller name: `{controller_name}`",
                f"- Scenario YAML: `{bundled_suite.scenario}`",
                f"- Suite YAML: `{bundled_suite_path.name}`",
                f"- Checkpoint: `{manifest['checkpoint'] or 'none'}`",
                f"- Uncertainty artifact: `{manifest['uncertainty_artifact'] or 'none'}`",
                f"- Calibration artifact: `{manifest['calibration_artifact'] or 'none'}`",
                "",
                "## Recommended Command",
                "",
                f"`{manifest['recommended_command']}`",
                "",
                "This package is intended to be self-contained for later controller comparisons.",
            ]
        ),
        encoding="utf-8",
    )
    return package_dir


def run_benchmark(
    *,
    suite: BenchmarkSuite,
    output_dir: str | Path,
    controller_kind: str,
    checkpoint: str | Path | None = None,
    controller_spec: str | None = None,
    controller_kwargs_json: str | None = None,
    driver_dataset: str | None = None,
    speed_profile_quantile: float = 0.65,
    speed_profile_scale: float = 0.55,
    target_speed: float = 14.0,
    min_speed: float = 8.0,
    seeds: Sequence[int] = (0, 1, 2, 3),
    uncertainty_artifact: str | Path | None = None,
    calibration_artifact: str | Path | None = None,
    modes: Sequence[str] | None = None,
    gaussian_std: Sequence[float] | None = None,
    write_suite_path: str | Path | None = None,
    package_dir: str | Path | None = None,
    render_cases: Sequence[str] = (),
    render_mode: str = "rgb_array_follow",
    render_width: int = 640,
    render_height: int = 360,
    render_stride: int = 1,
) -> BenchmarkArtifacts:
    controller = _build_controller(
        scenario=suite.scenario,
        controller_kind=controller_kind,
        checkpoint=checkpoint,
        controller_spec=controller_spec,
        controller_kwargs_json=controller_kwargs_json,
        driver_dataset=driver_dataset,
        speed_profile_quantile=speed_profile_quantile,
        speed_profile_scale=speed_profile_scale,
        target_speed=target_speed,
        min_speed=min_speed,
    )
    resolved_output_dir = ensure_dir(output_dir)
    resolved_suite = BenchmarkSuite(
        name=suite.name,
        scenario=suite.scenario,
        cases=suite.cases,
        modes=tuple(modes or suite.modes),
        gaussian_std=tuple(float(x) for x in (gaussian_std or suite.gaussian_std)),
        notes=suite.notes,
    )
    suite_path = write_benchmark_suite(resolved_suite, write_suite_path) if write_suite_path is not None else None

    episode_rows: list[dict[str, Any]] = []
    representative_rollouts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for mode in resolved_suite.modes:
        env = _make_env(
            suite=resolved_suite,
            mode=mode,
            uncertainty_artifact=uncertainty_artifact,
            calibration_artifact=calibration_artifact,
            output_dir=resolved_output_dir,
            gaussian_std=resolved_suite.gaussian_std,
        )
        try:
            for case in resolved_suite.cases:
                for episode_index, seed in enumerate(seeds):
                    capture_rollout = bool(render_cases) and episode_index == 0 and case.case_id in set(render_cases)
                    row, rollout_rows = _run_episode(
                        env=env,
                        controller=controller,
                        case=case,
                        mode=mode,
                        seed=int(seed),
                        episode_index=episode_index,
                        capture_rollout=capture_rollout,
                    )
                    episode_rows.append(row)
                    if rollout_rows:
                        representative_rollouts[(case.case_id, mode)] = rollout_rows
        finally:
            env.close()

    episode_frame = pd.DataFrame(episode_rows)
    aggregate_frame = _aggregate_episode_rows(episode_frame)
    episode_csv_path = resolved_output_dir / "episode_metrics.csv"
    aggregate_csv_path = resolved_output_dir / "aggregate_metrics.csv"
    episode_frame.to_csv(episode_csv_path, index=False)
    aggregate_frame.to_csv(aggregate_csv_path, index=False)

    plot_paths = _save_benchmark_plots(aggregate_frame, resolved_output_dir)
    video_paths: list[str] = []
    if representative_rollouts:
        scenario = load_scenario(resolved_suite.scenario)
        track = TrackModel.from_config(scenario.track)
        video_dir = ensure_dir(resolved_output_dir / "videos")
        for (case_id, mode), rollout_rows in representative_rollouts.items():
            video_path = video_dir / f"{case_id}_{mode}.mp4"
            rendered = _render_rollout_video(
                scenario=scenario,
                track=track,
                rollout_rows=rollout_rows,
                output_path=video_path,
                render_mode=render_mode,
                width=render_width,
                height=render_height,
                stride=render_stride,
            )
            if rendered is not None:
                video_paths.append(rendered.as_posix())

    package_path = None
    if package_dir is not None:
        package_path = package_baseline_bundle(
            suite=resolved_suite,
            suite_path=suite_path,
            controller_kind=controller_kind,
            controller_name=getattr(controller, "name", type(controller).__name__),
            checkpoint=checkpoint,
            uncertainty_artifact=uncertainty_artifact,
            calibration_artifact=calibration_artifact,
            output_dir=package_dir,
        )

    summary_path = _write_summary_markdown(
        output_path=resolved_output_dir / "benchmark_summary.md",
        suite=resolved_suite,
        controller_name=getattr(controller, "name", type(controller).__name__),
        aggregate=aggregate_frame,
        episode_rows=episode_frame,
        plot_paths=plot_paths,
        video_paths=video_paths,
        suite_path=suite_path,
        package_dir=package_path,
    )
    return BenchmarkArtifacts(
        summary_path=summary_path,
        aggregate_csv_path=aggregate_csv_path,
        episode_csv_path=episode_csv_path,
        suite_path=suite_path,
        package_dir=package_path,
        plot_paths=plot_paths,
        video_paths=video_paths,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark controllers on nominal, Gaussian, and empirical uncertainty suites.")
    parser.add_argument("--suite", default=None, help="Benchmark suite YAML. If omitted, auto-generate from --scenario.")
    parser.add_argument("--scenario", default="package://scenarios/ks_barcelona_layout_gp_dallara_f317_rl_long.yaml")
    parser.add_argument("--output-dir", default="output/benchmark_run")
    parser.add_argument("--controller-kind", default="ppo_checkpoint", choices=["ppo_checkpoint", "centerline", "profiled_centerline", "python", "mppi_jax", "smooth_mppi_jax"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--controller-spec", default=None)
    parser.add_argument("--controller-kwargs-json", default=None)
    parser.add_argument("--driver-dataset", default=None)
    parser.add_argument("--speed-profile-quantile", type=float, default=0.65)
    parser.add_argument("--speed-profile-scale", type=float, default=0.55)
    parser.add_argument("--target-speed", type=float, default=14.0)
    parser.add_argument("--min-speed", type=float, default=8.0)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--modes", nargs="*", default=None, choices=list(DEFAULT_MODES))
    parser.add_argument("--gaussian-std", nargs=4, type=float, default=list(DEFAULT_GAUSSIAN_STD))
    parser.add_argument("--uncertainty-artifact", default=None)
    parser.add_argument("--calibration-artifact", default=None)
    parser.add_argument("--write-suite", default=None, help="Optional path to save the resolved suite YAML.")
    parser.add_argument("--package-dir", default=None, help="Optional self-contained baseline bundle output directory.")
    parser.add_argument("--render-cases", nargs="*", default=[])
    parser.add_argument("--render-mode", default="rgb_array_follow")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=360)
    parser.add_argument("--render-stride", type=int, default=1)
    return parser


def benchmark_main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    suite = load_benchmark_suite(args.suite) if args.suite is not None else build_default_stress_suite(args.scenario)
    artifacts = run_benchmark(
        suite=suite,
        output_dir=args.output_dir,
        controller_kind=args.controller_kind,
        checkpoint=args.checkpoint,
        controller_spec=args.controller_spec,
        controller_kwargs_json=args.controller_kwargs_json,
        driver_dataset=args.driver_dataset,
        speed_profile_quantile=args.speed_profile_quantile,
        speed_profile_scale=args.speed_profile_scale,
        target_speed=args.target_speed,
        min_speed=args.min_speed,
        seeds=tuple(args.seeds),
        uncertainty_artifact=args.uncertainty_artifact,
        calibration_artifact=args.calibration_artifact,
        modes=None if args.modes is None or len(args.modes) == 0 else tuple(args.modes),
        gaussian_std=tuple(args.gaussian_std),
        write_suite_path=args.write_suite,
        package_dir=args.package_dir,
        render_cases=tuple(args.render_cases),
        render_mode=args.render_mode,
        render_width=args.render_width,
        render_height=args.render_height,
        render_stride=args.render_stride,
    )
    print(
        json.dumps(
            {
                "summary_path": artifacts.summary_path.as_posix(),
                "aggregate_csv_path": artifacts.aggregate_csv_path.as_posix(),
                "episode_csv_path": artifacts.episode_csv_path.as_posix(),
                "suite_path": None if artifacts.suite_path is None else artifacts.suite_path.as_posix(),
                "package_dir": None if artifacts.package_dir is None else artifacts.package_dir.as_posix(),
                "plot_paths": artifacts.plot_paths,
                "video_paths": artifacts.video_paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    benchmark_main()
