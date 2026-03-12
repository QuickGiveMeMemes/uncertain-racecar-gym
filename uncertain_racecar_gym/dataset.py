from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from uncertain_racecar_gym.common import ensure_dir
from uncertain_racecar_gym.controllers import CenterlineDriver
from uncertain_racecar_gym.dynamics import DynamicBicycleModel
from uncertain_racecar_gym.scenario import Scenario, load_scenario
from uncertain_racecar_gym.track import TrackModel

CANONICAL_COLUMNS = [
    "trajectory_id",
    "frame_index",
    "track_id",
    "car_id",
    "t",
    "dt",
    "x",
    "y",
    "yaw",
    "progress",
    "lateral_error",
    "heading_error",
    "curvature",
    "vx",
    "vy",
    "yaw_rate",
    "steer",
    "throttle",
    "brake",
    "wheel_rotation",
    "lap_count",
]


def _normalize_time_seconds(df: pd.DataFrame) -> pd.Series:
    if "t" in df:
        return df["t"].astype(float)
    if "time" in df:
        return df["time"].astype(float)
    if "currentTime" in df:
        t = df["currentTime"].astype(float)
        diffs = t.diff().dropna()
        if not diffs.empty and diffs.median() > 1.0:
            return t / 1000.0
        return t
    return pd.Series(np.arange(len(df), dtype=float) * 0.05)


def _normalize_steer(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    if values.abs().max() <= 1.5:
        return values.clip(-1.0, 1.0)
    return (values / 450.0).clip(-1.0, 1.0)


def _load_records(input_path: Path) -> pd.DataFrame:
    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)
    if input_path.suffix in {".csv", ".tsv"}:
        sep = "\t" if input_path.suffix == ".tsv" else ","
        return pd.read_csv(input_path, sep=sep)
    if input_path.suffix == ".pkl":
        with input_path.open("rb") as handle:
            raw = pickle.load(handle)
        if isinstance(raw, dict) and "states" in raw:
            return pd.DataFrame(raw["states"])
        return pd.DataFrame(raw)
    if input_path.suffix == ".json":
        with input_path.open("r", encoding="utf-8") as handle:
            return pd.DataFrame(json.load(handle))
    raise ValueError(f"Unsupported dataset input: {input_path}")


def canonicalize_dataframe(
    df: pd.DataFrame,
    track: TrackModel,
    track_id: str,
    car_id: str,
    trajectory_id: str,
) -> pd.DataFrame:
    if set(CANONICAL_COLUMNS).issubset(df.columns):
        return df[CANONICAL_COLUMNS].copy()

    t = _normalize_time_seconds(df)
    dt = t.diff().fillna(t.diff().dropna().median() if len(df) > 1 else 0.05)
    x = df.get("world_position_x", df.get("x")).astype(float)
    y = df.get("world_position_y", df.get("y")).astype(float)
    yaw = df.get("yaw", pd.Series(np.zeros(len(df), dtype=float))).astype(float)
    vx = df.get("local_velocity_x", df.get("vx", df.get("speed", pd.Series(np.zeros(len(df)))))).astype(float)
    vy = df.get("local_velocity_y", df.get("vy", pd.Series(np.zeros(len(df))))).astype(float)
    yaw_rate = df.get(
        "angular_velocity_y",
        df.get("yaw_rate", pd.Series(np.gradient(yaw.to_numpy(), np.maximum(t.to_numpy(), 1e-3)))),
    ).astype(float)
    steer = _normalize_steer(df.get("steerAngle", df.get("steer", pd.Series(np.zeros(len(df)))))).astype(float)
    throttle = df.get("accStatus", df.get("throttle", pd.Series(np.zeros(len(df))))).astype(float).clip(0.0, 1.0)
    brake = df.get("brakeStatus", df.get("brake", pd.Series(np.zeros(len(df))))).astype(float).clip(0.0, 1.0)

    rows = []
    wheel_rotation = 0.0
    last_t = float(t.iloc[0]) if len(df) else 0.0
    for frame_index in range(len(df)):
        projection = track.project(float(x.iloc[frame_index]), float(y.iloc[frame_index]))
        delta_t = float(dt.iloc[frame_index]) if frame_index else float(dt.iloc[0] or 0.05)
        wheel_rotation += float(vx.iloc[frame_index]) * delta_t / 0.33
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "frame_index": frame_index,
                "track_id": track_id,
                "car_id": car_id,
                "t": float(t.iloc[frame_index]),
                "dt": delta_t,
                "x": float(x.iloc[frame_index]),
                "y": float(y.iloc[frame_index]),
                "yaw": float(yaw.iloc[frame_index]),
                "progress": projection.progress,
                "lateral_error": projection.lateral_error,
                "heading_error": float(yaw.iloc[frame_index] - projection.heading),
                "curvature": projection.curvature,
                "vx": float(vx.iloc[frame_index]),
                "vy": float(vy.iloc[frame_index]),
                "yaw_rate": float(yaw_rate.iloc[frame_index]),
                "steer": float(steer.iloc[frame_index]),
                "throttle": float(throttle.iloc[frame_index]),
                "brake": float(brake.iloc[frame_index]),
                "wheel_rotation": float(wheel_rotation),
                "lap_count": int(frame_index > 0 and projection.progress + 0.5 < rows[-1]["progress"]) + (rows[-1]["lap_count"] if rows else 0),
            }
        )
        last_t = float(t.iloc[frame_index])
    canonical = pd.DataFrame(rows)
    canonical["heading_error"] = ((canonical["heading_error"] + np.pi) % (2.0 * np.pi)) - np.pi
    return canonical[CANONICAL_COLUMNS]


def build_demo_dataset(
    scenario: Scenario,
    output_path: str | Path,
    episodes: int = 6,
    steps_per_episode: int = 220,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    track = TrackModel.from_config(scenario.track)
    model = DynamicBicycleModel(scenario.vehicle)
    driver = CenterlineDriver()

    rows = []
    for episode in range(episodes):
        progress = float(rng.uniform(0.0, 1.0))
        state = model.initial_state(track, progress=progress, speed=float(rng.uniform(8.0, 14.0)))
        for step in range(steps_per_episode):
            action = driver.act(state, track)
            projection = track.project(state.x, state.y)
            rows.append(
                {
                    "trajectory_id": f"demo_{episode:03d}",
                    "frame_index": step,
                    "track_id": scenario.name,
                    "car_id": "demo_racecar",
                    "t": step * scenario.simulation.dt,
                    "dt": scenario.simulation.dt,
                    "x": state.x,
                    "y": state.y,
                    "yaw": state.yaw,
                    "progress": state.progress,
                    "lateral_error": state.lateral_error,
                    "heading_error": state.heading_error,
                    "curvature": projection.curvature,
                    "vx": state.vx,
                    "vy": state.vy,
                    "yaw_rate": state.yaw_rate,
                    "steer": action[0],
                    "throttle": action[1],
                    "brake": action[2],
                    "wheel_rotation": state.wheel_rotation,
                    "lap_count": state.lap_count,
                }
            )

            residual = np.zeros(3, dtype=float)
            if abs(projection.curvature) > 0.015:
                residual[1] = rng.choice([-0.35, 0.35]) * (0.25 + abs(projection.curvature) * 10.0)
                residual[2] = rng.choice([-0.18, 0.18]) * (0.6 + abs(projection.curvature) * 15.0)
            residual[0] = 0.1 * np.sin(episode + projection.progress * np.pi * 8.0) + rng.normal(0.0, 0.03)
            residual += rng.normal(0.0, [0.02, 0.03, 0.015], size=3)

            state = model.step(state, action, track, scenario.simulation.dt, residual=residual)
            if track.out_of_bounds(state.lateral_error):
                break

    canonical = pd.DataFrame(rows)[CANONICAL_COLUMNS]
    output = Path(output_path)
    ensure_dir(output.parent)
    canonical.to_parquet(output, index=False)
    return output


def build_canonical_dataset(
    inputs: Iterable[str | Path],
    output_path: str | Path,
    scenario: Scenario,
    track_id: str,
    car_id: str,
) -> Path:
    track = TrackModel.from_config(scenario.track)
    frames = []
    for input_path in inputs:
        path = Path(input_path)
        raw = _load_records(path)
        frames.append(canonicalize_dataframe(raw, track=track, track_id=track_id, car_id=car_id, trajectory_id=path.stem))

    canonical = pd.concat(frames, ignore_index=True)
    output = Path(output_path)
    ensure_dir(output.parent)
    canonical.to_parquet(output, index=False)
    return output
