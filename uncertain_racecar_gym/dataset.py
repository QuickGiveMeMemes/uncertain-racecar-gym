from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
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


@dataclass(slots=True)
class LoadedRecords:
    frame: pd.DataFrame
    metadata: dict[str, str]
    source_format: str


def _parse_time_value(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return float("nan")
        if ":" in stripped:
            try:
                total = 0.0
                for part in stripped.split(":"):
                    total = total * 60.0 + float(part)
                return total
            except ValueError:
                return float("nan")
        try:
            return float(stripped)
        except ValueError:
            return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _unwrap_lap_time_seconds(
    lap_time: pd.Series,
    lap_count: pd.Series | None = None,
    last_lap_time: pd.Series | None = None,
) -> pd.Series:
    values = lap_time.map(_parse_time_value).astype(float).to_numpy()
    lap_count_values = lap_count.astype(float).to_numpy() if lap_count is not None else None
    last_lap_values = last_lap_time.map(_parse_time_value).astype(float).to_numpy() if last_lap_time is not None else None

    absolute = np.zeros(len(values), dtype=float)
    offset = 0.0
    previous_value = float(values[0]) if len(values) else 0.0
    previous_lap = float(lap_count_values[0]) if lap_count_values is not None and len(lap_count_values) else np.nan

    for index, value in enumerate(values):
        current_value = float(value) if np.isfinite(value) else previous_value
        if index > 0:
            lap_changed = False
            if lap_count_values is not None and np.isfinite(lap_count_values[index]) and np.isfinite(previous_lap):
                lap_changed = lap_count_values[index] > previous_lap
            if lap_changed or current_value + 1e-9 < previous_value:
                lap_duration = previous_value
                if last_lap_values is not None and np.isfinite(last_lap_values[index]) and last_lap_values[index] > 0.0:
                    lap_duration = float(last_lap_values[index])
                offset += lap_duration
        absolute[index] = offset + current_value
        previous_value = current_value
        if lap_count_values is not None and np.isfinite(lap_count_values[index]):
            previous_lap = float(lap_count_values[index])

    return pd.Series(absolute, index=lap_time.index, dtype=float)


def _positive_dt_from_time(t: pd.Series, default_dt: float = 0.05) -> pd.Series:
    diffs = t.diff()
    positive = diffs[diffs > 1e-9]
    fallback = float(positive.median()) if not positive.empty else default_dt
    dt = diffs.where(diffs > 1e-9, fallback).fillna(fallback)
    return dt.astype(float)


def _sanitize_identifier(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _metadata_from_static_info(static_info: object) -> dict[str, str]:
    if not isinstance(static_info, dict):
        return {}

    track_name = static_info.get("TrackName") or static_info.get("track") or static_info.get("venue")
    track_configuration = static_info.get("TrackConfiguration") or static_info.get("track_configuration")
    if track_name and track_configuration:
        track_id = f"{track_name}-{track_configuration}"
    else:
        track_id = track_name

    car_id = static_info.get("CarName") or static_info.get("vehicleid") or static_info.get("car")
    return {
        "track_id": _sanitize_identifier(track_id, "") if track_id else "",
        "car_id": _sanitize_identifier(car_id, "") if car_id else "",
    }


def _metadata_from_path(path: Path) -> dict[str, str]:
    parts = list(path.parts)
    if "data_sets" in parts:
        index = parts.index("data_sets")
        if index + 2 < len(parts):
            return {
                "track_id": _sanitize_identifier(parts[index + 1], ""),
                "car_id": _sanitize_identifier(parts[index + 2], ""),
            }
    return {}


def _normalize_time_seconds(df: pd.DataFrame) -> pd.Series:
    if "t" in df:
        return df["t"].astype(float)
    if "time" in df:
        return df["time"].astype(float)
    if "lap time" in df:
        return _unwrap_lap_time_seconds(
            df["lap time"],
            lap_count=df["LapCount"] if "LapCount" in df else None,
            last_lap_time=df["lastLapTime"] if "lastLapTime" in df else None,
        )
    if "session time left" in df:
        session_left = df["session time left"].map(_parse_time_value).astype(float)
        if session_left.notna().any():
            return (float(session_left.iloc[0]) - session_left).astype(float)
    if "currentTime" in df:
        t = df["currentTime"].map(_parse_time_value).astype(float)
        diffs = t.diff().dropna()
        if not diffs.empty and diffs.median() > 1.0:
            return t / 1000.0
        if not diffs.empty and diffs.median() < 1e-3:
            for rate_key in ("raw data sample rate", "fps"):
                if rate_key in df:
                    rate = pd.to_numeric(df[rate_key], errors="coerce").dropna()
                    if not rate.empty and float(rate.median()) > 1.0:
                        dt = 1.0 / float(rate.median())
                        return pd.Series(np.arange(len(df), dtype=float) * dt, index=df.index, dtype=float)
        return t
    return pd.Series(np.arange(len(df), dtype=float) * 0.05, index=df.index, dtype=float)


def _normalize_steer(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    if values.abs().max() <= 1.5:
        return values.clip(-1.0, 1.0)
    return (values / 450.0).clip(-1.0, 1.0)


def load_records(input_path: str | Path) -> LoadedRecords:
    path = Path(input_path)
    path_metadata = _metadata_from_path(path)
    if path.suffix == ".parquet":
        return LoadedRecords(frame=pd.read_parquet(path), metadata=path_metadata, source_format="parquet")
    if path.suffix in {".csv", ".tsv"}:
        sep = "\t" if path.suffix == ".tsv" else ","
        return LoadedRecords(frame=pd.read_csv(path, sep=sep), metadata=path_metadata, source_format=path.suffix.lstrip("."))
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        if isinstance(raw, dict) and "telemetry" in raw:
            metadata = {**path_metadata, **_metadata_from_static_info(raw.get("static_info"))}
            return LoadedRecords(
                frame=pd.DataFrame(raw["telemetry"]),
                metadata=metadata,
                source_format="assetto_telemetry_pickle",
            )
        if isinstance(raw, dict) and "states" in raw:
            metadata = {**path_metadata, **_metadata_from_static_info(raw.get("static_info"))}
            return LoadedRecords(
                frame=pd.DataFrame(raw["states"]),
                metadata=metadata,
                source_format="assetto_state_pickle",
            )
        if isinstance(raw, dict):
            return LoadedRecords(frame=pd.DataFrame(raw), metadata=path_metadata, source_format="pickle_mapping")
        return LoadedRecords(frame=pd.DataFrame(raw), metadata=path_metadata, source_format="pickle_records")
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict) and "telemetry" in raw:
            metadata = {**path_metadata, **_metadata_from_static_info(raw.get("static_info"))}
            return LoadedRecords(
                frame=pd.DataFrame(raw["telemetry"]),
                metadata=metadata,
                source_format="assetto_telemetry_json",
            )
        if isinstance(raw, dict) and "states" in raw:
            metadata = {**path_metadata, **_metadata_from_static_info(raw.get("static_info"))}
            return LoadedRecords(
                frame=pd.DataFrame(raw["states"]),
                metadata=metadata,
                source_format="assetto_state_json",
            )
        if isinstance(raw, dict):
            return LoadedRecords(frame=pd.DataFrame(raw), metadata=path_metadata, source_format="json_mapping")
        return LoadedRecords(frame=pd.DataFrame(raw), metadata=path_metadata, source_format="json_records")
    raise ValueError(f"Unsupported dataset input: {path}")


def _load_records(input_path: Path) -> pd.DataFrame:
    return load_records(input_path).frame


def _resolve_ids(track_id: str | None, car_id: str | None, metadata: dict[str, str], fallback_track_id: str) -> tuple[str, str]:
    resolved_track_id = _sanitize_identifier(track_id or metadata.get("track_id"), fallback_track_id)
    resolved_car_id = _sanitize_identifier(car_id or metadata.get("car_id"), "assetto_car")
    return resolved_track_id, resolved_car_id


def _fill_metadata_columns(canonical: pd.DataFrame, track_id: str, car_id: str, trajectory_id: str) -> pd.DataFrame:
    canonical = canonical.copy()
    canonical["track_id"] = canonical["track_id"].replace("", pd.NA).fillna(track_id)
    canonical["car_id"] = canonical["car_id"].replace("", pd.NA).fillna(car_id)
    canonical["trajectory_id"] = canonical["trajectory_id"].replace("", pd.NA).fillna(trajectory_id)
    return canonical[CANONICAL_COLUMNS]


def canonicalize_dataframe(
    df: pd.DataFrame,
    track: TrackModel,
    track_id: str,
    car_id: str,
    trajectory_id: str,
) -> pd.DataFrame:
    if set(CANONICAL_COLUMNS).issubset(df.columns):
        return _fill_metadata_columns(df[CANONICAL_COLUMNS].copy(), track_id=track_id, car_id=car_id, trajectory_id=trajectory_id)

    t = _normalize_time_seconds(df)
    dt = _positive_dt_from_time(t)
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

    progress_series = None
    if "NormalizedSplinePosition" in df:
        progress_series = np.mod(df["NormalizedSplinePosition"].astype(float).to_numpy(), 1.0)
    elif "progress" in df:
        progress_series = np.mod(df["progress"].astype(float).to_numpy(), 1.0)

    rows = []
    wheel_rotation = 0.0
    last_t = float(t.iloc[0]) if len(df) else 0.0
    for frame_index in range(len(df)):
        if progress_series is not None:
            progress = float(progress_series[frame_index])
            projection = track.sample(progress)
            tangent = np.array([np.cos(projection.heading), np.sin(projection.heading)])
            normal = np.array([-tangent[1], tangent[0]])
            point = np.array([float(x.iloc[frame_index]), float(y.iloc[frame_index])], dtype=float)
            center = np.array([projection.x, projection.y], dtype=float)
            lateral_error = float(np.dot(point - center, normal))
            heading_error = float(yaw.iloc[frame_index] - projection.heading)
        else:
            projection = track.project(float(x.iloc[frame_index]), float(y.iloc[frame_index]))
            progress = projection.progress
            lateral_error = projection.lateral_error
            heading_error = float(yaw.iloc[frame_index] - projection.heading)
        delta_t = float(dt.iloc[frame_index]) if frame_index else float(dt.iloc[0] or 0.05)
        wheel_rotation += float(vx.iloc[frame_index]) * delta_t / 0.33
        if "LapCount" in df:
            lap_count = int(df["LapCount"].iloc[frame_index])
        else:
            lap_count = int(frame_index > 0 and projection.progress + 0.5 < rows[-1]["progress"]) + (rows[-1]["lap_count"] if rows else 0)
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
                "progress": progress,
                "lateral_error": lateral_error,
                "heading_error": heading_error,
                "curvature": projection.curvature,
                "vx": float(vx.iloc[frame_index]),
                "vy": float(vy.iloc[frame_index]),
                "yaw_rate": float(yaw_rate.iloc[frame_index]),
                "steer": float(steer.iloc[frame_index]),
                "throttle": float(throttle.iloc[frame_index]),
                "brake": float(brake.iloc[frame_index]),
                "wheel_rotation": float(wheel_rotation),
                "lap_count": lap_count,
            }
        )
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
    track_id: str | None,
    car_id: str | None,
) -> Path:
    track = TrackModel.from_config(scenario.track)
    frames = []
    manifest = []
    for input_path in inputs:
        path = Path(input_path)
        loaded = load_records(path)
        resolved_track_id, resolved_car_id = _resolve_ids(track_id, car_id, loaded.metadata, fallback_track_id=path.stem)
        canonical = canonicalize_dataframe(
            loaded.frame,
            track=track,
            track_id=resolved_track_id,
            car_id=resolved_car_id,
            trajectory_id=path.stem,
        )
        frames.append(canonical)
        manifest.append(
            {
                "input_path": str(path),
                "source_format": loaded.source_format,
                "track_id": resolved_track_id,
                "car_id": resolved_car_id,
                "rows": int(len(canonical)),
            }
        )

    canonical = pd.concat(frames, ignore_index=True)
    output = Path(output_path)
    ensure_dir(output.parent)
    canonical.to_parquet(output, index=False)
    manifest_path = output.with_name(f"{output.stem}_sources.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output
