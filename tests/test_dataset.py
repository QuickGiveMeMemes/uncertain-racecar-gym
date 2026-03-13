from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uncertain_racecar_gym.dataset import CANONICAL_COLUMNS, build_canonical_dataset, build_demo_dataset, canonicalize_dataframe, load_records
from uncertain_racecar_gym.scenario import load_scenario
from uncertain_racecar_gym.track import TrackModel
from uncertain_racecar_gym.track_builder import build_track_from_dataset


def test_build_demo_dataset(tmp_path: Path) -> None:
    scenario = load_scenario()
    output = tmp_path / "demo.parquet"
    path = build_demo_dataset(scenario, output, episodes=2, steps_per_episode=20, seed=3)
    frame = pd.read_parquet(path)
    assert list(frame.columns) == CANONICAL_COLUMNS
    assert len(frame) > 0
    assert frame["trajectory_id"].nunique() == 2


def test_canonicalize_assetto_like_dataframe() -> None:
    scenario = load_scenario()
    track = TrackModel.from_config(scenario.track)
    raw = pd.DataFrame(
        {
            "currentTime": [0.0, 0.05, 0.1],
            "world_position_x": [10.0, 10.2, 10.4],
            "world_position_y": [0.0, 0.1, 0.2],
            "yaw": [0.0, 0.02, 0.03],
            "local_velocity_x": [8.0, 8.2, 8.4],
            "local_velocity_y": [0.1, 0.0, -0.1],
            "angular_velocity_y": [0.01, 0.02, 0.03],
            "steerAngle": [0.0, 30.0, 45.0],
            "accStatus": [0.4, 0.5, 0.6],
            "brakeStatus": [0.0, 0.0, 0.1],
        }
    )
    canonical = canonicalize_dataframe(raw, track, track_id="demo_track", car_id="demo_car", trajectory_id="traj")
    assert list(canonical.columns) == CANONICAL_COLUMNS
    assert canonical["track_id"].iloc[0] == "demo_track"
    assert canonical["car_id"].iloc[0] == "demo_car"
    assert canonical["steer"].between(-1.0, 1.0).all()


def test_canonicalize_prefers_lap_time_and_unwraps_resets() -> None:
    scenario = load_scenario()
    track = TrackModel.from_config(scenario.track)
    raw = pd.DataFrame(
        {
            "currentTime": [0.17489, 0.17493, 0.00002, 0.0000633333333],
            "lap time": [174.89, 174.93, 0.02, 0.0633333333],
            "lastLapTime": [0.0, 0.0, 174.95, 174.95],
            "LapCount": [0, 0, 1, 1],
            "world_position_x": [10.0, 10.2, 10.4, 10.6],
            "world_position_y": [0.0, 0.1, 0.2, 0.3],
            "yaw": [0.0, 0.02, 0.03, 0.04],
            "local_velocity_x": [44.4, 44.6, 44.8, 45.0],
            "local_velocity_y": [0.1, 0.0, -0.1, -0.1],
            "angular_velocity_y": [0.01, 0.02, 0.03, 0.03],
            "steerAngle": [0.0, 10.0, 15.0, 20.0],
            "accStatus": [0.4, 0.5, 0.6, 0.6],
            "brakeStatus": [0.0, 0.0, 0.1, 0.0],
        }
    )
    canonical = canonicalize_dataframe(raw, track, track_id="demo_track", car_id="demo_car", trajectory_id="traj")
    assert np.all(np.diff(canonical["t"].to_numpy()) > 0.0)
    assert canonical["dt"].iloc[1] == pytest.approx(0.04, abs=1e-6)
    assert canonical["dt"].iloc[2] == pytest.approx(0.04, abs=1e-6)
    assert canonical["dt"].iloc[3] == pytest.approx(0.0433333333, abs=1e-6)


def test_load_assetto_telemetry_pickle_and_infer_metadata(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry_sample.pkl"
    raw = {
        "telemetry": [
            {
                "currentTime": 0,
                "world_position_x": 10.0,
                "world_position_y": 0.0,
                "yaw": 0.0,
                "local_velocity_x": 8.0,
                "local_velocity_y": 0.1,
                "angular_velocity_y": 0.01,
                "steerAngle": 0.0,
                "accStatus": 0.4,
                "brakeStatus": 0.0,
            },
            {
                "currentTime": 50,
                "world_position_x": 10.2,
                "world_position_y": 0.1,
                "yaw": 0.02,
                "local_velocity_x": 8.2,
                "local_velocity_y": 0.0,
                "angular_velocity_y": 0.02,
                "steerAngle": 15.0,
                "accStatus": 0.5,
                "brakeStatus": 0.0,
            },
        ],
        "static_info": {
            "TrackName": "ks_barcelona",
            "TrackConfiguration": "layout_gp",
            "CarName": "dallara_f317",
        },
    }
    with telemetry_path.open("wb") as handle:
        pickle.dump(raw, handle, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = load_records(telemetry_path)
    assert loaded.source_format == "assetto_telemetry_pickle"
    assert loaded.metadata["track_id"] == "ks_barcelona-layout_gp"
    assert loaded.metadata["car_id"] == "dallara_f317"
    assert len(loaded.frame) == 2


def test_load_assetto_state_pickle_infers_metadata_from_path_when_static_info_is_empty(tmp_path: Path) -> None:
    state_path = tmp_path / "data_sets" / "ks_barcelona-layout_gp" / "dallara_f317" / "session" / "laps" / "lap.pkl"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "states": [
            {
                "lap time": 0.0,
                "world_position_x": 10.0,
                "world_position_y": 0.0,
                "yaw": 0.0,
                "local_velocity_x": 8.0,
                "local_velocity_y": 0.1,
                "angular_velocity_y": 0.01,
                "steerAngle": 0.0,
                "accStatus": 0.4,
                "brakeStatus": 0.0,
            }
        ],
        "static_info": "",
    }
    with state_path.open("wb") as handle:
        pickle.dump(raw, handle, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = load_records(state_path)
    assert loaded.source_format == "assetto_state_pickle"
    assert loaded.metadata["track_id"] == "ks_barcelona-layout_gp"
    assert loaded.metadata["car_id"] == "dallara_f317"


def test_build_canonical_dataset_from_assetto_telemetry_pickle(tmp_path: Path) -> None:
    scenario = load_scenario()
    telemetry_path = tmp_path / "telemetry_sample.pkl"
    output_path = tmp_path / "canonical.parquet"
    raw = {
        "telemetry": [
            {
                "currentTime": 0,
                "world_position_x": 10.0,
                "world_position_y": 0.0,
                "yaw": 0.0,
                "local_velocity_x": 8.0,
                "local_velocity_y": 0.1,
                "angular_velocity_y": 0.01,
                "steerAngle": 0.0,
                "accStatus": 0.4,
                "brakeStatus": 0.0,
            },
            {
                "currentTime": 50,
                "world_position_x": 10.2,
                "world_position_y": 0.1,
                "yaw": 0.02,
                "local_velocity_x": 8.2,
                "local_velocity_y": 0.0,
                "angular_velocity_y": 0.02,
                "steerAngle": 15.0,
                "accStatus": 0.5,
                "brakeStatus": 0.0,
            },
            {
                "currentTime": 100,
                "world_position_x": 10.4,
                "world_position_y": 0.2,
                "yaw": 0.03,
                "local_velocity_x": 8.4,
                "local_velocity_y": -0.1,
                "angular_velocity_y": 0.03,
                "steerAngle": 30.0,
                "accStatus": 0.6,
                "brakeStatus": 0.1,
            },
        ],
        "static_info": {
            "TrackName": "ks_barcelona",
            "TrackConfiguration": "layout_gp",
            "CarName": "dallara_f317",
        },
    }
    with telemetry_path.open("wb") as handle:
        pickle.dump(raw, handle, protocol=pickle.HIGHEST_PROTOCOL)

    path = build_canonical_dataset(
        inputs=[telemetry_path],
        output_path=output_path,
        scenario=scenario,
        track_id=None,
        car_id=None,
    )
    canonical = pd.read_parquet(path)
    manifest = json.loads((tmp_path / "canonical_sources.json").read_text(encoding="utf-8"))

    assert list(canonical.columns) == CANONICAL_COLUMNS
    assert canonical["track_id"].iloc[0] == "ks_barcelona-layout_gp"
    assert canonical["car_id"].iloc[0] == "dallara_f317"
    assert canonical["dt"].iloc[1] == 0.05
    assert manifest[0]["source_format"] == "assetto_telemetry_pickle"


def test_canonicalize_uses_normalized_spline_progress() -> None:
    scenario = load_scenario()
    track = TrackModel.from_config(scenario.track)
    raw = pd.DataFrame(
        {
            "currentTime": [0, 50, 100],
            "NormalizedSplinePosition": [0.1, 0.15, 0.2],
            "world_position_x": [10.0, 10.1, 10.2],
            "world_position_y": [0.0, 0.2, 0.4],
            "yaw": [0.0, 0.02, 0.03],
            "local_velocity_x": [8.0, 8.2, 8.4],
            "local_velocity_y": [0.1, 0.0, -0.1],
            "angular_velocity_y": [0.01, 0.02, 0.03],
            "steerAngle": [0.0, 30.0, 45.0],
            "accStatus": [0.4, 0.5, 0.6],
            "brakeStatus": [0.0, 0.0, 0.1],
            "LapCount": [1, 1, 1],
        }
    )
    canonical = canonicalize_dataframe(raw, track, track_id="demo_track", car_id="demo_car", trajectory_id="traj")
    assert np.allclose(canonical["progress"].to_numpy(), [0.1, 0.15, 0.2])
    assert (canonical["lap_count"] == 1).all()


def test_canonicalize_uses_sample_rate_when_current_time_is_tiny() -> None:
    scenario = load_scenario()
    track = TrackModel.from_config(scenario.track)
    raw = pd.DataFrame(
        {
            "currentTime": [0.00001, 0.00007, 0.00013],
            "raw data sample rate": [50.0, 50.0, 50.0],
            "world_position_x": [10.0, 10.1, 10.2],
            "world_position_y": [0.0, 0.2, 0.4],
            "yaw": [0.0, 0.02, 0.03],
            "local_velocity_x": [8.0, 8.2, 8.4],
            "local_velocity_y": [0.1, 0.0, -0.1],
            "angular_velocity_y": [0.01, 0.02, 0.03],
            "steerAngle": [0.0, 30.0, 45.0],
            "accStatus": [0.4, 0.5, 0.6],
            "brakeStatus": [0.0, 0.0, 0.1],
        }
    )
    canonical = canonicalize_dataframe(raw, track, track_id="demo_track", car_id="demo_car", trajectory_id="traj")
    assert canonical["dt"].iloc[1] == pytest.approx(0.02, abs=1e-9)


def test_build_track_from_progress_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "track.csv"
    scenario_path = tmp_path / "scenario.yaml"
    report_dir = tmp_path / "report"
    inputs = []
    for idx, lateral_offset in enumerate([0.2, -0.15]):
        progress = np.linspace(0.0, 0.999, 500)
        angle = progress * 2.0 * np.pi
        raw = pd.DataFrame(
            {
                "NormalizedSplinePosition": progress,
                "world_position_x": 20.0 * np.cos(angle) + lateral_offset * np.cos(angle),
                "world_position_y": 20.0 * np.sin(angle) + lateral_offset * np.sin(angle),
            }
        )
        path = tmp_path / f"sample_{idx}.pkl"
        with path.open("wb") as handle:
            pickle.dump({"states": raw.to_dict(orient="records"), "static_info": {"TrackName": "circle", "CarName": "demo"}}, handle)
        inputs.append(path)

    artifacts = build_track_from_dataset(
        inputs=inputs,
        output_csv=csv_path,
        scenario_output=scenario_path,
        report_dir=report_dir,
        scenario_name="circle_demo",
        num_bins=300,
        smoothing_window=9,
    )
    track_frame = pd.read_csv(artifacts.csv_path)
    assert len(track_frame) == 300
    assert artifacts.scenario_path == scenario_path
    assert artifacts.report_path == report_dir / "track_report.md"
