from __future__ import annotations

from pathlib import Path

import pandas as pd

from uncertain_racecar_gym.dataset import CANONICAL_COLUMNS, build_demo_dataset, canonicalize_dataframe
from uncertain_racecar_gym.scenario import load_scenario
from uncertain_racecar_gym.track import TrackModel


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
