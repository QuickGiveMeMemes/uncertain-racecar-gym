from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

import uncertain_racecar_gym  # noqa: F401
from uncertain_racecar_gym.dataset import build_demo_dataset
from uncertain_racecar_gym.scenario import load_scenario
from uncertain_racecar_gym.uncertainty import EmpiricalUncertaintyModel


def test_env_smoke_nominal() -> None:
    env = gym.make("UncertainRacecar-v0", renderer=None)
    obs, info = env.reset(seed=1)
    assert obs.shape[0] == env.observation_space.shape[0]
    assert "render_state" in info
    for _ in range(8):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.3, 0.0], dtype=np.float32))
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()


def test_env_empirical_mode_changes_trajectory(tmp_path: Path) -> None:
    scenario = load_scenario()
    dataset_path = build_demo_dataset(scenario, tmp_path / "demo.parquet", episodes=4, steps_per_episode=35, seed=10)
    canonical = pd.read_parquet(dataset_path)
    artifact = EmpiricalUncertaintyModel.fit(canonical, scenario)
    artifact_path = artifact.save(tmp_path / "artifact.pkl")

    nominal = gym.make("UncertainRacecar-v0", renderer=None)
    empirical = gym.make(
        "UncertainRacecar-v0",
        renderer=None,
        uncertainty="empirical",
        uncertainty_artifact=artifact_path,
    )
    nominal.reset(seed=5)
    empirical.reset(seed=5, options={"uncertainty_mode": "empirical"})

    nominal_positions = []
    empirical_positions = []
    action = np.array([0.0, 0.35, 0.0], dtype=np.float32)
    for _ in range(12):
        _, _, nominal_done, nominal_trunc, nominal_info = nominal.step(action)
        _, _, empirical_done, empirical_trunc, empirical_info = empirical.step(action)
        nominal_positions.append((nominal_info["state"]["x"], nominal_info["state"]["y"]))
        empirical_positions.append((empirical_info["state"]["x"], empirical_info["state"]["y"]))
        if nominal_done or nominal_trunc or empirical_done or empirical_trunc:
            break

    nominal.close()
    empirical.close()
    assert nominal_positions != empirical_positions
