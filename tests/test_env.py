from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

import uncertain_racecar_gym  # noqa: F401
from uncertain_racecar_gym.dataset import build_demo_dataset
from uncertain_racecar_gym.analysis import compute_residual_table
from uncertain_racecar_gym.deterministic import HybridCalibrationModel, fit_longitudinal_correction, load_calibration_model
from uncertain_racecar_gym.scenario import load_scenario
from uncertain_racecar_gym.uncertainty import EmpiricalUncertaintyModel, regime_key_from_feature_vector


def test_env_smoke_nominal() -> None:
    env = gym.make("UncertainRacecar-v0", renderer=None)
    obs, info = env.reset(seed=1)
    assert obs.shape[0] == env.observation_space.shape[0]
    assert "render_state" in info
    assert info["uncertainty"]["mode"] == "none"
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


def test_env_gaussian_mode_changes_trajectory_and_can_switch_runtime() -> None:
    nominal = gym.make("UncertainRacecar-v0", renderer=None)
    gaussian = gym.make(
        "UncertainRacecar-v0",
        renderer=None,
        uncertainty="gaussian",
        gaussian_noise_std=[0.25, 0.15, 0.08, 0.03],
    )
    nominal.reset(seed=7)
    gaussian.reset(seed=7, options={"uncertainty_mode": "gaussian"})

    action = np.array([0.05, 0.35, 0.0], dtype=np.float32)
    nominal_positions = []
    gaussian_positions = []
    for _ in range(12):
        _, _, done_n, trunc_n, info_n = nominal.step(action)
        _, _, done_g, trunc_g, info_g = gaussian.step(action)
        nominal_positions.append((info_n["state"]["x"], info_n["state"]["y"]))
        gaussian_positions.append((info_g["state"]["x"], info_g["state"]["y"]))
        if done_n or trunc_n or done_g or trunc_g:
            break

    assert nominal_positions != gaussian_positions
    gaussian.unwrapped.set_uncertainty(None)
    assert gaussian.unwrapped.uncertainty_mode is None
    nominal.close()
    gaussian.close()


def test_fit_from_residual_table_and_gate_resolution(tmp_path: Path) -> None:
    scenario = load_scenario()
    dataset_path = build_demo_dataset(scenario, tmp_path / "demo.parquet", episodes=3, steps_per_episode=25, seed=8)
    canonical = pd.read_parquet(dataset_path)
    residual_table = compute_residual_table(canonical, scenario)
    artifact = EmpiricalUncertaintyModel.fit_from_residual_table(residual_table, scenario)

    assert artifact.gate_prefixes
    gate_key = artifact.resolve_gate_key(progress_bin=3, track_id="wrong_track", car_id="wrong_car")
    assert gate_key[2] == 3

    row = residual_table.iloc[0]
    predicted_mean, _ = artifact.predict_mean(np.asarray(row["feature_vector"], dtype=float), gate_key)
    assert predicted_mean.shape == (3,)


def test_hybrid_calibration_artifact_roundtrip(tmp_path: Path) -> None:
    scenario = load_scenario()
    dataset_path = build_demo_dataset(scenario, tmp_path / "demo.parquet", episodes=4, steps_per_episode=40, seed=9)
    canonical = pd.read_parquet(dataset_path)
    residual_table = compute_residual_table(canonical, scenario)
    trajectory_ids = sorted(residual_table["trajectory_id"].unique())
    train = residual_table[residual_table["trajectory_id"].isin(trajectory_ids[:-1])].copy()
    test = residual_table[residual_table["trajectory_id"].isin(trajectory_ids[-1:])].copy()

    longitudinal_model, _, centered_train, _ = fit_longitudinal_correction(train, test, scenario)
    residual_model = EmpiricalUncertaintyModel.fit_from_residual_table(centered_train, scenario)
    hybrid = HybridCalibrationModel(longitudinal_model=longitudinal_model, residual_model=residual_model)
    artifact_path = hybrid.save(tmp_path / "hybrid_calibration.pkl")

    loaded = load_calibration_model(artifact_path)
    row = test.iloc[0]
    gate_key = loaded.resolve_gate_key(progress_bin=int(row.progress_bin), track_id=str(row.track_id), car_id=str(row.car_id))
    predicted_mean, info = loaded.predict_mean(np.asarray(row.feature_vector, dtype=float), gate_key, dt=float(row["dt"]))
    assert predicted_mean.shape == (3,)
    assert "longitudinal" in info or "residual" in info


def test_sampler_persists_mode_key_across_samples(tmp_path: Path) -> None:
    scenario = load_scenario()
    dataset_path = build_demo_dataset(scenario, tmp_path / "demo.parquet", episodes=6, steps_per_episode=40, seed=12)
    canonical = pd.read_parquet(dataset_path)
    trajectory_ids = sorted(canonical["trajectory_id"].unique())
    rename_map = {
        trajectory_id: f"{'alpha' if index % 2 == 0 else 'beta'}_{index:03d}"
        for index, trajectory_id in enumerate(trajectory_ids)
    }
    canonical["trajectory_id"] = canonical["trajectory_id"].map(rename_map)
    residual_table = compute_residual_table(canonical, scenario)
    artifact = EmpiricalUncertaintyModel.fit_from_residual_table(residual_table, scenario)

    row = residual_table.iloc[0]
    gate_key = artifact.resolve_gate_key(progress_bin=int(row.progress_bin), track_id=str(row.track_id), car_id=str(row.car_id))
    runtime_state = artifact.make_runtime_state()
    rng = np.random.default_rng(3)
    _, info0 = artifact.sample(np.asarray(row.feature_vector, dtype=float), gate_key, rng, runtime_state)
    runtime_state.remaining_block = 0
    _, info1 = artifact.sample(np.asarray(row.feature_vector, dtype=float), gate_key, rng, runtime_state)

    assert runtime_state.active_mode_key is not None
    assert info0["mode_key"] == runtime_state.active_mode_key
    assert info1["mode_key"] == runtime_state.active_mode_key


def test_regime_specific_channel_mask_can_enable_delta_vx_locally(tmp_path: Path) -> None:
    scenario = load_scenario()
    dataset_path = build_demo_dataset(scenario, tmp_path / "demo.parquet", episodes=6, steps_per_episode=50, seed=21)
    canonical = pd.read_parquet(dataset_path)
    residual_table = compute_residual_table(canonical, scenario)
    artifact = EmpiricalUncertaintyModel.fit_from_residual_table(residual_table, scenario)

    chosen_row = None
    chosen_mean = None
    chosen_gate = None
    for row in residual_table.itertuples(index=False):
        gate_key = artifact.resolve_gate_key(progress_bin=int(row.progress_bin), track_id=str(row.track_id), car_id=str(row.car_id))
        predicted_mean, _ = artifact.predict_mean(np.asarray(row.feature_vector, dtype=float), gate_key)
        if abs(float(predicted_mean[0])) > 1e-4:
            chosen_row = row
            chosen_mean = predicted_mean
            chosen_gate = gate_key
            break

    assert chosen_row is not None
    chosen_feature = np.asarray(chosen_row.feature_vector, dtype=float)
    chosen_regime = regime_key_from_feature_vector(chosen_feature)
    masked = artifact.copy().with_channel_masks(
        global_channel_mask=np.array([False, True, True], dtype=bool),
        regime_channel_masks={chosen_regime: np.array([True, True, True], dtype=bool)},
    )

    local_mean, _ = masked.predict_mean(chosen_feature, chosen_gate)
    assert abs(float(local_mean[0])) > 1e-4

    alternate_feature = chosen_feature.copy()
    alternate_feature[6] = 0.0
    alternate_feature[7] = 0.9
    if regime_key_from_feature_vector(alternate_feature) == chosen_regime:
        alternate_feature[7] = 0.0
        alternate_feature[6] = 0.95

    alternate_mean, info = masked.predict_mean(alternate_feature, chosen_gate)
    assert regime_key_from_feature_vector(alternate_feature) != chosen_regime
    assert alternate_mean[0] == 0.0
    assert info["channel_mask"][0] == 0
