from __future__ import annotations

from pathlib import Path

import pandas as pd

from uncertain_racecar_gym.dataset import build_demo_dataset
from uncertain_racecar_gym.features import FEATURE_NAMES
from uncertain_racecar_gym.analysis import compute_residual_table
from uncertain_racecar_gym.replay_eval import generate_replay_evaluation
from uncertain_racecar_gym.scenario import load_scenario
from uncertain_racecar_gym.uncertainty import EmpiricalUncertaintyModel


def test_residual_table_uses_richer_feature_vector(tmp_path: Path) -> None:
    scenario = load_scenario()
    dataset_path = build_demo_dataset(scenario, tmp_path / "demo.parquet", episodes=3, steps_per_episode=25, seed=4)
    canonical = pd.read_parquet(dataset_path)
    residual_table = compute_residual_table(canonical, scenario)
    feature_vector = residual_table.iloc[0]["feature_vector"]
    expected_dim = len(FEATURE_NAMES) + 3 * scenario.uncertainty.history_length
    assert len(feature_vector) == expected_dim
    assert "speed_gap" in FEATURE_NAMES
    assert "rear_slip_ratio_mean" in residual_table.columns


def test_replay_evaluation_smoke(tmp_path: Path) -> None:
    scenario = load_scenario()
    dataset_path = build_demo_dataset(scenario, tmp_path / "demo.parquet", episodes=5, steps_per_episode=35, seed=6)
    canonical = pd.read_parquet(dataset_path)
    artifact_path = EmpiricalUncertaintyModel.fit(canonical, scenario).save(tmp_path / "artifact.pkl")

    artifacts = generate_replay_evaluation(
        dataset_path=dataset_path,
        scenario_path=scenario.source_path,
        output_dir=tmp_path / "replay_eval",
        calibration_artifact_path=None,
        uncertainty_artifact_path=artifact_path,
        trajectory_limit=1,
        empirical_seeds=(3, 5),
    )
    assert artifacts.report_path.exists()
    assert artifacts.metrics_path.exists()
    assert "Replay Evaluation Report" in artifacts.report_path.read_text(encoding="utf-8")
