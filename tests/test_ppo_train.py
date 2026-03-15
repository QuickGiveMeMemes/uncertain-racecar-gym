from __future__ import annotations

from pathlib import Path

import pytest


jax = pytest.importorskip("jax")
pytest.importorskip("optax")
pytest.importorskip("wandb")

from uncertain_racecar_gym.ppo_train import PPOConfig, train_ppo


def test_train_ppo_smoke(tmp_path: Path) -> None:
    scenario = Path("/Users/ktk/Desktop/mycode/uncertain-racecar-gym/uncertain_racecar_gym/assets/scenarios/sample_oval.yaml")
    config = PPOConfig(
        scenario=scenario.as_posix(),
        output_dir=tmp_path.as_posix(),
        run_name="ppo_smoke_test",
        total_timesteps=2048,
        num_envs=8,
        num_steps=32,
        num_minibatches=4,
        update_epochs=2,
        eval_interval_updates=1,
        eval_episodes=2,
        disable_wandb=True,
        render_after_training=False,
        start_mode="grid",
        bc_epochs=0,
    )

    result = train_ppo(config)

    assert Path(result["checkpoint_path"]).exists()
    assert Path(result["history_csv"]).exists()
    assert Path(result["summary_path"]).exists()
    assert result["best_eval"]["mean_length"] > 0.0
