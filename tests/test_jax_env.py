from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import uncertain_racecar_gym  # noqa: F401

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from uncertain_racecar_gym.jax_env import NominalJaxRacecarEnv
from uncertain_racecar_gym.scenario import load_scenario


def _state_vector_from_python_env(env) -> np.ndarray:
    state = env.unwrapped._state
    return np.asarray(
        [
            state.x,
            state.y,
            state.yaw,
            state.progress,
            state.lateral_error,
            state.heading_error,
            state.vx,
            state.vy,
            state.yaw_rate,
            state.steer,
            state.throttle,
            state.brake,
            state.wheel_rotation,
            float(state.lap_count),
            float(state.step_count),
        ],
        dtype=float,
    )


def _state_vector_from_jax(state) -> np.ndarray:
    return np.asarray(
        [
            state.x,
            state.y,
            state.yaw,
            state.progress,
            state.lateral_error,
            state.heading_error,
            state.vx,
            state.vy,
            state.yaw_rate,
            state.steer,
            state.throttle,
            state.brake,
            state.wheel_rotation,
            state.lap_count,
            state.step_count,
        ],
        dtype=float,
    )


def test_nominal_jax_env_reset_matches_python_grid_reset() -> None:
    scenario = load_scenario()
    python_env = gym.make("UncertainRacecar-v0", scenario=scenario.source_path, renderer=None, uncertainty=None)
    obs_python, _ = python_env.reset(seed=0, options={"start_mode": "grid", "uncertainty_mode": None})

    jax_env = NominalJaxRacecarEnv(scenario.source_path)
    reset_output = jax_env.reset(jax.random.PRNGKey(0), start_mode="grid")

    np.testing.assert_allclose(np.asarray(obs_python), np.asarray(reset_output.observation), atol=1e-6)
    np.testing.assert_allclose(_state_vector_from_python_env(python_env), _state_vector_from_jax(reset_output.state), atol=3e-6)
    python_env.close()


def test_nominal_jax_env_step_matches_python_nominal_step() -> None:
    scenario = load_scenario()
    action = np.asarray([0.12, 0.3, 0.0], dtype=np.float32)

    python_env = gym.make("UncertainRacecar-v0", scenario=scenario.source_path, renderer=None, uncertainty=None)
    python_env.reset(seed=0, options={"start_mode": "grid", "uncertainty_mode": None})
    obs_python, reward_python, terminated_python, truncated_python, _ = python_env.step(action)

    jax_env = NominalJaxRacecarEnv(scenario.source_path)
    state = jax_env.reset(jax.random.PRNGKey(0), start_mode="grid").state
    step_output = jax_env.step(state, jnp.asarray(action))

    np.testing.assert_allclose(np.asarray(obs_python), np.asarray(step_output.observation), atol=1e-6)
    np.testing.assert_allclose(_state_vector_from_python_env(python_env), _state_vector_from_jax(step_output.state), atol=1e-6)
    np.testing.assert_allclose(np.asarray(reward_python), np.asarray(step_output.reward), atol=1e-6)
    assert bool(terminated_python) == bool(np.asarray(step_output.terminated))
    assert bool(truncated_python) == bool(np.asarray(step_output.truncated))
    python_env.close()


def test_nominal_jax_env_step_is_jittable() -> None:
    env = NominalJaxRacecarEnv()
    reset_output = env.reset(jax.random.PRNGKey(0), start_mode="random")
    action = jnp.asarray([0.0, 0.25, 0.0], dtype=jnp.float32)
    jitted = env.step_jit
    output = jitted(reset_output.state, action)

    assert output.observation.shape[0] == env.observation_size
    assert np.isfinite(np.asarray(output.reward))


def test_nominal_jax_env_custom_reset_matches_python_initial_conditions() -> None:
    scenario = load_scenario()
    python_env = gym.make("UncertainRacecar-v0", scenario=scenario.source_path, renderer=None, uncertainty=None)
    obs_python, _ = python_env.reset(
        seed=0,
        options={
            "start_mode": "grid",
            "uncertainty_mode": None,
            "initial_progress": 0.33,
            "initial_lateral_error": 0.08,
            "initial_heading_error": -0.02,
            "initial_speed": 10.2,
        },
    )

    jax_env = NominalJaxRacecarEnv(scenario.source_path)
    reset_output = jax_env.reset_custom(progress=0.33, lateral_error=0.08, heading_error=-0.02, speed=10.2)

    np.testing.assert_allclose(np.asarray(obs_python), np.asarray(reset_output.observation), atol=1e-6)
    np.testing.assert_allclose(_state_vector_from_python_env(python_env), _state_vector_from_jax(reset_output.state), atol=3e-6)
    python_env.close()
