from __future__ import annotations

import gymnasium as gym
import numpy as np

import uncertain_racecar_gym  # noqa: F401
from uncertain_racecar_gym.controllers import ControllerStepContext
from uncertain_racecar_gym.mppi_jax import JaxMPPIConfig, JaxMPPIController
from uncertain_racecar_gym.smooth_mppi_jax import JaxSmoothMPPIConfig, JaxSmoothMPPIController


def _roll_controller(controller, *, steps: int = 6) -> None:
    env = gym.make("UncertainRacecar-v0", scenario="package://scenarios/sample_oval.yaml", renderer=None, uncertainty=None)
    observation, info = env.reset(seed=0, options={"uncertainty_mode": None, "start_mode": "grid"})
    controller.reset(seed=0, case_id="smoke")
    saw_debug = False
    for step_index in range(steps):
        action = controller.act(
            observation,
            context=ControllerStepContext(
                observation=np.asarray(observation, dtype=np.float32),
                step_index=step_index,
                episode_index=0,
                case_id="smoke",
                mode=None,
                env=env.unwrapped,
                info=info,
            ),
        )
        assert action.shape == (3,)
        debug = controller.get_render_debug()
        if debug is not None:
            saw_debug = True
            assert debug["candidate_xy"].ndim == 3
            assert debug["candidate_xy"].shape[2] == 2
            assert debug["candidate_xy"].shape[0] <= 100
            assert debug["final_xy"].ndim == 2
            assert debug["final_xy"].shape[1] == 2
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    assert saw_debug


def test_jax_mppi_controller_smoke() -> None:
    controller = JaxMPPIController(
        scenario="package://scenarios/sample_oval.yaml",
        config=JaxMPPIConfig(horizon=10, num_samples=48, optimization_steps=1, replan_interval=1, target_speed=10.0),
    )
    _roll_controller(controller)


def test_jax_smooth_mppi_controller_smoke() -> None:
    controller = JaxSmoothMPPIController(
        scenario="package://scenarios/sample_oval.yaml",
        config=JaxSmoothMPPIConfig(horizon=10, num_samples=48, optimization_steps=1, replan_interval=1, target_speed=10.0),
    )
    _roll_controller(controller)
