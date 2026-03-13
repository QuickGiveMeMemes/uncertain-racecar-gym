from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from uncertain_racecar_gym.common import ensure_dir
from uncertain_racecar_gym.deterministic import load_calibration_model
from uncertain_racecar_gym.dynamics import DynamicBicycleModel, VehicleState
from uncertain_racecar_gym.rendering import PyBulletMirrorRenderer
from uncertain_racecar_gym.scenario import DEFAULT_SCENARIO, Scenario, load_scenario
from uncertain_racecar_gym.track import TrackModel
from uncertain_racecar_gym.uncertainty import EmpiricalUncertaintyModel


class UncertainRacecarEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array_follow", "rgb_array_birds_eye", "rgb_array_cinematic"],
    }

    def __init__(
        self,
        scenario: str | Path | None = None,
        uncertainty: str = "nominal",
        uncertainty_artifact: str | Path | None = None,
        calibration_artifact: str | Path | None = None,
        renderer: str | None = None,
        render_mode: str | None = None,
        output_dir: str | Path = "output",
    ) -> None:
        super().__init__()
        self.scenario: Scenario = load_scenario(scenario or DEFAULT_SCENARIO)
        self.track = TrackModel.from_config(self.scenario.track)
        self.dynamics = DynamicBicycleModel(self.scenario.vehicle)
        self.output_dir = ensure_dir(output_dir)
        self.default_uncertainty_mode = uncertainty
        self.renderer_kind = renderer
        self.render_mode = render_mode
        self.renderer = None
        self.reset_rows = None

        self.uncertainty_model = None
        self.calibration_model = None
        if uncertainty_artifact:
            self.uncertainty_model = EmpiricalUncertaintyModel.load(uncertainty_artifact)
        if calibration_artifact:
            self.calibration_model = load_calibration_model(calibration_artifact)
        self._sampler_state = self.uncertainty_model.make_runtime_state() if self.uncertainty_model else None

        self._history = deque(maxlen=self.scenario.uncertainty.history_length)
        self._episode_history: list[dict[str, Any]] = []
        self._state: VehicleState | None = None
        self._uncertainty_mode = self.default_uncertainty_mode

        lookahead = self.scenario.simulation.lookahead_points
        obs_dim = 7 + lookahead + (3 * self.scenario.uncertainty.history_length)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, 0.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)

    @property
    def episode_history(self) -> list[dict[str, Any]]:
        return self._episode_history

    def load_reset_dataset(self, canonical_path: str | Path) -> None:
        self.reset_rows = pd.read_parquet(canonical_path)

    def _initial_state(self, start_mode: str) -> VehicleState:
        if start_mode == "dataset_match" and self.reset_rows is not None and len(self.reset_rows):
            row = self.reset_rows.iloc[int(self.np_random.integers(0, len(self.reset_rows)))]
            return self.dynamics.state_from_canonical_row(row)
        if start_mode == "random":
            progress = float(self.np_random.uniform(0.0, 1.0))
            lateral_error = float(self.np_random.uniform(-0.2, 0.2))
            heading_error = float(self.np_random.uniform(-0.08, 0.08))
            speed = float(self.np_random.uniform(7.0, 12.0))
            return self.dynamics.initial_state(self.track, progress=progress, lateral_error=lateral_error, heading_error=heading_error, speed=speed)
        return self.dynamics.initial_state(self.track, progress=0.0, speed=8.0)

    def _feature_vector(self) -> np.ndarray:
        assert self._state is not None
        feature = [
            self.track.sample(self._state.progress).curvature,
            self._state.progress,
            self._state.vx,
            self._state.vy,
            self._state.yaw_rate,
            self._state.steer,
            self._state.throttle,
            self._state.brake,
        ]
        history = list(self._history)
        while len(history) < self.scenario.uncertainty.history_length:
            history.insert(0, np.zeros(3, dtype=float))
        return np.concatenate([np.asarray(feature, dtype=float), np.asarray(history, dtype=float).reshape(-1)])

    def _observation(self) -> np.ndarray:
        assert self._state is not None
        lookahead_curvature = self.track.lookahead_curvatures(
            self._state.progress,
            count=self.scenario.simulation.lookahead_points,
            spacing_m=self.scenario.simulation.lookahead_spacing_m,
        )
        history = list(self._history)
        while len(history) < self.scenario.uncertainty.history_length:
            history.insert(0, np.zeros(3, dtype=float))
        obs = np.concatenate(
            [
                np.array(
                    [
                        self._state.progress,
                        self._state.lateral_error,
                        self._state.heading_error,
                        self._state.vx,
                        self._state.vy,
                        self._state.yaw_rate,
                        self.track.sample(self._state.progress).curvature,
                    ],
                    dtype=np.float32,
                ),
                lookahead_curvature.astype(np.float32),
                np.asarray(history, dtype=np.float32).reshape(-1),
            ]
        )
        return obs

    def _render_state(self) -> dict[str, Any]:
        assert self._state is not None
        return {
            "x": self._state.x,
            "y": self._state.y,
            "yaw": self._state.yaw,
            "steering_angle": self._state.steer * self.scenario.vehicle.max_steer_rad,
            "wheel_rotation": self._state.wheel_rotation,
            "progress": self._state.progress,
            "frame_index": self._state.step_count,
            "speed": self._state.vx,
        }

    def _reward(self, previous_progress: float, state: VehicleState) -> float:
        delta = state.progress - previous_progress
        if delta < -0.5:
            delta += 1.0
        penalty = 0.03 * abs(state.lateral_error) + 0.01 * abs(state.heading_error)
        return float(delta * 100.0 + 0.05 * state.vx - penalty)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        start_mode = options.get("start_mode", options.get("mode", "grid"))
        self._uncertainty_mode = options.get("uncertainty_mode", self.default_uncertainty_mode)
        self._state = self._initial_state(start_mode)
        self._history.clear()
        self._episode_history = []
        if self.uncertainty_model:
            self._sampler_state = self.uncertainty_model.make_runtime_state()

        obs = self._observation()
        info = {
            "state": self._render_state(),
            "render_state": self._render_state(),
            "uncertainty": {"mode": self._uncertainty_mode},
        }
        self._episode_history.append({**self._render_state(), "reward": 0.0, "uncertainty": info["uncertainty"]})
        return obs, info

    def step(self, action):
        assert self._state is not None, "Call reset() before step()."
        action = np.asarray(action, dtype=float)
        previous_progress = self._state.progress
        feature_vector = self._feature_vector()
        progress_bin = int(self._state.progress * self.track.progress_bins) % self.track.progress_bins
        if self.uncertainty_model is not None:
            gate_key = self.uncertainty_model.resolve_gate_key(
                progress_bin=progress_bin,
                track_id=self.scenario.name,
                car_id="demo_racecar",
            )
        elif self.calibration_model is not None and hasattr(self.calibration_model, "resolve_gate_key"):
            gate_key = self.calibration_model.resolve_gate_key(
                progress_bin=progress_bin,
                track_id=self.scenario.name,
                car_id="demo_racecar",
            )
        else:
            gate_key = (self.scenario.name, "demo_racecar", progress_bin)

        residual = np.zeros(3, dtype=float)
        uncertainty_info = {"mode": self._uncertainty_mode}
        calibration_info = None
        if self.calibration_model is not None:
            mean_residual, calibration_info = self.calibration_model.predict_mean(
                feature_vector,
                gate_key,
                dt=float(self.scenario.simulation.dt),
            )
            residual = residual + mean_residual
        if self._uncertainty_mode == "empirical" and self.uncertainty_model is not None:
            sampled_residual, uncertainty_info = self.uncertainty_model.sample(feature_vector, gate_key, self.np_random, self._sampler_state)
            residual = residual + sampled_residual
        elif self.calibration_model is not None:
            uncertainty_info = {"mode": "calibrated_nominal"}

        self._state = self.dynamics.step(self._state, action, self.track, self.scenario.simulation.dt, residual=residual)
        self._history.append(np.array(action, dtype=float))

        reward = self._reward(previous_progress, self._state)
        terminated = self.track.out_of_bounds(self._state.lateral_error)
        truncated = self._state.step_count >= self.scenario.simulation.max_steps

        render_state = self._render_state()
        info = {
            "state": render_state,
            "render_state": render_state,
            "uncertainty": uncertainty_info,
            "calibration": calibration_info,
            "lap_count": self._state.lap_count,
        }
        self._episode_history.append({**render_state, "reward": reward, "uncertainty": uncertainty_info})
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None or self.renderer_kind != "pybullet":
            return None
        if self.renderer is None:
            self.renderer = PyBulletMirrorRenderer(self.scenario, self.track, self.render_mode)
        return self.renderer.render(self._render_state())

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
