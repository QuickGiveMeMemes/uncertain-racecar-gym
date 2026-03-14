from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd

from uncertain_racecar_gym.common import ensure_dir
from uncertain_racecar_gym.deterministic import load_calibration_model
from uncertain_racecar_gym.dynamics import DynamicBicycleModel, VehicleState
from uncertain_racecar_gym.features import build_feature_vector_from_state
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
        uncertainty: str | None = None,
        uncertainty_artifact: str | Path | None = None,
        calibration_artifact: str | Path | None = None,
        apply_mean_correction: bool = False,
        gaussian_noise_mean: Sequence[float] | None = None,
        gaussian_noise_std: Sequence[float] | None = None,
        renderer: str | None = None,
        render_mode: str | None = None,
        output_dir: str | Path = "output",
    ) -> None:
        super().__init__()
        self.scenario: Scenario = load_scenario(scenario or DEFAULT_SCENARIO)
        self.track = TrackModel.from_config(self.scenario.track)
        self.dynamics = DynamicBicycleModel(self.scenario.vehicle)
        self.output_dir = ensure_dir(output_dir)
        self.default_uncertainty_mode = self._normalize_uncertainty_mode(uncertainty)
        self.renderer_kind = renderer
        self.render_mode = render_mode
        self.renderer = None
        self.reset_rows = None
        self.apply_mean_correction = bool(apply_mean_correction)
        self.gaussian_noise_mean = self._normalize_gaussian_vector(gaussian_noise_mean, default=np.zeros(4, dtype=float))
        self.gaussian_noise_std = self._normalize_gaussian_vector(
            gaussian_noise_std,
            default=np.array([0.12, 0.08, 0.05, 0.015], dtype=float),
            nonnegative=True,
        )

        self.uncertainty_model = None
        self.calibration_model = None
        if uncertainty_artifact:
            self.uncertainty_model = EmpiricalUncertaintyModel.load(uncertainty_artifact)
        if calibration_artifact:
            self.calibration_model = load_calibration_model(calibration_artifact)
        self._sampler_state = self.uncertainty_model.make_runtime_state() if self.uncertainty_model else None
        self.runtime_track_id, self.runtime_car_id = self._resolve_runtime_ids()

        self._history = deque(maxlen=self.scenario.uncertainty.history_length)
        self._episode_history: list[dict[str, Any]] = []
        self._state: VehicleState | None = None
        self._previous_feature_state: VehicleState | None = None
        self._uncertainty_mode = self.default_uncertainty_mode

        lookahead = self.scenario.simulation.lookahead_points
        obs_dim = 7 + lookahead + (3 * self.scenario.uncertainty.history_length)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, 0.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)

    @property
    def episode_history(self) -> list[dict[str, Any]]:
        return self._episode_history

    @staticmethod
    def _normalize_uncertainty_mode(mode: str | None) -> str | None:
        if mode is None:
            return None
        normalized = str(mode).strip().lower()
        if normalized in {"", "none", "nominal"}:
            return None
        if normalized in {"gaussian", "empirical"}:
            return normalized
        raise ValueError(f"Unsupported uncertainty mode: {mode}")

    @staticmethod
    def _normalize_gaussian_vector(
        values: Sequence[float] | None,
        default: np.ndarray,
        nonnegative: bool = False,
    ) -> np.ndarray:
        if values is None:
            vector = np.asarray(default, dtype=float).copy()
        else:
            vector = np.asarray(list(values), dtype=float).reshape(-1)
        if len(vector) != 4:
            raise ValueError(f"Gaussian uncertainty vectors must have length 4, got {len(vector)}")
        if nonnegative:
            vector = np.clip(vector, 0.0, np.inf)
        return vector.astype(float)

    @property
    def uncertainty_mode(self) -> str | None:
        return self._uncertainty_mode

    def set_uncertainty(
        self,
        mode: str | None,
        *,
        gaussian_noise_mean: Sequence[float] | None = None,
        gaussian_noise_std: Sequence[float] | None = None,
        apply_mean_correction: bool | None = None,
    ) -> None:
        self._uncertainty_mode = self._normalize_uncertainty_mode(mode)
        if self._uncertainty_mode == "empirical" and self.uncertainty_model is None:
            raise ValueError("Empirical uncertainty mode requires an uncertainty_artifact.")
        if gaussian_noise_mean is not None:
            self.gaussian_noise_mean = self._normalize_gaussian_vector(gaussian_noise_mean, default=self.gaussian_noise_mean)
        if gaussian_noise_std is not None:
            self.gaussian_noise_std = self._normalize_gaussian_vector(
                gaussian_noise_std,
                default=self.gaussian_noise_std,
                nonnegative=True,
            )
        if apply_mean_correction is not None:
            self.apply_mean_correction = bool(apply_mean_correction)

    def load_reset_dataset(self, canonical_path: str | Path) -> None:
        self.reset_rows = pd.read_parquet(canonical_path)

    def _resolve_runtime_ids(self) -> tuple[str, str]:
        if self.uncertainty_model is not None and len(self.uncertainty_model.gate_prefixes) == 1:
            return self.uncertainty_model.gate_prefixes[0]
        residual_model = getattr(self.calibration_model, "residual_model", None)
        if residual_model is not None and len(residual_model.gate_prefixes) == 1:
            return residual_model.gate_prefixes[0]
        return self.scenario.name, "demo_racecar"

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
        history = list(self._history)
        while len(history) < self.scenario.uncertainty.history_length:
            history.insert(0, np.zeros(3, dtype=float))
        return build_feature_vector_from_state(
            self._state,
            curvature=self.track.sample(self._state.progress).curvature,
            action_history=history,
            vehicle_config=self.scenario.vehicle,
            dt=float(self.scenario.simulation.dt),
            previous_state=self._previous_feature_state,
        )

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

    def _gaussian_uncertainty(self) -> tuple[np.ndarray, dict[str, Any]]:
        sample = self.np_random.normal(loc=self.gaussian_noise_mean, scale=self.gaussian_noise_std).astype(float)
        return sample, {
            "mode": "gaussian",
            "mean": self.gaussian_noise_mean.tolist(),
            "std": self.gaussian_noise_std.tolist(),
            "sample": sample.tolist(),
            "channels": ["delta_vx", "delta_vy", "delta_yaw_rate", "delta_steer"],
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        start_mode = options.get("start_mode", options.get("mode", "grid"))
        self.set_uncertainty(
            options.get("uncertainty_mode", self.default_uncertainty_mode),
            gaussian_noise_mean=options.get("gaussian_noise_mean"),
            gaussian_noise_std=options.get("gaussian_noise_std"),
            apply_mean_correction=options.get("apply_mean_correction"),
        )
        self._state = self._initial_state(start_mode)
        self._history.clear()
        self._episode_history = []
        self._previous_feature_state = None
        if self.uncertainty_model:
            self._sampler_state = self.uncertainty_model.make_runtime_state()

        obs = self._observation()
        info = {
            "state": self._render_state(),
            "render_state": self._render_state(),
            "uncertainty": {"mode": self._uncertainty_mode or "none"},
        }
        self._episode_history.append({**self._render_state(), "reward": 0.0, "uncertainty": info["uncertainty"]})
        return obs, info

    def step(self, action):
        assert self._state is not None, "Call reset() before step()."
        action = np.asarray(action, dtype=float)
        previous_progress = self._state.progress
        feature_vector = self._feature_vector()
        progress_bin = int(self._state.progress * self.track.progress_bins) % self.track.progress_bins
        previous_state_for_features = self._state
        if self.uncertainty_model is not None:
            gate_key = self.uncertainty_model.resolve_gate_key(
                progress_bin=progress_bin,
                track_id=self.runtime_track_id,
                car_id=self.runtime_car_id,
            )
        elif self.calibration_model is not None and hasattr(self.calibration_model, "resolve_gate_key"):
            gate_key = self.calibration_model.resolve_gate_key(
                progress_bin=progress_bin,
                track_id=self.runtime_track_id,
                car_id=self.runtime_car_id,
            )
        else:
            gate_key = (self.runtime_track_id, self.runtime_car_id, progress_bin)

        residual = np.zeros(4, dtype=float)
        uncertainty_info = {"mode": self._uncertainty_mode or "none"}
        calibration_info = None
        if self.apply_mean_correction and self.calibration_model is not None:
            mean_residual, calibration_info = self.calibration_model.predict_mean(
                feature_vector,
                gate_key,
                dt=float(self.scenario.simulation.dt),
            )
            residual[:3] = residual[:3] + mean_residual
        if self._uncertainty_mode == "empirical" and self.uncertainty_model is not None:
            sampled_residual, uncertainty_info = self.uncertainty_model.sample(feature_vector, gate_key, self.np_random, self._sampler_state)
            residual[:3] = residual[:3] + sampled_residual
        elif self._uncertainty_mode == "gaussian":
            sampled_residual, uncertainty_info = self._gaussian_uncertainty()
            residual = residual + sampled_residual
        elif self.apply_mean_correction and self.calibration_model is not None:
            uncertainty_info = {"mode": "calibrated_nominal"}

        self._state = self.dynamics.step(self._state, action, self.track, self.scenario.simulation.dt, residual=residual)
        self._previous_feature_state = previous_state_for_features
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
