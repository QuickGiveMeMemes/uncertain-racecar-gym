from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from uncertain_racecar_gym.dynamics import VehicleState
from uncertain_racecar_gym.track import TrackModel


class CenterlineDriver:
    def __init__(self, target_speed: float = 14.0, min_speed: float = 6.0):
        self.target_speed = target_speed
        self.min_speed = min_speed

    def act(self, state: VehicleState, track: TrackModel) -> np.ndarray:
        curvature = abs(track.sample(state.progress).curvature)
        speed_target = max(self.min_speed, self.target_speed / (1.0 + curvature * 50.0))
        speed_error = speed_target - state.vx

        steer = np.clip(-(0.18 * state.lateral_error + 0.85 * state.heading_error), -1.0, 1.0)
        throttle = float(np.clip(speed_error * 0.22, 0.0, 1.0))
        brake = float(np.clip(-speed_error * 0.15, 0.0, 1.0))
        return np.array([steer, throttle, brake], dtype=np.float32)


def build_speed_profile(
    canonical: pd.DataFrame,
    progress_bins: int = 160,
    speed_quantile: float = 0.65,
    speed_scale: float = 0.55,
    min_speed: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame = canonical[["progress", "vx"]].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame["progress_bin"] = np.floor(frame["progress"].to_numpy(dtype=float) * progress_bins).astype(int) % progress_bins
    grouped = frame.groupby("progress_bin", observed=False)["vx"].quantile(speed_quantile)
    profile = np.full(progress_bins, float(min_speed), dtype=float)
    for bin_index, value in grouped.items():
        profile[int(bin_index)] = max(float(min_speed), float(value) * speed_scale)

    valid = np.flatnonzero(np.isfinite(profile))
    if len(valid) and len(valid) < progress_bins:
        profile = np.interp(np.arange(progress_bins), valid, profile[valid], period=progress_bins)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel = kernel / kernel.sum()
    padded = np.pad(profile, (2, 2), mode="wrap")
    smoothed = np.convolve(padded, kernel, mode="valid")
    progress = (np.arange(progress_bins, dtype=float) + 0.5) / float(progress_bins)
    return progress, smoothed


class ProfiledCenterlineDriver:
    def __init__(self, progress_points: np.ndarray, target_speeds: np.ndarray, min_speed: float = 8.0):
        self.progress_points = np.asarray(progress_points, dtype=float)
        self.target_speeds = np.asarray(target_speeds, dtype=float)
        self.min_speed = float(min_speed)

    @classmethod
    def from_canonical_dataframe(
        cls,
        canonical: pd.DataFrame,
        progress_bins: int = 160,
        speed_quantile: float = 0.65,
        speed_scale: float = 0.55,
        min_speed: float = 8.0,
    ) -> "ProfiledCenterlineDriver":
        progress_points, target_speeds = build_speed_profile(
            canonical,
            progress_bins=progress_bins,
            speed_quantile=speed_quantile,
            speed_scale=speed_scale,
            min_speed=min_speed,
        )
        return cls(progress_points=progress_points, target_speeds=target_speeds, min_speed=min_speed)

    def target_speed(self, progress: float) -> float:
        clipped = float(progress % 1.0)
        return float(max(self.min_speed, np.interp(clipped, self.progress_points, self.target_speeds, period=1.0)))

    def act(self, state: VehicleState, track: TrackModel) -> np.ndarray:
        speed_target = self.target_speed(state.progress)
        speed_error = speed_target - state.vx
        steer = np.clip(-(0.18 * state.lateral_error + 0.85 * state.heading_error), -1.0, 1.0)
        throttle = float(np.clip(speed_error * 0.18, 0.0, 1.0))
        brake = float(np.clip(-speed_error * 0.18, 0.0, 1.0))
        return np.array([steer, throttle, brake], dtype=np.float32)


@dataclass(slots=True)
class ControllerStepContext:
    observation: np.ndarray
    step_index: int
    episode_index: int
    case_id: str
    mode: str | None
    env: Any | None = None
    info: Mapping[str, Any] | None = None


class BenchmarkController(Protocol):
    name: str

    def reset(self, *, seed: int | None = None, case_id: str | None = None) -> None: ...

    def act(self, observation: np.ndarray, *, context: ControllerStepContext) -> np.ndarray: ...


class DriverControllerAdapter:
    def __init__(self, driver: CenterlineDriver | ProfiledCenterlineDriver, name: str | None = None):
        self.driver = driver
        self.name = name or type(driver).__name__

    def reset(self, *, seed: int | None = None, case_id: str | None = None) -> None:
        return None

    def act(self, observation: np.ndarray, *, context: ControllerStepContext) -> np.ndarray:
        if context.env is None or getattr(context.env, "_state", None) is None:
            raise ValueError("DriverControllerAdapter requires a live env with an initialized `_state`.")
        action = self.driver.act(context.env._state, context.env.track)
        return np.asarray(action, dtype=np.float32)


class PPOCheckpointController:
    def __init__(self, checkpoint_path: str | Path, name: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.name = name or self.checkpoint_path.stem
        import jax
        import jax.numpy as jnp

        from uncertain_racecar_gym.ppo_train import load_checkpoint

        payload = load_checkpoint(self.checkpoint_path)
        self.params = payload["params"]
        self.obs_norm = payload["obs_norm"]
        from uncertain_racecar_gym.ppo_train import deterministic_policy_action

        self._policy_fn = jax.jit(
            lambda observation: deterministic_policy_action(self.params, self.obs_norm, observation)[0]
        )
        obs_dim = int(np.asarray(self.obs_norm.mean).shape[0])
        self._policy_fn(jnp.zeros((obs_dim,), dtype=jnp.float32))

    def reset(self, *, seed: int | None = None, case_id: str | None = None) -> None:
        return None

    def act(self, observation: np.ndarray, *, context: ControllerStepContext) -> np.ndarray:
        import jax.numpy as jnp
        env_action = self._policy_fn(jnp.asarray(observation, dtype=jnp.float32))
        return np.asarray(env_action, dtype=np.float32)


def load_python_controller(spec: str, init_kwargs: dict[str, Any] | None = None) -> BenchmarkController:
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError("Python controller spec must be '<module_or_path>:<ClassName>'.")
    if module_name.endswith(".py") or Path(module_name).exists():
        module_path = Path(module_name).resolve()
        dynamic_name = f"custom_controller_{module_path.stem}"
        module_spec = importlib.util.spec_from_file_location(dynamic_name, module_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Unable to load controller module from {module_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    controller_cls = getattr(module, class_name)
    controller = controller_cls(**(init_kwargs or {}))
    if not hasattr(controller, "act"):
        raise TypeError(f"Controller class {class_name} must define an act(...) method.")
    if not hasattr(controller, "reset"):
        raise TypeError(f"Controller class {class_name} must define a reset(...) method.")
    return controller
