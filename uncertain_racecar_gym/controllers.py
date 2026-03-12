from __future__ import annotations

import numpy as np

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
