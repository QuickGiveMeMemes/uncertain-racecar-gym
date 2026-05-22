from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from uncertain_racecar_gym.common import wrap_angle
from uncertain_racecar_gym.scenario import VehicleConfig
from uncertain_racecar_gym.track import TrackModel


@dataclass(slots=True)
class VehicleState:
    x: float
    y: float
    yaw: float
    progress: float
    lateral_error: float
    heading_error: float
    vx: float
    vy: float
    yaw_rate: float
    steer: float
    throttle: float
    brake: float
    wheel_rotation: float
    lap_count: int = 0
    step_count: int = 0


@dataclass(slots=True)
class DynamicPrediction:
    vx: float
    vy: float
    yaw_rate: float
    steer: float
    throttle: float
    brake: float
    wheel_rotation: float


class DynamicBicycleModel:
    def __init__(self, config: VehicleConfig):
        self.config = config

    def initial_state(self, track: TrackModel, progress: float = 0.0, lateral_error: float = 0.0, heading_error: float = 0.0, speed: float = 6.0) -> VehicleState:
        x, y, yaw = track.spawn_pose(progress, lateral_error=lateral_error, heading_error=heading_error)
        projection = track.project(x, y)
        return VehicleState(
            x=x,
            y=y,
            yaw=yaw,
            progress=projection.progress,
            lateral_error=projection.lateral_error,
            heading_error=wrap_angle(yaw - projection.heading),
            vx=speed,
            vy=0.0,
            yaw_rate=0.0,
            steer=0.0,
            throttle=0.0,
            brake=0.0,
            wheel_rotation=0.0,
        )

    def state_from_canonical_row(self, row) -> VehicleState:
        return VehicleState(
            x=float(row["x"]),
            y=float(row["y"]),
            yaw=float(row["yaw"]),
            progress=float(row["progress"]),
            lateral_error=float(row["lateral_error"]),
            heading_error=float(row["heading_error"]),
            vx=float(row["vx"]),
            vy=float(row["vy"]),
            yaw_rate=float(row["yaw_rate"]),
            steer=float(row["steer"]),
            throttle=float(row["throttle"]),
            brake=float(row["brake"]),
            wheel_rotation=float(row.get("wheel_rotation", 0.0)),
            lap_count=int(row.get("lap_count", 0)),
            step_count=int(row.get("frame_index", 0)),
        )

    def predict(self, state: VehicleState, action: np.ndarray, dt: float) -> DynamicPrediction:
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], 0.0, 1.0))
        brake_cmd = float(np.clip(action[2], 0.0, 1.0))

        steer = state.steer + (steer_cmd - state.steer) * min(1.0, dt * 8.0)
        throttle = throttle_cmd
        brake = brake_cmd

        vx_safe = max(abs(state.vx), 0.5)
        steer_angle = steer * self.config.max_steer_rad
        alpha_f = steer_angle - np.arctan2(state.vy + self.config.lf * state.yaw_rate, vx_safe)
        alpha_r = -np.arctan2(state.vy - self.config.lr * state.yaw_rate, vx_safe)

        fyf = self.config.cornering_stiffness_front * alpha_f
        fyr = self.config.cornering_stiffness_rear * alpha_r
        longitudinal_acc = (
            throttle * self.config.max_accel
            - brake * self.config.max_brake
            - self.config.drag_coefficient * state.vx * abs(state.vx) / max(self.config.mass, 1.0)
        )

        vx_dot = longitudinal_acc + state.vy * state.yaw_rate
        vy_dot = (fyf * np.cos(steer_angle) + fyr) / self.config.mass - state.vx * state.yaw_rate
        yaw_rate_dot = (
            self.config.lf * fyf * np.cos(steer_angle) - self.config.lr * fyr
        ) / self.config.inertia_z

        next_vx = state.vx + vx_dot * dt
        next_vy = state.vy + vy_dot * dt
        next_yaw_rate = state.yaw_rate + yaw_rate_dot * dt
        wheel_rotation = state.wheel_rotation + (next_vx * dt / max(self.config.wheel_radius, 1e-6))

        return DynamicPrediction(
            vx=float(next_vx),
            vy=float(next_vy),
            yaw_rate=float(next_yaw_rate),
            steer=steer,
            throttle=throttle,
            brake=brake,
            wheel_rotation=float(wheel_rotation),
        )

    def integrate(self, state: VehicleState, prediction: DynamicPrediction, track: TrackModel, dt: float) -> VehicleState:
        avg_vx = 0.5 * (state.vx + prediction.vx)
        avg_vy = 0.5 * (state.vy + prediction.vy)
        avg_yaw_rate = 0.5 * (state.yaw_rate + prediction.yaw_rate)

        x_dot = avg_vx * np.cos(state.yaw) - avg_vy * np.sin(state.yaw)
        y_dot = avg_vx * np.sin(state.yaw) + avg_vy * np.cos(state.yaw)
        next_x = state.x + x_dot * dt
        next_y = state.y + y_dot * dt
        next_yaw = wrap_angle(state.yaw + avg_yaw_rate * dt)

        projection = track.project(next_x, next_y)
        lap_count = state.lap_count + int(projection.progress + 0.5 < state.progress)
        return VehicleState(
            x=next_x,
            y=next_y,
            yaw=next_yaw,
            progress=projection.progress,
            lateral_error=projection.lateral_error,
            heading_error=wrap_angle(next_yaw - projection.heading),
            vx=prediction.vx,
            vy=prediction.vy,
            yaw_rate=prediction.yaw_rate,
            steer=prediction.steer,
            throttle=prediction.throttle,
            brake=prediction.brake,
            wheel_rotation=prediction.wheel_rotation,
            lap_count=lap_count,
            step_count=state.step_count + 1,
        )

    def step(self, state: VehicleState, action: np.ndarray, track: TrackModel, dt: float, residual: np.ndarray | None = None) -> VehicleState:
        prediction = self.predict(state, action, dt)
        if residual is not None:
            residual_values = np.asarray(residual, dtype=float).reshape(-1)
            prediction = replace(
                prediction,
                vx=max(0.0, prediction.vx + float(residual_values[0] if len(residual_values) > 0 else 0.0)),
                vy=prediction.vy + float(residual_values[1] if len(residual_values) > 1 else 0.0),
                yaw_rate=prediction.yaw_rate + float(residual_values[2] if len(residual_values) > 2 else 0.0),
                steer=float(np.clip(prediction.steer + float(residual_values[3] if len(residual_values) > 3 else 0.0), -1.0, 1.0)),
            )
        return self.integrate(state, prediction, track, dt)
