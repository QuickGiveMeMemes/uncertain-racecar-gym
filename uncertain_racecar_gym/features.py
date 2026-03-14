from __future__ import annotations

from typing import Any

import numpy as np


CORE_FEATURE_NAMES = [
    "curvature",
    "progress",
    "vx",
    "vy",
    "yaw_rate",
    "steer",
    "throttle",
    "brake",
]

TELEMETRY_CANONICAL_COLUMNS = [
    "accel_x",
    "accel_y",
    "drive_train_speed",
    "rpm",
    "gear",
    "rear_slip_ratio_mean",
    "rear_slip_angle_mean",
    "tc_active",
    "abs_active",
]

EXTRA_FEATURE_NAMES = [
    "accel_x",
    "accel_y",
    "drive_train_speed",
    "speed_gap",
    "rear_slip_ratio_mean",
    "rear_slip_angle_mean",
    "gear_norm",
    "rpm_norm",
    "tc_active",
    "abs_active",
]

FEATURE_NAMES = CORE_FEATURE_NAMES + EXTRA_FEATURE_NAMES
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
ACTION_HISTORY_WIDTH = 3


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return numeric


def _series_get(row: Any, name: str, default: float = 0.0) -> float:
    if hasattr(row, "get"):
        return _safe_float(row.get(name, default), default=default)
    return _safe_float(getattr(row, name, default), default=default)


def estimate_gear_from_speed(speed_mps: float) -> float:
    thresholds = [6.0, 13.0, 21.0, 30.0, 40.0, 51.0]
    for index, threshold in enumerate(thresholds, start=1):
        if speed_mps < threshold:
            return float(index)
    return 7.0


def normalize_gear(gear: float) -> float:
    return float(np.clip(_safe_float(gear, 0.0) / 7.0, 0.0, 1.0))


def estimate_rpm_from_drive_speed(drive_speed_mps: float, gear: float, wheel_radius: float) -> float:
    drive_speed = max(abs(_safe_float(drive_speed_mps, 0.0)), 0.0)
    gear_value = max(1.0, _safe_float(gear, 1.0))
    gear_ratio = {
        1.0: 13.5,
        2.0: 9.6,
        3.0: 7.2,
        4.0: 5.8,
        5.0: 4.8,
        6.0: 4.0,
        7.0: 3.5,
    }.get(float(round(gear_value)), 4.2)
    wheel_rad_s = drive_speed / max(_safe_float(wheel_radius, 0.33), 1e-6)
    rpm = wheel_rad_s * gear_ratio * 60.0 / (2.0 * np.pi) + 1000.0
    return float(np.clip(rpm, 900.0, 9000.0))


def normalize_rpm(rpm: float) -> float:
    return float(np.clip((_safe_float(rpm, 1000.0) - 1000.0) / 8000.0, 0.0, 1.0))


def rear_slip_angle_proxy(vx: float, vy: float, yaw_rate: float, lr: float) -> float:
    vx_safe = max(abs(_safe_float(vx, 0.0)), 1.0)
    return float(np.arctan2(_safe_float(vy, 0.0) - _safe_float(lr, 0.0) * _safe_float(yaw_rate, 0.0), vx_safe))


def runtime_telemetry_proxy(
    state: Any,
    previous_state: Any | None,
    dt: float,
    vehicle_config: Any,
) -> dict[str, float]:
    vx = _safe_float(getattr(state, "vx", 0.0))
    vy = _safe_float(getattr(state, "vy", 0.0))
    yaw_rate = _safe_float(getattr(state, "yaw_rate", 0.0))
    throttle = _safe_float(getattr(state, "throttle", 0.0))
    brake = _safe_float(getattr(state, "brake", 0.0))
    wheel_rotation = _safe_float(getattr(state, "wheel_rotation", 0.0))

    if previous_state is not None:
        previous_vx = _safe_float(getattr(previous_state, "vx", vx))
        previous_wheel_rotation = _safe_float(getattr(previous_state, "wheel_rotation", wheel_rotation))
        accel_x = (vx - previous_vx) / max(_safe_float(dt, 0.05), 1e-6)
        drive_train_speed = (
            (wheel_rotation - previous_wheel_rotation)
            * _safe_float(getattr(vehicle_config, "wheel_radius", 0.33), 0.33)
            / max(_safe_float(dt, 0.05), 1e-6)
        )
    else:
        accel_x = throttle * _safe_float(getattr(vehicle_config, "max_accel", 0.0)) - brake * _safe_float(getattr(vehicle_config, "max_brake", 0.0))
        drive_train_speed = vx

    accel_y = vx * yaw_rate
    speed_gap = drive_train_speed - vx
    rear_slip_ratio = speed_gap / max(abs(vx), 1.0)
    rear_slip_angle = rear_slip_angle_proxy(vx, vy, yaw_rate, _safe_float(getattr(vehicle_config, "lr", 0.0), 0.0))
    gear = estimate_gear_from_speed(vx)
    rpm = estimate_rpm_from_drive_speed(drive_train_speed, gear, _safe_float(getattr(vehicle_config, "wheel_radius", 0.33), 0.33))
    tc_active = float(throttle > 0.35 and rear_slip_ratio > 0.08)
    abs_active = float(brake > 0.25 and rear_slip_ratio < -0.08)

    return {
        "accel_x": float(accel_x),
        "accel_y": float(accel_y),
        "drive_train_speed": float(drive_train_speed),
        "speed_gap": float(speed_gap),
        "rear_slip_ratio_mean": float(rear_slip_ratio),
        "rear_slip_angle_mean": float(rear_slip_angle),
        "gear_norm": normalize_gear(gear),
        "rpm_norm": normalize_rpm(rpm),
        "tc_active": float(tc_active),
        "abs_active": float(abs_active),
    }


def telemetry_from_canonical_row(
    row: Any,
    previous_row: Any | None,
    dt: float,
    vehicle_config: Any,
) -> dict[str, float]:
    vx = _series_get(row, "vx")
    vy = _series_get(row, "vy")
    yaw_rate = _series_get(row, "yaw_rate")

    drive_train_speed = _series_get(row, "drive_train_speed", np.nan)
    if not np.isfinite(drive_train_speed):
        wheel_rotation = _series_get(row, "wheel_rotation")
        previous_wheel_rotation = _series_get(previous_row, "wheel_rotation", wheel_rotation) if previous_row is not None else wheel_rotation
        drive_train_speed = (
            (wheel_rotation - previous_wheel_rotation)
            * _safe_float(getattr(vehicle_config, "wheel_radius", 0.33), 0.33)
            / max(_safe_float(dt, 0.05), 1e-6)
        )
        if not np.isfinite(drive_train_speed) or abs(drive_train_speed) < 1e-6:
            drive_train_speed = vx

    accel_x = _series_get(row, "accel_x", np.nan)
    if not np.isfinite(accel_x):
        previous_vx = _series_get(previous_row, "vx", vx) if previous_row is not None else vx
        accel_x = (vx - previous_vx) / max(_safe_float(dt, 0.05), 1e-6)

    accel_y = _series_get(row, "accel_y", np.nan)
    if not np.isfinite(accel_y):
        accel_y = vx * yaw_rate

    speed_gap = drive_train_speed - vx

    rear_slip_ratio = _series_get(row, "rear_slip_ratio_mean", np.nan)
    if not np.isfinite(rear_slip_ratio):
        rear_slip_ratio = speed_gap / max(abs(vx), 1.0)

    rear_slip_angle = _series_get(row, "rear_slip_angle_mean", np.nan)
    if not np.isfinite(rear_slip_angle):
        rear_slip_angle = rear_slip_angle_proxy(vx, vy, yaw_rate, _safe_float(getattr(vehicle_config, "lr", 0.0), 0.0))

    gear = _series_get(row, "gear", np.nan)
    if not np.isfinite(gear) or gear <= 0.0:
        gear = estimate_gear_from_speed(vx)

    rpm = _series_get(row, "rpm", np.nan)
    if not np.isfinite(rpm):
        rpm = estimate_rpm_from_drive_speed(drive_train_speed, gear, _safe_float(getattr(vehicle_config, "wheel_radius", 0.33), 0.33))

    tc_active = _series_get(row, "tc_active", np.nan)
    if not np.isfinite(tc_active):
        tc_active = float(_series_get(row, "throttle") > 0.35 and rear_slip_ratio > 0.08)

    abs_active = _series_get(row, "abs_active", np.nan)
    if not np.isfinite(abs_active):
        abs_active = float(_series_get(row, "brake") > 0.25 and rear_slip_ratio < -0.08)

    return {
        "accel_x": float(accel_x),
        "accel_y": float(accel_y),
        "drive_train_speed": float(drive_train_speed),
        "speed_gap": float(speed_gap),
        "rear_slip_ratio_mean": float(rear_slip_ratio),
        "rear_slip_angle_mean": float(rear_slip_angle),
        "gear_norm": normalize_gear(gear),
        "rpm_norm": normalize_rpm(rpm),
        "tc_active": float(tc_active),
        "abs_active": float(abs_active),
    }


def build_feature_vector_from_row(
    row: Any,
    action_history: list[np.ndarray] | np.ndarray,
    vehicle_config: Any,
    previous_row: Any | None = None,
) -> np.ndarray:
    dt = _series_get(row, "dt", 0.05)
    telemetry = telemetry_from_canonical_row(row, previous_row=previous_row, dt=dt, vehicle_config=vehicle_config)
    feature = [
        _series_get(row, "curvature"),
        _series_get(row, "progress"),
        _series_get(row, "vx"),
        _series_get(row, "vy"),
        _series_get(row, "yaw_rate"),
        _series_get(row, "steer"),
        _series_get(row, "throttle"),
        _series_get(row, "brake"),
        telemetry["accel_x"],
        telemetry["accel_y"],
        telemetry["drive_train_speed"],
        telemetry["speed_gap"],
        telemetry["rear_slip_ratio_mean"],
        telemetry["rear_slip_angle_mean"],
        telemetry["gear_norm"],
        telemetry["rpm_norm"],
        telemetry["tc_active"],
        telemetry["abs_active"],
    ]
    history = np.asarray(action_history, dtype=float).reshape(-1)
    return np.concatenate([np.asarray(feature, dtype=float), history])


def build_feature_vector_from_state(
    state: Any,
    curvature: float,
    action_history: list[np.ndarray] | np.ndarray,
    vehicle_config: Any,
    dt: float,
    previous_state: Any | None = None,
) -> np.ndarray:
    telemetry = runtime_telemetry_proxy(state, previous_state=previous_state, dt=dt, vehicle_config=vehicle_config)
    feature = [
        _safe_float(curvature),
        _safe_float(getattr(state, "progress", 0.0)),
        _safe_float(getattr(state, "vx", 0.0)),
        _safe_float(getattr(state, "vy", 0.0)),
        _safe_float(getattr(state, "yaw_rate", 0.0)),
        _safe_float(getattr(state, "steer", 0.0)),
        _safe_float(getattr(state, "throttle", 0.0)),
        _safe_float(getattr(state, "brake", 0.0)),
        telemetry["accel_x"],
        telemetry["accel_y"],
        telemetry["drive_train_speed"],
        telemetry["speed_gap"],
        telemetry["rear_slip_ratio_mean"],
        telemetry["rear_slip_angle_mean"],
        telemetry["gear_norm"],
        telemetry["rpm_norm"],
        telemetry["tc_active"],
        telemetry["abs_active"],
    ]
    history = np.asarray(action_history, dtype=float).reshape(-1)
    return np.concatenate([np.asarray(feature, dtype=float), history])


def history_means_from_feature_vector(feature_vector: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(feature_vector, dtype=float)
    if len(values) <= len(FEATURE_NAMES):
        return 0.0, 0.0, 0.0
    history = values[len(FEATURE_NAMES) :].reshape(-1, ACTION_HISTORY_WIDTH)
    return float(history[:, 0].mean()), float(history[:, 1].mean()), float(history[:, 2].mean())


def feature_value(feature_vector: np.ndarray, name: str, default: float = 0.0) -> float:
    index = FEATURE_INDEX.get(name)
    if index is None or len(feature_vector) <= index:
        return float(default)
    return _safe_float(np.asarray(feature_vector, dtype=float)[index], default=default)
