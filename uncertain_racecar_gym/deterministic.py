from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uncertain_racecar_gym.scenario import Scenario
from uncertain_racecar_gym.uncertainty import EmpiricalUncertaintyModel, RESIDUAL_NAMES


LONGITUDINAL_FEATURE_NAMES = [
    "bias",
    "vx",
    "vx_abs_vx",
    "throttle",
    "brake",
    "abs_steer",
    "steer_sq",
    "curvature",
    "abs_curvature",
    "abs_vy",
    "abs_yaw_rate",
    "throttle_vx",
    "brake_vx",
    "steer_vx",
    "throttle_abs_curvature",
    "brake_abs_curvature",
    "sin_progress_1",
    "cos_progress_1",
    "sin_progress_2",
    "cos_progress_2",
    "history_steer_mean",
    "history_throttle_mean",
    "history_brake_mean",
]


def _history_means(feature_vector: np.ndarray) -> tuple[float, float, float]:
    if len(feature_vector) <= 8:
        return 0.0, 0.0, 0.0
    history = np.asarray(feature_vector[8:], dtype=float).reshape(-1, 3)
    return float(history[:, 0].mean()), float(history[:, 1].mean()), float(history[:, 2].mean())


def build_longitudinal_design_vector(feature_vector: np.ndarray) -> np.ndarray:
    curvature, progress, vx, vy, yaw_rate, steer, throttle, brake = np.asarray(feature_vector[:8], dtype=float)
    history_steer_mean, history_throttle_mean, history_brake_mean = _history_means(feature_vector)
    return np.array(
        [
            1.0,
            vx,
            vx * abs(vx),
            throttle,
            brake,
            abs(steer),
            steer**2,
            curvature,
            abs(curvature),
            abs(vy),
            abs(yaw_rate),
            throttle * vx,
            brake * vx,
            abs(steer) * vx,
            throttle * abs(curvature),
            brake * abs(curvature),
            np.sin(2.0 * np.pi * progress),
            np.cos(2.0 * np.pi * progress),
            np.sin(4.0 * np.pi * progress),
            np.cos(4.0 * np.pi * progress),
            history_steer_mean,
            history_throttle_mean,
            history_brake_mean,
        ],
        dtype=float,
    )


def build_longitudinal_design_matrix(feature_vectors: list[np.ndarray] | pd.Series) -> np.ndarray:
    return np.vstack([build_longitudinal_design_vector(np.asarray(vector, dtype=float)) for vector in feature_vectors])


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


@dataclass(slots=True)
class LongitudinalCorrectionModel:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    weights: np.ndarray
    min_vx: float
    max_abs_vy: float
    max_abs_yaw_rate: float
    max_abs_steer: float
    max_abs_delta_vx: float
    default_dt: float
    active: bool = True

    def eligible(self, feature_vector: np.ndarray) -> bool:
        _, _, vx, vy, yaw_rate, steer, _, _ = np.asarray(feature_vector[:8], dtype=float)
        return (
            self.active
            and vx >= self.min_vx
            and abs(vy) <= self.max_abs_vy
            and abs(yaw_rate) <= self.max_abs_yaw_rate
            and abs(steer) <= self.max_abs_steer
        )

    def predict_delta_vx(self, feature_vector: np.ndarray, dt: float | None = None) -> tuple[float, dict[str, Any]]:
        if not self.eligible(feature_vector):
            return 0.0, {"mode": "inactive_or_out_of_regime", "applied": False}
        vector = build_longitudinal_design_vector(np.asarray(feature_vector, dtype=float))
        normalized = (vector - self.feature_mean) / self.feature_std
        correction_accel = float(normalized @ self.weights)
        delta_t = float(self.default_dt if dt is None else dt)
        correction = float(np.clip(correction_accel * delta_t, -self.max_abs_delta_vx, self.max_abs_delta_vx))
        return correction, {
            "mode": "parametric_longitudinal",
            "applied": True,
            "correction_accel": correction_accel,
            "delta_vx_correction": correction,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "weights": self.weights,
            "min_vx": self.min_vx,
            "max_abs_vy": self.max_abs_vy,
            "max_abs_yaw_rate": self.max_abs_yaw_rate,
            "max_abs_steer": self.max_abs_steer,
            "max_abs_delta_vx": self.max_abs_delta_vx,
            "default_dt": self.default_dt,
            "active": self.active,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LongitudinalCorrectionModel":
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_std=np.asarray(payload["feature_std"], dtype=float),
            weights=np.asarray(payload["weights"], dtype=float),
            min_vx=float(payload["min_vx"]),
            max_abs_vy=float(payload["max_abs_vy"]),
            max_abs_yaw_rate=float(payload["max_abs_yaw_rate"]),
            max_abs_steer=float(payload["max_abs_steer"]),
            max_abs_delta_vx=float(payload["max_abs_delta_vx"]),
            default_dt=float(payload["default_dt"]),
            active=bool(payload.get("active", True)),
        )


@dataclass(slots=True)
class HybridCalibrationModel:
    longitudinal_model: LongitudinalCorrectionModel | None = None
    residual_model: EmpiricalUncertaintyModel | None = None

    def resolve_gate_key(
        self,
        progress_bin: int,
        track_id: str | None = None,
        car_id: str | None = None,
    ) -> tuple[str, str, int]:
        if self.residual_model is not None:
            return self.residual_model.resolve_gate_key(progress_bin=progress_bin, track_id=track_id, car_id=car_id)
        return str(track_id or ""), str(car_id or ""), int(progress_bin)

    def predict_mean(
        self,
        feature_vector: np.ndarray,
        gate_key: tuple[str, str, int],
        dt: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        prediction = np.zeros(len(RESIDUAL_NAMES), dtype=float)
        info: dict[str, Any] = {}
        if self.longitudinal_model is not None:
            delta_vx, longitudinal_info = self.longitudinal_model.predict_delta_vx(feature_vector, dt=dt)
            prediction[0] = delta_vx
            info["longitudinal"] = longitudinal_info
        if self.residual_model is not None:
            residual_mean, residual_info = self.residual_model.predict_mean(feature_vector, gate_key)
            prediction = prediction + residual_mean
            info["residual"] = residual_info
        return prediction, info

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_type": "hybrid_calibration",
            "longitudinal_model": self.longitudinal_model.to_payload() if self.longitudinal_model is not None else None,
            "residual_model": self.residual_model.to_payload() if self.residual_model is not None else None,
        }

    def save(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        with path.open("wb") as handle:
            pickle.dump(self.to_payload(), handle)
        return path

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "HybridCalibrationModel":
        return cls(
            longitudinal_model=(
                LongitudinalCorrectionModel.from_payload(payload["longitudinal_model"])
                if payload.get("longitudinal_model") is not None
                else None
            ),
            residual_model=(
                EmpiricalUncertaintyModel.from_payload(payload["residual_model"])
                if payload.get("residual_model") is not None
                else None
            ),
        )


def load_calibration_model(input_path: str | Path) -> HybridCalibrationModel | EmpiricalUncertaintyModel:
    with Path(input_path).open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and payload.get("model_type") == "hybrid_calibration":
        return HybridCalibrationModel.from_payload(payload)
    if isinstance(payload, dict) and "buckets" in payload and "feature_mean" in payload:
        return EmpiricalUncertaintyModel.from_payload(payload)
    raise ValueError(f"Unsupported calibration artifact: {input_path}")


def longitudinal_training_mask(
    residual_table: pd.DataFrame,
    scenario: Scenario,
    min_vx: float,
    max_abs_delta_vx: float,
) -> pd.Series:
    return (
        (residual_table["vx"] >= min_vx)
        & (residual_table["dt"] > 0.02)
        & (residual_table["dt"] < 0.08)
        & (residual_table["heading_error"].abs() < 1.2)
        & (residual_table["lateral_error"].abs() < scenario.track.width * 0.45)
        & (residual_table["delta_vx"].abs() <= max_abs_delta_vx)
    )


def fit_longitudinal_correction(
    train_residuals: pd.DataFrame,
    test_residuals: pd.DataFrame,
    scenario: Scenario,
    ridge_lambda: float = 5.0,
    robust_iterations: int = 6,
) -> tuple[LongitudinalCorrectionModel, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    min_vx = 2.0
    max_abs_delta_vx = float(min(12.0, max(1.0, train_residuals["delta_vx"].abs().quantile(0.995))))
    train_mask = longitudinal_training_mask(train_residuals, scenario, min_vx=min_vx, max_abs_delta_vx=max_abs_delta_vx)
    selected = train_residuals.loc[train_mask].copy()
    if selected.empty:
        model = LongitudinalCorrectionModel(
            feature_mean=np.zeros(len(LONGITUDINAL_FEATURE_NAMES), dtype=float),
            feature_std=np.ones(len(LONGITUDINAL_FEATURE_NAMES), dtype=float),
            weights=np.zeros(len(LONGITUDINAL_FEATURE_NAMES), dtype=float),
            min_vx=min_vx,
            max_abs_vy=5.0,
            max_abs_yaw_rate=2.0,
            max_abs_steer=1.0,
            max_abs_delta_vx=0.0,
            default_dt=float(scenario.simulation.dt),
            active=False,
        )
        summary = {
            "active": False,
            "train_rows": int(len(train_residuals)),
            "selected_rows": 0,
            "stable_test_rows": 0,
            "reason": "no_rows_selected",
        }
        return model, summary, train_residuals.copy(), test_residuals.copy()

    design = build_longitudinal_design_matrix(selected["feature_vector"].tolist())
    targets = (selected["delta_vx"] / selected["dt"]).to_numpy(dtype=float)
    feature_mean = design.mean(axis=0)
    feature_std = design.std(axis=0)
    feature_mean[0] = 0.0
    feature_std[feature_std < 1e-6] = 1.0
    feature_std[0] = 1.0
    normalized = (design - feature_mean) / feature_std

    weights = np.ones(len(targets), dtype=float)
    coefficients = np.zeros(normalized.shape[1], dtype=float)
    eye = np.eye(normalized.shape[1], dtype=float)
    for _ in range(robust_iterations):
        sqrt_weights = np.sqrt(weights)
        weighted_x = normalized * sqrt_weights[:, None]
        weighted_y = targets * sqrt_weights
        coefficients = np.linalg.solve(weighted_x.T @ weighted_x + ridge_lambda * eye, weighted_x.T @ weighted_y)
        residual = targets - normalized @ coefficients
        scale = float(np.median(np.abs(residual)) / 0.6745 + 1e-6)
        cutoff = 1.5 * scale
        weights = np.where(np.abs(residual) <= cutoff, 1.0, cutoff / np.maximum(np.abs(residual), 1e-9))

    max_abs_vy = float(max(1.0, selected["vy"].abs().quantile(0.995)))
    max_abs_yaw_rate = float(max(0.2, selected["yaw_rate"].abs().quantile(0.995)))
    max_abs_steer = float(max(0.2, selected["steer"].abs().quantile(0.995)))
    predicted_train = ((normalized @ coefficients) * selected["dt"].to_numpy(dtype=float)).astype(float)
    max_abs_prediction = float(
        max(
            0.25,
            min(
                max_abs_delta_vx,
                np.quantile(np.abs(predicted_train), 0.98),
            ),
        )
    )

    model = LongitudinalCorrectionModel(
        feature_mean=feature_mean,
        feature_std=feature_std,
        weights=coefficients,
        min_vx=min_vx,
        max_abs_vy=max_abs_vy,
        max_abs_yaw_rate=max_abs_yaw_rate,
        max_abs_steer=max_abs_steer,
        max_abs_delta_vx=max_abs_prediction,
        default_dt=float(scenario.simulation.dt),
        active=True,
    )

    centered_train = apply_longitudinal_correction(train_residuals, model)
    centered_test = apply_longitudinal_correction(test_residuals, model)
    stable_test_mask = longitudinal_training_mask(test_residuals, scenario, min_vx=min_vx, max_abs_delta_vx=max_abs_delta_vx).to_numpy(dtype=bool)

    raw_test = test_residuals["delta_vx"].to_numpy(dtype=float)
    centered_test_values = centered_test["delta_vx"].to_numpy(dtype=float)
    raw_test_stable = test_residuals.reset_index(drop=True).loc[stable_test_mask, "delta_vx"].to_numpy(dtype=float)
    centered_test_stable = centered_test.reset_index(drop=True).loc[stable_test_mask, "delta_vx"].to_numpy(dtype=float)

    summary = {
        "active": True,
        "train_rows": int(len(train_residuals)),
        "selected_rows": int(len(selected)),
        "selected_fraction": float(len(selected) / max(len(train_residuals), 1)),
        "stable_test_rows": int(stable_test_mask.sum()),
        "runtime_regime": {
            "min_vx": model.min_vx,
            "max_abs_vy": model.max_abs_vy,
            "max_abs_yaw_rate": model.max_abs_yaw_rate,
            "max_abs_steer": model.max_abs_steer,
            "max_abs_delta_vx_correction": model.max_abs_delta_vx,
        },
        "all_test_metrics": {
            "rmse_raw": _rmse(raw_test),
            "rmse_centered": _rmse(centered_test_values),
            "mean_abs_raw": float(np.mean(np.abs(raw_test))),
            "mean_abs_centered": float(np.mean(np.abs(centered_test_values))),
        },
        "stable_test_metrics": {
            "rmse_raw": _rmse(raw_test_stable) if len(raw_test_stable) else float("nan"),
            "rmse_centered": _rmse(centered_test_stable) if len(centered_test_stable) else float("nan"),
            "mean_abs_raw": float(np.mean(np.abs(raw_test_stable))) if len(raw_test_stable) else float("nan"),
            "mean_abs_centered": float(np.mean(np.abs(centered_test_stable))) if len(centered_test_stable) else float("nan"),
        },
        "top_coefficients": [
            {"feature": name, "weight": float(value)}
            for name, value in sorted(
                zip(LONGITUDINAL_FEATURE_NAMES, coefficients / feature_std, strict=False),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:8]
        ],
    }
    return model, summary, centered_train, centered_test


def apply_longitudinal_correction(
    residual_table: pd.DataFrame,
    model: LongitudinalCorrectionModel | None,
) -> pd.DataFrame:
    if model is None:
        return residual_table.reset_index(drop=True).copy()
    centered = residual_table.reset_index(drop=True).copy()
    predicted = []
    for row in centered.itertuples(index=False):
        delta_vx_mean, info = model.predict_delta_vx(np.asarray(row.feature_vector, dtype=float), dt=float(row.dt))
        predicted.append(
            {
                "delta_vx_longitudinal_mean": float(delta_vx_mean),
                "longitudinal_applied": bool(info.get("applied", False)),
            }
        )
    predicted_frame = pd.DataFrame(predicted)
    centered["delta_vx_raw"] = centered["delta_vx"]
    centered["delta_vx"] = centered["delta_vx"].to_numpy(dtype=float) - predicted_frame["delta_vx_longitudinal_mean"].to_numpy(dtype=float)
    return pd.concat([centered.reset_index(drop=True), predicted_frame], axis=1)
