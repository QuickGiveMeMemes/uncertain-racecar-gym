from __future__ import annotations

import pickle
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from uncertain_racecar_gym.common import softmax_sample_weights
from uncertain_racecar_gym.dynamics import DynamicBicycleModel
from uncertain_racecar_gym.scenario import Scenario
from uncertain_racecar_gym.track import TrackModel


FEATURE_NAMES = [
    "curvature",
    "progress",
    "vx",
    "vy",
    "yaw_rate",
    "steer",
    "throttle",
    "brake",
]
RESIDUAL_NAMES = ["delta_vx", "delta_vy", "delta_yaw_rate"]


@dataclass(slots=True)
class SamplerRuntimeState:
    active_row_id: int | None = None
    remaining_block: int = 0
    active_mode_key: str | None = None


@dataclass(slots=True)
class Bucket:
    gate: tuple[str, str, int]
    features: np.ndarray
    residuals: np.ndarray
    row_ids: np.ndarray
    trajectory_ids: np.ndarray
    frame_indices: np.ndarray
    mode_keys: np.ndarray
    tree: cKDTree | None = None
    mode_indices: dict[str, np.ndarray] | None = None
    mode_trees: dict[str, cKDTree] | None = None


def _mode_key_from_trajectory_id(trajectory_id: str) -> str:
    text = str(trajectory_id)
    if "&_&" in text:
        parts = [part.strip() for part in text.split("&_&") if part.strip()]
        if len(parts) >= 3:
            return parts[2]
    tokens = [token for token in text.replace("/", "_").split("_") if token]
    return tokens[0] if tokens else text


class EmpiricalUncertaintyModel:
    @staticmethod
    def _build_from_feature_rows(
        feature_rows: list[np.ndarray],
        residual_rows: list[np.ndarray],
        gate_rows: list[tuple[str, str, int]],
        trajectory_ids: list[str],
        mode_keys: list[str],
        frame_indices: list[int],
        continuation_map: dict[int, int],
        history_length: int,
        neighbor_count: int,
        progress_bins: int,
        block_length: int,
    ) -> "EmpiricalUncertaintyModel":
        feature_matrix = np.asarray(feature_rows, dtype=float)
        feature_mean = feature_matrix.mean(axis=0)
        feature_std = feature_matrix.std(axis=0)
        feature_std[feature_std < 1e-6] = 1.0
        normalized = (feature_matrix - feature_mean) / feature_std

        buckets = {}
        row_ids_array = np.arange(len(feature_rows), dtype=int)
        residual_matrix = np.asarray(residual_rows, dtype=float)
        trajectory_array = np.asarray(trajectory_ids, dtype=object)
        mode_key_array = np.asarray(mode_keys, dtype=object)
        frame_array = np.asarray(frame_indices, dtype=int)
        for gate in sorted(set(gate_rows)):
            mask = np.array([candidate == gate for candidate in gate_rows], dtype=bool)
            buckets[gate] = Bucket(
                gate=gate,
                features=normalized[mask],
                residuals=residual_matrix[mask],
                row_ids=row_ids_array[mask],
                trajectory_ids=trajectory_array[mask],
                frame_indices=frame_array[mask],
                mode_keys=mode_key_array[mask],
            )

        return EmpiricalUncertaintyModel(
            history_length=history_length,
            neighbor_count=neighbor_count,
            progress_bins=progress_bins,
            block_length=block_length,
            feature_mean=feature_mean,
            feature_std=feature_std,
            buckets=buckets,
            continuation_map=continuation_map,
        )

    def __init__(
        self,
        history_length: int,
        neighbor_count: int,
        progress_bins: int,
        block_length: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        buckets: dict[tuple[str, str, int], Bucket],
        continuation_map: dict[int, int],
    ):
        self.history_length = history_length
        self.neighbor_count = neighbor_count
        self.progress_bins = progress_bins
        self.block_length = block_length
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.buckets = buckets
        self.continuation_map = continuation_map
        self._gate_prefixes = sorted({(gate[0], gate[1]) for gate in buckets})
        self._row_lookup = {}
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._row_lookup = {}
        for gate, bucket in self.buckets.items():
            bucket.tree = cKDTree(bucket.features) if len(bucket.features) else None
            bucket.mode_indices = {}
            bucket.mode_trees = {}
            for mode_key in sorted(set(bucket.mode_keys.tolist())):
                local_indices = np.flatnonzero(bucket.mode_keys == mode_key)
                bucket.mode_indices[str(mode_key)] = local_indices
                if len(local_indices):
                    bucket.mode_trees[str(mode_key)] = cKDTree(bucket.features[local_indices])
            for index, row_id in enumerate(bucket.row_ids):
                self._row_lookup[int(row_id)] = (gate, index)

    @classmethod
    def fit(
        cls,
        canonical: pd.DataFrame,
        scenario: Scenario,
        history_length: int | None = None,
        neighbor_count: int | None = None,
        block_length: int | None = None,
    ) -> "EmpiricalUncertaintyModel":
        history_length = history_length or scenario.uncertainty.history_length
        neighbor_count = neighbor_count or scenario.uncertainty.neighbor_count
        block_length = block_length or scenario.uncertainty.block_length

        model = DynamicBicycleModel(scenario.vehicle)
        track = TrackModel.from_config(scenario.track)
        canonical = canonical.sort_values(["trajectory_id", "frame_index"]).reset_index(drop=True)

        feature_rows = []
        residual_rows = []
        gate_rows = []
        row_ids = []
        trajectory_ids = []
        mode_keys = []
        frame_indices = []
        continuation_map = {}

        global_row_id = 0
        for _, group in canonical.groupby("trajectory_id", sort=False):
            group = group.reset_index(drop=True)
            action_history = [np.zeros(3, dtype=float) for _ in range(history_length)]
            previous_row_id = None
            for index in range(len(group) - 1):
                current = group.iloc[index]
                nxt = group.iloc[index + 1]
                action = np.array([current["steer"], current["throttle"], current["brake"]], dtype=float)
                state = model.state_from_canonical_row(current)
                prediction = model.predict(state, action, float(current["dt"]))
                feature_vector = np.concatenate(
                    [
                        np.array(
                            [
                                current["curvature"],
                                current["progress"],
                                current["vx"],
                                current["vy"],
                                current["yaw_rate"],
                                current["steer"],
                                current["throttle"],
                                current["brake"],
                            ],
                            dtype=float,
                        ),
                        np.array(action_history, dtype=float).reshape(-1),
                    ]
                )
                residual_vector = np.array(
                    [
                        float(nxt["vx"]) - prediction.vx,
                        float(nxt["vy"]) - prediction.vy,
                        float(nxt["yaw_rate"]) - prediction.yaw_rate,
                    ],
                    dtype=float,
                )
                gate_key = (
                    str(current["track_id"]),
                    str(current["car_id"]),
                    int(min(track.progress_bins - 1, max(0, np.floor(float(current["progress"]) * track.progress_bins)))),
                )

                feature_rows.append(feature_vector)
                residual_rows.append(residual_vector)
                gate_rows.append(gate_key)
                row_ids.append(global_row_id)
                trajectory_ids.append(str(current["trajectory_id"]))
                mode_keys.append(_mode_key_from_trajectory_id(str(current["trajectory_id"])))
                frame_indices.append(int(current["frame_index"]))

                if previous_row_id is not None:
                    continuation_map[previous_row_id] = global_row_id
                previous_row_id = global_row_id
                global_row_id += 1

                action_history.pop(0)
                action_history.append(action)

        return cls._build_from_feature_rows(
            feature_rows=feature_rows,
            residual_rows=residual_rows,
            gate_rows=gate_rows,
            trajectory_ids=trajectory_ids,
            mode_keys=mode_keys,
            frame_indices=frame_indices,
            continuation_map=continuation_map,
            history_length=history_length,
            neighbor_count=neighbor_count,
            progress_bins=track.progress_bins,
            block_length=block_length,
        )

    @classmethod
    def fit_from_residual_table(
        cls,
        residual_table: pd.DataFrame,
        scenario: Scenario,
        history_length: int | None = None,
        neighbor_count: int | None = None,
        block_length: int | None = None,
    ) -> "EmpiricalUncertaintyModel":
        history_length = history_length or scenario.uncertainty.history_length
        neighbor_count = neighbor_count or scenario.uncertainty.neighbor_count
        block_length = block_length or scenario.uncertainty.block_length
        track = TrackModel.from_config(scenario.track)

        ordered = residual_table.sort_values(["trajectory_id", "frame_index"]).reset_index(drop=True)
        feature_rows = [np.asarray(item, dtype=float) for item in ordered["feature_vector"].tolist()]
        residual_rows = [ordered.loc[index, RESIDUAL_NAMES].to_numpy(dtype=float) for index in range(len(ordered))]
        gate_rows = [
            (
                str(row.track_id),
                str(row.car_id),
                int(min(track.progress_bins - 1, max(0, int(row.progress_bin)))),
            )
            for row in ordered.itertuples(index=False)
        ]
        trajectory_ids = [str(value) for value in ordered["trajectory_id"].tolist()]
        mode_keys = [_mode_key_from_trajectory_id(value) for value in trajectory_ids]
        frame_indices = [int(value) for value in ordered["frame_index"].tolist()]

        continuation_map: dict[int, int] = {}
        previous_key: tuple[str, int] | None = None
        previous_row_id: int | None = None
        for row_id, row in enumerate(ordered.itertuples(index=False)):
            current_key = (str(row.trajectory_id), int(row.frame_index))
            if previous_key is not None and current_key[0] == previous_key[0] and current_key[1] == previous_key[1] + 1 and previous_row_id is not None:
                continuation_map[previous_row_id] = row_id
            previous_key = current_key
            previous_row_id = row_id

        return cls._build_from_feature_rows(
            feature_rows=feature_rows,
            residual_rows=residual_rows,
            gate_rows=gate_rows,
            trajectory_ids=trajectory_ids,
            mode_keys=mode_keys,
            frame_indices=frame_indices,
            continuation_map=continuation_map,
            history_length=history_length,
            neighbor_count=neighbor_count,
            progress_bins=track.progress_bins,
            block_length=block_length,
        )

    def make_runtime_state(self) -> SamplerRuntimeState:
        return SamplerRuntimeState()

    @property
    def gate_prefixes(self) -> list[tuple[str, str]]:
        return list(self._gate_prefixes)

    def resolve_gate_key(
        self,
        progress_bin: int,
        track_id: str | None = None,
        car_id: str | None = None,
    ) -> tuple[str, str, int]:
        progress_bin = int(progress_bin) % max(self.progress_bins, 1)
        requested = (str(track_id or ""), str(car_id or ""))
        if requested in self._gate_prefixes:
            return requested[0], requested[1], progress_bin
        if len(self._gate_prefixes) == 1:
            only_track_id, only_car_id = self._gate_prefixes[0]
            return only_track_id, only_car_id, progress_bin
        matching_track = [prefix for prefix in self._gate_prefixes if prefix[0] == requested[0]]
        if len(matching_track) == 1:
            return matching_track[0][0], matching_track[0][1], progress_bin
        matching_car = [prefix for prefix in self._gate_prefixes if prefix[1] == requested[1]]
        if len(matching_car) == 1:
            return matching_car[0][0], matching_car[0][1], progress_bin
        first_track_id, first_car_id = self._gate_prefixes[0]
        return first_track_id, first_car_id, progress_bin

    def _select_bucket(self, track_id: str, car_id: str, progress_bin: int) -> Bucket | None:
        exact = (track_id, car_id, progress_bin)
        if exact in self.buckets:
            return self.buckets[exact]
        for offset in range(1, self.progress_bins):
            for candidate in (
                (track_id, car_id, (progress_bin - offset) % self.progress_bins),
                (track_id, car_id, (progress_bin + offset) % self.progress_bins),
            ):
                if candidate in self.buckets:
                    return self.buckets[candidate]
        for gate, bucket in self.buckets.items():
            if gate[0] == track_id and gate[1] == car_id:
                return bucket
        return next(iter(self.buckets.values()), None)

    def sample(
        self,
        feature_vector: np.ndarray,
        gate_key: tuple[str, str, int],
        rng: np.random.Generator,
        runtime_state: SamplerRuntimeState,
    ) -> tuple[np.ndarray, dict]:
        if not self.buckets:
            return np.zeros(3, dtype=float), {"mode": "empty"}

        if runtime_state.remaining_block > 0 and runtime_state.active_row_id in self.continuation_map:
            next_row_id = self.continuation_map[runtime_state.active_row_id]
            if next_row_id in self._row_lookup:
                gate, local_index = self._row_lookup[next_row_id]
                bucket = self.buckets[gate]
                runtime_state.active_row_id = next_row_id
                runtime_state.remaining_block -= 1
                runtime_state.active_mode_key = str(bucket.mode_keys[local_index])
                return bucket.residuals[local_index], {
                    "mode": "block",
                    "row_id": int(next_row_id),
                    "gate": gate,
                    "mode_key": runtime_state.active_mode_key,
                }

        normalized = (feature_vector - self.feature_mean) / self.feature_std
        bucket = self._select_bucket(*gate_key)
        if bucket is None or bucket.tree is None:
            return np.zeros(3, dtype=float), {"mode": "fallback"}

        candidate_tree = bucket.tree
        candidate_indices = np.arange(len(bucket.features), dtype=int)
        if (
            runtime_state.active_mode_key is not None
            and bucket.mode_indices is not None
            and bucket.mode_trees is not None
            and runtime_state.active_mode_key in bucket.mode_indices
            and len(bucket.mode_indices[runtime_state.active_mode_key]) >= max(16, self.neighbor_count // 2)
        ):
            candidate_indices = bucket.mode_indices[runtime_state.active_mode_key]
            candidate_tree = bucket.mode_trees[runtime_state.active_mode_key]

        query_k = min(self.neighbor_count, len(candidate_indices))
        distances, local_indices = candidate_tree.query(normalized, k=query_k)
        distances = np.atleast_1d(distances).astype(float)
        local_indices = np.atleast_1d(local_indices).astype(int)
        indices = candidate_indices[local_indices]
        weights = softmax_sample_weights(distances)
        choice_index = int(rng.choice(indices, p=weights))
        row_id = int(bucket.row_ids[choice_index])
        runtime_state.active_row_id = row_id
        runtime_state.remaining_block = int(rng.integers(1, self.block_length + 1))
        runtime_state.active_mode_key = str(bucket.mode_keys[choice_index])
        return bucket.residuals[choice_index], {
            "mode": "knn",
            "row_id": row_id,
            "gate": bucket.gate,
            "mode_key": runtime_state.active_mode_key,
            "distance_mean": float(distances.mean()),
            "sample_weight": float(weights[np.where(indices == choice_index)[0][0]]),
        }

    def predict_mean(
        self,
        feature_vector: np.ndarray,
        gate_key: tuple[str, str, int],
        dt: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        if not self.buckets:
            return np.zeros(3, dtype=float), {"mode": "empty"}
        bucket = self._select_bucket(*gate_key)
        if bucket is None or bucket.tree is None:
            return np.zeros(3, dtype=float), {"mode": "fallback"}

        normalized = (feature_vector - self.feature_mean) / self.feature_std
        query_k = min(self.neighbor_count, len(bucket.features))
        distances, indices = bucket.tree.query(normalized, k=query_k)
        distances = np.atleast_1d(distances).astype(float)
        indices = np.atleast_1d(indices).astype(int)
        weights = softmax_sample_weights(distances)
        mean_residual = np.average(bucket.residuals[indices], axis=0, weights=weights)
        return mean_residual, {
            "mode": "knn_mean",
            "gate": bucket.gate,
            "distance_mean": float(distances.mean()),
            "neighbors": int(query_k),
        }

    def to_payload(self) -> dict:
        return {
            "history_length": self.history_length,
            "neighbor_count": self.neighbor_count,
            "progress_bins": self.progress_bins,
            "block_length": self.block_length,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "continuation_map": self.continuation_map,
            "buckets": {
                gate: {
                    "features": bucket.features,
                    "residuals": bucket.residuals,
                    "row_ids": bucket.row_ids,
                    "trajectory_ids": bucket.trajectory_ids,
                    "frame_indices": bucket.frame_indices,
                    "mode_keys": bucket.mode_keys,
                }
                for gate, bucket in self.buckets.items()
            },
        }
    
    def save(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        payload = self.to_payload()
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        return path

    def copy(self) -> "EmpiricalUncertaintyModel":
        return copy.deepcopy(self)

    def zero_residual_channels(self, channel_names: list[str]) -> "EmpiricalUncertaintyModel":
        channel_indices = [RESIDUAL_NAMES.index(name) for name in channel_names]
        for bucket in self.buckets.values():
            bucket.residuals[:, channel_indices] = 0.0
        return self

    @classmethod
    def from_payload(cls, payload: dict) -> "EmpiricalUncertaintyModel":
        buckets = {
            gate: Bucket(
                gate=gate,
                features=data["features"],
                residuals=data["residuals"],
                row_ids=data["row_ids"],
                trajectory_ids=data["trajectory_ids"],
                frame_indices=data["frame_indices"],
                mode_keys=data.get(
                    "mode_keys",
                    np.asarray([_mode_key_from_trajectory_id(value) for value in data["trajectory_ids"]], dtype=object),
                ),
            )
            for gate, data in payload["buckets"].items()
        }
        return cls(
            history_length=payload["history_length"],
            neighbor_count=payload["neighbor_count"],
            progress_bins=payload["progress_bins"],
            block_length=payload["block_length"],
            feature_mean=payload["feature_mean"],
            feature_std=payload["feature_std"],
            buckets=buckets,
            continuation_map=payload["continuation_map"],
        )

    @classmethod
    def load(cls, input_path: str | Path) -> "EmpiricalUncertaintyModel":
        with Path(input_path).open("rb") as handle:
            payload = pickle.load(handle)
        return cls.from_payload(payload)
