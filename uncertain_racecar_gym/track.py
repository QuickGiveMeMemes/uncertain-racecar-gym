from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from uncertain_racecar_gym.common import wrap_angle
from uncertain_racecar_gym.scenario import TrackConfig


@dataclass(slots=True)
class TrackProjection:
    progress: float
    arc_length: float
    x: float
    y: float
    heading: float
    lateral_error: float
    curvature: float


class TrackModel:
    def __init__(self, centerline: np.ndarray, width: float, closed: bool = True, progress_bins: int = 24):
        if centerline.shape[0] < 4:
            raise ValueError("Track centerline must contain at least four points.")
        if np.allclose(centerline[0], centerline[-1]):
            centerline = centerline[:-1]
        centerline = np.asarray(centerline, dtype=float)
        deduped = [centerline[0]]
        for point in centerline[1:]:
            if np.linalg.norm(point - deduped[-1]) > 1e-4:
                deduped.append(point)
        centerline = np.asarray(deduped, dtype=float)
        if centerline.shape[0] < 4:
            raise ValueError("Track centerline must contain at least four distinct points.")
        self.centerline = centerline
        self.width = float(width)
        self.closed = bool(closed)
        self.progress_bins = int(progress_bins)

        extended = np.vstack([self.centerline, self.centerline[0]])
        self._segments = np.diff(extended, axis=0)
        self._segment_lengths = np.linalg.norm(self._segments, axis=1)
        self._segment_length_sq = self._segment_lengths * self._segment_lengths
        self._segment_valid = self._segment_lengths > 1e-9
        self._segment_tangents = self._segments / np.maximum(self._segment_lengths[:, None], 1e-9)
        self._segment_normals = np.stack([-self._segment_tangents[:, 1], self._segment_tangents[:, 0]], axis=1)
        self._segment_headings = np.arctan2(self._segments[:, 1], self._segments[:, 0])
        self._cumulative = np.concatenate([[0.0], np.cumsum(self._segment_lengths)])
        self.length = float(self._cumulative[-1])

        self._headings = np.append(self._segment_headings, self._segment_headings[0])

        arc = self._cumulative[:-1]
        heading_unwrapped = np.unwrap(self._headings[:-1])
        curvature = np.gradient(heading_unwrapped, arc, edge_order=1)
        self._arc_samples = arc
        self._curvature_samples = curvature

    @classmethod
    def from_config(cls, config: TrackConfig) -> "TrackModel":
        frame = pd.read_csv(Path(config.csv))
        if {"x", "y"}.issubset(frame.columns):
            centerline = frame[["x", "y"]].to_numpy(dtype=float)
        else:
            centerline = frame.iloc[:, :2].to_numpy(dtype=float)
        return cls(centerline=centerline, width=config.width, closed=config.closed, progress_bins=config.progress_bins)

    def progress_to_arc(self, progress: float) -> float:
        return float(progress % 1.0) * self.length

    def arc_to_progress(self, arc_length: float) -> float:
        return float((arc_length % self.length) / self.length)

    def sample(self, progress: float) -> TrackProjection:
        arc = self.progress_to_arc(progress)
        index = int(np.searchsorted(self._cumulative, arc, side="right") - 1) % len(self._segments)
        segment_start = self.centerline[index]
        segment = self._segments[index]
        segment_length = self._segment_lengths[index]
        local_t = 0.0 if segment_length < 1e-9 else (arc - self._cumulative[index]) / segment_length
        point = segment_start + segment * np.clip(local_t, 0.0, 1.0)
        heading = float(np.arctan2(segment[1], segment[0]))
        tangent = segment / max(segment_length, 1e-9)
        normal = np.array([-tangent[1], tangent[0]])
        curvature = float(np.interp(arc, self._arc_samples, self._curvature_samples, period=self.length))
        return TrackProjection(
            progress=self.arc_to_progress(arc),
            arc_length=arc,
            x=float(point[0]),
            y=float(point[1]),
            heading=heading,
            lateral_error=0.0,
            curvature=curvature,
        )

    def project(self, x: float, y: float) -> TrackProjection:
        point = np.asarray([x, y], dtype=float)
        delta_from_start = point - self.centerline
        t = np.divide(
            np.einsum("ij,ij->i", delta_from_start, self._segments),
            self._segment_length_sq,
            out=np.zeros_like(self._segment_lengths),
            where=self._segment_valid,
        )
        t = np.clip(t, 0.0, 1.0)
        projected = self.centerline + self._segments * t[:, None]
        delta = point - projected
        distance_sq = np.einsum("ij,ij->i", delta, delta)
        distance_sq = np.where(self._segment_valid, distance_sq, np.inf)
        index = int(np.argmin(distance_sq))
        if not np.isfinite(distance_sq[index]):
            raise RuntimeError("Unable to project point onto track.")

        signed_offset = float(np.dot(point - projected[index], self._segment_normals[index]))
        arc = float(self._cumulative[index] + t[index] * self._segment_lengths[index])
        return TrackProjection(
            progress=self.arc_to_progress(arc),
            arc_length=arc,
            x=float(projected[index, 0]),
            y=float(projected[index, 1]),
            heading=float(self._segment_headings[index]),
            lateral_error=signed_offset,
            curvature=float(np.interp(arc, self._arc_samples, self._curvature_samples, period=self.length)),
        )

    def spawn_pose(self, progress: float, lateral_error: float = 0.0, heading_error: float = 0.0) -> tuple[float, float, float]:
        projection = self.sample(progress)
        tangent = np.array([np.cos(projection.heading), np.sin(projection.heading)])
        normal = np.array([-tangent[1], tangent[0]])
        position = np.array([projection.x, projection.y]) + normal * lateral_error
        yaw = wrap_angle(projection.heading + heading_error)
        return float(position[0]), float(position[1]), float(yaw)

    def lookahead_curvatures(self, progress: float, count: int, spacing_m: float) -> np.ndarray:
        base_arc = self.progress_to_arc(progress)
        arcs = base_arc + np.arange(count, dtype=float) * spacing_m
        return np.interp(arcs % self.length, self._arc_samples, self._curvature_samples, period=self.length)

    def out_of_bounds(self, lateral_error: float) -> bool:
        return abs(lateral_error) > (self.width * 0.5)
