from __future__ import annotations

import numpy as np

from uncertain_racecar_gym.scenario import load_scenario
from uncertain_racecar_gym.track import TrackModel, TrackProjection


def _brute_force_project(track: TrackModel, x: float, y: float) -> TrackProjection:
    point = np.array([x, y], dtype=float)
    best_distance = float("inf")
    best_projection = None
    for index, segment in enumerate(track._segments):
        p0 = track.centerline[index]
        seg_len = track._segment_lengths[index]
        if seg_len < 1e-9:
            continue
        t = np.clip(np.dot(point - p0, segment) / (seg_len * seg_len), 0.0, 1.0)
        projected = p0 + t * segment
        distance = float(np.linalg.norm(point - projected))
        if distance < best_distance:
            tangent = segment / seg_len
            normal = np.array([-tangent[1], tangent[0]])
            signed_offset = float(np.dot(point - projected, normal))
            arc = track._cumulative[index] + t * seg_len
            best_distance = distance
            best_projection = TrackProjection(
                progress=track.arc_to_progress(arc),
                arc_length=arc,
                x=float(projected[0]),
                y=float(projected[1]),
                heading=float(np.arctan2(segment[1], segment[0])),
                lateral_error=signed_offset,
                curvature=float(np.interp(arc, track._arc_samples, track._curvature_samples, period=track.length)),
            )
    assert best_projection is not None
    return best_projection


def test_vectorized_projection_matches_bruteforce_on_dense_track() -> None:
    scenario = load_scenario("package://scenarios/ks_barcelona_layout_gp_dallara_f317_rl_long.yaml")
    track = TrackModel.from_config(scenario.track)

    for progress in np.linspace(0.0, 0.96, 13):
        sample = track.sample(float(progress))
        tangent = np.array([np.cos(sample.heading), np.sin(sample.heading)])
        normal = np.array([-tangent[1], tangent[0]])
        for lateral_offset in (-1.25, -0.1, 0.0, 0.8):
            point = np.array([sample.x, sample.y]) + normal * lateral_offset
            actual = track.project(float(point[0]), float(point[1]))
            expected = _brute_force_project(track, float(point[0]), float(point[1]))

            np.testing.assert_allclose(actual.progress, expected.progress, atol=1e-12)
            np.testing.assert_allclose(actual.arc_length, expected.arc_length, atol=1e-9)
            np.testing.assert_allclose([actual.x, actual.y], [expected.x, expected.y], atol=1e-12)
            np.testing.assert_allclose(actual.heading, expected.heading, atol=1e-12)
            np.testing.assert_allclose(actual.lateral_error, expected.lateral_error, atol=1e-12)
            np.testing.assert_allclose(actual.curvature, expected.curvature, atol=1e-12)
