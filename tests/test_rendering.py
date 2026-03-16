from __future__ import annotations

from pathlib import Path

import json
import numpy as np

from uncertain_racecar_gym.cli import export_replay_main, record_rollout_main
from uncertain_racecar_gym.replay import export_replay_bundle
from uncertain_racecar_gym.rendering import PyBulletMirrorRenderer
from uncertain_racecar_gym.scenario import load_scenario
from uncertain_racecar_gym.track import TrackModel


def test_record_rollout_and_export(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    rc = record_rollout_main(
        [
            "--output-dir",
            str(output_dir),
            "--name",
            "smoke_rollout",
            "--steps",
            "20",
            "--seed",
            "2",
            "--render-mode",
            "rgb_array_follow",
            "--uncertainty-mode",
            "nominal",
        ]
    )
    assert rc == 0
    assert (output_dir / "smoke_rollout.mp4").exists()
    assert (output_dir / "smoke_rollout.json").exists()
    history = json.loads((output_dir / "smoke_rollout.json").read_text(encoding="utf-8"))
    assert len(history) > 0

    rc = export_replay_main(
        [
            "--rollout-json",
            str(output_dir / "smoke_rollout.json"),
            "--output-dir",
            str(output_dir / "bundle"),
            "--video-path",
            str(output_dir / "smoke_rollout.mp4"),
        ]
    )
    assert rc == 0
    assert (output_dir / "bundle" / "trajectory.json").exists()
    assert (output_dir / "bundle" / "scene_manifest.json").exists()
    assert (output_dir / "bundle" / "track_centerline.csv").exists()
    assert (output_dir / "bundle" / "scenario.yaml").exists()
    manifest = json.loads((output_dir / "bundle" / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["track_csv"] == "track_centerline.csv"
    assert manifest["scenario_yaml"] == "scenario.yaml"
    assert manifest["track"]["width"] > 0.0
    assert manifest["fps"] >= 1


def test_renderer_planner_overlay_smoke() -> None:
    scenario = load_scenario("package://scenarios/sample_oval.yaml")
    track = TrackModel.from_config(scenario.track)
    renderer = PyBulletMirrorRenderer(scenario, track, "rgb_array_follow", width=320, height=180)
    frame = renderer.render(
        {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "steering_angle": 0.0,
            "wheel_rotation": 0.0,
            "frame_index": 0,
            "progress": 0.0,
            "speed": 8.0,
        },
        planner_debug={
            "candidate_xy": np.asarray(
                [
                    [[0.0, 0.0], [1.0, 0.4], [2.0, 0.6]],
                    [[0.0, 0.0], [0.8, -0.2], [1.8, -0.4]],
                ],
                dtype=np.float32,
            ),
            "final_xy": np.asarray([[0.0, 0.0], [1.2, 0.1], [2.5, 0.15]], dtype=np.float32),
        },
    )
    renderer.close()
    assert frame is not None
    assert frame.shape == (180, 320, 3)


def test_export_replay_bundle_serializes_planner_debug_numpy(tmp_path: Path) -> None:
    scenario = load_scenario("package://scenarios/sample_oval.yaml")
    bundle_dir = export_replay_bundle(
        [
            {
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "steering_angle": 0.0,
                "wheel_rotation": 0.0,
                "planner_debug": {
                    "candidate_xy": np.asarray([[[0.0, 0.0], [1.0, 0.1]]], dtype=np.float32),
                    "final_xy": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
                },
            }
        ],
        scenario,
        tmp_path / "bundle",
    )
    payload = json.loads((bundle_dir / "trajectory.json").read_text(encoding="utf-8"))
    assert payload[0]["planner_debug"]["candidate_xy"][0][1] == [1.0, 0.10000000149011612]
    assert payload[0]["planner_debug"]["final_xy"][1] == [1.0, 0.0]
