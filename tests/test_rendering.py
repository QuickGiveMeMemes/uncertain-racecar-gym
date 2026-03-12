from __future__ import annotations

from pathlib import Path

import json

from uncertain_racecar_gym.cli import export_replay_main, record_rollout_main


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
