from __future__ import annotations

import json
from pathlib import Path

from uncertain_racecar_gym.blender_render import BlenderRenderConfig, build_blender_render_command, render_replay_bundle


def test_build_blender_render_command_uses_bundle_and_output(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    output_path = tmp_path / "render.mp4"
    fake_blender = tmp_path / "blender"
    fake_blender.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_blender.chmod(0o755)

    command = build_blender_render_command(
        BlenderRenderConfig(
            bundle_dir=bundle_dir,
            output_path=output_path,
            blender_executable=fake_blender,
            vehicle_asset=tmp_path / "racecar.glb",
            frame_limit=120,
            save_blend_path=tmp_path / "scene.blend",
        )
    )

    assert command[0] == str(fake_blender)
    assert "--bundle-dir" in command
    assert str(bundle_dir) in command
    assert "--frames-dir" in command
    assert str(tmp_path / "render_frames") in command
    assert "--frame-limit" in command
    assert "--save-blend-path" in command
    assert "--vehicle-asset" in command


def test_render_replay_bundle_dry_run(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "scene_manifest.json").write_text(
        json.dumps({"fps": 20, "frame_count": 42}),
        encoding="utf-8",
    )
    fake_blender = tmp_path / "blender"
    fake_blender.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_blender.chmod(0o755)

    result = render_replay_bundle(
        BlenderRenderConfig(
            bundle_dir=bundle_dir,
            output_path=tmp_path / "render.mp4",
            blender_executable=fake_blender,
            dry_run=True,
        )
    )

    assert result["bundle_dir"] == bundle_dir.as_posix()
    assert result["output_path"] == (tmp_path / "render.mp4").as_posix()
    assert result["frames_dir"] == (tmp_path / "render_frames").as_posix()
    assert result["engine"] == "BLENDER_EEVEE"
    assert result["resolution"] == [1920, 1080]
    assert result["frame_limit"] is None
    assert any(str(fake_blender) == part for part in result["command"])
