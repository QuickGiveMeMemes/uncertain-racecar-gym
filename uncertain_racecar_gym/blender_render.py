from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess

import imageio.v2 as imageio

from uncertain_racecar_gym.common import package_asset_path


DEFAULT_MAC_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")


@dataclass(slots=True)
class BlenderRenderConfig:
    bundle_dir: Path
    output_path: Path
    blender_executable: Path | None = None
    vehicle_asset: Path | None = None
    engine: str = "BLENDER_EEVEE"
    samples: int = 128
    resolution_x: int = 1920
    resolution_y: int = 1080
    frame_limit: int | None = None
    save_blend_path: Path | None = None
    keep_frames: bool = False
    dry_run: bool = False


def find_blender_executable(explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    env_candidate = os.environ.get("BLENDER_BIN")
    if env_candidate:
        candidates.append(Path(env_candidate))
    which_candidate = shutil.which("blender")
    if which_candidate:
        candidates.append(Path(which_candidate))
    candidates.append(DEFAULT_MAC_BLENDER)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to find Blender. Install Blender or set BLENDER_BIN to the executable path."
    )


def build_blender_render_command(config: BlenderRenderConfig) -> list[str]:
    blender_executable = find_blender_executable(config.blender_executable)
    script_path = package_asset_path("blender/render_replay.py")
    frames_dir = config.output_path.parent / f"{config.output_path.stem}_frames"
    command = [
        str(blender_executable),
        "-b",
        "-P",
        str(script_path),
        "--",
        "--bundle-dir",
        str(config.bundle_dir),
        "--frames-dir",
        str(frames_dir),
        "--engine",
        str(config.engine),
        "--samples",
        str(int(config.samples)),
        "--resolution-x",
        str(int(config.resolution_x)),
        "--resolution-y",
        str(int(config.resolution_y)),
    ]
    if config.frame_limit is not None:
        command.extend(["--frame-limit", str(int(config.frame_limit))])
    if config.save_blend_path is not None:
        command.extend(["--save-blend-path", str(config.save_blend_path)])
    if config.vehicle_asset is not None:
        command.extend(["--vehicle-asset", str(config.vehicle_asset)])
    return command


def render_replay_bundle(config: BlenderRenderConfig) -> dict[str, object]:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = config.output_path.parent / f"{config.output_path.stem}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    command = build_blender_render_command(config)
    result = {
        "bundle_dir": config.bundle_dir.as_posix(),
        "output_path": config.output_path.as_posix(),
        "frames_dir": frames_dir.as_posix(),
        "command": command,
        "engine": config.engine,
        "samples": int(config.samples),
        "resolution": [int(config.resolution_x), int(config.resolution_y)],
        "frame_limit": int(config.frame_limit) if config.frame_limit is not None else None,
    }
    if config.dry_run:
        return result

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result["stdout"] = completed.stdout
    result["stderr"] = completed.stderr
    result["output_exists"] = config.output_path.exists()
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(
            "Blender finished without producing rendered frames.\n"
            f"Command: {' '.join(command)}\n"
            f"stderr:\n{completed.stderr}"
        )
    manifest_path = config.bundle_dir / "scene_manifest.json"
    fps = 20
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result["manifest_fps"] = manifest.get("fps")
        result["frame_count"] = manifest.get("frame_count")
        fps = int(manifest.get("fps", fps))

    with imageio.get_writer(config.output_path, fps=fps) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    result["output_exists"] = config.output_path.exists()
    result["rendered_frames"] = len(frame_paths)
    if not config.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return result
