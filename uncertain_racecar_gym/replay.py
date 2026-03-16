from __future__ import annotations

import json
from pathlib import Path

from uncertain_racecar_gym.common import ensure_dir
from uncertain_racecar_gym.scenario import Scenario


def export_replay_bundle(
    history: list[dict],
    scenario: Scenario,
    output_dir: str | Path,
    video_path: str | Path | None = None,
) -> Path:
    bundle_dir = ensure_dir(output_dir)

    trajectory_path = bundle_dir / "trajectory.json"
    manifest_path = bundle_dir / "scene_manifest.json"
    camera_path = bundle_dir / "camera_script.json"
    readme_path = bundle_dir / "README.md"

    trajectory_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "scenario": scenario.name,
                "track_csv": scenario.track.csv,
                "vehicle_asset": "package://vehicles/simple_racecar.urdf",
                "video_path": str(video_path) if video_path else None,
                "frame_count": len(history),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    camera_path.write_text(
        json.dumps(
            {
                "shots": [
                    {"name": "follow", "type": "follow", "start": 0, "end": max(0, len(history) // 3)},
                    {"name": "orbit", "type": "cinematic", "start": max(0, len(history) // 3), "end": max(0, 2 * len(history) // 3)},
                    {"name": "trackside", "type": "birds_eye", "start": max(0, 2 * len(history) // 3), "end": len(history)},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    readme_path.write_text(
        "\n".join(
            [
                "# Blender Replay Bundle",
                "",
                "This bundle contains the simulator trajectory, camera script, and scene metadata for offline rendering.",
                "",
                "Suggested workflow:",
                "1. Import or reference the packaged vehicle asset.",
                "2. Build the track mesh from the centerline CSV in `scene_manifest.json`.",
                "3. Animate the vehicle using `trajectory.json`.",
                "4. Apply the shot plan from `camera_script.json`.",
            ]
        ),
        encoding="utf-8",
    )
    return bundle_dir
