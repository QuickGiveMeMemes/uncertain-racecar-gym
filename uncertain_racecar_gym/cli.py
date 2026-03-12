from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from uncertain_racecar_gym.controllers import CenterlineDriver
from uncertain_racecar_gym.analysis import generate_default_report
from uncertain_racecar_gym.dataset import build_canonical_dataset, build_demo_dataset
from uncertain_racecar_gym.env import UncertainRacecarEnv
from uncertain_racecar_gym.replay import export_replay_bundle
from uncertain_racecar_gym.rendering import write_video
from uncertain_racecar_gym.scenario import load_scenario
from uncertain_racecar_gym.uncertainty import EmpiricalUncertaintyModel


def build_dataset_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a canonical dataset for uncertain-racecar-gym.")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--demo-episodes", type=int, default=0)
    parser.add_argument("--steps-per-episode", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--track-id", default="assetto_track")
    parser.add_argument("--car-id", default="assetto_car")
    parser.add_argument("inputs", nargs="*")
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    if args.demo_episodes > 0:
        path = build_demo_dataset(
            scenario=scenario,
            output_path=args.output,
            episodes=args.demo_episodes,
            steps_per_episode=args.steps_per_episode,
            seed=args.seed,
        )
    else:
        if not args.inputs:
            parser.error("Provide input dataset files or use --demo-episodes.")
        path = build_canonical_dataset(
            inputs=args.inputs,
            output_path=args.output,
            scenario=scenario,
            track_id=args.track_id,
            car_id=args.car_id,
        )
    print(path)
    return 0


def fit_uncertainty_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit an empirical uncertainty model from a canonical dataset.")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--neighbor-count", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=None)
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    canonical = pd.read_parquet(args.input)
    artifact = EmpiricalUncertaintyModel.fit(
        canonical=canonical,
        scenario=scenario,
        history_length=args.history_length,
        neighbor_count=args.neighbor_count,
        block_length=args.block_length,
    )
    print(artifact.save(args.output))
    return 0


def record_rollout_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a rollout video and rollout log.")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--name", default="rollout")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-mode", default="rgb_array_follow")
    parser.add_argument("--uncertainty-mode", default="nominal")
    parser.add_argument("--uncertainty-artifact", default=None)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    scenario = load_scenario(args.scenario)
    env = UncertainRacecarEnv(
        scenario=scenario.source_path,
        uncertainty=args.uncertainty_mode,
        uncertainty_artifact=args.uncertainty_artifact,
        renderer="pybullet",
        render_mode=args.render_mode,
        output_dir=output_dir,
    )
    controller = CenterlineDriver()
    obs, info = env.reset(seed=args.seed, options={"uncertainty_mode": args.uncertainty_mode, "start_mode": "random"})

    frames = []
    rewards = []
    for _ in range(args.steps):
        state = env._state
        action = controller.act(state, env.track)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        rewards.append(reward)
        if terminated or truncated:
            break
    env.close()

    video_path = output_dir / f"{args.name}.mp4"
    if frames:
        write_video(frames, video_path, fps=round(1.0 / scenario.simulation.dt))
    rollout_json = output_dir / f"{args.name}.json"
    rollout_json.write_text(json.dumps(env.episode_history, indent=2), encoding="utf-8")
    markdown_path = output_dir / f"{args.name}.md"
    markdown_path.write_text(
        "\n".join(
            [
                f"# Rollout: {args.name}",
                "",
                f"- Scenario: `{scenario.name}`",
                f"- Uncertainty mode: `{args.uncertainty_mode}`",
                f"- Frames: `{len(frames)}`",
                f"- Total reward: `{sum(rewards):.3f}`",
                f"- Video: `{video_path.name if frames else 'not generated'}`",
                f"- Replay JSON: `{rollout_json.name}`",
            ]
        ),
        encoding="utf-8",
    )
    print(video_path if frames else rollout_json)
    return 0


def export_replay_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a Blender-friendly replay bundle from rollout JSON.")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--rollout-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-path", default=None)
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    history = json.loads(Path(args.rollout_json).read_text(encoding="utf-8"))
    bundle = export_replay_bundle(history, scenario, args.output_dir, video_path=args.video_path)
    print(bundle)
    return 0


def analyze_uncertainty_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a technical uncertainty report with plots.")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args(argv)

    artifacts = generate_default_report(output_dir=args.output_dir, scenario_path=args.scenario)
    print(artifacts.report_path)
    return 0
