from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from uncertain_racecar_gym.calibration import generate_nominal_calibration_package
from uncertain_racecar_gym.controllers import CenterlineDriver, ProfiledCenterlineDriver
from uncertain_racecar_gym.analysis import generate_default_report
from uncertain_racecar_gym.dataset import build_canonical_dataset, build_demo_dataset
from uncertain_racecar_gym.env import UncertainRacecarEnv
from uncertain_racecar_gym.replay import export_replay_bundle
from uncertain_racecar_gym.replay_eval import generate_replay_evaluation
from uncertain_racecar_gym.rendering import PyBulletMirrorRenderer, write_video
from uncertain_racecar_gym.scenario import load_scenario
from uncertain_racecar_gym.track_builder import build_track_from_dataset
from uncertain_racecar_gym.uncertainty import EmpiricalUncertaintyModel


def _build_controller(
    driver_dataset: str | None,
    speed_profile_quantile: float,
    speed_profile_scale: float,
    target_speed: float,
    min_speed: float,
):
    if driver_dataset is not None:
        canonical = pd.read_parquet(driver_dataset)
        return ProfiledCenterlineDriver.from_canonical_dataframe(
            canonical,
            speed_quantile=speed_profile_quantile,
            speed_scale=speed_profile_scale,
            min_speed=min_speed,
        )
    return CenterlineDriver(target_speed=target_speed, min_speed=min_speed)


def build_dataset_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a canonical dataset for uncertain-racecar-gym.")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--demo-episodes", type=int, default=0)
    parser.add_argument("--steps-per-episode", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--track-id", default=None, help="Optional override. If omitted, try to infer from Assetto static_info.")
    parser.add_argument("--car-id", default=None, help="Optional override. If omitted, try to infer from Assetto static_info.")
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


def build_track_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconstruct a track centerline from progress-labeled trajectory data.")
    parser.add_argument("--output", required=True, help="Output CSV path for the reconstructed centerline.")
    parser.add_argument("--scenario-output", default=None, help="Optional scenario YAML to write.")
    parser.add_argument("--report-dir", default=None, help="Optional report directory for reconstruction plots and markdown.")
    parser.add_argument("--scenario-name", default="ks_barcelona_layout_gp_dallara_f317")
    parser.add_argument("--width", type=float, default=None, help="Optional manual track width override.")
    parser.add_argument("--num-bins", type=int, default=2000)
    parser.add_argument("--smoothing-window", type=int, default=31)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args(argv)

    artifacts = build_track_from_dataset(
        inputs=args.inputs,
        output_csv=args.output,
        scenario_output=args.scenario_output,
        report_dir=args.report_dir,
        scenario_name=args.scenario_name,
        width=args.width,
        num_bins=args.num_bins,
        smoothing_window=args.smoothing_window,
    )
    print(artifacts.csv_path)
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


def calibrate_nominal_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit a deterministic nominal calibration artifact and regenerate stochastic uncertainty outputs.")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-description", default=None)
    args = parser.parse_args(argv)

    artifacts = generate_nominal_calibration_package(
        dataset_path=args.dataset,
        scenario_path=args.scenario,
        output_dir=args.output_dir,
        source_description=args.source_description,
    )
    print(artifacts.calibration_report_path)
    return 0


def record_rollout_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a rollout video and rollout log.")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--name", default="rollout")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-mode", default="rgb_array_follow")
    parser.add_argument("--uncertainty-mode", default="none", choices=["none", "nominal", "gaussian", "empirical"])
    parser.add_argument("--uncertainty-artifact", default=None)
    parser.add_argument("--calibration-artifact", default=None)
    parser.add_argument("--gaussian-mean", nargs=4, type=float, default=None, metavar=("VX", "VY", "YAW", "STEER"))
    parser.add_argument("--gaussian-std", nargs=4, type=float, default=None, metavar=("VX", "VY", "YAW", "STEER"))
    parser.add_argument("--driver-dataset", default=None, help="Optional canonical parquet used to build a progress-dependent speed profile.")
    parser.add_argument("--speed-profile-quantile", type=float, default=0.6)
    parser.add_argument("--speed-profile-scale", type=float, default=0.6)
    parser.add_argument("--target-speed", type=float, default=14.0)
    parser.add_argument("--min-speed", type=float, default=8.0)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    scenario = load_scenario(args.scenario)
    env = UncertainRacecarEnv(
        scenario=scenario.source_path,
        uncertainty=args.uncertainty_mode,
        uncertainty_artifact=args.uncertainty_artifact,
        calibration_artifact=args.calibration_artifact,
        apply_mean_correction=bool(args.calibration_artifact),
        gaussian_noise_mean=args.gaussian_mean,
        gaussian_noise_std=args.gaussian_std,
        renderer="pybullet",
        render_mode=args.render_mode,
        output_dir=output_dir,
    )
    controller = _build_controller(
        driver_dataset=args.driver_dataset,
        speed_profile_quantile=args.speed_profile_quantile,
        speed_profile_scale=args.speed_profile_scale,
        target_speed=args.target_speed,
        min_speed=args.min_speed,
    )
    obs, info = env.reset(
        seed=args.seed,
        options={
            "uncertainty_mode": args.uncertainty_mode,
            "gaussian_noise_mean": args.gaussian_mean,
            "gaussian_noise_std": args.gaussian_std,
            "start_mode": "random",
        },
    )

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
                f"- Driver dataset: `{Path(args.driver_dataset).name if args.driver_dataset else 'none'}`",
                f"- Video: `{video_path.name if frames else 'not generated'}`",
                f"- Replay JSON: `{rollout_json.name}`",
            ]
        ),
        encoding="utf-8",
    )
    print(video_path if frames else rollout_json)
    return 0


def compare_rollouts_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a comparison video with the empirical rollout overlaid against a translucent nominal ghost.")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--name", default="comparison_rollout")
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-mode", default="rgb_array_follow")
    parser.add_argument("--uncertainty-artifact", required=True)
    parser.add_argument("--calibration-artifact", default=None)
    parser.add_argument("--driver-dataset", default=None, help="Optional canonical parquet used to build a progress-dependent speed profile.")
    parser.add_argument("--speed-profile-quantile", type=float, default=0.6)
    parser.add_argument("--speed-profile-scale", type=float, default=0.6)
    parser.add_argument("--target-speed", type=float, default=14.0)
    parser.add_argument("--min-speed", type=float, default=8.0)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    scenario = load_scenario(args.scenario)
    nominal = UncertainRacecarEnv(
        scenario=scenario.source_path,
        uncertainty=None,
        calibration_artifact=args.calibration_artifact,
        apply_mean_correction=bool(args.calibration_artifact),
        renderer=None,
        render_mode=None,
        output_dir=output_dir,
    )
    empirical = UncertainRacecarEnv(
        scenario=scenario.source_path,
        uncertainty="empirical",
        uncertainty_artifact=args.uncertainty_artifact,
        calibration_artifact=args.calibration_artifact,
        apply_mean_correction=bool(args.calibration_artifact),
        renderer=None,
        render_mode=None,
        output_dir=output_dir,
    )
    controller = _build_controller(
        driver_dataset=args.driver_dataset,
        speed_profile_quantile=args.speed_profile_quantile,
        speed_profile_scale=args.speed_profile_scale,
        target_speed=args.target_speed,
        min_speed=args.min_speed,
    )
    nominal.reset(seed=args.seed, options={"uncertainty_mode": None, "start_mode": "random"})
    empirical.reset(seed=args.seed, options={"uncertainty_mode": "empirical", "start_mode": "random"})

    renderer = PyBulletMirrorRenderer(scenario, nominal.track, args.render_mode)
    frames = []
    rewards = {"nominal": 0.0, "empirical": 0.0}
    step_count = 0
    while step_count < args.steps:
        action = controller.act(nominal._state, nominal.track)
        _, nominal_reward, nominal_done, nominal_trunc, nominal_info = nominal.step(action)
        _, empirical_reward, empirical_done, empirical_trunc, empirical_info = empirical.step(action)
        rewards["nominal"] += nominal_reward
        rewards["empirical"] += empirical_reward
        frame = renderer.render(
            empirical_info["render_state"],
            comparison_state=nominal_info["render_state"],
        )
        if frame is not None:
            frames.append(frame)
        step_count += 1
        if nominal_done or nominal_trunc or empirical_done or empirical_trunc:
            break
    renderer.close()
    nominal.close()
    empirical.close()

    video_path = output_dir / f"{args.name}.mp4"
    if frames:
        write_video(frames, video_path, fps=round(1.0 / scenario.simulation.dt))
    markdown_path = output_dir / f"{args.name}.md"
    markdown_path.write_text(
        "\n".join(
            [
                f"# Comparison Rollout: {args.name}",
                "",
                f"- Scenario: `{scenario.name}`",
                f"- Primary vehicle: empirical rollout",
                f"- Ghost vehicle: calibrated nominal rollout",
                f"- Shared action source: nominal controller state",
                f"- Driver dataset: `{Path(args.driver_dataset).name if args.driver_dataset else 'none'}`",
                f"- Frames: `{len(frames)}`",
                f"- Nominal total reward: `{rewards['nominal']:.3f}`",
                f"- Empirical total reward: `{rewards['empirical']:.3f}`",
                f"- Video: `{video_path.name if frames else 'not generated'}`",
            ]
        ),
        encoding="utf-8",
    )
    print(video_path if frames else markdown_path)
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
    parser.add_argument("--dataset", default=None, help="Optional canonical parquet to analyze instead of rebuilding the synthetic demo dataset.")
    parser.add_argument("--source-description", default=None, help="Optional text label describing the uncertainty data source in the report.")
    parser.add_argument("--calibration-artifact", default=None, help="Optional deterministic calibration artifact to apply before fitting stochastic uncertainty.")
    args = parser.parse_args(argv)

    artifacts = generate_default_report(
        output_dir=args.output_dir,
        scenario_path=args.scenario,
        dataset_path=args.dataset,
        source_description=args.source_description,
        calibration_artifact_path=args.calibration_artifact,
    )
    print(artifacts.report_path)
    return 0


def replay_evaluate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay recorded Assetto actions and compare actual data against nominal, calibrated, and empirical rollouts.")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-artifact", default=None)
    parser.add_argument("--uncertainty-artifact", default=None)
    parser.add_argument("--trajectory-limit", type=int, default=2)
    parser.add_argument("--empirical-seeds", nargs="*", type=int, default=[11, 17, 23, 29])
    args = parser.parse_args(argv)

    artifacts = generate_replay_evaluation(
        dataset_path=args.dataset,
        scenario_path=args.scenario,
        output_dir=args.output_dir,
        calibration_artifact_path=args.calibration_artifact,
        uncertainty_artifact_path=args.uncertainty_artifact,
        trajectory_limit=args.trajectory_limit,
        empirical_seeds=tuple(args.empirical_seeds),
    )
    print(artifacts.report_path)
    return 0
