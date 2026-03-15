from __future__ import annotations

from pathlib import Path

import pandas as pd

from uncertain_racecar_gym.benchmark import BenchmarkCase, BenchmarkSuite, load_benchmark_suite, run_benchmark


def test_benchmark_smoke_with_centerline_controller(tmp_path: Path) -> None:
    suite = BenchmarkSuite(
        name="sample_oval_smoke_suite",
        scenario="package://scenarios/sample_oval.yaml",
        cases=(
            BenchmarkCase(
                case_id="full_lap_short",
                description="Short nominal sanity check from the grid start.",
                start_mode="grid",
                max_steps=40,
            ),
            BenchmarkCase(
                case_id="custom_turn_in",
                description="Explicit benchmark reset to verify custom initial conditions.",
                start_mode="grid",
                initial_progress=0.2,
                initial_speed=9.0,
                max_steps=30,
                tags=("custom_reset",),
            ),
        ),
        modes=("nominal", "gaussian"),
        gaussian_std=(0.2, 0.1, 0.05, 0.02),
    )

    artifacts = run_benchmark(
        suite=suite,
        output_dir=tmp_path / "benchmark",
        controller_kind="centerline",
        seeds=(0,),
        modes=("nominal", "gaussian"),
        gaussian_std=suite.gaussian_std,
        write_suite_path=tmp_path / "benchmark" / "suite.yaml",
        package_dir=tmp_path / "benchmark" / "package",
    )

    assert artifacts.summary_path.exists()
    assert artifacts.aggregate_csv_path.exists()
    assert artifacts.episode_csv_path.exists()
    assert artifacts.suite_path is not None and artifacts.suite_path.exists()
    assert artifacts.package_dir is not None and artifacts.package_dir.exists()

    aggregate = pd.read_csv(artifacts.aggregate_csv_path)
    assert set(aggregate["case_id"]) == {"full_lap_short", "custom_turn_in"}
    assert set(aggregate["mode"]) == {"nominal", "gaussian"}
    assert "mean_traveled_distance_m" in aggregate.columns

    loaded_suite = load_benchmark_suite(artifacts.suite_path)
    assert loaded_suite.name == suite.name
