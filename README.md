# uncertain-racecar-gym

A racecar-focused Gymnasium environment with:

- a nominal dynamic-bicycle simulator
- an empirical uncertainty model fitted from canonicalized trajectory data
- real-time 3D PyBullet rendering
- offline replay bundle export for higher-fidelity review workflows

## Rendering status

The current saved MP4s are **Tier 1 diagnostic renders**, not the final publication-quality output path.

- Tier 1:
  - real-time PyBullet mirror rendering
  - good for debugging, controller inspection, and quick comparisons
- Tier 2:
  - replay export for offline scene work
  - intended path for publication-ready animation quality

For a deeper explanation of the current renderer quality gap and the uncertainty model itself, see [UNCERTAINTY_TECHNICAL_README.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/UNCERTAINTY_TECHNICAL_README.md). For the latest telemetry-conditioned real-data review package, start with [phase_12_telemetry_and_replay_eval.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/phase_12_telemetry_and_replay_eval.md).

## Quick start

```bash
uv sync --extra dev
uv run pytest -q
```

For the JAX nominal-training path:

```bash
uv sync --extra dev --extra jax
uv run --extra jax pytest -q tests/test_jax_env.py
```

## Clean Gym API

The training-facing API keeps a standard Gymnasium `step(action)` signature.

Switch uncertainty with the env constructor, `reset(options=...)`, or `env.unwrapped.set_uncertainty(...)`:

```python
import gymnasium as gym
import uncertain_racecar_gym  # registers UncertainRacecar-v0

env = gym.make("UncertainRacecar-v0", uncertainty=None)
obs, info = env.reset(seed=0)

# Pure nominal rollout
obs, reward, terminated, truncated, info = env.step(action)

# Switch to fixed Gaussian uncertainty on [delta_vx, delta_vy, delta_yaw_rate, delta_steer]
env.unwrapped.set_uncertainty("gaussian", gaussian_noise_std=[0.12, 0.08, 0.05, 0.015])

# Switch to empirical uncertainty
env = gym.make(
    "UncertainRacecar-v0",
    uncertainty="empirical",
    uncertainty_artifact="path/to/analysis_uncertainty.pkl",
)
```

Mode summary:

- `uncertainty=None` or `"nominal"`: pure dynamic-bicycle rollout
- `uncertainty="gaussian"`: zero-mean Gaussian perturbation on modeled dynamic states
- `uncertainty="empirical"`: empirical residual sampler from fitted data artifact

Important: the RL observation stays limited to the bicycle-model state and track context. Signals like `drive_train_speed`, `rpm`, and `gear` are not exposed as policy state.

## JAX nominal env

There is now a partial JAX counterpart for nominal training in [jax_env.py](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/uncertain_racecar_gym/jax_env.py).

This JAX port currently includes:

- track sampling and projection
- nominal dynamic-bicycle rollout
- reset / step / observation / reward / termination
- JIT-friendly pure-state API

It intentionally does not include:

- PyBullet rendering
- empirical uncertainty
- replay/report tooling

Example:

```python
import jax
import jax.numpy as jnp

from uncertain_racecar_gym.jax_env import NominalJaxRacecarEnv

env = NominalJaxRacecarEnv("output/barcelona_real/ks_barcelona_layout_gp_dallara_f317.yaml")
key = jax.random.PRNGKey(0)

reset_out = env.reset(key, start_mode="random")
state = reset_out.state
obs = reset_out.observation

action = jnp.array([0.0, 0.2, 0.0], dtype=jnp.float32)
step_out = env.step(state, action)

# JIT-friendly
step_out = env.step_jit(state, action)
```

## Demo workflow

Build a synthetic canonical dataset:

```bash
uv run uncertain-racecar-build-dataset \
  --output output/demo_dataset.parquet \
  --demo-episodes 2 \
  --steps-per-episode 50
```

Fit the empirical uncertainty artifact:

```bash
uv run uncertain-racecar-fit-uncertainty \
  --input output/demo_dataset.parquet \
  --output output/demo_uncertainty.pkl
```

Record nominal and empirical rollouts:

```bash
uv run uncertain-racecar-record-rollout \
  --output-dir output \
  --name nominal_rollout \
  --steps 140 \
  --render-mode rgb_array_follow \
  --uncertainty-mode none

uv run uncertain-racecar-record-rollout \
  --output-dir output \
  --name gaussian_rollout \
  --steps 140 \
  --render-mode rgb_array_follow \
  --uncertainty-mode gaussian \
  --gaussian-std 0.12 0.08 0.05 0.015

uv run uncertain-racecar-record-rollout \
  --output-dir output \
  --name empirical_rollout \
  --steps 140 \
  --render-mode rgb_array_follow \
  --uncertainty-mode empirical \
  --uncertainty-artifact output/demo_uncertainty.pkl
```

Export the replay bundle:

```bash
uv run uncertain-racecar-export-replay \
  --rollout-json output/empirical_rollout.json \
  --output-dir output/empirical_replay_bundle \
  --video-path output/empirical_rollout.mp4
```

Generate the uncertainty analysis report with plots:

```bash
uv run uncertain-racecar-analyze-uncertainty --output-dir output
```

Replay recorded actions against actual data:

```bash
uv run uncertain-racecar-replay-evaluate \
  --scenario output/barcelona_real/ks_barcelona_layout_gp_dallara_f317.yaml \
  --dataset output/barcelona_telemetry_real/barcelona_real_canonical.parquet \
  --output-dir output/barcelona_telemetry_regime_calibrated/replay_eval \
  --calibration-artifact output/barcelona_telemetry_regime_calibrated/nominal_calibration.pkl \
  --uncertainty-artifact output/barcelona_telemetry_regime_calibrated/stochastic_report/analysis_uncertainty.pkl
```

## Assetto-style ingestion

The dataset builder now accepts two Assetto-style offline formats:

- plugin telemetry pickles with `{"telemetry": [...], "static_info": {...}}`
- converted state pickles with `{"states": [...], "static_info": {...}}`

Build a canonical parquet from one or more Assetto logs:

```bash
uv run uncertain-racecar-build-dataset \
  --scenario uncertain_racecar_gym/assets/scenarios/sample_oval.yaml \
  --output output/assetto_canonical.parquet \
  path/to/telemetry_001.pkl \
  path/to/telemetry_002.pkl
```

If `static_info` contains `TrackName`, `TrackConfiguration`, and `CarName`, the builder will infer `track_id` and `car_id` automatically. You can still override them:

```bash
uv run uncertain-racecar-build-dataset \
  --scenario uncertain_racecar_gym/assets/scenarios/sample_oval.yaml \
  --output output/assetto_canonical.parquet \
  --track-id ks_barcelona-layout_gp \
  --car-id dallara_f317 \
  path/to/lap.pkl
```

Analyze a canonical real-data parquet directly:

```bash
uv run uncertain-racecar-analyze-uncertainty \
  --output-dir output \
  --dataset output/assetto_canonical.parquet \
  --source-description "canonical Assetto telemetry dataset"
```

Note: the scenario centerline must match the track used to collect the logs, otherwise the projected `progress`, `lateral_error`, and `curvature` fields will be wrong.

## Real Assetto workflow

For real Assetto data, the intended order is:

1. reconstruct a track centerline from progress-labeled laps,
2. write a matching scenario YAML,
3. build the canonical parquet,
4. fit and analyze the empirical uncertainty model,
5. record rollout videos and export replay bundles.

Example commands:

```bash
uv run uncertain-racecar-build-track \
  --output output/barcelona_real/ks_barcelona_layout_gp_centerline.csv \
  --scenario-output output/barcelona_real/ks_barcelona_layout_gp_dallara_f317.yaml \
  --report-dir output/barcelona_real/track_report \
  --scenario-name ks_barcelona_layout_gp_dallara_f317 \
  data/AssettoCorsaGymDataSet/data_sets/ks_barcelona-layout_gp/dallara_f317/*/laps/*.pkl

uv run uncertain-racecar-build-dataset \
  --scenario output/barcelona_real/ks_barcelona_layout_gp_dallara_f317.yaml \
  --output output/barcelona_real/barcelona_real_canonical.parquet \
  data/AssettoCorsaGymDataSet/data_sets/ks_barcelona-layout_gp/dallara_f317/*/laps/*.pkl

uv run uncertain-racecar-analyze-uncertainty \
  --scenario output/barcelona_real/ks_barcelona_layout_gp_dallara_f317.yaml \
  --output-dir output/barcelona_real/report \
  --dataset output/barcelona_real/barcelona_real_canonical.parquet \
  --source-description "Real Assetto Corsa Barcelona / dallara_f317 state-pickle subset"
```

The current example outputs from that workflow are already available here:

- [phase_5_barcelona_real_data.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/phase_5_barcelona_real_data.md)
- [phase_6_dtfix_and_multimodal_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/phase_6_dtfix_and_multimodal_report.md)
- [phase_7_calibrated_nominal_and_ghost_compare.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/phase_7_calibrated_nominal_and_ghost_compare.md)
- [phase_8_hybrid_longitudinal_calibration.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/phase_8_hybrid_longitudinal_calibration.md)
- [phase_9_cross_track_uncertainty_and_renderer.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/phase_9_cross_track_uncertainty_and_renderer.md)
- [phase_10_regime_conditioned_uncertainty.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/phase_10_regime_conditioned_uncertainty.md)
- [phase_11_trajectory_id_fix.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/phase_11_trajectory_id_fix.md)
- [phase_12_telemetry_and_replay_eval.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/phase_12_telemetry_and_replay_eval.md)
- [track_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/barcelona_real/track_report/track_report.md)
- [telemetry-conditioned Barcelona uncertainty_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/barcelona_telemetry_regime_calibrated/stochastic_report/uncertainty_report.md)
- [telemetry-conditioned Monza uncertainty_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/monza_telemetry_regime_calibrated/stochastic_report/uncertainty_report.md)
- [Barcelona replay_eval_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/barcelona_telemetry_regime_calibrated/replay_eval/replay_eval_report.md)
- [Monza replay_eval_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/monza_telemetry_regime_calibrated/replay_eval/replay_eval_report.md)

## Project layout

- `uncertain_racecar_gym/`: package code, packaged assets, CLI entry points
- `tests/`: focused tests for the new repo only
- `output/`: ignored local artifacts for videos, logs, datasets, and replay bundles
