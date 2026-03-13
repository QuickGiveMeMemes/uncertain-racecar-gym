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

For a deeper explanation of the current renderer quality gap and the uncertainty model itself, see [UNCERTAINTY_TECHNICAL_README.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/UNCERTAINTY_TECHNICAL_README.md). For the latest calibrated real-data review package, start with [phase_9_cross_track_uncertainty_and_renderer.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/phase_9_cross_track_uncertainty_and_renderer.md).

## Quick start

```bash
uv sync --extra dev
uv run pytest -q
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
  --uncertainty-mode nominal

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
- [track_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/barcelona_real/track_report/track_report.md)
- [uncertainty_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/barcelona_real/report/uncertainty_report.md)
- [corrected uncertainty_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/barcelona_real_dtfix/report/uncertainty_report.md)
- [calibrated uncertainty_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/barcelona_calibrated/stochastic_report/uncertainty_report.md)
- [hybrid calibrated uncertainty_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/barcelona_hybrid_calibrated/stochastic_report/uncertainty_report.md)
- [Monza hybrid calibrated uncertainty_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/monza_hybrid_calibrated/stochastic_report/uncertainty_report.md)

## Project layout

- `uncertain_racecar_gym/`: package code, packaged assets, CLI entry points
- `tests/`: focused tests for the new repo only
- `output/`: ignored local artifacts for videos, logs, datasets, and replay bundles
