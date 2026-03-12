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

For a deeper explanation of the current renderer quality gap and the uncertainty model itself, see [UNCERTAINTY_TECHNICAL_README.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/UNCERTAINTY_TECHNICAL_README.md). After generating artifacts, the step-by-step plot report is also available in [output/uncertainty_report.md](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/output/uncertainty_report.md).

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

## Project layout

- `uncertain_racecar_gym/`: package code, packaged assets, CLI entry points
- `tests/`: focused tests for the new repo only
- `output/`: ignored local artifacts for videos, logs, datasets, and replay bundles
