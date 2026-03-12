# `uncertain_racecar_gym` with first-class 3D rendering

**Summary**
- Keep the main recommendation: Assetto Corsa should be the offline data/calibration source, not the required runtime.
- Update the architecture so 3D is a required feature, not a later add-on.
- V1 should use a split design: an empirical uncertainty simulator as the source of truth, plus a 3D render layer that mirrors simulator state in real time.
- Ship two visual tiers from the start:
  - Tier 1: real-time PyBullet 3D rendering for debugging, controller demos, and MP4 export.
  - Tier 2: offline high-quality replay export for polished videos, using Blender replay packages.
- As of March 12, 2026, the official [Assetto Corsa Steam listing](https://store.steampowered.com/app/244210/Assetto%5C_Corsa/) is Windows-focused, so Assetto should remain optional and off the critical path for Mac-friendly runtime use.

**Key Interfaces**
- Main env:
  - `gymnasium.make("UncertainRacecar-v0", scenario=..., uncertainty="empirical", renderer="pybullet" | None, render_mode="human" | "rgb_array_follow" | "rgb_array_birds_eye" | "rgb_array_cinematic")`
- Step API:
  - `env.step(action)` advances only the empirical dynamics core.
  - The renderer consumes `render_state` from the simulator and never owns physics in v1.
- Reset options:
  - `start_mode={"grid","random","dataset_match"}`
  - `uncertainty_mode={"nominal","empirical"}`
- Info contract:
  - `info["uncertainty"]`: sampled residual metadata and debug IDs.
  - `info["render_state"]`: pose, heading, steering angle, wheel spin, track progress, camera target.
- Tooling CLIs:
  - `build_dataset`
  - `fit_uncertainty`
  - `record_rollout`
  - `export_replay`

**Implementation Changes**
- Core simulator:
  - Use a nominal dynamic-bicycle model plus a continuous empirical residual sampler fitted from Assetto telemetry.
  - Condition the sampler on track/car ID, local curvature context, vehicle state, action, and short recent history.
  - Sample residuals for `v_x`, `v_y`, and `yaw_rate`, then integrate to the next state.
- Data pipeline:
  - Normalize Assetto plugin logs, MoTeC conversions, and public dataset laps into one canonical parquet schema.
  - Base the schema on fields already present in the local Assetto repos, especially pose, local velocities, yaw, yaw rate, controls, slip angles, and slip ratios.
- Tier 1 renderer:
  - Reuse the `racecar_gym` idea of `human` and `rgb_array_*` camera modes, but make PyBullet a visualization backend, not the physics authority.
  - Build a static 3D track mesh, wall geometry, and a kinematic racecar visual model updated from simulator state each step.
  - Support headless offscreen rendering and MP4 export for controller rollouts.
- Tier 2 renderer:
  - Export a replay package containing trajectory, steering, wheel rotation, camera script, and asset references.
  - Target Blender for offline cinematic rendering because it is open source and supports both [glTF 2.0](https://docs.blender.org/manual/en/latest/addons/import_export/scene_gltf2.html) and [USD](https://docs.blender.org/manual/en/latest/files/import_export/usd.html).
  - Ship one default cinematic template first: chase, orbit, and trackside cuts.
- Explicitly defer:
  - Direct Assetto runtime integration.
  - Real Bullet-based vehicle dynamics with uncertainty injected into the Bullet solver.
  - Multi-agent racing.

**Test Plan**
- Gymnasium API tests for `reset`, `step`, `seed`, `close`, and all `render_mode` variants.
- Determinism tests:
  - identical seeded rollouts in `nominal` mode,
  - reproducible but stochastic rollouts in `empirical` mode.
- Renderer tests:
  - headless `rgb_array_*` frames on macOS/Linux,
  - MP4 recording from Tier 1,
  - replay export then Blender import smoke test for Tier 2.
- Calibration tests on held-out Assetto laps:
  - residual quantile accuracy,
  - multimodality preservation,
  - rollout stability without excessive off-track failures.
- Consistency tests that Tier 1 render playback and Tier 2 replay export reproduce the same vehicle trajectory.

**Assumptions**
- Single-agent only in v1.
- Racing-first repo, not a general robotics framework in v1.
- Tier 1 visual quality target is “good 3D demo/debug,” not photorealistic real time.
- MetaDrive is a useful reference for a current open-source 3D Gym-style simulator, but not the base here because it is traffic-oriented rather than racing-oriented: [MetaDrive](https://github.com/metadriverse/metadrive). Waymax is a useful reference for data-driven simulation structure, but not for racing visuals: [Waymax](https://waymo.com/research/waymax/).
- Local anchors for this plan: [render utils](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/racecar_gym/racecar_gym/bullet/util.py), [Gymnasium render modes](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/racecar_gym/racecar_gym/envs/gym_api/single_agent_race.py), [Assetto telemetry fields](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/structures.py), [MoTeC channel schema](/Users/ktk/Desktop/mycode/uncertain-racecar-gym/assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaEnv/motec_loader.py).
