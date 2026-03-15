from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from uncertain_racecar_gym.scenario import DEFAULT_SCENARIO, Scenario, load_scenario
from uncertain_racecar_gym.track import TrackModel


Array = jax.Array


class JaxTrackProjection(NamedTuple):
    progress: Array
    arc_length: Array
    x: Array
    y: Array
    heading: Array
    lateral_error: Array
    curvature: Array


class JaxTrackData(NamedTuple):
    centerline: Array
    segments: Array
    segment_lengths: Array
    cumulative: Array
    arc_samples: Array
    curvature_samples: Array
    curvature_interp_arc: Array
    curvature_interp_values: Array
    width: Array
    road_half_width: Array
    length: Array


class JaxVehicleParams(NamedTuple):
    wheelbase: Array
    lf: Array
    lr: Array
    mass: Array
    inertia_z: Array
    cornering_stiffness_front: Array
    cornering_stiffness_rear: Array
    max_steer_rad: Array
    max_accel: Array
    max_brake: Array
    drag_coefficient: Array
    wheel_radius: Array


class JaxSimulationParams(NamedTuple):
    dt: Array
    max_steps: Array
    lookahead_offsets: Array
    history_template: Array


class JaxRewardParams(NamedTuple):
    progress_coef: Array
    speed_coef: Array
    lateral_error_coef: Array
    heading_error_coef: Array


class NominalJaxEnvParams(NamedTuple):
    track: JaxTrackData
    vehicle: JaxVehicleParams
    simulation: JaxSimulationParams
    reward: JaxRewardParams


class JaxRacecarState(NamedTuple):
    x: Array
    y: Array
    yaw: Array
    progress: Array
    lateral_error: Array
    heading_error: Array
    vx: Array
    vy: Array
    yaw_rate: Array
    steer: Array
    throttle: Array
    brake: Array
    wheel_rotation: Array
    lap_count: Array
    step_count: Array
    action_history: Array


class JaxResetOutput(NamedTuple):
    state: JaxRacecarState
    observation: Array


class JaxStepOutput(NamedTuple):
    state: JaxRacecarState
    observation: Array
    reward: Array
    terminated: Array
    truncated: Array


def _wrap_angle(angle: Array) -> Array:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _interp_periodic(x: Array, xp: Array, fp: Array) -> Array:
    return jnp.interp(x, xp, fp)


def _sample_track(track: JaxTrackData, progress: Array) -> JaxTrackProjection:
    arc = jnp.mod(progress, 1.0) * track.length
    index = jnp.clip(jnp.searchsorted(track.cumulative, arc, side="right") - 1, 0, track.segments.shape[0] - 1)
    segment_start = track.centerline[index]
    segment = track.segments[index]
    segment_length = track.segment_lengths[index]
    local_t = jnp.where(segment_length < 1e-9, 0.0, (arc - track.cumulative[index]) / segment_length)
    clipped_t = jnp.clip(local_t, 0.0, 1.0)
    point = segment_start + segment * clipped_t
    heading = jnp.arctan2(segment[1], segment[0])
    curvature = _interp_periodic(arc, track.curvature_interp_arc, track.curvature_interp_values)
    return JaxTrackProjection(
        progress=jnp.mod(arc / track.length, 1.0),
        arc_length=arc,
        x=point[0],
        y=point[1],
        heading=heading,
        lateral_error=jnp.asarray(0.0),
        curvature=curvature,
    )


def _project_to_track(track: JaxTrackData, x: Array, y: Array) -> JaxTrackProjection:
    point = jnp.stack([x, y])
    p0 = track.centerline
    segments = track.segments
    seg_len = track.segment_lengths
    denom = jnp.maximum(seg_len * seg_len, 1e-9)
    t = jnp.clip(jnp.sum((point - p0) * segments, axis=1) / denom, 0.0, 1.0)
    projected = p0 + segments * t[:, None]
    delta = point[None, :] - projected
    distance_sq = jnp.sum(delta * delta, axis=1)
    valid_distance_sq = jnp.where(seg_len > 1e-9, distance_sq, jnp.inf)
    index = jnp.argmin(valid_distance_sq)

    chosen_segment = segments[index]
    chosen_length = seg_len[index]
    tangent = chosen_segment / jnp.maximum(chosen_length, 1e-9)
    normal = jnp.stack([-tangent[1], tangent[0]])
    chosen_projected = projected[index]
    signed_offset = jnp.sum((point - chosen_projected) * normal)
    arc = track.cumulative[index] + t[index] * chosen_length
    curvature = _interp_periodic(arc, track.curvature_interp_arc, track.curvature_interp_values)
    heading = jnp.arctan2(chosen_segment[1], chosen_segment[0])
    return JaxTrackProjection(
        progress=jnp.mod(arc / track.length, 1.0),
        arc_length=arc,
        x=chosen_projected[0],
        y=chosen_projected[1],
        heading=heading,
        lateral_error=signed_offset,
        curvature=curvature,
    )


def _spawn_pose(track: JaxTrackData, progress: Array, lateral_error: Array, heading_error: Array) -> tuple[Array, Array, Array]:
    projection = _sample_track(track, progress)
    tangent = jnp.stack([jnp.cos(projection.heading), jnp.sin(projection.heading)])
    normal = jnp.stack([-tangent[1], tangent[0]])
    position = jnp.stack([projection.x, projection.y]) + normal * lateral_error
    yaw = _wrap_angle(projection.heading + heading_error)
    return position[0], position[1], yaw


def _lookahead_curvatures(track: JaxTrackData, progress: Array, lookahead_offsets: Array) -> Array:
    base_arc = jnp.mod(progress, 1.0) * track.length
    arcs = jnp.mod(base_arc + lookahead_offsets, track.length)
    return jnp.interp(arcs, track.curvature_interp_arc, track.curvature_interp_values)


def _initial_state(
    params: NominalJaxEnvParams,
    progress: Array,
    lateral_error: Array,
    heading_error: Array,
    speed: Array,
) -> JaxRacecarState:
    x, y, yaw = _spawn_pose(params.track, progress, lateral_error, heading_error)
    projection = _project_to_track(params.track, x, y)
    return JaxRacecarState(
        x=x,
        y=y,
        yaw=yaw,
        progress=projection.progress,
        lateral_error=projection.lateral_error,
        heading_error=_wrap_angle(yaw - projection.heading),
        vx=speed,
        vy=jnp.asarray(0.0, dtype=jnp.float32),
        yaw_rate=jnp.asarray(0.0, dtype=jnp.float32),
        steer=jnp.asarray(0.0, dtype=jnp.float32),
        throttle=jnp.asarray(0.0, dtype=jnp.float32),
        brake=jnp.asarray(0.0, dtype=jnp.float32),
        wheel_rotation=jnp.asarray(0.0, dtype=jnp.float32),
        lap_count=jnp.asarray(0, dtype=jnp.int32),
        step_count=jnp.asarray(0, dtype=jnp.int32),
        action_history=params.simulation.history_template,
    )


def _observation(params: NominalJaxEnvParams, state: JaxRacecarState) -> Array:
    curvature = _sample_track(params.track, state.progress).curvature
    lookahead_curvature = _lookahead_curvatures(params.track, state.progress, params.simulation.lookahead_offsets)
    return jnp.concatenate(
        [
            jnp.asarray(
                [
                    state.progress,
                    state.lateral_error,
                    state.heading_error,
                    state.vx,
                    state.vy,
                    state.yaw_rate,
                    curvature,
                ],
                dtype=jnp.float32,
            ),
            lookahead_curvature.astype(jnp.float32),
            state.action_history.astype(jnp.float32).reshape(-1),
        ]
    )


def _reward(params: NominalJaxEnvParams, previous_progress: Array, state: JaxRacecarState) -> Array:
    delta = state.progress - previous_progress
    delta = jnp.where(delta < -0.5, delta + 1.0, delta)
    penalty = (
        params.reward.lateral_error_coef * jnp.abs(state.lateral_error)
        + params.reward.heading_error_coef * jnp.abs(state.heading_error)
    )
    return delta * params.reward.progress_coef + params.reward.speed_coef * state.vx - penalty


def reset_nominal(
    params: NominalJaxEnvParams,
    key: Array,
    start_mode: str = "grid",
) -> JaxResetOutput:
    if start_mode == "random":
        key_progress, key_lateral, key_heading, key_speed = jax.random.split(key, 4)
        progress = jax.random.uniform(key_progress, shape=(), minval=0.0, maxval=1.0)
        lateral_error = jax.random.uniform(key_lateral, shape=(), minval=-0.2, maxval=0.2)
        heading_error = jax.random.uniform(key_heading, shape=(), minval=-0.08, maxval=0.08)
        speed = jax.random.uniform(key_speed, shape=(), minval=7.0, maxval=12.0)
    else:
        progress = jnp.asarray(0.0, dtype=jnp.float32)
        lateral_error = jnp.asarray(0.0, dtype=jnp.float32)
        heading_error = jnp.asarray(0.0, dtype=jnp.float32)
        speed = jnp.asarray(8.0, dtype=jnp.float32)
    state = _initial_state(params, progress, lateral_error, heading_error, speed)
    return JaxResetOutput(state=state, observation=_observation(params, state))


def reset_nominal_custom(
    params: NominalJaxEnvParams,
    *,
    progress: Array,
    lateral_error: Array = jnp.asarray(0.0, dtype=jnp.float32),
    heading_error: Array = jnp.asarray(0.0, dtype=jnp.float32),
    speed: Array = jnp.asarray(8.0, dtype=jnp.float32),
) -> JaxResetOutput:
    state = _initial_state(
        params,
        jnp.mod(jnp.asarray(progress, dtype=jnp.float32), 1.0),
        jnp.asarray(lateral_error, dtype=jnp.float32),
        jnp.asarray(heading_error, dtype=jnp.float32),
        jnp.maximum(jnp.asarray(speed, dtype=jnp.float32), 0.0),
    )
    return JaxResetOutput(state=state, observation=_observation(params, state))


def step_nominal(
    params: NominalJaxEnvParams,
    state: JaxRacecarState,
    action: Array,
) -> JaxStepOutput:
    action = jnp.asarray(action, dtype=jnp.float32)
    steer_cmd = jnp.clip(action[0], -1.0, 1.0)
    throttle_cmd = jnp.clip(action[1], 0.0, 1.0)
    brake_cmd = jnp.clip(action[2], 0.0, 1.0)
    dt = params.simulation.dt
    vehicle = params.vehicle

    steer = state.steer + (steer_cmd - state.steer) * jnp.minimum(1.0, dt * 8.0)
    throttle = throttle_cmd
    brake = brake_cmd

    vx_safe = jnp.maximum(jnp.abs(state.vx), 0.5)
    steer_angle = steer * vehicle.max_steer_rad
    alpha_f = steer_angle - jnp.arctan2(state.vy + vehicle.lf * state.yaw_rate, vx_safe)
    alpha_r = -jnp.arctan2(state.vy - vehicle.lr * state.yaw_rate, vx_safe)

    fyf = vehicle.cornering_stiffness_front * alpha_f
    fyr = vehicle.cornering_stiffness_rear * alpha_r
    longitudinal_acc = (
        throttle * vehicle.max_accel
        - brake * vehicle.max_brake
        - vehicle.drag_coefficient * state.vx * jnp.abs(state.vx) / jnp.maximum(vehicle.mass, 1.0)
    )

    vx_dot = longitudinal_acc + state.vy * state.yaw_rate
    vy_dot = (fyf * jnp.cos(steer_angle) + fyr) / vehicle.mass - state.vx * state.yaw_rate
    yaw_rate_dot = (vehicle.lf * fyf * jnp.cos(steer_angle) - vehicle.lr * fyr) / vehicle.inertia_z

    next_vx = jnp.maximum(0.0, state.vx + vx_dot * dt)
    next_vy = state.vy + vy_dot * dt
    next_yaw_rate = state.yaw_rate + yaw_rate_dot * dt
    wheel_rotation = state.wheel_rotation + next_vx * dt / jnp.maximum(vehicle.wheel_radius, 1e-6)

    avg_vx = 0.5 * (state.vx + next_vx)
    avg_vy = 0.5 * (state.vy + next_vy)
    avg_yaw_rate = 0.5 * (state.yaw_rate + next_yaw_rate)

    next_x = state.x + (avg_vx * jnp.cos(state.yaw) - avg_vy * jnp.sin(state.yaw)) * dt
    next_y = state.y + (avg_vx * jnp.sin(state.yaw) + avg_vy * jnp.cos(state.yaw)) * dt
    next_yaw = _wrap_angle(state.yaw + avg_yaw_rate * dt)

    projection = _project_to_track(params.track, next_x, next_y)
    lap_increment = jnp.where(projection.progress + 0.5 < state.progress, 1, 0).astype(jnp.int32)
    next_history = jnp.concatenate([state.action_history[1:], action.reshape(1, 3)], axis=0)
    next_state = JaxRacecarState(
        x=next_x,
        y=next_y,
        yaw=next_yaw,
        progress=projection.progress,
        lateral_error=projection.lateral_error,
        heading_error=_wrap_angle(next_yaw - projection.heading),
        vx=next_vx,
        vy=next_vy,
        yaw_rate=next_yaw_rate,
        steer=steer,
        throttle=throttle,
        brake=brake,
        wheel_rotation=wheel_rotation,
        lap_count=state.lap_count + lap_increment,
        step_count=state.step_count + jnp.asarray(1, dtype=jnp.int32),
        action_history=next_history,
    )
    reward = _reward(params, state.progress, next_state)
    terminated = jnp.abs(next_state.lateral_error) > params.track.road_half_width
    truncated = next_state.step_count >= params.simulation.max_steps
    return JaxStepOutput(
        state=next_state,
        observation=_observation(params, next_state),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
    )


def build_nominal_jax_params(scenario: str | Scenario | None = None) -> tuple[NominalJaxEnvParams, Scenario]:
    resolved_scenario = scenario if isinstance(scenario, Scenario) else load_scenario(scenario or DEFAULT_SCENARIO)
    track_model = TrackModel.from_config(resolved_scenario.track)

    arc_samples = np.asarray(track_model._arc_samples, dtype=np.float32)
    curvature_samples = np.asarray(track_model._curvature_samples, dtype=np.float32)
    length = np.asarray(track_model.length, dtype=np.float32)
    curvature_interp_arc = np.concatenate([arc_samples, arc_samples[1:] + float(length)], axis=0)
    curvature_interp_values = np.concatenate([curvature_samples, curvature_samples[1:]], axis=0)

    params = NominalJaxEnvParams(
        track=JaxTrackData(
            centerline=jnp.asarray(track_model.centerline, dtype=jnp.float32),
            segments=jnp.asarray(track_model._segments, dtype=jnp.float32),
            segment_lengths=jnp.asarray(track_model._segment_lengths, dtype=jnp.float32),
            cumulative=jnp.asarray(track_model._cumulative, dtype=jnp.float32),
            arc_samples=jnp.asarray(arc_samples, dtype=jnp.float32),
            curvature_samples=jnp.asarray(curvature_samples, dtype=jnp.float32),
            curvature_interp_arc=jnp.asarray(curvature_interp_arc, dtype=jnp.float32),
            curvature_interp_values=jnp.asarray(curvature_interp_values, dtype=jnp.float32),
            width=jnp.asarray(track_model.width, dtype=jnp.float32),
            road_half_width=jnp.asarray(track_model.width * 0.5, dtype=jnp.float32),
            length=jnp.asarray(track_model.length, dtype=jnp.float32),
        ),
        vehicle=JaxVehicleParams(
            wheelbase=jnp.asarray(resolved_scenario.vehicle.wheelbase, dtype=jnp.float32),
            lf=jnp.asarray(resolved_scenario.vehicle.lf, dtype=jnp.float32),
            lr=jnp.asarray(resolved_scenario.vehicle.lr, dtype=jnp.float32),
            mass=jnp.asarray(resolved_scenario.vehicle.mass, dtype=jnp.float32),
            inertia_z=jnp.asarray(resolved_scenario.vehicle.inertia_z, dtype=jnp.float32),
            cornering_stiffness_front=jnp.asarray(resolved_scenario.vehicle.cornering_stiffness_front, dtype=jnp.float32),
            cornering_stiffness_rear=jnp.asarray(resolved_scenario.vehicle.cornering_stiffness_rear, dtype=jnp.float32),
            max_steer_rad=jnp.asarray(resolved_scenario.vehicle.max_steer_rad, dtype=jnp.float32),
            max_accel=jnp.asarray(resolved_scenario.vehicle.max_accel, dtype=jnp.float32),
            max_brake=jnp.asarray(resolved_scenario.vehicle.max_brake, dtype=jnp.float32),
            drag_coefficient=jnp.asarray(resolved_scenario.vehicle.drag_coefficient, dtype=jnp.float32),
            wheel_radius=jnp.asarray(resolved_scenario.vehicle.wheel_radius, dtype=jnp.float32),
        ),
        simulation=JaxSimulationParams(
            dt=jnp.asarray(resolved_scenario.simulation.dt, dtype=jnp.float32),
            max_steps=jnp.asarray(resolved_scenario.simulation.max_steps, dtype=jnp.int32),
            lookahead_offsets=jnp.asarray(
                np.arange(resolved_scenario.simulation.lookahead_points, dtype=np.float32)
                * float(resolved_scenario.simulation.lookahead_spacing_m),
                dtype=jnp.float32,
            ),
            history_template=jnp.zeros((resolved_scenario.uncertainty.history_length, 3), dtype=jnp.float32),
        ),
        reward=JaxRewardParams(
            progress_coef=jnp.asarray(resolved_scenario.reward.progress_coef, dtype=jnp.float32),
            speed_coef=jnp.asarray(resolved_scenario.reward.speed_coef, dtype=jnp.float32),
            lateral_error_coef=jnp.asarray(resolved_scenario.reward.lateral_error_coef, dtype=jnp.float32),
            heading_error_coef=jnp.asarray(resolved_scenario.reward.heading_error_coef, dtype=jnp.float32),
        ),
    )
    return params, resolved_scenario


class NominalJaxRacecarEnv:
    def __init__(self, scenario: str | Scenario | None = None):
        self.params, self.scenario = build_nominal_jax_params(scenario)
        self.action_size = 3
        self.observation_size = int(7 + len(self.params.simulation.lookahead_offsets) + self.params.simulation.history_template.size)
        self.reset_grid_jit = jax.jit(lambda key: reset_nominal(self.params, key, start_mode="grid"))
        self.reset_random_jit = jax.jit(lambda key: reset_nominal(self.params, key, start_mode="random"))
        self.reset_custom_jit = jax.jit(
            lambda progress, lateral_error, heading_error, speed: reset_nominal_custom(
                self.params,
                progress=progress,
                lateral_error=lateral_error,
                heading_error=heading_error,
                speed=speed,
            )
        )
        self.step_jit = jax.jit(lambda state, action: step_nominal(self.params, state, action))

    def reset(self, key: Array, start_mode: str = "grid") -> JaxResetOutput:
        return reset_nominal(self.params, key, start_mode=start_mode)

    def reset_custom(
        self,
        *,
        progress: float,
        lateral_error: float = 0.0,
        heading_error: float = 0.0,
        speed: float = 8.0,
    ) -> JaxResetOutput:
        return reset_nominal_custom(
            self.params,
            progress=jnp.asarray(progress, dtype=jnp.float32),
            lateral_error=jnp.asarray(lateral_error, dtype=jnp.float32),
            heading_error=jnp.asarray(heading_error, dtype=jnp.float32),
            speed=jnp.asarray(speed, dtype=jnp.float32),
        )

    def step(self, state: JaxRacecarState, action: Array) -> JaxStepOutput:
        return step_nominal(self.params, state, action)
