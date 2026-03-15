from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import wandb

from uncertain_racecar_gym.common import ensure_dir
from uncertain_racecar_gym.controllers import CenterlineDriver, ProfiledCenterlineDriver
from uncertain_racecar_gym.env import UncertainRacecarEnv
from uncertain_racecar_gym.jax_env import JaxRacecarState, NominalJaxEnvParams, NominalJaxRacecarEnv, reset_nominal, step_nominal
from uncertain_racecar_gym.scenario import load_scenario
from uncertain_racecar_gym.track import TrackModel


Array = jax.Array


@dataclass(slots=True)
class PPOConfig:
    scenario: str
    output_dir: str = "output"
    run_name: str | None = None
    seed: int = 7
    total_timesteps: int = 2_000_000
    num_envs: int = 64
    num_steps: int = 256
    num_minibatches: int = 16
    update_epochs: int = 6
    learning_rate: float = 3e-4
    gamma: float = 0.992
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    eval_interval_updates: int = 10
    eval_episodes: int = 8
    hidden_sizes: tuple[int, int] = (256, 256)
    initial_log_std: float = -0.7
    start_mode: str = "random"
    wandb_project: str = "uncertain-racecar-gym-rl"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_mode: str = "online"
    disable_wandb: bool = False
    gaussian_std: tuple[float, float, float, float] = (0.24, 0.16, 0.10, 0.025)
    empirical_artifact: str | None = None
    calibration_artifact: str | None = None
    render_steps: int = 900
    render_mode: str = "rgb_array_follow"
    render_width: int = 1280
    render_height: int = 720
    render_after_training: bool = False
    bc_canonical_dataset: str | None = None
    bc_episodes: int = 28
    bc_steps_per_episode: int = 1200
    bc_epochs: int = 40
    bc_batch_size: int = 2048
    bc_learning_rate: float = 1e-3
    bc_max_rows: int = 120_000


class RunningNorm(NamedTuple):
    count: Array
    mean: Array
    m2: Array


class ActorCriticParams(NamedTuple):
    actor_layers: tuple[dict[str, Array], ...]
    actor_mean: dict[str, Array]
    critic_layers: tuple[dict[str, Array], ...]
    critic_out: dict[str, Array]
    log_std: Array


class RolloutBatch(NamedTuple):
    observations: Array
    actions: Array
    log_probs: Array
    rewards: Array
    dones: Array
    values: Array


class UpdateState(NamedTuple):
    env_state: JaxRacecarState
    observation: Array
    rng_key: Array
    episode_returns: Array
    episode_lengths: Array


class EvalSummary(NamedTuple):
    mean_return: float
    mean_length: float
    mean_progress: float
    lap_rate: float
    offtrack_rate: float


def _init_dense(key: Array, in_dim: int, out_dim: int, gain: float) -> dict[str, Array]:
    weight = jax.nn.initializers.orthogonal(scale=gain)(key, (in_dim, out_dim), dtype=jnp.float32)
    bias = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": weight, "b": bias}


def init_actor_critic_params(key: Array, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...], initial_log_std: float) -> ActorCriticParams:
    keys = jax.random.split(key, len(hidden_sizes) * 2 + 2)
    actor_layers: list[dict[str, Array]] = []
    critic_layers: list[dict[str, Array]] = []
    last_dim = obs_dim
    for idx, hidden in enumerate(hidden_sizes):
        actor_layers.append(_init_dense(keys[idx], last_dim, hidden, gain=math.sqrt(2.0)))
        critic_layers.append(_init_dense(keys[idx + len(hidden_sizes)], last_dim, hidden, gain=math.sqrt(2.0)))
        last_dim = hidden
    actor_mean = _init_dense(keys[-2], last_dim, action_dim, gain=0.01)
    if action_dim == 3:
        actor_mean["b"] = jnp.asarray([0.0, 0.15, -1.2], dtype=jnp.float32)
    critic_out = _init_dense(keys[-1], last_dim, 1, gain=1.0)
    return ActorCriticParams(
        actor_layers=tuple(actor_layers),
        actor_mean=actor_mean,
        critic_layers=tuple(critic_layers),
        critic_out=critic_out,
        log_std=jnp.full((action_dim,), float(initial_log_std), dtype=jnp.float32),
    )


def _dense(params: dict[str, Array], x: Array) -> Array:
    return jnp.dot(x, params["w"]) + params["b"]


def _mlp(layers: tuple[dict[str, Array], ...], x: Array) -> Array:
    for layer in layers:
        x = jnp.tanh(_dense(layer, x))
    return x


def _normalize_obs(observation: Array, obs_norm: RunningNorm) -> Array:
    variance = jnp.maximum(obs_norm.m2 / jnp.maximum(obs_norm.count - 1.0, 1.0), 1e-6)
    normalized = (observation - obs_norm.mean) / jnp.sqrt(variance)
    return jnp.clip(normalized, -10.0, 10.0)


def actor_critic_apply(params: ActorCriticParams, observation: Array, obs_norm: RunningNorm) -> tuple[Array, Array, Array]:
    normalized = _normalize_obs(observation, obs_norm)
    actor_hidden = _mlp(params.actor_layers, normalized)
    critic_hidden = _mlp(params.critic_layers, normalized)
    mean = _dense(params.actor_mean, actor_hidden)
    value = _dense(params.critic_out, critic_hidden).squeeze(-1)
    return mean, params.log_std, value


def _gaussian_log_prob(pre_tanh_action: Array, mean: Array, log_std: Array) -> Array:
    variance = jnp.exp(2.0 * log_std)
    log_prob = -0.5 * (((pre_tanh_action - mean) ** 2) / variance + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    return jnp.sum(log_prob, axis=-1)


def _squash_log_prob(pre_tanh_action: Array, mean: Array, log_std: Array) -> Array:
    action = jnp.tanh(pre_tanh_action)
    correction = jnp.sum(jnp.log(jnp.maximum(1.0 - action * action, 1e-6)), axis=-1)
    return _gaussian_log_prob(pre_tanh_action, mean, log_std) - correction


def _normalized_to_env_action(normalized_action: Array) -> Array:
    steer = jnp.clip(normalized_action[..., 0], -1.0, 1.0)
    throttle = jnp.clip(normalized_action[..., 1], 0.0, 1.0)
    brake = jnp.clip(normalized_action[..., 2], 0.0, 1.0)
    return jnp.stack([steer, throttle, brake], axis=-1)


def sample_policy_action(params: ActorCriticParams, obs_norm: RunningNorm, observation: Array, key: Array) -> tuple[Array, Array, Array, Array]:
    mean, log_std, value = actor_critic_apply(params, observation, obs_norm)
    key, noise_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, shape=mean.shape)
    pre_tanh = mean + jnp.exp(log_std) * noise
    normalized_action = jnp.tanh(pre_tanh)
    log_prob = _squash_log_prob(pre_tanh, mean, log_std)
    env_action = _normalized_to_env_action(normalized_action)
    return key, env_action, normalized_action, log_prob, value


def deterministic_policy_action(params: ActorCriticParams, obs_norm: RunningNorm, observation: Array) -> tuple[Array, Array]:
    mean, _, value = actor_critic_apply(params, observation, obs_norm)
    normalized_action = jnp.tanh(mean)
    env_action = _normalized_to_env_action(normalized_action)
    return env_action, value


def update_running_norm(obs_norm: RunningNorm, observations: np.ndarray) -> RunningNorm:
    batch = np.asarray(observations, dtype=np.float64).reshape(-1, observations.shape[-1])
    if batch.size == 0:
        return obs_norm
    batch_count = float(batch.shape[0])
    batch_mean = batch.mean(axis=0)
    centered = batch - batch_mean
    batch_m2 = np.square(centered).sum(axis=0)

    count = float(obs_norm.count)
    mean = np.asarray(obs_norm.mean, dtype=np.float64)
    m2 = np.asarray(obs_norm.m2, dtype=np.float64)

    delta = batch_mean - mean
    total_count = count + batch_count
    new_mean = mean + delta * (batch_count / total_count)
    new_m2 = m2 + batch_m2 + np.square(delta) * count * batch_count / total_count
    return RunningNorm(
        count=jnp.asarray(total_count, dtype=jnp.float32),
        mean=jnp.asarray(new_mean, dtype=jnp.float32),
        m2=jnp.asarray(new_m2, dtype=jnp.float32),
    )


def init_running_norm(obs_dim: int) -> RunningNorm:
    return RunningNorm(
        count=jnp.asarray(1e-4, dtype=jnp.float32),
        mean=jnp.zeros((obs_dim,), dtype=jnp.float32),
        m2=jnp.ones((obs_dim,), dtype=jnp.float32),
    )


def build_train_step(env_params: NominalJaxEnvParams, start_mode: str):
    reset_many = jax.jit(jax.vmap(lambda key: reset_nominal(env_params, key, start_mode=start_mode)))
    step_many = jax.jit(jax.vmap(lambda state, action: step_nominal(env_params, state, action)))

    @jax.jit
    def rollout_step(params: ActorCriticParams, obs_norm: RunningNorm, update_state: UpdateState) -> tuple[UpdateState, RolloutBatch, Array, Array]:
        keys = jax.random.split(update_state.rng_key, update_state.observation.shape[0] + 1)
        next_rng = keys[0]
        sample_keys = keys[1:]
        sample_fn = jax.vmap(lambda obs, key: sample_policy_action(params, obs_norm, obs, key), in_axes=(0, 0))
        _, env_action, norm_action, log_prob, value = sample_fn(update_state.observation, sample_keys)
        step_out = step_many(update_state.env_state, env_action)
        done = jnp.logical_or(step_out.terminated, step_out.truncated)

        new_returns = update_state.episode_returns + step_out.reward
        new_lengths = update_state.episode_lengths + 1
        finished_returns = jnp.where(done, new_returns, jnp.nan)
        finished_lengths = jnp.where(done, new_lengths, jnp.nan)

        reset_keys = jax.random.split(next_rng, update_state.observation.shape[0] + 1)
        carry_rng = reset_keys[0]
        reset_out = reset_many(reset_keys[1:])

        def select_done(stepped: Array, reset: Array) -> Array:
            mask = done.reshape(done.shape + (1,) * max(stepped.ndim - done.ndim, 0))
            return jnp.where(mask, reset, stepped)

        next_state = jax.tree.map(select_done, step_out.state, reset_out.state)
        next_observation = jnp.where(done[:, None], reset_out.observation, step_out.observation)
        carry_returns = jnp.where(done, 0.0, new_returns)
        carry_lengths = jnp.where(done, 0.0, new_lengths)

        next_update_state = UpdateState(
            env_state=next_state,
            observation=next_observation,
            rng_key=carry_rng,
            episode_returns=carry_returns,
            episode_lengths=carry_lengths,
        )
        batch = RolloutBatch(
            observations=update_state.observation,
            actions=norm_action,
            log_probs=log_prob,
            rewards=step_out.reward,
            dones=done.astype(jnp.float32),
            values=value,
        )
        return next_update_state, batch, finished_returns, finished_lengths

    return reset_many, step_many, rollout_step


def compute_gae(rewards: Array, dones: Array, values: Array, last_value: Array, gamma: float, gae_lambda: float) -> tuple[Array, Array]:
    advantages = []
    gae = jnp.zeros_like(last_value)
    next_value = last_value
    for t in range(rewards.shape[0] - 1, -1, -1):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.append(gae)
        next_value = values[t]
    advantages = jnp.stack(advantages[::-1], axis=0)
    returns = advantages + values
    return advantages, returns


def minibatch_indices(key: Array, batch_size: int, minibatch_size: int) -> Array:
    permutation = jax.random.permutation(key, batch_size)
    return permutation.reshape((-1, minibatch_size))


def ppo_loss(
    params: ActorCriticParams,
    obs_norm: RunningNorm,
    observations: Array,
    actions: Array,
    old_log_probs: Array,
    old_values: Array,
    advantages: Array,
    returns: Array,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[Array, dict[str, Array]]:
    mean, log_std, value = actor_critic_apply(params, observations, obs_norm)
    clipped_action = jnp.clip(actions, -0.999999, 0.999999)
    pre_tanh = jnp.arctanh(clipped_action)
    new_log_probs = _squash_log_prob(pre_tanh, mean, log_std)
    entropy = jnp.sum(log_std + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)), axis=-1)

    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss_1 = -normalized_advantages * ratio
    pg_loss_2 = -normalized_advantages * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    policy_loss = jnp.maximum(pg_loss_1, pg_loss_2).mean()

    value_clipped = old_values + jnp.clip(value - old_values, -clip_coef, clip_coef)
    value_loss_unclipped = (value - returns) ** 2
    value_loss_clipped = (value_clipped - returns) ** 2
    value_loss = 0.5 * jnp.maximum(value_loss_unclipped, value_loss_clipped).mean()

    entropy_loss = entropy.mean()
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32))
    return total_loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_loss,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "mean_value": value.mean(),
    }


def make_update_fn(optimizer: optax.GradientTransformation):
    @jax.jit
    def update_step(
        params: ActorCriticParams,
        opt_state: optax.OptState,
        obs_norm: RunningNorm,
        observations: Array,
        actions: Array,
        old_log_probs: Array,
        old_values: Array,
        advantages: Array,
        returns: Array,
        clip_coef: float,
        vf_coef: float,
        ent_coef: float,
    ) -> tuple[ActorCriticParams, optax.OptState, dict[str, Array]]:
        (loss, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
            params,
            obs_norm,
            observations,
            actions,
            old_log_probs,
            old_values,
            advantages,
            returns,
            clip_coef,
            vf_coef,
            ent_coef,
        )
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        grad_norm = optax.tree.norm(grads)
        metrics = {**metrics, "loss": loss, "grad_norm": grad_norm}
        return next_params, next_opt_state, metrics

    return update_step


def evaluate_policy(
    env: NominalJaxRacecarEnv,
    params: ActorCriticParams,
    obs_norm: RunningNorm,
    *,
    episodes: int,
    seed: int,
    start_mode: str = "grid",
    max_steps: int | None = None,
) -> EvalSummary:
    returns: list[float] = []
    lengths: list[int] = []
    progresses: list[float] = []
    laps: list[float] = []
    offtracks: list[float] = []
    current_max_steps = max_steps or int(env.scenario.simulation.max_steps)
    for episode in range(episodes):
        reset_out = env.reset(jax.random.PRNGKey(seed + episode), start_mode=start_mode)
        state = reset_out.state
        observation = reset_out.observation
        episode_return = 0.0
        offtrack = 0.0
        for step_idx in range(current_max_steps):
            env_action, _ = deterministic_policy_action(params, obs_norm, observation)
            step_out = env.step_jit(state, env_action)
            episode_return += float(step_out.reward)
            state = step_out.state
            observation = step_out.observation
            if bool(step_out.terminated):
                offtrack = 1.0
                lengths.append(step_idx + 1)
                break
            if bool(step_out.truncated):
                lengths.append(step_idx + 1)
                break
        else:
            lengths.append(current_max_steps)
        returns.append(episode_return)
        progresses.append(float(state.progress) + float(state.lap_count))
        laps.append(float(state.lap_count > 0))
        offtracks.append(offtrack)
    return EvalSummary(
        mean_return=float(np.mean(returns)),
        mean_length=float(np.mean(lengths)),
        mean_progress=float(np.mean(progresses)),
        lap_rate=float(np.mean(laps)),
        offtrack_rate=float(np.mean(offtracks)),
    )


def save_checkpoint(path: Path, *, config: PPOConfig, params: ActorCriticParams, obs_norm: RunningNorm, train_history: list[dict[str, Any]]) -> Path:
    payload = {
        "config": asdict(config),
        "params": jax.device_get(params),
        "obs_norm": jax.device_get(obs_norm),
        "train_history": train_history,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def render_policy_rollout(
    *,
    scenario: str,
    checkpoint_path: Path,
    output_dir: Path,
    video_name: str,
    uncertainty_mode: str | None,
    gaussian_std: tuple[float, float, float, float],
    uncertainty_artifact: str | None,
    calibration_artifact: str | None,
    steps: int,
    seed: int,
    render_mode: str,
    wandb_run: Any | None,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    params = checkpoint["params"]
    obs_norm = checkpoint["obs_norm"]
    env = UncertainRacecarEnv(
        scenario=scenario,
        uncertainty=uncertainty_mode,
        uncertainty_artifact=uncertainty_artifact,
        calibration_artifact=calibration_artifact,
        apply_mean_correction=bool(calibration_artifact),
        gaussian_noise_std=gaussian_std,
        renderer="pybullet",
        render_mode=render_mode,
        output_dir=output_dir,
    )
    observation, _ = env.reset(seed=seed, options={"start_mode": "grid"})
    rollout_rows: list[dict[str, Any]] = []
    total_reward = 0.0
    terminated = False
    truncated = False
    video_path = output_dir / f"{video_name}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(video_path, fps=round(1.0 / env.scenario.simulation.dt))
    for step_idx in range(steps):
        obs_array = jnp.asarray(observation, dtype=jnp.float32)
        env_action, _ = deterministic_policy_action(params, obs_norm, obs_array)
        action_np = np.asarray(env_action, dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action_np)
        frame = env.render()
        if frame is not None:
            writer.append_data(frame.astype(np.uint8))
        total_reward += float(reward)
        rollout_rows.append(
            {
                "step": step_idx,
                "reward": float(reward),
                "progress": float(info["state"]["progress"]),
                "speed": float(info["state"]["speed"]),
                "x": float(info["state"]["x"]),
                "y": float(info["state"]["y"]),
                "lateral_error": float(env._state.lateral_error if env._state is not None else 0.0),
                "heading_error": float(env._state.heading_error if env._state is not None else 0.0),
                "mode": uncertainty_mode or "nominal",
            }
        )
        if terminated or truncated:
            break
    writer.close()
    env.close()

    json_path = output_dir / f"{video_name}.json"
    md_path = output_dir / f"{video_name}.md"
    json_path.write_text(json.dumps(rollout_rows, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                f"# {video_name}",
                "",
                f"- Uncertainty mode: `{uncertainty_mode or 'nominal'}`",
                f"- Steps executed: `{len(rollout_rows)}`",
                f"- Terminated: `{terminated}`",
                f"- Truncated: `{truncated}`",
                f"- Total reward: `{total_reward:.3f}`",
                f"- Final progress: `{rollout_rows[-1]['progress'] if rollout_rows else 0.0:.4f}`",
                f"- Video: `{video_path.name}`",
                f"- Rollout JSON: `{json_path.name}`",
            ]
        ),
        encoding="utf-8",
    )
    if wandb_run is not None and video_path.exists():
        wandb_run.log({f"videos/{video_name}": wandb.Video(str(video_path), fps=round(1.0 / env.scenario.simulation.dt), format="mp4")})
    return {
        "video_path": video_path.as_posix(),
        "json_path": json_path.as_posix(),
        "markdown_path": md_path.as_posix(),
        "total_reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "steps": len(rollout_rows),
        "final_progress": float(rollout_rows[-1]["progress"]) if rollout_rows else 0.0,
    }


def _training_plots(output_dir: Path, history: list[dict[str, Any]]) -> dict[str, str]:
    if not history:
        return {}
    steps = np.asarray([row["env_steps"] for row in history], dtype=float)
    eval_returns = np.asarray([row["eval_return"] for row in history], dtype=float)
    eval_progress = np.asarray([row["eval_progress"] for row in history], dtype=float)
    policy_loss = np.asarray([row["policy_loss"] for row in history], dtype=float)
    value_loss = np.asarray([row["value_loss"] for row in history], dtype=float)
    entropy = np.asarray([row["entropy"] for row in history], dtype=float)

    paths: dict[str, str] = {}

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    axes[0].plot(steps, eval_returns, color="#113f67", linewidth=2.0)
    axes[0].set_title("Evaluation Return")
    axes[0].set_xlabel("Environment Steps")
    axes[0].set_ylabel("Return")

    axes[1].plot(steps, eval_progress, color="#5c5470", linewidth=2.0)
    axes[1].set_title("Evaluation Progress")
    axes[1].set_xlabel("Environment Steps")
    axes[1].set_ylabel("Progress + Laps")

    axes[2].plot(steps, policy_loss, label="policy_loss", color="#c06c84")
    axes[2].plot(steps, value_loss, label="value_loss", color="#355c7d")
    axes[2].plot(steps, entropy, label="entropy", color="#f67280")
    axes[2].set_title("Optimization Diagnostics")
    axes[2].set_xlabel("Environment Steps")
    axes[2].legend()
    learning_curve_path = output_dir / "training_curves.png"
    fig.savefig(learning_curve_path, dpi=180)
    plt.close(fig)
    paths["training_curves"] = learning_curve_path.as_posix()
    return paths


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> Path:
    if not history:
        return path
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    return path


def _write_summary_markdown(
    path: Path,
    *,
    config: PPOConfig,
    best_eval: EvalSummary,
    final_eval: EvalSummary,
    noisy_evals: dict[str, dict[str, Any]],
    checkpoint_path: Path,
    history_csv: Path,
    plot_paths: dict[str, str],
    wandb_url: str | None,
) -> Path:
    lines = [
        "# Nominal PPO Training Summary",
        "",
        "## Setup",
        "",
        f"- Scenario: `{config.scenario}`",
        f"- Seed: `{config.seed}`",
        f"- Timesteps: `{config.total_timesteps}`",
        f"- Num envs: `{config.num_envs}`",
        f"- Rollout horizon: `{config.num_steps}`",
        f"- Update epochs: `{config.update_epochs}`",
        f"- Minibatches: `{config.num_minibatches}`",
        f"- Behavior cloning epochs: `{config.bc_epochs}`",
        f"- Behavior cloning dataset: `{config.bc_canonical_dataset or ('profiled nominal expert' if config.bc_epochs > 0 else 'none')}`",
        f"- Checkpoint: `{checkpoint_path.name}`",
        f"- History CSV: `{history_csv.name}`",
    ]
    if wandb_url:
        lines.extend(["", f"- wandb run: {wandb_url}"])
    if plot_paths:
        lines.extend(["", "## Training Curves", ""])
        for name, plot_path in plot_paths.items():
            lines.append(f"![{name}]({Path(plot_path).resolve().as_posix()})")
            lines.append("")
    lines.extend(
        [
            "## Evaluation",
            "",
            f"- Best nominal eval return: `{best_eval.mean_return:.3f}`",
            f"- Best nominal eval progress: `{best_eval.mean_progress:.3f}`",
            f"- Best nominal lap rate: `{best_eval.lap_rate:.3f}`",
            f"- Best nominal off-track rate: `{best_eval.offtrack_rate:.3f}`",
            f"- Final nominal eval return: `{final_eval.mean_return:.3f}`",
            f"- Final nominal eval progress: `{final_eval.mean_progress:.3f}`",
            "",
            "## Noise Stress Test",
            "",
        ]
    )
    for name, metrics in noisy_evals.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Total reward: `{metrics['total_reward']:.3f}`",
                f"- Final progress: `{metrics['final_progress']:.4f}`",
                f"- Steps: `{metrics['steps']}`",
                f"- Terminated: `{metrics['terminated']}`",
                f"- Truncated: `{metrics['truncated']}`",
                f"- Video: `{Path(metrics['video_path']).name}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_eval_summary_markdown(
    path: Path,
    *,
    checkpoint_path: Path,
    evaluation_rows: dict[str, dict[str, Any]],
    wandb_url: str | None,
) -> Path:
    lines = [
        "# PPO Policy Evaluation",
        "",
        f"- Checkpoint: `{checkpoint_path.name}`",
    ]
    if wandb_url:
        lines.append(f"- wandb run: {wandb_url}")
    lines.extend(["", "## Rollouts", ""])
    for name, metrics in evaluation_rows.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Reward: `{metrics['total_reward']:.3f}`",
                f"- Final progress: `{metrics['final_progress']:.4f}`",
                f"- Steps: `{metrics['steps']}`",
                f"- Terminated: `{metrics['terminated']}`",
                f"- Truncated: `{metrics['truncated']}`",
                f"- Video: `{Path(metrics['video_path']).name}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_behavior_cloning_dataset(
    *,
    scenario: str,
    canonical_dataset: str | None,
    output_dir: Path,
    episodes: int,
    steps_per_episode: int,
    seed: int,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, Path]:
    output_dir = ensure_dir(output_dir)
    if canonical_dataset:
        scenario_obj = load_scenario(scenario)
        track = TrackModel.from_config(scenario_obj.track)
        frame = pd.read_parquet(
            canonical_dataset,
            columns=[
                "trajectory_id",
                "frame_index",
                "progress",
                "lateral_error",
                "heading_error",
                "vx",
                "vy",
                "yaw_rate",
                "steer",
                "throttle",
                "brake",
            ],
        ).sort_values(["trajectory_id", "frame_index"])
        if len(frame) > 0 and len(frame) > max_rows:
            stride = max(1, len(frame) // max_rows)
            frame = frame.iloc[::stride].copy()
        history_length = scenario_obj.uncertainty.history_length
        lookahead_count = scenario_obj.simulation.lookahead_points
        lookahead_spacing = scenario_obj.simulation.lookahead_spacing_m
        observations_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []
        zeros = np.zeros((history_length, 3), dtype=np.float32)
        for _, trajectory in frame.groupby("trajectory_id", sort=False):
            history = zeros.copy()
            for row in trajectory.itertuples(index=False):
                lookahead = track.lookahead_curvatures(
                    float(row.progress),
                    count=lookahead_count,
                    spacing_m=lookahead_spacing,
                ).astype(np.float32)
                obs = np.concatenate(
                    [
                        np.array(
                            [
                                float(row.progress),
                                float(row.lateral_error),
                                float(row.heading_error),
                                float(row.vx),
                                float(row.vy),
                                float(row.yaw_rate),
                                float(track.sample(float(row.progress)).curvature),
                            ],
                            dtype=np.float32,
                        ),
                        lookahead,
                        history.reshape(-1),
                    ]
                )
                action = np.array([float(row.steer), float(row.throttle), float(row.brake)], dtype=np.float32)
                observations_list.append(obs)
                actions_list.append(action)
                history = np.concatenate([history[1:], action.reshape(1, 3)], axis=0)
        observations = np.asarray(observations_list, dtype=np.float32)
        actions = np.asarray(actions_list, dtype=np.float32)
    else:
        env = UncertainRacecarEnv(scenario=scenario, uncertainty=None)
        driver = CenterlineDriver(target_speed=14.0)
        observations_list = []
        actions_list = []
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode, options={"start_mode": "random"})
            for _ in range(steps_per_episode):
                assert env._state is not None
                action = driver.act(env._state, env.track).astype(np.float32)
                observations_list.append(np.asarray(observation, dtype=np.float32))
                actions_list.append(np.asarray(action, dtype=np.float32))
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        env.close()
        observations = np.asarray(observations_list, dtype=np.float32)
        actions = np.asarray(actions_list, dtype=np.float32)

    dataset_path = output_dir / "bc_dataset.npz"
    np.savez_compressed(dataset_path, observations=observations, actions=actions)
    return observations, actions, dataset_path


def _actor_env_action(params: ActorCriticParams, obs_norm: RunningNorm, observations: Array) -> Array:
    mean, _, _ = actor_critic_apply(params, observations, obs_norm)
    return _normalized_to_env_action(jnp.tanh(mean))


def pretrain_actor_with_behavior_cloning(
    *,
    params: ActorCriticParams,
    obs_norm: RunningNorm,
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> ActorCriticParams:
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    obs_array = jnp.asarray(observations, dtype=jnp.float32)
    act_array = jnp.asarray(actions, dtype=jnp.float32)
    n = observations.shape[0]

    @jax.jit
    def bc_step(current_params: ActorCriticParams, current_opt_state: optax.OptState, batch_obs: Array, batch_act: Array) -> tuple[ActorCriticParams, optax.OptState, Array]:
        def loss_fn(model_params: ActorCriticParams) -> Array:
            pred = _actor_env_action(model_params, obs_norm, batch_obs)
            return jnp.mean(jnp.square(pred - batch_act))

        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, loss

    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        permutation = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_indices = permutation[start : start + batch_size]
            params, opt_state, _ = bc_step(params, opt_state, obs_array[batch_indices], act_array[batch_indices])
    return params


def train_ppo(config: PPOConfig) -> dict[str, Any]:
    scenario_path = Path(config.scenario).resolve()
    if config.run_name is None:
        config.run_name = f"ppo_nominal_{scenario_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = ensure_dir(Path(config.output_dir) / config.run_name)

    env = NominalJaxRacecarEnv(scenario_path)
    obs_dim = env.observation_size
    action_dim = env.action_size
    batch_size = config.num_envs * config.num_steps
    minibatch_size = batch_size // config.num_minibatches
    num_updates = config.total_timesteps // batch_size

    rng = jax.random.PRNGKey(config.seed)
    rng, network_key, reset_key = jax.random.split(rng, 3)
    params = init_actor_critic_params(network_key, obs_dim, action_dim, config.hidden_sizes, config.initial_log_std)
    obs_norm = init_running_norm(obs_dim)
    learning_rate_schedule = optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=config.learning_rate * 0.2,
        transition_steps=max(num_updates, 1),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate_schedule),
    )
    opt_state = optimizer.init(params)
    update_fn = make_update_fn(optimizer)

    reset_many, _, rollout_step = build_train_step(env.params, config.start_mode)
    reset_keys = jax.random.split(reset_key, config.num_envs)
    reset_out = reset_many(reset_keys)
    update_state = UpdateState(
        env_state=reset_out.state,
        observation=reset_out.observation,
        rng_key=rng,
        episode_returns=jnp.zeros((config.num_envs,), dtype=jnp.float32),
        episode_lengths=jnp.zeros((config.num_envs,), dtype=jnp.float32),
    )

    bc_dataset_path = None
    if config.bc_epochs > 0:
        bc_observations, bc_actions, bc_dataset_path = build_behavior_cloning_dataset(
            scenario=config.scenario,
            canonical_dataset=config.bc_canonical_dataset,
            output_dir=output_dir,
            episodes=config.bc_episodes,
            steps_per_episode=config.bc_steps_per_episode,
            seed=config.seed + 123,
            max_rows=config.bc_max_rows,
        )
        obs_norm = update_running_norm(obs_norm, bc_observations)
        params = pretrain_actor_with_behavior_cloning(
            params=params,
            obs_norm=obs_norm,
            observations=bc_observations,
            actions=bc_actions,
            epochs=config.bc_epochs,
            batch_size=config.bc_batch_size,
            learning_rate=config.bc_learning_rate,
            seed=config.seed + 456,
        )
        opt_state = optimizer.init(params)

    wandb_run = None
    if not config.disable_wandb:
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=config.run_name,
            mode=config.wandb_mode,
            config={
                **asdict(config),
                "observation_dim": obs_dim,
                "action_dim": action_dim,
                "batch_size": batch_size,
                "minibatch_size": minibatch_size,
                "num_updates": num_updates,
                "bc_dataset": bc_dataset_path.as_posix() if bc_dataset_path is not None else None,
            },
            dir=output_dir.as_posix(),
        )

    history: list[dict[str, Any]] = []
    best_eval = EvalSummary(mean_return=-1e9, mean_length=0.0, mean_progress=0.0, lap_rate=0.0, offtrack_rate=1.0)
    best_params = params
    best_obs_norm = obs_norm

    last_eval = evaluate_policy(env, params, obs_norm, episodes=config.eval_episodes, seed=config.seed + 50_000, start_mode="grid")
    if last_eval.mean_return > best_eval.mean_return:
        best_eval = last_eval

    for update in range(num_updates):
        rollout_observations = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []
        finished_episode_returns: list[float] = []
        finished_episode_lengths: list[float] = []

        for _ in range(config.num_steps):
            update_state, batch, episode_return, episode_length = rollout_step(params, obs_norm, update_state)
            rollout_observations.append(np.asarray(batch.observations))
            rollout_actions.append(np.asarray(batch.actions))
            rollout_log_probs.append(np.asarray(batch.log_probs))
            rollout_rewards.append(np.asarray(batch.rewards))
            rollout_dones.append(np.asarray(batch.dones))
            rollout_values.append(np.asarray(batch.values))
            valid_returns = np.asarray(episode_return)
            valid_lengths = np.asarray(episode_length)
            finished_episode_returns.extend(valid_returns[~np.isnan(valid_returns)].tolist())
            finished_episode_lengths.extend(valid_lengths[~np.isnan(valid_lengths)].tolist())

        obs_norm = update_running_norm(obs_norm, np.asarray(rollout_observations))
        last_value = actor_critic_apply(params, update_state.observation, obs_norm)[2]

        rollout = RolloutBatch(
            observations=jnp.asarray(np.stack(rollout_observations), dtype=jnp.float32),
            actions=jnp.asarray(np.stack(rollout_actions), dtype=jnp.float32),
            log_probs=jnp.asarray(np.stack(rollout_log_probs), dtype=jnp.float32),
            rewards=jnp.asarray(np.stack(rollout_rewards), dtype=jnp.float32),
            dones=jnp.asarray(np.stack(rollout_dones), dtype=jnp.float32),
            values=jnp.asarray(np.stack(rollout_values), dtype=jnp.float32),
        )
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.dones,
            rollout.values,
            last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        b_obs = rollout.observations.reshape((batch_size, obs_dim))
        b_actions = rollout.actions.reshape((batch_size, action_dim))
        b_log_probs = rollout.log_probs.reshape((batch_size,))
        b_values = rollout.values.reshape((batch_size,))
        b_advantages = advantages.reshape((batch_size,))
        b_returns = returns.reshape((batch_size,))

        update_metrics: list[dict[str, float]] = []
        for epoch in range(config.update_epochs):
            rng, perm_key = jax.random.split(update_state.rng_key)
            update_state = update_state._replace(rng_key=rng)
            for minibatch in np.asarray(minibatch_indices(perm_key, batch_size, minibatch_size)):
                params, opt_state, metrics = update_fn(
                    params,
                    opt_state,
                    obs_norm,
                    b_obs[minibatch],
                    b_actions[minibatch],
                    b_log_probs[minibatch],
                    b_values[minibatch],
                    b_advantages[minibatch],
                    b_returns[minibatch],
                    config.clip_coef,
                    config.vf_coef,
                    config.ent_coef,
                )
                metric_row = {name: float(value) for name, value in metrics.items()}
                update_metrics.append(metric_row)
            mean_kl = float(np.mean([row["approx_kl"] for row in update_metrics[-config.num_minibatches :]]))
            if mean_kl > config.target_kl:
                break

        ran_eval = ((update + 1) % config.eval_interval_updates == 0) or (update == num_updates - 1)
        if ran_eval:
            last_eval = evaluate_policy(env, params, obs_norm, episodes=config.eval_episodes, seed=config.seed + 10_000 + update, start_mode="grid")
            if last_eval.mean_return > best_eval.mean_return:
                best_eval = last_eval
                best_params = params
                best_obs_norm = obs_norm

        metric_mean = {
            key: float(np.mean([row[key] for row in update_metrics])) for key in update_metrics[0]
        } if update_metrics else {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "mean_value": 0.0,
        }
        row = {
            "update": update + 1,
            "env_steps": int((update + 1) * batch_size),
            "loss": metric_mean["loss"],
            "policy_loss": metric_mean["policy_loss"],
            "value_loss": metric_mean["value_loss"],
            "entropy": metric_mean["entropy"],
            "approx_kl": metric_mean["approx_kl"],
            "clip_fraction": metric_mean["clip_fraction"],
            "grad_norm": metric_mean["grad_norm"],
            "learning_rate": float(learning_rate_schedule(update)),
            "episode_return": float(np.mean(finished_episode_returns)) if finished_episode_returns else float("nan"),
            "episode_length": float(np.mean(finished_episode_lengths)) if finished_episode_lengths else float("nan"),
            "eval_return": last_eval.mean_return,
            "eval_length": last_eval.mean_length,
            "eval_progress": last_eval.mean_progress,
            "eval_lap_rate": last_eval.lap_rate,
            "eval_offtrack_rate": last_eval.offtrack_rate,
            "ran_eval": float(ran_eval),
        }
        history.append(row)
        if wandb_run is not None:
            wandb_run.log(row, step=row["env_steps"])

    checkpoint_path = save_checkpoint(output_dir / "ppo_nominal_policy.pkl", config=config, params=best_params, obs_norm=best_obs_norm, train_history=history)
    history_csv = _write_history_csv(output_dir / "training_history.csv", history)
    plot_paths = _training_plots(output_dir, history)
    final_eval = evaluate_policy(env, best_params, best_obs_norm, episodes=max(config.eval_episodes * 2, 12), seed=config.seed + 90_000, start_mode="grid")

    noisy_evals: dict[str, dict[str, Any]] = {}
    if config.render_after_training:
        noisy_evals = {
            "nominal": render_policy_rollout(
                scenario=config.scenario,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                video_name="ppo_nominal_policy_nominal",
                uncertainty_mode=None,
                gaussian_std=config.gaussian_std,
                uncertainty_artifact=config.empirical_artifact,
                calibration_artifact=config.calibration_artifact,
                steps=config.render_steps,
                seed=config.seed + 1,
                render_mode=config.render_mode,
                wandb_run=wandb_run,
            ),
            "gaussian": render_policy_rollout(
                scenario=config.scenario,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                video_name="ppo_nominal_policy_gaussian",
                uncertainty_mode="gaussian",
                gaussian_std=config.gaussian_std,
                uncertainty_artifact=config.empirical_artifact,
                calibration_artifact=config.calibration_artifact,
                steps=config.render_steps,
                seed=config.seed + 2,
                render_mode=config.render_mode,
                wandb_run=wandb_run,
            ),
        }
        if config.empirical_artifact:
            noisy_evals["empirical"] = render_policy_rollout(
                scenario=config.scenario,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                video_name="ppo_nominal_policy_empirical",
                uncertainty_mode="empirical",
                gaussian_std=config.gaussian_std,
                uncertainty_artifact=config.empirical_artifact,
                calibration_artifact=config.calibration_artifact,
                steps=config.render_steps,
                seed=config.seed + 3,
                render_mode=config.render_mode,
                wandb_run=wandb_run,
            )

    summary_path = _write_summary_markdown(
        output_dir / "rl_training_summary.md",
        config=config,
        best_eval=best_eval,
        final_eval=final_eval,
        noisy_evals=noisy_evals,
        checkpoint_path=checkpoint_path,
        history_csv=history_csv,
        plot_paths=plot_paths,
        wandb_url=getattr(wandb_run, "url", None) if wandb_run is not None else None,
    )

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "best_eval_return": best_eval.mean_return,
                "best_eval_progress": best_eval.mean_progress,
                "best_eval_lap_rate": best_eval.lap_rate,
                "final_eval_return": final_eval.mean_return,
                "final_eval_progress": final_eval.mean_progress,
                "artifacts/checkpoint": checkpoint_path.as_posix(),
                "artifacts/summary": summary_path.as_posix(),
            }
        )
        wandb_run.finish()

    return {
        "output_dir": output_dir.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "history_csv": history_csv.as_posix(),
        "summary_path": summary_path.as_posix(),
        "best_eval": best_eval._asdict(),
        "final_eval": final_eval._asdict(),
        "noisy_evals": noisy_evals,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a nominal PPO policy for uncertain-racecar-gym.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--num-minibatches", type=int, default=16)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.992)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--eval-interval-updates", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--initial-log-std", type=float, default=-0.7)
    parser.add_argument("--start-mode", default="random", choices=["grid", "random"])
    parser.add_argument("--wandb-project", default="uncertain-racecar-gym-rl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--gaussian-std", type=float, nargs=4, default=[0.24, 0.16, 0.10, 0.025])
    parser.add_argument("--empirical-artifact", default=None)
    parser.add_argument("--calibration-artifact", default=None)
    parser.add_argument("--render-steps", type=int, default=900)
    parser.add_argument("--render-mode", default="rgb_array_follow")
    parser.add_argument("--render-after-training", action="store_true")
    parser.add_argument("--bc-canonical-dataset", default=None)
    parser.add_argument("--bc-episodes", type=int, default=28)
    parser.add_argument("--bc-steps-per-episode", type=int, default=1200)
    parser.add_argument("--bc-epochs", type=int, default=40)
    parser.add_argument("--bc-batch-size", type=int, default=2048)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-max-rows", type=int, default=120000)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", default="all", choices=["all", "nominal", "gaussian", "empirical"])
    parser.add_argument("--wandb-run-name", default=None)
    return parser


def train_ppo_main() -> None:
    args = build_arg_parser().parse_args()
    if args.checkpoint is not None:
        evaluate_ppo_main_from_args(args)
        return
    config = PPOConfig(
        scenario=args.scenario,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        eval_interval_updates=args.eval_interval_updates,
        eval_episodes=args.eval_episodes,
        hidden_sizes=tuple(args.hidden_sizes),
        initial_log_std=args.initial_log_std,
        start_mode=args.start_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        disable_wandb=args.disable_wandb,
        gaussian_std=tuple(args.gaussian_std),
        empirical_artifact=args.empirical_artifact,
        calibration_artifact=args.calibration_artifact,
        render_steps=args.render_steps,
        render_mode=args.render_mode,
        render_after_training=args.render_after_training,
        bc_canonical_dataset=args.bc_canonical_dataset,
        bc_episodes=args.bc_episodes,
        bc_steps_per_episode=args.bc_steps_per_episode,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_learning_rate=args.bc_learning_rate,
        bc_max_rows=args.bc_max_rows,
    )
    result = train_ppo(config)
    print(json.dumps(result, indent=2))


def evaluate_saved_policy(
    *,
    scenario: str,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    gaussian_std: tuple[float, float, float, float],
    empirical_artifact: str | None,
    calibration_artifact: str | None,
    render_steps: int,
    render_mode: str,
    seed: int,
    run_name: str | None,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_mode: str,
    disable_wandb: bool,
    modes: tuple[str, ...],
) -> dict[str, Any]:
    output_dir_path = ensure_dir(output_dir)
    wandb_run = None
    if not disable_wandb:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name or f"{Path(checkpoint_path).stem}_eval",
            mode=wandb_mode,
            config={
                "scenario": scenario,
                "checkpoint": str(checkpoint_path),
                "gaussian_std": list(gaussian_std),
                "modes": list(modes),
            },
            dir=output_dir_path.as_posix(),
        )

    evaluation_rows: dict[str, dict[str, Any]] = {}
    for offset, mode in enumerate(modes):
        evaluation_rows[mode] = render_policy_rollout(
            scenario=scenario,
            checkpoint_path=Path(checkpoint_path),
            output_dir=output_dir_path,
            video_name=f"{Path(checkpoint_path).stem}_{mode}",
            uncertainty_mode=None if mode == "nominal" else mode,
            gaussian_std=gaussian_std,
            uncertainty_artifact=empirical_artifact,
            calibration_artifact=calibration_artifact,
            steps=render_steps,
            seed=seed + offset,
            render_mode=render_mode,
            wandb_run=wandb_run,
        )

    summary_path = _write_eval_summary_markdown(
        output_dir_path / "ppo_policy_evaluation.md",
        checkpoint_path=Path(checkpoint_path),
        evaluation_rows=evaluation_rows,
        wandb_url=getattr(wandb_run, "url", None) if wandb_run is not None else None,
    )
    if wandb_run is not None:
        wandb_run.summary.update({"summary_markdown": summary_path.as_posix()})
        wandb_run.finish()
    return {
        "output_dir": output_dir_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "evaluations": evaluation_rows,
    }


def evaluate_ppo_main_from_args(args: argparse.Namespace) -> None:
    modes = ("nominal", "gaussian", "empirical") if args.mode == "all" else (args.mode,)
    result = evaluate_saved_policy(
        scenario=args.scenario,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        gaussian_std=tuple(args.gaussian_std),
        empirical_artifact=args.empirical_artifact,
        calibration_artifact=args.calibration_artifact,
        render_steps=args.render_steps,
        render_mode=args.render_mode,
        seed=args.seed,
        run_name=args.wandb_run_name,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        disable_wandb=args.disable_wandb,
        modes=modes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    train_ppo_main()
