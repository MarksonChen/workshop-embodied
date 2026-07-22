"""PPO post-training of the aligned SNN's continuous action readout.

The independently imitation-trained recurrent spiking core is frozen.  PPO
updates only the filtered-spike readout and exploration scale on a Demo A-style
1,000-step locomotion task.  No Demo H policy, checkpoint, or activation is
loaded by this module.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from demo_j.control.aligned import (
    SPEED_ANCHORS,
    balanced_speed_indices,
    build_periodic_sequences,
)
from demo_j.control.aligned_tracking import AlignedLocomotion
from demo_j.experiments.aligned import load_aligned_checkpoint
from demo_j.artifacts import ALIGNED_OUTPUT_ROOT, save_pickle, sha256, write_json
from demo_j.control.config import ACTION_DIM
from demo_j.control.ppo import (
    critic_value,
    generalized_advantage_estimate,
    init_critic,
    normalize_advantage,
    tanh_normal_log_probability,
)
from demo_j.data.projection import DEFAULT_ROOT as PROJECTED_ROOT
from demo_j.data.projection import load_projected_reference
from demo_j.control.snn import LSNNParams, control_step, initial_state, sequence


class Actor(NamedTuple):
    snn: LSNNParams
    log_standard_deviation: jax.Array


class Rollout(NamedTuple):
    observation: jax.Array
    readout_feature: jax.Array
    raw_action: jax.Array
    action: jax.Array
    old_log_probability: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    valid: jax.Array
    speed: jax.Array
    target_speed: jax.Array
    track: jax.Array
    upright: jax.Array
    spike_probability: jax.Array


def _select(mask, selected, fallback):
    def choose(yes, no):
        shaped = mask.reshape(mask.shape + (1,) * (yes.ndim - mask.ndim))
        return jnp.where(shaped, yes, no)

    return jax.tree.map(choose, selected, fallback)


def _actor_sequence(actor, observation, config):
    state = initial_state((observation.shape[1],), config)
    return sequence(actor.snn, state, observation, config)[1]


def _policy_kl(old, new, observation, valid, config):
    old_mean, _ = _actor_sequence(old, observation, config)
    new_mean, _ = _actor_sequence(new, observation, config)
    old_log_std = jnp.clip(old.log_standard_deviation, -5.0, 1.0)
    new_log_std = jnp.clip(new.log_standard_deviation, -5.0, 1.0)
    old_variance = jnp.exp(2 * old_log_std)
    new_variance = jnp.exp(2 * new_log_std)
    kl = (
        new_log_std
        - old_log_std
        + (old_variance + jnp.square(old_mean - new_mean)) / (2 * new_variance)
        - 0.5
    ).sum(axis=-1)
    return (kl * valid).sum() / jnp.maximum(valid.sum(), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=PROJECTED_ROOT)
    parser.add_argument("--num-updates", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--minibatch-envs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=1000,
        help="recurrent PPO horizon; must equal episode-steps for aligned runs",
    )
    parser.add_argument("--clips-per-speed", type=int, default=64)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--actor-update",
        choices=("readout", "all"),
        default="readout",
    )
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--initial-log-std", type=float, default=-2.0)
    parser.add_argument("--entropy-coefficient", type=float, default=1e-3)
    parser.add_argument("--maximum-policy-kl", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--validation-envs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="aligned-readout")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.num_updates = 2
        args.num_envs = 16
        args.minibatch_envs = 8
        args.epochs = 1
        args.episode_steps = 100
        args.unroll_steps = 100
        args.eval_every = 1
        args.validation_envs = 6
    if args.num_envs % args.minibatch_envs:
        raise ValueError("num-envs must be divisible by minibatch-envs")
    if args.unroll_steps != args.episode_steps:
        raise ValueError(
            "aligned PPO must collect the same full horizon used for validation"
        )

    saved, tokenizer, config, initial_params = load_aligned_checkpoint(
        args.init_checkpoint
    )
    training_reference = load_projected_reference("train", args.reference_root)
    validation_reference = load_projected_reference("validation", args.reference_root)
    training_sequences = build_periodic_sequences(
        training_reference, tokenizer, preview_tokens=int(saved["preview_tokens"])
    )
    validation_sequences = build_periodic_sequences(
        validation_reference, tokenizer, preview_tokens=int(saved["preview_tokens"])
    )
    training_valid = balanced_speed_indices(training_sequences, args.clips_per_speed)
    validation_per_speed = max(
        1, int(np.ceil(args.validation_envs / len(SPEED_ANCHORS)))
    )
    validation_valid = balanced_speed_indices(
        validation_sequences, validation_per_speed
    )[: args.validation_envs]
    training_environment = AlignedLocomotion(
        training_reference, training_sequences, training_valid
    )
    validation_environment = AlignedLocomotion(
        validation_reference, validation_sequences, validation_valid
    )
    mean = jnp.asarray(saved["observation_mean"])
    std = jnp.asarray(saved["observation_std"])
    actor = Actor(
        initial_params,
        jnp.full((ACTION_DIM,), args.initial_log_std, jnp.float32),
    )
    key = jax.random.key(args.seed)
    key, critic_key = jax.random.split(key)
    critic = init_critic(critic_key, training_environment.observation_size)
    reset_training = jax.vmap(training_environment.reset)
    step_training = jax.vmap(training_environment.step)
    reset_validation = jax.vmap(validation_environment.reset_to)
    step_validation = jax.vmap(validation_environment.step)

    @jax.jit
    def collect(actor, critic, key):
        key, reset_key = jax.random.split(key)
        state = reset_training(jax.random.split(reset_key, args.num_envs))
        neuronal = initial_state((args.num_envs,), config)
        alive = jnp.ones((args.num_envs,), bool)

        def advance(carry, _):
            state, neuronal, alive, key = carry
            key, action_key = jax.random.split(key)
            observation = jnp.clip((state.obs - mean) / std, -10.0, 10.0)
            neuronal, (policy_mean, spikes) = control_step(
                actor.snn, neuronal, observation, config
            )
            raw_action = policy_mean + jnp.exp(
                jnp.clip(actor.log_standard_deviation, -5.0, 1.0)
            ) * jax.random.normal(action_key, policy_mean.shape)
            action = jnp.tanh(raw_action)
            log_probability = tanh_normal_log_probability(
                raw_action,
                action,
                policy_mean,
                actor.log_standard_deviation,
            )
            value = critic_value(critic, observation)
            stepped = step_training(state, action)
            terminal = stepped.done.astype(bool)
            valid = alive.astype(jnp.float32)
            output = Rollout(
                observation,
                neuronal.filtered_spikes,
                raw_action,
                action,
                log_probability,
                value,
                jnp.where(alive, stepped.reward, 0.0),
                jnp.where(alive, terminal, True).astype(jnp.float32),
                valid,
                stepped.metrics["speed"],
                stepped.metrics["target_speed"],
                stepped.metrics["track"],
                stepped.metrics["upright"],
                spikes.mean(axis=(0, 2)),
            )
            next_alive = alive & ~terminal
            state = _select(alive, stepped, state)
            return (state, neuronal, next_alive, key), output

        return jax.lax.scan(
            advance,
            (state, neuronal, alive, key),
            xs=None,
            length=args.unroll_steps,
        )[1]

    @jax.jit
    def evaluate(actor):
        count = len(validation_valid)
        state = reset_validation(jnp.asarray(validation_valid))
        neuronal = initial_state((count,), config)
        alive = jnp.ones((count,), bool)

        def advance(carry, _):
            state, neuronal, alive = carry
            observation = jnp.clip((state.obs - mean) / std, -10.0, 10.0)
            neuronal, (policy_mean, spikes) = control_step(
                actor.snn, neuronal, observation, config
            )
            action = jnp.tanh(policy_mean)
            stepped = step_validation(state, action)
            terminal = stepped.done.astype(bool)
            valid = alive.astype(jnp.float32)
            output = (
                jnp.where(alive, stepped.reward, 0.0),
                valid,
                stepped.metrics["speed"],
                stepped.metrics["target_speed"],
                stepped.metrics["track"],
                stepped.metrics["upright"],
                spikes.mean(),
                jnp.mean(jnp.abs(action) >= 0.99),
            )
            next_alive = alive & ~terminal
            state = _select(alive, stepped, state)
            return (state, neuronal, next_alive), output

        (_, _, final_alive), output = jax.lax.scan(
            advance,
            (state, neuronal, alive),
            xs=None,
            length=args.episode_steps,
        )
        reward, valid, speed, target, track, upright, spike, saturation = output
        bins = jnp.maximum(valid.sum(), 1.0)
        return {
            "survival_fraction": final_alive.mean(),
            "valid_bin_fraction": valid.mean(),
            "return_mean": reward.sum() / count,
            "track_mean_alive": (track * valid).sum() / bins,
            "speed_rmse_alive": jnp.sqrt(
                (jnp.square(speed - target) * valid).sum() / bins
            ),
            "upright_mean_alive": (upright * valid).sum() / bins,
            "spike_probability_per_5ms": spike.mean(),
            "action_saturation_fraction": saturation.mean(),
        }

    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(args.actor_learning_rate),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(args.critic_learning_rate),
    )
    actor_state = actor_optimizer.init(actor)
    critic_state = critic_optimizer.init(critic)

    def loss(actor, critic, batch, advantage, target_value):
        if args.actor_update == "readout":
            mean_action = (
                batch.readout_feature @ actor.snn.readout_weight
                + actor.snn.readout_bias
            )
            spike_probability = (
                batch.spike_probability * batch.valid
            ).sum() / jnp.maximum(batch.valid.sum(), 1.0)
        else:
            mean_action, spikes = _actor_sequence(actor, batch.observation, config)
            spike_mask = batch.valid[:, None, :, None]
            spike_probability = (spikes * spike_mask).sum() / jnp.maximum(
                batch.valid.sum() * config.substeps * config.neurons, 1.0
            )
        log_probability = tanh_normal_log_probability(
            batch.raw_action,
            batch.action,
            mean_action,
            actor.log_standard_deviation,
        )
        ratio = jnp.exp(
            jnp.clip(log_probability - batch.old_log_probability, -20.0, 20.0)
        )
        clipped = jnp.clip(ratio, 0.8, 1.2) * advantage
        count = jnp.maximum(batch.valid.sum(), 1.0)
        policy_loss = (
            -(jnp.minimum(ratio * advantage, clipped) * batch.valid).sum() / count
        )
        value = critic_value(critic, batch.observation)
        value_loss = (
            0.5 * (jnp.square(value - target_value) * batch.valid).sum() / count
        )
        entropy = jnp.sum(
            jnp.clip(actor.log_standard_deviation, -5.0, 1.0)
            + 0.5 * (1.0 + jnp.log(2 * jnp.pi))
        )
        total = policy_loss + 0.5 * value_loss - args.entropy_coefficient * entropy
        return total, {
            "loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "spike_probability": spike_probability,
            "approximate_kl": (
                (batch.old_log_probability - log_probability) * batch.valid
            ).sum()
            / count,
        }

    @jax.jit
    def update(
        actor, critic, actor_state, critic_state, batch, advantage, target, indices
    ):
        def subset(value):
            return jnp.take(value, indices, axis=1)

        minibatch = jax.tree.map(subset, batch)
        minibatch_advantage = subset(advantage)
        minibatch_target = subset(target)
        (_, metrics), (actor_gradient, critic_gradient) = jax.value_and_grad(
            loss, argnums=(0, 1), has_aux=True
        )(actor, critic, minibatch, minibatch_advantage, minibatch_target)
        if args.actor_update == "readout":
            zero = jax.tree.map(jnp.zeros_like, actor_gradient)
            actor_gradient = zero._replace(
                snn=zero.snn._replace(
                    readout_weight=actor_gradient.snn.readout_weight,
                    readout_bias=actor_gradient.snn.readout_bias,
                ),
                log_standard_deviation=actor_gradient.log_standard_deviation,
            )
        actor_updates, actor_state = actor_optimizer.update(
            actor_gradient, actor_state, actor
        )
        critic_updates, critic_state = critic_optimizer.update(
            critic_gradient, critic_state, critic
        )
        return (
            optax.apply_updates(actor, actor_updates),
            optax.apply_updates(critic, critic_updates),
            actor_state,
            critic_state,
            {**metrics, "actor_gradient_norm": optax.global_norm(actor_gradient)},
        )

    if args.actor_update == "readout":

        @jax.jit
        def policy_kl(old, new, rollout):
            old_mean = (
                rollout.readout_feature @ old.snn.readout_weight + old.snn.readout_bias
            )
            new_mean = (
                rollout.readout_feature @ new.snn.readout_weight + new.snn.readout_bias
            )
            old_log_std = jnp.clip(old.log_standard_deviation, -5.0, 1.0)
            new_log_std = jnp.clip(new.log_standard_deviation, -5.0, 1.0)
            old_variance = jnp.exp(2 * old_log_std)
            new_variance = jnp.exp(2 * new_log_std)
            kl = (
                new_log_std
                - old_log_std
                + (old_variance + jnp.square(old_mean - new_mean)) / (2 * new_variance)
                - 0.5
            ).sum(axis=-1)
            return (kl * rollout.valid).sum() / jnp.maximum(rollout.valid.sum(), 1.0)

    else:

        @jax.jit
        def policy_kl(old, new, rollout):
            return _policy_kl(old, new, rollout.observation, rollout.valid, config)

    @jax.jit
    def interpolate(old, new, scale):
        return jax.tree.map(lambda a, b: a + scale * (b - a), old, new)

    started = time.perf_counter()
    progress = []
    initial_metrics = {name: float(value) for name, value in evaluate(actor).items()}
    initial_metrics.update(update=0, seconds=time.perf_counter() - started)
    progress.append(initial_metrics)
    print(json.dumps(initial_metrics), flush=True)
    best_actor = actor
    best_qualified = initial_metrics["survival_fraction"] >= 0.95
    best_score = initial_metrics["return_mean"] if best_qualified else float("-inf")
    best_update = 0

    for update_index in range(1, args.num_updates + 1):
        key, rollout_key = jax.random.split(key)
        rollout = collect(actor, critic, rollout_key)
        old_actor = actor
        advantage, target_value = generalized_advantage_estimate(
            rollout.reward * 0.2,
            rollout.value,
            rollout.done,
            rollout.valid,
            discount=0.97,
            gae_lambda=0.95,
        )
        advantage = normalize_advantage(advantage, rollout.valid)
        last_metrics = {}
        for _ in range(args.epochs):
            key, permutation_key = jax.random.split(key)
            permutation = jax.random.permutation(permutation_key, args.num_envs)
            for start in range(0, args.num_envs, args.minibatch_envs):
                indices = permutation[start : start + args.minibatch_envs]
                actor, critic, actor_state, critic_state, last_metrics = update(
                    actor,
                    critic,
                    actor_state,
                    critic_state,
                    rollout,
                    advantage,
                    target_value,
                    indices,
                )
        proposed = actor
        kl = float(policy_kl(old_actor, proposed, rollout))
        accepted_scale = 1.0
        while kl > args.maximum_policy_kl and accepted_scale > 1 / 256:
            accepted_scale *= 0.5
            actor = interpolate(old_actor, proposed, accepted_scale)
            kl = float(policy_kl(old_actor, actor, rollout))
        if kl > args.maximum_policy_kl:
            actor = old_actor
            accepted_scale = 0.0
            kl = 0.0

        if update_index % args.eval_every == 0 or update_index == args.num_updates:
            metrics = {name: float(value) for name, value in evaluate(actor).items()}
            metrics.update(
                update=update_index,
                transitions=update_index * args.num_envs * args.unroll_steps,
                seconds=time.perf_counter() - started,
                policy_kl=kl,
                accepted_actor_scale=accepted_scale,
                **{
                    f"train_{name}": float(value)
                    for name, value in last_metrics.items()
                },
            )
            progress.append(metrics)
            print(json.dumps(metrics), flush=True)
            if metrics["survival_fraction"] >= 0.95 and (
                not best_qualified or metrics["return_mean"] > best_score
            ):
                best_actor = jax.device_get(actor)
                best_score = metrics["return_mean"]
                best_update = update_index
                best_qualified = True

    elapsed = time.perf_counter() - started
    ALIGNED_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    checkpoint = ALIGNED_OUTPUT_ROOT / f"snn_{args.run_id}_seed{args.seed}_{stamp}.pkl"
    payload = dict(saved)
    payload.update(
        params=best_actor.snn,
        rl_finetune={
            "schema": "demo-j-aligned-readout-ppo-v2",
            "initial_checkpoint": str(args.init_checkpoint),
            "initial_checkpoint_sha256": sha256(args.init_checkpoint),
            "best_update": best_update,
            "episode_steps": args.episode_steps,
            "ppo_unroll_steps": args.unroll_steps,
            "balanced_speed_sampling": True,
            "spiking_core_frozen": args.actor_update == "readout",
            "demo_h_policy_used": False,
        },
    )
    save_pickle(checkpoint, payload)
    report = {
        "schema": "demo-j-aligned-readout-ppo-training-v2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "initial_checkpoint": str(args.init_checkpoint),
        "initial_checkpoint_sha256": sha256(args.init_checkpoint),
        "seed": args.seed,
        "best_update": best_update,
        "best_validation_return": best_score,
        "best_checkpoint_passes_survival_gate": best_qualified,
        "num_updates": args.num_updates,
        "num_envs": args.num_envs,
        "episode_steps": args.episode_steps,
        "ppo_unroll_steps": args.unroll_steps,
        "speed_anchors": SPEED_ANCHORS.tolist(),
        "clips_per_speed": args.clips_per_speed,
        "validation_clips_per_speed": validation_per_speed,
        "transitions": args.num_updates * args.num_envs * args.unroll_steps,
        "training_seconds": elapsed,
        "transitions_per_second": (
            args.num_updates * args.num_envs * args.unroll_steps / elapsed
        ),
        "actor_update": args.actor_update,
        "spiking_core_frozen": args.actor_update == "readout",
        "reward": "speed tracking + 0.1 upright - 0.001 action squared",
        "naturalness_metrics_in_reward": False,
        "demo_h_policy_used": False,
        "progress": progress,
    }
    write_json(checkpoint.with_suffix(".json"), report)
    print(f"wrote {checkpoint}")


if __name__ == "__main__":
    main()
