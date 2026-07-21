"""Shaping-disabled, paired-seed evaluation of H0/H1/H2."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from demo_a.train_fetch import FetchV2, RawKeyVmapWrapper
from demo_g.metrics import GAIT_FIELDS, gait_distance, gait_statistics
from demo_h.config import COMMAND_HORIZON_SECONDS, OUT, TARGET_SPEED_FETCH
from demo_h.dataset.contract import DEFAULT_ROOT
from demo_h.dataset.loader import load_split
from demo_h.env import DemoHFetchRun
from demo_h.interfaces import (
    BASE_OBS_DIM,
    COMMAND_SLICE,
    FEATURE_BUFFER_SLICE,
    FEATURE_DIM,
)
from demo_h.policy import (
    compute_plans,
    diagonal_gaussian_kl,
    make_residual_ppo_networks,
    reference_parameters,
)
from demo_h.prior import DEFAULT_PRIOR, load_prior
from demo_h.wrappers import BatchedPlanWrapper


EVAL_SEEDS = (101, 211, 307, 401, 503)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_params(path: Path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def reference_summary(dataset_root: Path) -> dict:
    test = load_split("test", dataset_root)
    statistics = gait_statistics(test.features[:, :64])
    return {
        name: {"mean": float(value.mean()), "std": float(value.std())}
        for name, value in statistics.items()
    }


def make_environment(prior, target_speed=TARGET_SPEED_FETCH):
    environment = FetchV2(
        DemoHFetchRun(
            v_target=target_speed,
            sigma=max(target_speed / 3.0, 1e-3),
        )
    )
    environment = RawKeyVmapWrapper(environment)
    return BatchedPlanWrapper(environment, prior)


def _set_target_speeds(environment, prior, state, target_speeds):
    """Replace per-environment commands after one shared vectorized reset."""

    target_speeds = jnp.asarray(target_speeds, dtype=jnp.float32)
    commands = jnp.stack(
        (
            target_speeds * COMMAND_HORIZON_SECONDS,
            jnp.zeros_like(target_speeds),
            jnp.zeros_like(target_speeds),
        ),
        axis=-1,
    )
    pipeline = state.pipeline_state
    pipeline_info = dict(pipeline.info)
    pipeline_info.update(h_target_speed=target_speeds, h_command=commands)
    pipeline_metrics = dict(pipeline.metrics)
    pipeline_metrics["target_speed"] = target_speeds
    pipeline = pipeline.replace(
        obs=pipeline.obs.at[..., COMMAND_SLICE].set(commands),
        info=pipeline_info,
        metrics=pipeline_metrics,
    )
    metrics = dict(state.metrics)
    metrics["target_speed"] = target_speeds
    state = state.replace(
        pipeline_state=pipeline,
        obs=state.obs.at[..., COMMAND_SLICE].set(commands),
        metrics=metrics,
    )
    # Reset generated a plan for the environment's scalar construction speed.
    # Refresh it once after installing the actual batched commands.
    return environment._set_plan(state, compute_plans(prior, state.obs))


def make_actor(arm: str, checkpoint: Path, prior):
    params = load_params(checkpoint)
    if arm == "h0":
        network = ppo_networks.make_ppo_networks(
            BASE_OBS_DIM,
            10,
            preprocess_observations_fn=running_statistics.normalize,
        )
        inference = ppo_networks.make_inference_fn(network)(
            params, deterministic=True
        )

        def action(observation, key):
            return inference(observation[..., :BASE_OBS_DIM], key)[0]

        def parameters(observation):
            return network.policy_network.apply(
                params[0], params[1], observation[..., :BASE_OBS_DIM]
            )

    else:
        network = make_residual_ppo_networks((1094,), 10, prior=prior)
        inference = ppo_networks.make_inference_fn(network)(
            params, deterministic=True
        )

        def action(observation, key):
            return inference(observation, key)[0]

        def parameters(observation):
            return network.policy_network.apply(params[0], params[1], observation)

    return action, parameters


def rollout_arm(
    arm, checkpoint, prior, seeds, steps, target_speed=TARGET_SPEED_FETCH
):
    target_speeds = np.asarray(target_speed, dtype=np.float32)
    scalar_target = target_speeds.ndim == 0
    if not scalar_target and target_speeds.shape != (len(seeds),):
        raise ValueError(
            f"target speeds must be scalar or match {len(seeds)} seeds, "
            f"got {target_speeds.shape}"
        )
    construction_speed = float(
        target_speeds if scalar_target else target_speeds[0]
    )
    environment = make_environment(prior, construction_speed)
    action_fn, parameter_fn = make_actor(arm, checkpoint, prior)
    reset_keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    initial = environment.reset(reset_keys)
    if not scalar_target:
        initial = _set_target_speeds(environment, prior, initial, target_speeds)

    def rollout(initial_state):
        def step(carry, _):
            state, key = carry
            key, action_key = jax.random.split(key)
            parameters = parameter_fn(state.obs)
            reference = reference_parameters(prior, state.obs)
            kl = diagonal_gaussian_kl(parameters, reference)
            action = action_fn(state.obs, action_key)
            next_state = environment.step(state, action)
            feature = next_state.obs[..., FEATURE_BUFFER_SLICE].reshape(
                (len(seeds), 16, FEATURE_DIM)
            )[:, -1]
            output = (
                next_state.reward,
                next_state.done,
                next_state.metrics["speed"],
                next_state.metrics["track"],
                next_state.metrics["upright"],
                action,
                kl,
                feature,
                next_state.pipeline_state.qp,
            )
            return (next_state, key), output

        return jax.lax.scan(
            step,
            (initial_state, jax.random.PRNGKey(9_001)),
            xs=None,
            length=steps,
        )[1]

    stream = jax.jit(rollout)(initial)
    jax.block_until_ready(stream[0])
    stream = jax.tree_util.tree_map(np.asarray, stream)
    return environment, initial, stream


def summarize(stream, seeds, steps, target_speed=TARGET_SPEED_FETCH):
    reward, done, speed, track, upright, action, kl, feature, _ = stream
    target_speeds = np.asarray(target_speed, dtype=np.float32)
    if target_speeds.ndim == 0:
        target_speeds = np.repeat(target_speeds[None], len(seeds))
    if target_speeds.shape != (len(seeds),):
        raise ValueError(target_speeds.shape)
    episodes = []
    for episode_index, seed in enumerate(seeds):
        episode_target_speed = float(target_speeds[episode_index])
        terminal = done[:, episode_index].astype(bool)
        alive_before = np.concatenate(
            ((True,), np.cumprod(~terminal[:-1]).astype(bool))
        )
        living_features = feature[alive_before, episode_index]
        complete = len(living_features) // 64
        if complete:
            clips = living_features[: complete * 64].reshape(complete, 64, 60)
            gait = gait_statistics(clips)
            gait_values = {
                f"gait_{name}": float(np.mean(values))
                for name, values in gait.items()
            }
        else:
            gait_values = {f"gait_{name}": float("nan") for name in GAIT_FIELDS}
        mask = alive_before.astype(np.float32)
        count = max(mask.sum(), 1.0)
        episode = {
            "seed": int(seed),
            "metrics": {
                "task_return": float((reward[:, episode_index] * mask).sum()),
                "survival_fraction": float(mask.mean()),
                "track_mean": float((track[:, episode_index] * mask).sum() / count),
                "speed_mean": float((speed[:, episode_index] * mask).sum() / count),
                "speed_rmse": float(
                    np.sqrt(
                        (
                            np.square(speed[:, episode_index] - episode_target_speed)
                            * mask
                        ).sum()
                        / count
                    )
                ),
                "upright_mean": float(
                    (upright[:, episode_index] * mask).sum() / count
                ),
                "action_energy": float(
                    (np.square(action[:, episode_index]).mean(axis=-1) * mask).sum()
                    / count
                ),
                "action_saturation_fraction": float(
                    ((np.abs(action[:, episode_index]) >= 0.999).mean(axis=-1) * mask).sum()
                    / count
                ),
                "reference_kl_per_dimension": float(
                    (kl[:, episode_index] * mask).sum() / count
                ),
                **gait_values,
            },
        }
        episodes.append(episode)
    return episodes


def aggregate(episodes):
    return {
        name: {
            "mean": float(np.nanmean([row["metrics"][name] for row in episodes])),
            "std": float(np.nanstd([row["metrics"][name] for row in episodes])),
        }
        for name in episodes[0]["metrics"]
    }


def save_trace(output: Path, initial, stream, batch_index: int = 0):
    _, _, speed, _, upright, action, kl, feature, qps = stream
    initial_qp = jax.tree_util.tree_map(
        lambda value: np.asarray(value)[batch_index], initial.pipeline_state.qp
    )
    np.savez_compressed(
        output,
        initial_qp_pos=initial_qp.pos,
        initial_qp_rot=initial_qp.rot,
        initial_qp_vel=initial_qp.vel,
        initial_qp_ang=initial_qp.ang,
        qp_pos=qps.pos[:, batch_index],
        qp_rot=qps.rot[:, batch_index],
        qp_vel=qps.vel[:, batch_index],
        qp_ang=qps.ang[:, batch_index],
        controls=action[:, batch_index],
        contacts=feature[:, batch_index, 56:60].astype(np.uint8),
        speed=speed[:, batch_index],
        upright=upright[:, batch_index],
        reference_kl=kl[:, batch_index],
        features=feature[:, batch_index],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h0", type=Path, required=True)
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--h2", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=EVAL_SEEDS)
    parser.add_argument("--output", type=Path, default=OUT / "evaluation.json")
    args = parser.parse_args()
    prior = load_prior(args.prior)
    reference = reference_summary(args.dataset_root)
    arms = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for arm, checkpoint in (("h0", args.h0), ("h1", args.h1), ("h2", args.h2)):
        environment, initial, stream = rollout_arm(
            arm, checkpoint, prior, tuple(args.seeds), args.steps
        )
        episodes = summarize(stream, tuple(args.seeds), args.steps)
        for episode in episodes:
            episode["metrics"]["gait_reference_distance"] = gait_distance(
                episode["metrics"], reference
            )
        arms[arm] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "aggregate": aggregate(episodes),
            "episodes": episodes,
        }
        save_trace(args.output.parent / f"{arm}_trace.npz", initial, stream)
        print(f"evaluated {arm}: {json.dumps(arms[arm]['aggregate'])}", flush=True)

    h0, h1, h2 = (arms[name]["aggregate"] for name in ("h0", "h1", "h2"))
    comparison = {
        "h1_task_return_vs_h0": h1["task_return"]["mean"] - h0["task_return"]["mean"],
        "h2_task_retention_vs_h0": h2["task_return"]["mean"]
        / max(h0["task_return"]["mean"], 1e-8),
        "h2_task_retention_vs_h1": h2["task_return"]["mean"]
        / max(h1["task_return"]["mean"], 1e-8),
        "h2_kl_reduction_vs_h1": h1["reference_kl_per_dimension"]["mean"]
        - h2["reference_kl_per_dimension"]["mean"],
        "h2_gait_distance_reduction_vs_h1": h1["gait_reference_distance"]["mean"]
        - h2["gait_reference_distance"]["mean"],
    }
    comparison["passes_single_seed_task_kl_gate"] = bool(
        comparison["h2_task_retention_vs_h0"] >= 0.95
        and comparison["h2_kl_reduction_vs_h1"] > 0
    )
    report = {
        "schema": "demo-h-evaluation-v1",
        "evaluation_seeds": args.seeds,
        "episode_steps": args.steps,
        "shaping_disabled": True,
        "test_gait_reference": reference,
        "arms": arms,
        "comparison": comparison,
        "claim_boundary": "One training seed; three matched training seeds are required for an algorithm-level claim.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
