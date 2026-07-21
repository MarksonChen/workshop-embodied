"""Held-out, shaping-disabled evaluation for a matched Demo G checkpoint pair.

The evaluator is intentionally written before the full-budget runs.  It freezes
the comparison metric and rejects policies that trade away Demo A function for
an easier learned-model score.

    env -u LD_LIBRARY_PATH uv run --no-project --isolated \
      --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
      --with 'jaxlib==0.4.30' --with scipy \
      python -m demo_g.evaluate --g0 PATH --g1 PATH
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from demo_f.artifacts import sha256
from demo_f.dataset import load_split
from demo_f.dataset.contract import DYNAMIC_ROOT

from .env import BUFFER_FRAMES, DemoGFetchRun
from .metrics import GAIT_CLIP_FRAMES, GAIT_FIELDS, gait_distance, gait_statistics
from .prior import DEFAULT_PRIOR, load_prior


OUT = Path(__file__).resolve().parent / "out"
EVAL_SEEDS = (101, 211, 307, 401, 503)


def reference_summary(dataset_root: Path) -> dict:
    """Freeze direct-gait targets from Demo F's untouched test sessions."""

    test = load_split("test", dataset_root)
    statistics = gait_statistics(test.features)
    return {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for name, values in statistics.items()
    }


def load_params(checkpoint: Path):
    with Path(checkpoint).open("rb") as stream:
        return pickle.load(stream)


def make_rollout_runtime(prior):
    """Build one paired graph so an identity checkpoint control is exactly zero."""

    env = DemoGFetchRun(prior)
    networks = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_policy = ppo_networks.make_inference_fn(networks)

    @jax.jit
    def paired_step(states, keys, paired_params):
        def single_step(state, key, params):
            action, _ = make_policy(params, deterministic=True)(state.obs, key)
            return env.step(state, action), action

        return jax.vmap(single_step)(states, keys, paired_params)

    return env, jax.jit(jax.vmap(env.reset)), paired_step


def stack_params(left, right):
    return jax.tree_util.tree_map(
        lambda left_leaf, right_leaf: jnp.stack((left_leaf, right_leaf)), left, right
    )


def _summarize_stream(
    stream: dict, seed: int, steps: int, target_speed_fetch: float
) -> dict:
    feature_stream = np.stack(stream["features"])
    windows = np.stack(
        [
            feature_stream[index - BUFFER_FRAMES + 1 : index + 1]
            for index in range(BUFFER_FRAMES - 1, len(feature_stream), 4)
        ]
    ) if len(feature_stream) >= BUFFER_FRAMES else np.empty((0, BUFFER_FRAMES, 60), np.float32)
    complete = len(feature_stream) // GAIT_CLIP_FRAMES
    gait_clips = feature_stream[: complete * GAIT_CLIP_FRAMES].reshape(
        complete, GAIT_CLIP_FRAMES, 60
    )
    gait = gait_statistics(gait_clips) if complete else {
        name: np.asarray((np.nan,), np.float32) for name in GAIT_FIELDS
    }
    speed = np.asarray(stream["speeds"])
    metrics = {
        "task_return": float(np.sum(stream["rewards"])),
        "survival_fraction": len(stream["rewards"]) / steps,
        "track_mean": float(np.mean(stream["tracks"])),
        "speed_mean": float(np.mean(speed)),
        "speed_rmse": float(
            np.sqrt(np.mean(np.square(speed - target_speed_fetch)))
        ),
        "upright_mean": float(np.mean(stream["uprights"])),
        "action_energy": float(np.mean(stream["action_energy"])),
        **{f"gait_{name}": float(np.nanmean(values)) for name, values in gait.items()},
    }
    return {"seed": seed, "metrics": metrics, "score_windows": windows}


def paired_rollout(paired_params, runtime, seed: int, steps: int) -> tuple[dict, dict]:
    """Run G0/G1 together, with shaping disabled and no auto-reset."""

    env, reset, paired_step = runtime
    rng = jax.random.PRNGKey(seed)
    states = reset(jnp.stack((rng, rng)))
    names = ("features", "rewards", "speeds", "tracks", "uprights", "action_energy")
    streams = [{name: [] for name in names} for _ in range(2)]
    active = np.ones(2, dtype=bool)
    for _ in range(steps):
        rng, action_key = jax.random.split(rng)
        states, actions = paired_step(
            states, jnp.stack((action_key, action_key)), paired_params
        )
        feature = np.asarray(states.info["prior_features"][:, -1])
        reward = np.asarray(states.reward)
        speed = np.asarray(states.metrics["speed"])
        track = np.asarray(states.metrics["track"])
        upright = np.asarray(states.metrics["upright"])
        energy = np.asarray(jnp.square(actions).mean(axis=-1))
        done = np.asarray(states.done, dtype=bool)
        for arm in range(2):
            if active[arm]:
                streams[arm]["features"].append(feature[arm])
                streams[arm]["rewards"].append(float(reward[arm]))
                streams[arm]["speeds"].append(float(speed[arm]))
                streams[arm]["tracks"].append(float(track[arm]))
                streams[arm]["uprights"].append(float(upright[arm]))
                streams[arm]["action_energy"].append(float(energy[arm]))
        active &= ~done
        if not active.any():
            break
    return tuple(
        _summarize_stream(stream, seed, steps, env.v_target)
        for stream in streams
    )


def aggregate(episodes: list[dict]) -> dict:
    names = tuple(episodes[0]["metrics"])
    return {
        name: {
            "mean": float(np.mean([episode["metrics"][name] for episode in episodes])),
            "std": float(np.std([episode["metrics"][name] for episode in episodes])),
        }
        for name in names
    }


def finalize_pair(checkpoints, prior, episode_groups, reference) -> tuple[dict, dict]:
    """Score every arm/seed window in one graph and build JSON-sized reports."""

    flattened = [episode for episodes in episode_groups for episode in episodes]
    counts = [len(episode["score_windows"]) for episode in flattened]
    nonempty = [episode["score_windows"] for episode in flattened if len(episode["score_windows"])]
    if not nonempty:
        raise RuntimeError("all policies terminated before the prior warm-up")
    stacked = jnp.asarray(np.concatenate(nonempty))
    command = jnp.asarray(
        (prior.command_scale * prior.source_speed_mps, 0.0, 0.0), jnp.float32
    )
    score = jax.jit(jax.vmap(lambda window: prior.log_prob(window, command)))(stacked)
    score = np.asarray(score)
    offset = 0
    for episode, count in zip(flattened, counts, strict=True):
        values = score[offset : offset + count]
        offset += count
        episode["metrics"]["prior_logp"] = float(np.mean(values)) if count else float("nan")
        episode["metrics"]["prior_windows"] = count
        episode["metrics"]["gait_reference_distance"] = gait_distance(
            episode["metrics"], reference
        )
        del episode["score_windows"]
    return tuple(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "aggregate": aggregate(episodes),
            "episodes": episodes,
        }
        for checkpoint, episodes in zip(checkpoints, episode_groups, strict=True)
    )


def comparison(g0: dict, g1: dict) -> dict:
    g0_episodes, g1_episodes = g0["episodes"], g1["episodes"]

    def paired(name, higher_is_better=True):
        sign = 1.0 if higher_is_better else -1.0
        deltas = np.asarray(
            [
                sign * (right["metrics"][name] - left["metrics"][name])
                for left, right in zip(g0_episodes, g1_episodes, strict=True)
            ]
        )
        return {
            "mean_improvement": float(deltas.mean()),
            "std": float(deltas.std()),
            "wins": int((deltas > 0).sum()),
            "count": len(deltas),
        }

    raw = paired("prior_logp")
    gait = paired("gait_reference_distance", higher_is_better=False)
    track_ratio = g1["aggregate"]["track_mean"]["mean"] / max(
        g0["aggregate"]["track_mean"]["mean"], 1e-8
    )
    survival_ratio = g1["aggregate"]["survival_fraction"]["mean"] / max(
        g0["aggregate"]["survival_fraction"]["mean"], 1e-8
    )
    gates = {
        "track_retention_at_least_95pct": bool(track_ratio >= 0.95),
        "survival_retention_at_least_95pct": bool(survival_ratio >= 0.95),
        "raw_logp_mean_improves": bool(raw["mean_improvement"] > 0),
        "raw_logp_wins_majority": bool(raw["wins"] >= 3),
        "direct_gait_distance_mean_improves": bool(gait["mean_improvement"] > 0),
        "direct_gait_distance_wins_majority": bool(gait["wins"] >= 3),
    }
    return {
        "primary_objective": "paired held-out raw prior_logp improvement",
        "raw_logp": raw,
        "direct_gait_distance": gait,
        "track_retention_ratio": float(track_ratio),
        "survival_retention_ratio": float(survival_ratio),
        "gates": gates,
        "accepted_single_training_seed": all(gates.values()),
        "claim_boundary": (
            "This pairs held-out rollout seeds for one training seed. A workshop-level "
            "algorithm claim additionally requires three matched training seeds."
        ),
    }


def git_provenance() -> dict:
    return {
        "commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=True
        ).stdout.strip(),
        "dirty": bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--dataset-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=EVAL_SEEDS)
    parser.add_argument("--output", type=Path, default=OUT / "evaluation.json")
    args = parser.parse_args()

    prior = load_prior(args.prior)
    reference = reference_summary(args.dataset_root)
    runtime = make_rollout_runtime(prior)
    params = (load_params(args.g0), load_params(args.g1))
    paired_params = stack_params(*params)
    pairs = [paired_rollout(paired_params, runtime, seed, args.steps) for seed in args.seeds]
    episode_groups = tuple([pair[arm] for pair in pairs] for arm in range(2))
    g0, g1 = finalize_pair((args.g0, args.g1), prior, episode_groups, reference)
    report = {
        "schema": "demo-g-evaluation-v1",
        "git": git_provenance(),
        "prior": prior.metadata,
        "evaluation_seeds": args.seeds,
        "episode_steps": args.steps,
        "test_gait_reference": reference,
        "arms": {"g0": g0, "g1": g1},
        "comparison": comparison(g0, g1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["comparison"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
