"""Checkpoint and inference helpers shared by the three Demo E commands."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from brax.training.acme import running_statistics, specs as acme_specs
from brax.training.agents.ppo import networks as ppo_networks

from .config import TRAINING
from .provenance import read_pointer, validate_metadata


def resolve_run(arm: str, path: str | Path | None = None) -> Path:
    run = Path(path).resolve() if path is not None else read_pointer(f"{arm}_policy")
    config = json.loads((run / "config.json").read_text())
    validate_metadata(config)
    if config["demo_e"]["arm"] != arm:
        raise ValueError(
            f"requested {arm}, checkpoint metadata says {config['demo_e']['arm']}"
        )
    return run


def checkpoint_steps(run: Path) -> list[int]:
    return sorted(
        int(path.name)
        for path in run.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def resolve_step(run: Path, step: int | None = None) -> Path:
    steps = checkpoint_steps(run)
    if not steps:
        raise FileNotFoundError(f"no numeric checkpoint under {run}")
    chosen = steps[-1] if step is None else step
    path = run / str(chosen)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def make_network(observation_size, action_size):
    return ppo_networks.make_ppo_networks(
        observation_size,
        action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=TRAINING.policy_layers,
        value_hidden_layer_sizes=TRAINING.value_layers,
        policy_obs_key="state",
        value_obs_key="state",
    )


def load_policy(env, checkpoint: Path):
    state = env.reset(jax.random.PRNGKey(0))
    observation_size = jax.tree.map(lambda value: value.shape[-1], state.obs)
    network = make_network(observation_size, env.action_size)
    normalizer_target = running_statistics.init_state(
        {"state": acme_specs.Array((observation_size["state"],), jnp.float32)}
    )
    target = (
        normalizer_target,
        network.policy_network.init(jax.random.PRNGKey(1)),
        network.value_network.init(jax.random.PRNGKey(2)),
    )
    params = ocp.PyTreeCheckpointer().restore(str(checkpoint.resolve()), item=target)
    inference = ppo_networks.make_inference_fn(network)
    policy = jax.jit(inference(params, deterministic=True))
    return policy, params, network
