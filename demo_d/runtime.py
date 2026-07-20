"""Environment, checkpoint, and policy helpers shared by Demo D CLIs."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from brax.training.acme import running_statistics, specs as acme_specs
from brax.training.agents.ppo import networks as ppo_networks

from demo_d.config import ALL_CLIPS, TRAINING, VAL_CLIPS
from demo_d.env import HindsightCommandImitation, default_config
from demo_d.provenance import read_pointer, validate_scratch_metadata
from demo_d.reference import load_clips, split_loaded_clips


def load_split_environments():
    clips = load_clips(ALL_CLIPS)
    train_clips, val_clips = split_loaded_clips(clips)
    train_env = HindsightCommandImitation(config=default_config(training=True), clips=train_clips)
    val_env = HindsightCommandImitation(config=default_config(), clips=val_clips)
    return train_env, val_env


def load_validation_environment():
    return HindsightCommandImitation(config=default_config(), clips=load_clips(VAL_CLIPS))


def resolve_run(path: str | Path | None = None) -> Path:
    run = Path(path).resolve() if path is not None else read_pointer("policy")
    config_path = run / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing {config_path}")
    validate_scratch_metadata(json.loads(config_path.read_text()))
    return run


def checkpoint_steps(run: Path) -> list[int]:
    return sorted(int(p.name) for p in run.iterdir() if p.is_dir() and p.name.isdigit())


def resolve_step(run: Path, step: int | None = None) -> Path:
    steps = checkpoint_steps(run)
    if not steps:
        raise FileNotFoundError(f"no numeric checkpoints under {run}")
    chosen = steps[-1] if step is None else step
    path = run / str(chosen)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def make_network(obs_size, action_size):
    return ppo_networks.make_ppo_networks(
        obs_size,
        action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=TRAINING.policy_layers,
        value_hidden_layer_sizes=TRAINING.value_layers,
        policy_obs_key="state",
        value_obs_key="state",
    )


def load_policy(env, step_path: Path):
    step_path = step_path.resolve()
    state = env.reset(jax.random.PRNGKey(0))
    obs_size = jax.tree.map(lambda x: x.shape[-1], state.obs)
    network = make_network(obs_size, env.action_size)
    normalizer_target = running_statistics.init_state(
        {"state": acme_specs.Array((obs_size["state"],), jnp.float32)}
    )
    target = (
        normalizer_target,
        network.policy_network.init(jax.random.PRNGKey(1)),
        network.value_network.init(jax.random.PRNGKey(2)),
    )
    params = ocp.PyTreeCheckpointer().restore(str(step_path), item=target)
    make_policy = ppo_networks.make_inference_fn(network)
    return jax.jit(make_policy(params, deterministic=True)), params


def latest_metrics_path(run: Path) -> Path:
    files = sorted(run.glob("progress_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no progress log under {run}")
    return files[-1]
