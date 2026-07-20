import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from demo_b.constants import FULL_FM
from demo_b.evaluate import evaluate as evaluate_demo_b
from demo_b.features import full_motion_features, quat_to_mat
from demo_b.models import load_motor
from demo_b.splits import (
    COLTRANE_PRIOR_TEST_SESSIONS,
    COLTRANE_PRIOR_TRAIN_SESSIONS,
    COLTRANE_PRIOR_VAL_SESSIONS,
    validate_coltrane_prior_split,
)
from demo_e.config import ENV, MIMIC_CHECKPOINT, PIPELINE_VERSION, PRIOR_ASSET
from demo_e.env import (
    COMMAND_HORIZON_SECONDS,
    LikelihoodJoystick,
    build_env,
    normalize_logp,
    replace_command,
)
from demo_e.features import full_motion_feature, joystick_to_hindsight_command
from demo_e.prior import SOURCE_CONSTANT_STD_THRESHOLD, load_prior
from demo_e.provenance import validate_metadata


TORCH_PRIOR_ASSET = (
    Path(__file__).resolve().parents[2]
    / "demo_b"
    / "out"
    / "coltrane_281"
    / "research_contrastive_w10_m01_val_s0.pt"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_prior_split_is_fixed_disjoint_and_session_safe():
    validate_coltrane_prior_split()
    assert tuple(
        map(
            len,
            (
                COLTRANE_PRIOR_TRAIN_SESSIONS,
                COLTRANE_PRIOR_VAL_SESSIONS,
                COLTRANE_PRIOR_TEST_SESSIONS,
            ),
        )
    ) == (8, 5, 5)


def test_time_grid_is_exact():
    assert ENV.controls_per_feature == 2
    assert ENV.controls_per_prior == 8
    assert ENV.episode_length == 1000
    assert ENV.command_resample_steps == 512


def test_joystick_command_is_integrated_along_the_turning_arc():
    straight = np.asarray(
        joystick_to_hindsight_command(
            jnp.asarray([0.3, 0.0]), COMMAND_HORIZON_SECONDS
        )
    )
    np.testing.assert_allclose(straight, [0.186, 0.0, 0.0], atol=1e-7)

    velocity, yaw_rate = 0.3, 0.75
    angle = yaw_rate * COMMAND_HORIZON_SECONDS
    turning = np.asarray(
        joystick_to_hindsight_command(
            jnp.asarray([velocity, yaw_rate]), COMMAND_HORIZON_SECONDS
        )
    )
    np.testing.assert_allclose(
        turning,
        [
            velocity * np.sin(angle) / yaw_rate,
            velocity * (1 - np.cos(angle)) / yaw_rate,
            angle,
        ],
        atol=1e-7,
    )
    assert turning[1] > 0


def test_controller_likelihood_normalization_is_monotone_and_bounded():
    raw = jnp.asarray([-2.0, -1.5, -1.125, -0.75, 0.0])
    np.testing.assert_allclose(
        normalize_logp(raw), np.asarray([0.0, 0.0, 0.5, 1.0, 1.0])
    )


def test_jax_full_feature_matches_numpy_contract():
    rng = np.random.default_rng(4)
    qpos = np.zeros((2, 74), np.float32)
    qpos[:, 2] = 0.08
    qpos[:, 3] = 1.0
    qpos[:, 7:] = rng.normal(0, 0.1, (2, 67))
    qpos[1, :2] += [0.002, -0.001]
    yaw = 0.03
    qpos[1, 3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    local = rng.normal(0, 0.03, (2, 23, 3)).astype(np.float32)
    rotation = quat_to_mat(qpos[:, 3:7])
    world = qpos[:, None, :3] + np.einsum("tij,tkj->tki", rotation, local)
    numpy_value = full_motion_features(qpos, world.transpose(0, 2, 1) * 1000)[1]
    jax_value = np.asarray(
        full_motion_feature(
            jnp.asarray(qpos[0]),
            jnp.asarray(qpos[1]),
            jnp.asarray(local[0]),
            jnp.asarray(local[1]),
        )
    )
    assert numpy_value.shape == jax_value.shape == (FULL_FM,)
    np.testing.assert_allclose(jax_value, numpy_value, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(
    not PRIOR_ASSET.exists() or not TORCH_PRIOR_ASSET.exists(),
    reason="matching PyTorch/JAX prior pair not available",
)
def test_pytorch_jax_prior_parity():
    motion, transition, _, _ = load_motor(TORCH_PRIOR_ASSET)
    prior = load_prior()
    assert prior.metadata["source_asset_sha256"] == _sha256(TORCH_PRIOR_ASSET)
    assert prior.metadata["animal"] == "coltrane"
    assert prior.metadata["feature_dim"] == 281
    rng = np.random.default_rng(7)
    features = rng.normal(size=(3, 24, FULL_FM)).astype(np.float32)
    history = rng.normal(size=(3, 8, 16)).astype(np.float32)
    command = rng.normal(size=(3, 3)).astype(np.float32)
    device = next(transition.parameters()).device
    with torch.inference_mode():
        torch_latent = (
            motion.encode(torch.as_tensor(features, device=device))[0][:, -1]
            .cpu()
            .numpy()
        )
        torch_prediction = (
            transition.predict(
                torch.as_tensor(history, device=device),
                torch.as_tensor(command, device=device),
            )
            .cpu()
            .numpy()
        )
    np.testing.assert_allclose(
        np.asarray(prior.encode(features))[:, -1],
        torch_latent,
        atol=4e-4,
        rtol=4e-4,
    )
    np.testing.assert_allclose(
        prior.predict(history, command),
        torch_prediction,
        atol=3e-3,
        rtol=3e-3,
    )


@pytest.mark.skipif(not PRIOR_ASSET.exists(), reason="JAX prior not exported")
def test_source_constant_features_are_neutral_at_the_physics_bridge():
    prior = load_prior()
    constant = np.where(
        np.asarray(prior.norm["mstd"]) <= SOURCE_CONSTANT_STD_THRESHOLD
    )[0]
    np.testing.assert_array_equal(constant, [75, 142])
    one_standard_deviation = prior.norm["mmean"] + prior.norm["mstd"]
    normalized = np.asarray(prior.normalize_features(one_standard_deviation))
    np.testing.assert_array_equal(normalized[constant], 0.0)
    np.testing.assert_allclose(
        np.delete(normalized, constant), 1.0, atol=1e-5, rtol=1e-5
    )


def test_demo_b_likelihood_is_conditionally_eligible():
    report = evaluate_demo_b()
    speed = report["diagnostics"]["speed_likelihood"]
    assert report["eligible"]
    assert speed["diagonal_wins"] == 5
    assert speed["relative_peak_at_match"]


@pytest.mark.skipif(not PRIOR_ASSET.exists(), reason="JAX prior not exported")
def test_beta_changes_reward_but_not_physics():
    prior = load_prior()
    task = LikelihoodJoystick(beta=0.0, score_motion=True, prior=prior)
    scored = LikelihoodJoystick(beta=1.0, score_motion=True, prior=prior)
    key = jax.random.PRNGKey(9)
    a, b = task.reset(key), scored.reset(key)
    np.testing.assert_allclose(a.data.qpos, b.data.qpos, atol=1e-7)
    action = jnp.linspace(-0.2, 0.2, task.action_size)
    step_a, step_b = jax.jit(task.step), jax.jit(scored.step)
    for _ in range(8):
        a, b = step_a(a, action), step_b(b, action)
    # Separate beta constants compile as separate XLA programs. Their controls
    # are identical; contact-solver reduction order can amplify float32 noise
    # to roughly 1e-4 after eight controls.
    np.testing.assert_allclose(a.data.qpos, b.data.qpos, atol=1.2e-4, rtol=2e-3)
    assert float(a.metrics["r_prior_weighted"]) == 0.0
    assert float(b.metrics["r_prior_weighted"]) >= 0.0
    assert float(a.metrics["r_task"]) == pytest.approx(
        float(b.metrics["r_task"]), abs=2e-4, rel=2e-4
    )


def test_e0_reset_is_the_native_joystick_reset():
    task = LikelihoodJoystick(beta=0.0, score_motion=False)
    key = jax.random.PRNGKey(17)
    aligned = task.reset(key)
    native = super(LikelihoodJoystick, task).reset(key)
    np.testing.assert_array_equal(aligned.data.qpos, native.data.qpos)
    np.testing.assert_array_equal(aligned.data.qvel, native.data.qvel)
    np.testing.assert_array_equal(aligned.info["command"], native.info["command"])
    np.testing.assert_array_equal(aligned.reward, native.reward)
    np.testing.assert_array_equal(aligned.done, native.done)
    for aligned_leaf, native_leaf in zip(
        jax.tree.leaves(aligned.obs), jax.tree.leaves(native.obs), strict=True
    ):
        np.testing.assert_array_equal(aligned_leaf, native_leaf)


@pytest.mark.skipif(not PRIOR_ASSET.exists(), reason="JAX prior not exported")
def test_high_level_contract_and_command_pinning():
    env = build_env(beta=0.0, score_motion=True)
    state = env.reset(jax.random.PRNGKey(0))
    assert env.action_size == 16
    assert state.obs["state"].shape == (56,)
    assert state.info["latent_history"].shape == (8, 16)
    command = jnp.asarray([0.3, -0.5])
    pinned = replace_command(state, command)
    np.testing.assert_allclose(pinned.info["command"], command)
    np.testing.assert_allclose(pinned.obs["state"][-2:], command)
    np.testing.assert_allclose(
        pinned.info["_full_obs"]["state"]["task_obs"][-2:], command
    )


def test_provenance_requires_shared_frozen_decoder():
    valid = {
        "pipeline_version": PIPELINE_VERSION,
        "demo_e": {
            "arm": "e0",
            "trainable_policy": "high_level_intention_policy",
            "policy_from_scratch": True,
            "decoder_frozen": True,
            "decoder_checkpoint": str(MIMIC_CHECKPOINT.resolve()),
        },
    }
    validate_metadata(valid)
    invalid = json.loads(json.dumps(valid))
    invalid["demo_e"]["decoder_frozen"] = False
    with pytest.raises(ValueError, match="frozen"):
        validate_metadata(invalid)
    invalid = json.loads(json.dumps(valid))
    invalid["demo_e"]["decoder_checkpoint"] = "/tmp/other-decoder"
    with pytest.raises(ValueError, match="different"):
        validate_metadata(invalid)
