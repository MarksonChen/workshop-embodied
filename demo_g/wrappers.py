"""Batched frozen-prior reward wrapper for Demo G."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from brax.envs.base import Wrapper
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper

from demo_a.train_fetch import RawKeyVmapWrapper

from .config import DEFAULT_SCORE_STRIDE, PRIOR_LOGP_CENTER, PRIOR_LOGP_SCALE
from .env import BUFFER_FRAMES


class BatchedPriorRewardWrapper(Wrapper):
    """Score one synchronized batch every four simulation frames.

    This wrapper must sit outside ``RawKeyVmapWrapper``.  Its scalar condition is
    therefore a real XLA conditional, and the prior's matrix operations retain a
    batch dimension instead of being embedded once per environment.
    """

    def __init__(
        self,
        env,
        prior,
        beta,
        source_speed_mps=None,
        logp_center=PRIOR_LOGP_CENTER,
        logp_scale=PRIOR_LOGP_SCALE,
        score_stride=DEFAULT_SCORE_STRIDE,
    ):
        super().__init__(env)
        self.prior = prior
        self.beta = float(beta)
        self.logp_center = float(logp_center)
        self.logp_scale = float(logp_scale)
        self.score_stride = int(score_stride)
        if self.score_stride < 1:
            raise ValueError("score_stride must be positive")
        source_speed_mps = (
            prior.source_speed_mps
            if source_speed_mps is None
            else float(source_speed_mps)
        )
        self.command = jnp.asarray(
            (prior.command_scale * source_speed_mps, 0.0, 0.0),
            dtype=jnp.float32,
        )

    def reset(self, rng):
        state = self.env.reset(rng)
        info = dict(state.info)
        # Keep a batch-shaped synchronized phase: RawKeyVmapWrapper receives this
        # state again on step and requires every leaf to have a leading env axis.
        info["prior_phase"] = jnp.zeros_like(state.reward, dtype=jnp.int32)
        return state.replace(info=info)

    def _score_batch(self, feature_buffers):
        return jax.vmap(lambda features: self.prior.log_prob(features, self.command))(
            feature_buffers
        )

    def step(self, state, action):
        stepped = self.env.step(state, action)
        phase = (state.info["prior_phase"] + 1) % self.score_stride
        pipeline_state = stepped.pipeline_state
        pipeline_info = dict(pipeline_state.info)
        count = pipeline_info["prior_count"]
        ready = count >= BUFFER_FRAMES

        if self.beta == 0.0:
            # G0 is scientifically task-only, so avoid carrying a dead 7.8 MB
            # Transformer through its training graph.  Both arms are scored by
            # the same frozen prior afterward in the held-out evaluator.
            raw_logp = jnp.zeros_like(stepped.reward)
            prior_reward = jnp.zeros_like(stepped.reward)
        else:
            old_logp = pipeline_info["prior_logp"]
            update = phase[0] == 0
            candidate = jax.lax.cond(
                update,
                lambda features: self._score_batch(features),
                lambda _: old_logp,
                pipeline_info["prior_features"],
            )
            raw_logp = jnp.where(ready, candidate, 0.0)
            prior_reward = jnp.where(
                ready,
                jax.nn.sigmoid(
                    (raw_logp - self.logp_center) / self.logp_scale
                ),
                0.0,
            )

        task_reward = stepped.reward
        reward = (
            task_reward
            if self.beta == 0.0
            else task_reward + self.beta * prior_reward
        )
        pipeline_info.update(prior_logp=raw_logp, prior_reward=prior_reward)
        pipeline_state = pipeline_state.replace(info=pipeline_info)

        metrics = dict(stepped.metrics)
        metrics.update(
            task_reward=task_reward,
            prior_logp=raw_logp,
            prior_reward=prior_reward,
            prior_active=ready.astype(jnp.float32),
        )
        info = dict(stepped.info)
        info["prior_phase"] = phase
        return stepped.replace(
            pipeline_state=pipeline_state,
            reward=reward,
            metrics=metrics,
            info=info,
        )


def wrap_demo_g_for_training(
    env,
    episode_length=1000,
    action_repeat=1,
    *,
    prior,
    beta,
    score_stride=DEFAULT_SCORE_STRIDE,
    **_,
):
    """Raw-key-safe Demo A wrappers plus one batched Demo F scorer."""

    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = RawKeyVmapWrapper(env)
    env = BatchedPriorRewardWrapper(
        env, prior=prior, beta=beta, score_stride=score_stride
    )
    env = AutoResetWrapper(env)
    return env
