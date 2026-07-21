"""Batched reference cross-entropy whose entropy pair is exact KL."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from brax.envs.base import Wrapper
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from brax.training import distribution

from demo_a.train_fetch import RawKeyVmapWrapper
from demo_h.config import ACTION_DIM, PHASE_SLICE, PLAN_SLICE
from demo_h.policy import compute_plans, reference_parameters


class BatchedPlanWrapper(Wrapper):
    """Refresh the frozen motion plan once per four 50 Hz control frames."""

    def __init__(self, env, prior):
        super().__init__(env)
        self.prior = prior

    def _set_plan(self, state, plan):
        observation = state.obs.at[..., PLAN_SLICE].set(plan)
        pipeline = state.pipeline_state
        pipeline_info = dict(pipeline.info)
        pipeline_info["h_plan"] = plan
        pipeline = pipeline.replace(
            obs=pipeline.obs.at[..., PLAN_SLICE].set(plan),
            info=pipeline_info,
        )
        return state.replace(obs=observation, pipeline_state=pipeline)

    def reset(self, rng):
        state = self.env.reset(rng)
        return self._set_plan(state, compute_plans(self.prior, state.obs))

    def step(self, state, action):
        old_plan = state.obs[..., PLAN_SLICE]
        stepped = self.env.step(state, action)
        phase = jnp.argmax(stepped.obs[..., PHASE_SLICE], axis=-1)
        refresh = phase == 0
        candidate = jax.lax.cond(
            jnp.any(refresh),
            lambda observation: compute_plans(self.prior, observation),
            lambda _: old_plan,
            stepped.obs,
        )
        plan = jnp.where(refresh[..., None], candidate, old_plan)
        return self._set_plan(stepped, plan)


class BatchedReferenceWrapper(Wrapper):
    """Add beta * log p0(a|s) / dim to the physical task reward.

    Setting PPO's entropy coefficient to ``beta / dim`` makes this term plus
    policy entropy exactly ``-beta * KL(pi || p0) / dim`` in expectation.
    """

    def __init__(self, env, prior, beta):
        super().__init__(env)
        self.prior = prior
        self.beta = float(beta)
        self.distribution = distribution.NormalTanhDistribution(ACTION_DIM)

    def step(self, state, action):
        if self.beta:
            parameters = reference_parameters(self.prior, state.obs)
            raw_action = jnp.arctanh(jnp.clip(action, -0.999999, 0.999999))
            reference_logp = self.distribution.log_prob(parameters, raw_action) / ACTION_DIM
            reference_reward = self.beta * reference_logp
        else:
            reference_logp = jnp.zeros_like(state.reward)
            reference_reward = jnp.zeros_like(state.reward)
        stepped = self.env.step(state, action)
        task_reward = stepped.reward
        metrics = dict(stepped.metrics)
        metrics.update(
            task_reward=task_reward,
            reference_logp=reference_logp,
            reference_reward=reference_reward,
        )
        return stepped.replace(
            reward=task_reward + reference_reward,
            metrics=metrics,
        )


def wrap_demo_h_for_training(
    env,
    episode_length=1000,
    action_repeat=1,
    *,
    prior,
    beta,
    **_,
):
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = RawKeyVmapWrapper(env)
    env = BatchedPlanWrapper(env, prior)
    env = BatchedReferenceWrapper(env, prior, beta)
    return AutoResetWrapper(env)
