"""Demo A's FetchRun task plus Demo F's online feature history.

The expensive frozen-prior inference deliberately lives *outside* this v1
environment.  Brax vmaps v1 environments one at a time; putting the Transformer
here turns a per-environment conditional into a much larger compiled graph.
``demo_g.wrappers`` instead scores all environments in one batched operation.
"""

from __future__ import annotations

import jax.numpy as jnp

from brax.v1 import jumpy as jp

from demo_a.fetch_run import FetchRun
from demo_f.kinematics import fetch_feet

from .features import transition_feature


BUFFER_FRAMES = 32


class DemoGFetchRun(FetchRun):
    """Keep Demo A's task exact while collecting Demo F feature windows."""

    def __init__(
        self,
        prior,
        source_speed_mps: float | None = None,
        target_speed_fetch: float | None = None,
        **kwargs,
    ):
        source_speed_mps = (
            prior.source_speed_mps
            if source_speed_mps is None
            else float(source_speed_mps)
        )
        target_speed_fetch = (
            prior.target_speed_fetch
            if target_speed_fetch is None
            else float(target_speed_fetch)
        )
        # Preserve Demo A's dimensionless target-width ratio (sigma / speed =
        # 1/3) at the dynamically matched Fetch speed.  Keeping sigma=1 would
        # reward standing still at 65% of the maximum for the 0.925-unit/s task.
        kwargs.setdefault("sigma", target_speed_fetch / 3.0)
        super().__init__(v_target=target_speed_fetch, **kwargs)
        self.prior = prior
        self.source_speed_mps = source_speed_mps
        self.prior_command = jp.array(
            (prior.command_scale * source_speed_mps, 0.0, 0.0), dtype=jp.float32
        )
        self.foot_indices = jp.array(
            tuple(
                self.sys.body.index[name]
                for name in (
                    "Front Right Lower",
                    "Front Left Lower",
                    "Back Right Lower",
                    "Back Left Lower",
                )
            )
        )

    def _joint_state(self, qp):
        return self.sys.joints[0].angle_vel(qp)

    def _feature(self, previous_qp, qp, observation):
        previous_angles, _ = self._joint_state(previous_qp)
        angles, _ = self._joint_state(qp)
        previous_feet = fetch_feet(previous_angles)
        feet = fetch_feet(angles)
        # Fetch's final observation block is one contact bit per body.
        contacts = observation[-qp.pos.shape[0] :][self.foot_indices]
        return transition_feature(
            previous_qp.pos[self.torso_idx],
            qp.pos[self.torso_idx],
            previous_qp.rot[self.torso_idx],
            qp.rot[self.torso_idx],
            previous_angles,
            angles,
            previous_feet,
            feet,
            contacts,
        )

    def reset(self, rng):
        state = super().reset(rng)
        zero = state.reward
        info = dict(state.info)
        info.update(
            prior_features=jp.zeros((BUFFER_FRAMES, 60)),
            prior_count=jp.int32(0),
            prior_logp=zero,
            prior_reward=zero,
        )
        metrics = dict(state.metrics)
        metrics.update(
            task_reward=zero,
            prior_logp=zero,
            prior_reward=zero,
            prior_active=zero,
        )
        return state.replace(info=info, metrics=metrics)

    def step(self, state, action):
        stepped = super().step(state, action)
        feature = self._feature(state.qp, stepped.qp, stepped.obs)
        feature_buffer = jnp.roll(state.info["prior_features"], -1, axis=0)
        feature_buffer = feature_buffer.at[-1].set(feature)
        count = state.info["prior_count"] + 1
        task_reward = stepped.reward
        info = dict(stepped.info)
        info.update(
            prior_features=feature_buffer,
            prior_count=count,
        )
        metrics = dict(stepped.metrics)
        metrics.update(
            task_reward=task_reward,
        )
        return stepped.replace(info=info, metrics=metrics)
