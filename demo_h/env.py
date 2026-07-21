"""Demo A's exact task with the causal observations required by Demo H."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from brax.v1 import jumpy as jp

from demo_a.fetch_run import FetchRun
from demo_f.kinematics import fetch_feet
from demo_g.features import transition_feature
from demo_h.config import (
    COMMAND_HORIZON_SECONDS,
    TARGET_SPEED_FETCH,
)
from demo_h.interfaces import ACTION_DIM, BUFFER_FRAMES, OBS_DIM, PHASE_DIM


class DemoHFetchRun(FetchRun):
    """FetchRun plus a reset-safe 16-frame body/action history."""

    def __init__(
        self,
        command=None,
        task_speed_min=None,
        task_speed_max=None,
        **kwargs,
    ):
        kwargs.setdefault("v_target", TARGET_SPEED_FETCH)
        kwargs.setdefault("sigma", TARGET_SPEED_FETCH / 3.0)
        target_speed = float(kwargs["v_target"])
        super().__init__(**kwargs)
        if (task_speed_min is None) != (task_speed_max is None):
            raise ValueError("task speed range needs both endpoints")
        self.task_speed_min = (
            target_speed if task_speed_min is None else float(task_speed_min)
        )
        self.task_speed_max = (
            target_speed if task_speed_max is None else float(task_speed_max)
        )
        if not 0 < self.task_speed_min <= self.task_speed_max:
            raise ValueError((self.task_speed_min, self.task_speed_max))
        self.command_override = (
            None if command is None else jp.array(command, dtype=jp.float32)
        )
        self.foot_indices = jp.array((4, 6, 8, 10))

    def _command(self, target_speed):
        if self.command_override is not None:
            return self.command_override
        return jp.array(
            (target_speed * COMMAND_HORIZON_SECONDS, 0.0, 0.0),
            dtype=jp.float32,
        )

    @property
    def observation_size(self):
        return OBS_DIM

    def _feature(self, previous_qp, qp, observation):
        previous_angles, _ = self.sys.joints[0].angle_vel(previous_qp)
        angles, _ = self.sys.joints[0].angle_vel(qp)
        previous_feet = fetch_feet(previous_angles)
        feet = fetch_feet(angles)
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

    def _observation(self, base_observation, info):
        return jp.concatenate(
            (
                base_observation,
                info["h_feature_buffer"].reshape(-1),
                info["h_previous_control"],
                jax.nn.one_hot(info["h_phase"], PHASE_DIM),
                info["h_plan"],
                info["h_command"],
            )
        )

    def reset(self, rng):
        task_rng, environment_rng = jp.random_split(rng)
        state = super().reset(environment_rng)
        target_speed = jp.random_uniform(
            task_rng,
            low=self.task_speed_min,
            high=self.task_speed_max,
        )
        # A zero-width interval is useful for deterministic evaluation. JAX's
        # uniform sampler permits equal bounds, but this explicit expression
        # also works in NumPy-mode tests.
        target_speed = jp.where(
            self.task_speed_min == self.task_speed_max,
            jp.float32(self.task_speed_min),
            target_speed,
        )
        standing = self._feature(state.qp, state.qp, state.obs)
        info = dict(state.info)
        info.update(
            h_feature_buffer=jp.repeat(standing[None], BUFFER_FRAMES, axis=0),
            h_previous_control=jp.zeros(ACTION_DIM),
            h_phase=jp.int32(0),
            h_plan=jp.zeros(16),
            h_target_speed=target_speed,
            h_command=self._command(target_speed),
        )
        metrics = dict(state.metrics)
        zero = state.reward
        metrics.update(
            task_reward=zero,
            reference_logp=zero,
            reference_reward=zero,
            target_speed=target_speed,
            torso_height=state.qp.pos[self.torso_idx, 2],
            speed_reward=zero,
        )
        return state.replace(
            obs=self._observation(state.obs, info), info=info, metrics=metrics
        )

    def step(self, state, action):
        stepped = super().step(state, action)
        target_speed = state.info["h_target_speed"]
        sigma = target_speed / 3.0
        speed = stepped.metrics["speed"]
        track = jp.exp(-((speed - target_speed) ** 2) / (2.0 * sigma ** 2))
        upright = stepped.metrics["upright"]
        ctrl_cost = self.ctrl_w * jp.sum(jp.square(action))
        feature = self._feature(state.qp, stepped.qp, stepped.obs)
        speed_reward = track + self.upright_w * upright - ctrl_cost
        task_reward = speed_reward
        info = dict(stepped.info)
        buffer = jnp.roll(state.info["h_feature_buffer"], -1, axis=0)
        info.update(
            h_feature_buffer=buffer.at[-1].set(feature),
            h_previous_control=action,
            h_phase=(state.info["h_phase"] + 1) % PHASE_DIM,
            h_plan=state.info["h_plan"],
            h_target_speed=target_speed,
            h_command=state.info["h_command"],
        )
        metrics = dict(stepped.metrics)
        metrics.update(
            speed=speed,
            track=track,
            upright=upright,
            ctrl_cost=ctrl_cost,
            task_reward=task_reward,
            target_speed=target_speed,
            torso_height=stepped.qp.pos[self.torso_idx, 2],
            speed_reward=speed_reward,
        )
        return stepped.replace(
            obs=self._observation(stepped.obs, info),
            reward=task_reward,
            info=info,
            metrics=metrics,
        )
