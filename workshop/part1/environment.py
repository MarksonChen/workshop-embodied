from __future__ import annotations

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from brax.v1 import jumpy as jp
from brax.v1 import math
from brax.v1.envs.fetch import Fetch


class FetchRun(Fetch):
    def __init__(
        self,
        target_speed: float = 3.0,
        speed_width: float = 1.0,
        upright_weight: float = 0.1,
        control_weight: float = 1e-3,
    ):
        super().__init__()
        self.target_speed = target_speed
        self.speed_width = speed_width
        self.upright_weight = upright_weight
        self.control_weight = control_weight
        standing_height = float(self.sys.default_qp().pos[self.torso_idx, 2])
        self.minimum_height = 0.5 * standing_height

    def reset(self, rng):
        state = super().reset(rng)
        zero = state.reward
        metrics = {
            "speed": zero,
            "track": zero,
            "upright": zero,
            "ctrl_cost": zero,
        }
        return state.replace(metrics=metrics)

    def step(self, state, action):
        qp, info = self.sys.step(state.qp, action)
        observation = self._get_obs(qp, info)
        velocity = (
            qp.pos[self.torso_idx] - state.qp.pos[self.torso_idx]
        ) / self.sys.config.dt
        speed = velocity[0]
        track = jp.exp(
            -((speed - self.target_speed) ** 2) / (2.0 * self.speed_width**2)
        )
        world_up = jp.array([0.0, 0.0, 1.0])
        torso_up = math.rotate(world_up, qp.rot[self.torso_idx])
        upright = jp.dot(torso_up, world_up)
        ctrl_cost = self.control_weight * jp.sum(jp.square(action))
        reward = track + self.upright_weight * upright - ctrl_cost
        fell = (qp.pos[self.torso_idx, 2] < self.minimum_height) | (upright < 0.0)
        done = jp.where(fell, jp.float32(1), jp.float32(0))
        state.metrics.update(
            speed=speed,
            track=track,
            upright=upright,
            ctrl_cost=ctrl_cost,
        )
        return state.replace(
            qp=qp,
            obs=observation,
            reward=reward,
            done=done,
        )


class FetchV2(Env):
    def __init__(self, inner: FetchRun):
        self._env = inner

    def reset(self, rng) -> State:
        state = self._env.reset(rng)
        return State(
            pipeline_state=state,
            obs=state.obs,
            reward=state.reward,
            done=state.done,
            metrics=dict(state.metrics),
            info={},
        )

    def step(self, state: State, action) -> State:
        next_state = self._env.step(state.pipeline_state, action)
        metrics = dict(state.metrics)
        metrics.update(next_state.metrics)
        return state.replace(
            pipeline_state=next_state,
            obs=next_state.obs,
            reward=next_state.reward,
            done=next_state.done,
            metrics=metrics,
        )

    @property
    def observation_size(self):
        return self._env.observation_size

    @property
    def action_size(self):
        return self._env.action_size

    @property
    def backend(self):
        return "generalized"


class RawKeyVmapWrapper(Wrapper):
    def reset(self, rng) -> State:
        if rng.dtype != jnp.uint32:
            rng = jax.random.key_data(rng)
        return jax.vmap(self.env.reset)(rng)

    def step(self, state: State, action) -> State:
        return jax.vmap(self.env.step)(state, action)


def wrap_training(environment, episode_length=1000, action_repeat=1, **_):
    environment = EpisodeWrapper(environment, episode_length, action_repeat)
    environment = RawKeyVmapWrapper(environment)
    return AutoResetWrapper(environment)
