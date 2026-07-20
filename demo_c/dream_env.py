"""Vectorized short-horizon navigation inside the frozen Demo B world model."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from demo_c.config import BASE_OBS_SIZE, TASK, WAM_CONTEXT_SIZE
from demo_c.motor import FrozenMotor


@dataclass
class DreamState:
    history: torch.Tensor
    xy: torch.Tensor
    yaw: torch.Tensor
    goal: torch.Tensor
    body_velocity: torch.Tensor
    previous_action: torch.Tensor
    elapsed: torch.Tensor
    episode_return: torch.Tensor


class RodentDreamEnv:
    """A pedagogical model-based environment: goals/reward are known, motion is learned."""

    action_size = 2

    def __init__(
        self,
        motor: FrozenMotor,
        num_envs: int,
        use_wam_context: bool,
        seed: int,
        auto_reset: bool = True,
    ):
        self.motor = motor
        self.device = motor.device
        self.num_envs = num_envs
        self.use_wam_context = use_wam_context
        self.observation_size = BASE_OBS_SIZE + (WAM_CONTEXT_SIZE if use_wam_context else 0)
        self.auto_reset = auto_reset
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.state = self._fresh_state(num_envs)

    def _fresh_state(self, count: int) -> DreamState:
        history = self.motor.sample_histories(count, self.generator)
        xy = torch.zeros((count, 2), device=self.device)
        yaw = (torch.rand(count, generator=self.generator, device=self.device) * 2 - 1) * torch.pi
        relative_bearing = (torch.rand(count, generator=self.generator, device=self.device) * 2 - 1) * TASK.goal_bearing_max
        bearing = yaw + relative_bearing
        radius = TASK.goal_radius_min + torch.rand(
            count, generator=self.generator, device=self.device
        ) * (TASK.goal_radius_max - TASK.goal_radius_min)
        goal = torch.stack((torch.cos(bearing), torch.sin(bearing)), -1) * radius[:, None]
        return DreamState(
            history=history,
            xy=xy,
            yaw=yaw,
            goal=goal,
            body_velocity=torch.zeros((count, 3), device=self.device),
            previous_action=torch.zeros((count, 2), device=self.device),
            elapsed=torch.zeros(count, dtype=torch.long, device=self.device),
            episode_return=torch.zeros(count, device=self.device),
        )

    @staticmethod
    def _select(mask: torch.Tensor, fresh: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        shape = (len(mask),) + (1,) * (old.ndim - 1)
        return torch.where(mask.reshape(shape), fresh, old)

    def _reset_done(self, done: torch.Tensor) -> None:
        if not done.any():
            return
        fresh = self._fresh_state(self.num_envs)
        for name in DreamState.__dataclass_fields__:
            old_value = getattr(self.state, name)
            fresh_value = getattr(fresh, name)
            setattr(self.state, name, self._select(done, fresh_value, old_value))

    def base_observation(self) -> torch.Tensor:
        state = self.state
        dxy = state.goal - state.xy
        c, s = torch.cos(-state.yaw), torch.sin(-state.yaw)
        goal_local = torch.stack((c * dxy[:, 0] - s * dxy[:, 1], s * dxy[:, 0] + c * dxy[:, 1]), -1)
        goal_local = goal_local / TASK.goal_radius_max
        distance = dxy.norm(dim=-1, keepdim=True) / TASK.goal_radius_max
        velocity_scale = torch.tensor((0.35, 0.20, 1.2), device=self.device)
        velocity = (state.body_velocity / velocity_scale).clamp(-2.0, 2.0)
        obs = torch.cat((goal_local, distance, velocity, state.previous_action), dim=-1)
        assert obs.shape == (self.num_envs, BASE_OBS_SIZE)
        return obs

    @torch.inference_mode()
    def observe(self) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.motor.context(self.state.history)
        base = self.base_observation()
        return (torch.cat((base, context), -1) if self.use_wam_context else base), context

    @torch.inference_mode()
    def step(self, action: torch.Tensor, context: torch.Tensor | None = None):
        state = self.state
        old_distance = (state.goal - state.xy).norm(dim=-1)
        future, body_delta, invalid = self.motor.advance(state.history, action, context)
        c, s = torch.cos(state.yaw), torch.sin(state.yaw)
        world_delta = torch.stack(
            (c * body_delta[:, 0] - s * body_delta[:, 1], s * body_delta[:, 0] + c * body_delta[:, 1]), -1
        )
        state.history = future
        state.xy = state.xy + world_delta
        state.yaw = torch.atan2(
            torch.sin(state.yaw + body_delta[:, 2]), torch.cos(state.yaw + body_delta[:, 2])
        )
        state.body_velocity = body_delta / TASK.step_seconds
        state.previous_action = action.clamp(-1.0, 1.0)
        state.elapsed += 1

        new_distance = (state.goal - state.xy).norm(dim=-1)
        success = new_distance < TASK.reach_radius
        timeout = state.elapsed >= TASK.horizon
        done = success | timeout | invalid
        reward = (
            TASK.progress_scale * (old_distance - new_distance)
            + TASK.arrival_bonus * success.float()
            - TASK.time_cost
            - TASK.turn_cost * action[:, 1].square()
            - TASK.invalid_penalty * invalid.float()
        )
        state.episode_return += reward
        terminal_return = state.episode_return.clone()
        terminal_length = state.elapsed.clone()
        info = {
            "success": success,
            "invalid": invalid,
            "distance": new_distance,
            "body_delta": body_delta,
            "episode_return": terminal_return,
            "episode_length": terminal_length,
        }
        if self.auto_reset:
            self._reset_done(done)
        return reward, done, info
