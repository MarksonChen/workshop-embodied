"""Small PPO actor-critic shared by both matched rodent conditions."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from demo_c.config import POLICY, TASK

LOG_2PI = 1.8378770664093453


def _squashed_log_prob(mean, log_std, pre_tanh):
    inv_var = torch.exp(-2.0 * log_std)
    normal = -0.5 * ((pre_tanh - mean).square() * inv_var + 2.0 * log_std + LOG_2PI)
    # Stable log(1 - tanh(x)^2), from the SAC change-of-variables identity.
    correction = 2.0 * (torch.log(torch.tensor(2.0, device=mean.device)) - pre_tanh - F.softplus(-2.0 * pre_tanh))
    return (normal - correction).sum(-1)


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int = 2):
        super().__init__()
        layers = []
        width = observation_size
        for _ in range(POLICY.hidden_layers):
            layer = nn.Linear(width, POLICY.hidden_size)
            nn.init.orthogonal_(layer.weight, gain=2 ** 0.5); nn.init.zeros_(layer.bias)
            layers += [layer, nn.Tanh()]
            width = POLICY.hidden_size
        self.encoder = nn.Sequential(*layers)
        self.actor = nn.Linear(width, action_size)
        self.critic = nn.Linear(width, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01); nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0); nn.init.zeros_(self.critic.bias)
        self.log_std = nn.Parameter(torch.full((action_size,), POLICY.initial_log_std))
        self.observation_size = observation_size

    def features(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)

    def forward(self, observation: torch.Tensor):
        feature = self.features(observation)
        return self.actor(feature), self.critic(feature).squeeze(-1)

    def act(self, observation: torch.Tensor, deterministic: bool = False):
        mean, value = self(observation)
        pre_tanh = mean if deterministic else mean + self.log_std.exp() * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        log_prob = _squashed_log_prob(mean, self.log_std, pre_tanh)
        return action, pre_tanh, log_prob, value

    def evaluate_actions(self, observation: torch.Tensor, pre_tanh: torch.Tensor):
        mean, value = self(observation)
        log_prob = _squashed_log_prob(mean, self.log_std, pre_tanh)
        entropy = (self.log_std + 0.5 * (1.0 + LOG_2PI)).sum().expand_as(log_prob)
        return log_prob, entropy, value


def save_policy(path: Path, model: ActorCritic, variant: str, seed: int, metrics: dict, provenance: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "observation_size": model.observation_size,
            "variant": variant,
            "seed": seed,
            "metrics": metrics,
            "provenance": provenance,
        },
        path,
    )


def load_policy(path: Path, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(payload["observation_size"]).to(device)
    model.load_state_dict(payload["state_dict"]); model.eval()
    return model, payload


def heuristic_action(base_observation: torch.Tensor) -> torch.Tensor:
    """Non-learning go-to-goal control; reported honestly as a task-ceiling reference."""
    goal = base_observation[:, :2]
    bearing = torch.atan2(goal[:, 1], goal[:, 0])
    turn = (bearing / TASK.turn_max).clamp(-1.0, 1.0)
    alignment = torch.cos(bearing).clamp(0.0, 1.0)
    forward = 2.0 * (0.25 + 0.75 * alignment) - 1.0
    return torch.stack((forward, turn), -1)
