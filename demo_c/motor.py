"""Thin adapter around Demo B's frozen action-conditioned motor model.

The transition is the world factor learned with SSL in Demo B. Demo C never updates its
weights: PPO sees it only as an environment transition and (for the ``wam`` condition)
as a predictive context provider.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from demo_c.config import TASK

ROOT = Path(__file__).resolve().parents[1]
DEMO_B = ROOT / "demo_b"
if str(DEMO_B) not in sys.path:
    sys.path.insert(0, str(DEMO_B))

# Demo B is intentionally runnable as a standalone directory and therefore uses local
# imports. Adding its directory above is its supported integration boundary.
from constants import CLIP, DM, FM, FPS, H, K  # type: ignore  # noqa: E402
from models import ASSETS, MotionVAE, SimpleTrans  # type: ignore  # noqa: E402


def _sixd_yaw(d6: torch.Tensor) -> torch.Tensor:
    """Yaw of a 6-D rotation representation, with safe Gram--Schmidt."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=-1, eps=1e-6)
    # Rotation columns are [b1,b2,b3], so yaw = atan2(R10, R00).
    return torch.atan2(b1[..., 1], b1[..., 0])


class FrozenMotor:
    """Frozen Demo B tokenizer + transition with batched planar integration."""

    def __init__(self, device: str | torch.device = "cuda", checkpoint: Path | None = None):
        self.device = torch.device(device)
        broadened = ROOT / "demo_c" / "out" / "world" / "world_seed0.pt"
        checkpoint = checkpoint or (broadened if broadened.exists() else ASSETS / "motor_standalone.pt")
        self.checkpoint = Path(checkpoint).resolve()
        payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self.world_training = payload.get("world_training", {})
        self.motion = MotionVAE().to(self.device)
        self.motion.load_state_dict(payload["motion"])
        self.transition = SimpleTrans(**payload["model_cfg"]).to(self.device)
        self.transition.load_state_dict(payload["trans"])
        self.motion.eval(); self.transition.eval()
        for module in (self.motion, self.transition):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.norms = {
            key: torch.as_tensor(payload[key], dtype=torch.float32, device=self.device)
            for key in ("zmean", "zstd", "cmean", "cstd", "mmean", "mstd")
        }
        self.seed_feat = np.asarray(payload["seed_feat"], np.float32)
        self.seed_name = str(payload.get("seed_name", "bundled locomotion seed"))
        self.initial_histories = self._build_initial_histories()

    @torch.inference_mode()
    def _build_initial_histories(self) -> torch.Tensor:
        """Encode overlapping *real contiguous* seed windows; never stitch clips."""
        starts = list(range(0, len(self.seed_feat) - CLIP + 1, CLIP // 2))
        clips = np.stack([self.seed_feat[s:s + CLIP] for s in starts])
        x = torch.as_tensor(clips, device=self.device)
        x = (x - self.norms["mmean"]) / self.norms["mstd"]
        mu = self.motion.encode(x)[0]
        zn = (mu - self.norms["zmean"]) / self.norms["zstd"]
        # Both halves are genuine histories. More reset diversity reduces policy reliance
        # on a single gait phase while retaining real temporal continuity.
        return torch.cat((zn[:, :H], zn[:, -H:]), dim=0).contiguous()

    def sample_histories(self, count: int, generator: torch.Generator) -> torch.Tensor:
        idx = torch.randint(
            len(self.initial_histories), (count,), generator=generator, device=self.device
        )
        return self.initial_histories[idx].clone()

    @torch.inference_mode()
    def context(self, history: torch.Tensor) -> torch.Tensor:
        return self.transition.context(history)

    def action_to_command(self, action: torch.Tensor) -> torch.Tensor:
        """Map bounded PPO actions to in-distribution Demo B displacements."""
        forward = TASK.forward_min + (action[:, 0] + 1.0) * 0.5 * (
            TASK.forward_max - TASK.forward_min
        )
        turn = TASK.turn_max * action[:, 1]
        return torch.stack((forward, torch.zeros_like(forward), turn), dim=-1)

    def command_to_action(self, command: torch.Tensor) -> torch.Tensor:
        forward = 2.0 * (command[..., 0] - TASK.forward_min) / (
            TASK.forward_max - TASK.forward_min
        ) - 1.0
        turn = command[..., 2] / TASK.turn_max
        return torch.stack((forward, turn), dim=-1).clamp(-1.0, 1.0)

    @torch.inference_mode()
    def advance(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
        decode: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one 0.64-s dream step.

        Returns ``(future_history, body_delta=[dx,dy,dyaw], invalid)``. The planar
        displacement is integrated from the decoder's predicted local velocity and 6-D
        orientation channels; it is not copied from the command.
        """
        command = self.action_to_command(action.clamp(-1.0, 1.0))
        command_n = (command - self.norms["cmean"]) / self.norms["cstd"]
        if context is None:
            context = self.transition.context(history)
        future = self.transition.predict_from_context(context, command_n)
        invalid = ~torch.isfinite(future).flatten(1).all(1)
        future = torch.nan_to_num(future)

        if not decode:
            empty = torch.zeros((len(history), 3), device=self.device)
            return future, empty, invalid

        latent = torch.cat((history, future), dim=1)
        latent = latent * self.norms["zstd"] + self.norms["zmean"]
        features = self.motion.decode(latent) * self.norms["mstd"] + self.norms["mmean"]
        features = features[:, CLIP // 2:]
        vxy = features[..., :2]
        dyaw = _sixd_yaw(features[..., 3:9])

        rel_xy = torch.zeros((len(history), 2), device=self.device)
        rel_yaw = torch.zeros(len(history), device=self.device)
        for frame in range(features.shape[1]):
            c, s = torch.cos(rel_yaw), torch.sin(rel_yaw)
            vx, vy = vxy[:, frame, 0] / FPS, vxy[:, frame, 1] / FPS
            rel_xy[:, 0] += c * vx - s * vy
            rel_xy[:, 1] += s * vx + c * vy
            rel_yaw += dyaw[:, frame]
        rel_yaw = torch.atan2(torch.sin(rel_yaw), torch.cos(rel_yaw))
        delta = torch.cat((rel_xy, rel_yaw[:, None]), dim=-1)
        invalid |= ~torch.isfinite(delta).all(1)
        # A hard gate prevents a single corrupt prediction from poisoning a PPO batch;
        # it is surfaced as terminal failure rather than silently counted as success.
        invalid |= (rel_xy.norm(dim=-1) > 0.5) | (rel_yaw.abs() > math.pi)
        delta = torch.nan_to_num(delta)
        return future, delta, invalid


__all__ = ["FrozenMotor", "CLIP", "DM", "FM", "FPS", "H", "K"]
