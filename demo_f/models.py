"""Small, independently tunable conditional motion model for Demo F."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel: int, stride: int = 1):
        super().__init__()
        self.padding = kernel - stride
        self.convolution = nn.Conv1d(inputs, outputs, kernel, stride=stride)

    def forward(self, values):
        return self.convolution(F.pad(values, (self.padding, 0)))


class CausalConvTranspose1d(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel: int, stride: int):
        super().__init__()
        self.trim = kernel - stride
        self.convolution = nn.ConvTranspose1d(inputs, outputs, kernel, stride=stride)

    def forward(self, values):
        output = self.convolution(values)
        return output[..., :-self.trim] if self.trim else output


class MotionAutoencoder(nn.Module):
    """A causal 4x temporal tokenizer: ``(B,64,F) <-> (B,16,D)``."""

    def __init__(self, feature_dim: int, hidden: int = 192, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            CausalConv1d(feature_dim, hidden, 5),
            nn.SiLU(),
            CausalConv1d(hidden, hidden, 4, stride=2),
            nn.SiLU(),
            CausalConv1d(hidden, latent_dim, 4, stride=2),
        )
        self.decoder = nn.Sequential(
            CausalConv1d(latent_dim, hidden, 3),
            nn.SiLU(),
            CausalConvTranspose1d(hidden, hidden, 4, stride=2),
            nn.SiLU(),
            CausalConvTranspose1d(hidden, feature_dim, 4, stride=2),
        )

    def encode(self, features):
        return self.encoder(features.transpose(1, 2)).transpose(1, 2)

    def decode(self, tokens):
        return self.decoder(tokens.transpose(1, 2)).transpose(1, 2)

    def forward(self, features):
        return self.decode(self.encode(features))


def sinusoidal_positions(length: int, width: int, device) -> torch.Tensor:
    position = torch.arange(length, device=device)[:, None].float()
    frequency = torch.exp(
        torch.arange(0, width, 2, device=device).float() * (-math.log(10_000.0) / width)
    )
    output = torch.zeros(length, width, device=device)
    output[:, 0::2] = torch.sin(position * frequency)
    output[:, 1::2] = torch.cos(position * frequency)
    return output


class ConditionalTransformer(nn.Module):
    """Predict future motion-token means from causal history and a command."""

    def __init__(
        self,
        latent_dim: int = 16,
        future_tokens: int = 8,
        width: int = 192,
        layers: int = 4,
        heads: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.future_tokens = future_tokens
        self.input = nn.Linear(latent_dim, width)
        block = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(width)
        self.command = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64))
        self.output = nn.Sequential(
            nn.Linear(width + 64, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, future_tokens * latent_dim),
        )

    def context(self, history):
        hidden = self.input(history)
        hidden = hidden + sinusoidal_positions(
            hidden.shape[1], hidden.shape[2], hidden.device
        )[None]
        return self.norm(self.transformer(hidden))[:, -1]

    def predict_from_context(self, context, command):
        output = self.output(torch.cat((context, self.command(command)), dim=-1))
        return output.view(-1, self.future_tokens, self.latent_dim)

    def predict(self, history, command):
        return self.predict_from_context(self.context(history), command)

    def log_prob(self, history, future, command, sigma):
        sigma = torch.as_tensor(sigma, dtype=history.dtype, device=history.device)
        residual = (future - self.predict(history, command)) / sigma
        return -0.5 * (
            residual.square() + 2 * sigma.log() + math.log(2 * math.pi)
        ).mean(dim=(-1, -2))
