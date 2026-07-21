"""Pure-JAX frozen Demo H planner and feedback action distribution."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from demo_f.artifacts import sha256
from demo_f.config import (
    FEATURE_CONTRACT_VERSION,
    LEGACY_FEATURE_CONTRACT_VERSION,
)
from demo_f.jax_models import causal_conv, layer_norm, linear, sinusoidal_positions
from demo_h.config import PRIOR_CONTROL_LIMIT, PriorConfig


DEFAULT_PRIOR = (
    Path(__file__).resolve().parent / "out" / "prior_retime_1p75_jax.npz"
)


@dataclass(frozen=True)
class DemoHPrior:
    tokenizer: dict
    predictor: dict
    action_decoder: dict
    feature_mean: jax.Array
    feature_std: jax.Array
    token_mean: jax.Array
    token_std: jax.Array
    command_mean: jax.Array
    command_std: jax.Array
    state_sigma: jax.Array
    metadata: dict
    artifact_sha256: str

    @property
    def heads(self) -> int:
        return int(self.metadata["config"]["transformer_heads"])

    @property
    def layers(self) -> int:
        return int(self.metadata["config"]["transformer_layers"])

    @property
    def action_log_std(self):
        return self.action_decoder["log_std"]

    def encode(self, raw_features):
        values = (raw_features - self.feature_mean) / self.feature_std
        for index, (stride, activation) in enumerate(
            ((1, True), (2, True), (2, False))
        ):
            prefix = f"encoder.{2 * index}.convolution"
            values = causal_conv(
                values,
                self.tokenizer[f"{prefix}.weight"],
                self.tokenizer[f"{prefix}.bias"],
                stride,
            )
            if activation:
                values = jax.nn.silu(values)
        return (values - self.token_mean) / self.token_std

    def _attention(self, values, layer):
        prefix = f"transformer.layers.{layer}"
        qkv = linear(
            values,
            self.predictor[f"{prefix}.self_attn.in_proj_weight"],
            self.predictor[f"{prefix}.self_attn.in_proj_bias"],
        )
        query, key, value = jnp.split(qkv, 3, axis=-1)
        head_dim = values.shape[-1] // self.heads

        def split_heads(tensor):
            return tensor.reshape(tensor.shape[0], self.heads, head_dim).transpose(
                1, 0, 2
            )

        query, key, value = map(split_heads, (query, key, value))
        weights = jax.nn.softmax(
            jnp.einsum("htd,hsd->hts", query, key) / math.sqrt(head_dim), axis=-1
        )
        attended = jnp.einsum("hts,hsd->htd", weights, value)
        attended = attended.transpose(1, 0, 2).reshape(values.shape)
        return linear(
            attended,
            self.predictor[f"{prefix}.self_attn.out_proj.weight"],
            self.predictor[f"{prefix}.self_attn.out_proj.bias"],
        )

    def _context(self, history):
        values = linear(
            history, self.predictor["input.weight"], self.predictor["input.bias"]
        )
        values = values + sinusoidal_positions(
            values.shape[0], values.shape[1], values.dtype
        )
        for layer in range(self.layers):
            prefix = f"transformer.layers.{layer}"
            normalized = layer_norm(
                values,
                self.predictor[f"{prefix}.norm1.weight"],
                self.predictor[f"{prefix}.norm1.bias"],
            )
            values = values + self._attention(normalized, layer)
            normalized = layer_norm(
                values,
                self.predictor[f"{prefix}.norm2.weight"],
                self.predictor[f"{prefix}.norm2.bias"],
            )
            hidden = linear(
                normalized,
                self.predictor[f"{prefix}.linear1.weight"],
                self.predictor[f"{prefix}.linear1.bias"],
            )
            hidden = jax.nn.gelu(hidden, approximate=False)
            values = values + linear(
                hidden,
                self.predictor[f"{prefix}.linear2.weight"],
                self.predictor[f"{prefix}.linear2.bias"],
            )
        values = layer_norm(
            values, self.predictor["norm.weight"], self.predictor["norm.bias"]
        )
        return values[-1]

    def predict_plan(self, history, raw_command):
        command = (raw_command - self.command_mean) / self.command_std
        command = jax.nn.silu(
            linear(
                command,
                self.predictor["command.0.weight"],
                self.predictor["command.0.bias"],
            )
        )
        command = linear(
            command,
            self.predictor["command.2.weight"],
            self.predictor["command.2.bias"],
        )
        hidden = jnp.concatenate((self._context(history), command))
        hidden = jax.nn.silu(
            linear(
                hidden,
                self.predictor["output.0.weight"],
                self.predictor["output.0.bias"],
            )
        )
        return linear(
            hidden,
            self.predictor["output.2.weight"],
            self.predictor["output.2.bias"],
        )

    def state_log_prob(self, history, realized_plan, raw_command):
        """Gaussian next-state-token log likelihood per latent dimension."""

        residual = (realized_plan - self.predict_plan(history, raw_command)) / self.state_sigma
        return -0.5 * (
            jnp.square(residual)
            + 2.0 * jnp.log(self.state_sigma)
            + math.log(2.0 * math.pi)
        ).mean(axis=-1)

    def action_mean(
        self,
        raw_feature,
        plan,
        previous_control,
        phase,
        raw_command,
    ):
        feature = (raw_feature - self.feature_mean) / self.feature_std
        command = (raw_command - self.command_mean) / self.command_std
        previous_control = jnp.clip(
            previous_control, -PRIOR_CONTROL_LIMIT, PRIOR_CONTROL_LIMIT
        )
        values = jnp.concatenate(
            (feature, plan, previous_control, phase, command), axis=-1
        )
        values = jax.nn.silu(
            linear(
                values,
                self.action_decoder["network.0.weight"],
                self.action_decoder["network.0.bias"],
            )
        )
        values = jax.nn.silu(
            linear(
                values,
                self.action_decoder["network.2.weight"],
                self.action_decoder["network.2.bias"],
            )
        )
        correction = linear(
            values,
            self.action_decoder["network.4.weight"],
            self.action_decoder["network.4.bias"],
        )
        previous_mean = jnp.arctanh(previous_control)
        return previous_mean + correction


def load_prior(path: Path = DEFAULT_PRIOR) -> DemoHPrior:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing JAX prior {path}; run `python -m demo_h.export_jax`"
        )
    with np.load(path) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata.get("schema") != "demo-h-jax-prior-v1":
            raise ValueError(f"unsupported schema {metadata.get('schema')!r}")
        feature_contract = metadata.get(
            "feature_contract_version", LEGACY_FEATURE_CONTRACT_VERSION
        )
        if feature_contract != FEATURE_CONTRACT_VERSION:
            raise ValueError(
                f"prior feature contract {feature_contract!r}; "
                f"expected {FEATURE_CONTRACT_VERSION!r}"
            )
        PriorConfig(**metadata["config"]).validate_online_contract()
        groups = {}
        for group in ("tokenizer", "predictor", "action_decoder"):
            groups[group] = {
                name.removeprefix(f"{group}::"): jnp.asarray(archive[name])
                for name in archive.files
                if name.startswith(f"{group}::")
            }
        constants = {
            name: jnp.asarray(archive[name])
            for name in (
                "feature_mean",
                "feature_std",
                "token_mean",
                "token_std",
                "command_mean",
                "command_std",
                "state_sigma",
            )
        }
    return DemoHPrior(
        metadata=metadata,
        artifact_sha256=sha256(path),
        **groups,
        **constants,
    )
