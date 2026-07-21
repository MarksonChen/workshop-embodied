"""Pure-JAX inference for Demo F's frozen conditional Gaussian prior."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from demo_f.config import (
    FEATURE_CONTRACT_VERSION,
    LEGACY_FEATURE_CONTRACT_VERSION,
)
from demo_f.jax_models import causal_conv, layer_norm, linear, sinusoidal_positions

DEFAULT_PRIOR = Path(__file__).resolve().parents[1] / "demo_f" / "out" / "prior_jax.npz"
# Retargeted root orientation is deliberately yaw-only.  These channels encode
# the two out-of-plane entries of the 6-D rotation and roll/pitch angular
# velocity; their training variance is numerical zero, so physical body rocking
# must not dominate the motor-likelihood score.
PLANAR_UNSUPPORTED_FEATURES = (7, 8, 9, 10)


@dataclass(frozen=True)
class DemoFPrior:
    tokenizer: dict
    predictor: dict
    feature_mean: jax.Array
    feature_std: jax.Array
    token_mean: jax.Array
    token_std: jax.Array
    command_mean: jax.Array
    command_std: jax.Array
    sigma: jax.Array
    validation_logp_quantiles: jax.Array
    metadata: dict

    @property
    def heads(self):
        return int(self.metadata["config"]["transformer_heads"])

    @property
    def layers(self):
        return int(self.metadata["config"]["transformer_layers"])

    @property
    def command_scale(self):
        return float(self.metadata["command_scale_fetch_displacement_per_mps"])

    @property
    def source_speed_mps(self):
        scaling = self.metadata.get("dynamic_scaling") or {}
        return float(scaling.get("reference_source_speed_mps", 0.20))

    @property
    def target_speed_fetch(self):
        scaling = self.metadata.get("dynamic_scaling") or {}
        # Legacy kinematic priors were paired with Demo A's 3-unit/s task.
        return float(scaling.get("recommended_fetch_speed", 3.0))

    def encode(self, features):
        values = (features - self.feature_mean) / self.feature_std
        values = values.at[..., jnp.asarray(PLANAR_UNSUPPORTED_FEATURES)].set(0.0)
        for index, (stride, activation) in enumerate(((1, True), (2, True), (2, False))):
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

    def attention(self, values, layer):
        prefix = f"transformer.layers.{layer}"
        qkv = linear(
            values,
            self.predictor[f"{prefix}.self_attn.in_proj_weight"],
            self.predictor[f"{prefix}.self_attn.in_proj_bias"],
        )
        query, key, value = jnp.split(qkv, 3, axis=-1)
        head_dim = values.shape[-1] // self.heads

        def split_heads(tensor):
            return tensor.reshape(tensor.shape[0], self.heads, head_dim).transpose(1, 0, 2)

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

    def context(self, history):
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
            values = values + self.attention(normalized, layer)
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

    def predict(self, history, raw_command):
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
        hidden = jnp.concatenate((self.context(history), command))
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

    def log_prob(self, feature_buffer, raw_command):
        tokens = self.encode(feature_buffer)
        history, realized = tokens[-5:-1], tokens[-1]
        residual = (realized - self.predict(history, raw_command)) / self.sigma
        return -0.5 * (
            jnp.square(residual)
            + 2 * jnp.log(self.sigma)
            + math.log(2 * math.pi)
        ).mean()


def load_prior(path: Path = DEFAULT_PRIOR) -> DemoFPrior:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing JAX prior {path}; run `python -m demo_f.export_jax`")
    with np.load(path) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata.get("schema") != "demo-f-jax-prior-v1":
            raise ValueError(f"unsupported JAX prior schema {metadata.get('schema')!r}")
        feature_contract = metadata.get(
            "feature_contract_version", LEGACY_FEATURE_CONTRACT_VERSION
        )
        if feature_contract != FEATURE_CONTRACT_VERSION:
            raise ValueError(
                f"prior feature contract {feature_contract!r}; "
                f"expected {FEATURE_CONTRACT_VERSION!r}"
            )
        tokenizer = {
            name.removeprefix("tokenizer::"): jnp.asarray(archive[name])
            for name in archive.files
            if name.startswith("tokenizer::")
        }
        predictor = {
            name.removeprefix("predictor::"): jnp.asarray(archive[name])
            for name in archive.files
            if name.startswith("predictor::")
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
                "sigma",
                "validation_logp_quantiles",
            )
        }
    return DemoFPrior(
        tokenizer=tokenizer, predictor=predictor, metadata=metadata, **constants
    )
