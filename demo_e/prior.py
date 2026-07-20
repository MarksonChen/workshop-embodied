"""Pure-JAX inference for Demo B's frozen causal Gaussian motion prior."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import PRIOR_ASSET


SOURCE_CONSTANT_STD_THRESHOLD = 1.0001e-4


def _linear(x, weight, bias):
    return x @ weight.T + bias


def _layer_norm(x, weight, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * weight + bias


def _causal_conv(x, weight, bias, stride):
    """PyTorch Conv1d cross-correlation with Demo B's exact left padding."""
    unbatched = x.ndim == 2
    if unbatched:
        x = x[None]
    kernel = jnp.transpose(weight, (2, 1, 0))
    width = kernel.shape[0]
    left = (width - 1) + (1 - stride)
    output = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(stride,),
        padding=((left, 0),),
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    output = output + bias
    return output[0] if unbatched else output


def _position_encoding(length: int, width: int, dtype):
    position = jnp.arange(length, dtype=dtype)[:, None]
    frequency = jnp.exp(
        jnp.arange(0, width, 2, dtype=dtype) * (-math.log(10000.0) / width)
    )
    phase = position * frequency[None]
    encoding = jnp.zeros((length, width), dtype=dtype)
    encoding = encoding.at[:, 0::2].set(jnp.sin(phase))
    return encoding.at[:, 1::2].set(jnp.cos(phase))


class FrozenMotionPrior:
    """Inference-only arrays; this object is never part of PPO parameters."""

    def __init__(self, arrays: dict[str, np.ndarray], metadata: dict):
        self.parameters = {
            key: jnp.asarray(value) for key, value in arrays.items() if key.startswith(("motion/", "trans/"))
        }
        self.norm = {
            key.removeprefix("norm/"): jnp.asarray(value)
            for key, value in arrays.items()
            if key.startswith("norm/")
        }
        # Demo B stores latent statistics with a leading singleton batch
        # dimension for PyTorch broadcasting.  Removing it here preserves the
        # same arithmetic while keeping unbatched MJX rollouts unbatched.
        self.norm["zmean"] = self.norm["zmean"].reshape(-1)
        self.norm["zstd"] = self.norm["zstd"].reshape(-1)
        self.metadata = metadata
        self.layers = int(metadata["model_cfg"]["layers"])
        self.heads = int(metadata["model_cfg"]["heads"])

    def _p(self, name):
        return self.parameters[name]

    def encode(self, normalized_features):
        """Encode a causal feature stream; only ``mu`` is used by Demo E."""
        x = _causal_conv(
            normalized_features,
            self._p("motion/enc.0.conv.weight"),
            self._p("motion/enc.0.conv.bias"),
            1,
        )
        x = jax.nn.silu(x)
        x = _causal_conv(x, self._p("motion/enc.2.conv.weight"), self._p("motion/enc.2.conv.bias"), 2)
        x = jax.nn.silu(x)
        x = _causal_conv(x, self._p("motion/enc.4.conv.weight"), self._p("motion/enc.4.conv.bias"), 2)
        x = jax.nn.silu(x)
        return _causal_conv(x, self._p("motion/to_mu.conv.weight"), self._p("motion/to_mu.conv.bias"), 1)

    def normalize_features(self, features):
        """Apply Demo B normalization without extrapolating constants.

        A feature with effectively zero variance in recorded motion carries no
        learned likelihood information.  Physics may move such a passive
        coordinate, but dividing it by the variance floor would turn harmless
        numerical motion into an enormous out-of-distribution input.
        """
        normalized = (features - self.norm["mmean"]) / self.norm["mstd"]
        supported = self.norm["mstd"] > SOURCE_CONSTANT_STD_THRESHOLD
        return jnp.where(supported, normalized, 0.0)

    def encode_last(self, normalized_feature_buffer):
        latent = self.encode(normalized_feature_buffer)[..., -1, :]
        return (latent - self.norm["zmean"]) / self.norm["zstd"]

    def _attention(self, x, layer: int):
        prefix = f"trans/enc.layers.{layer}."
        normalized = _layer_norm(
            x, self._p(prefix + "norm1.weight"), self._p(prefix + "norm1.bias")
        )
        qkv = _linear(
            normalized,
            self._p(prefix + "self_attn.in_proj_weight"),
            self._p(prefix + "self_attn.in_proj_bias"),
        )
        query, key, value = jnp.split(qkv, 3, axis=-1)
        batch, length, width = query.shape
        head_width = width // self.heads

        def split_heads(tensor):
            return tensor.reshape(batch, length, self.heads, head_width).transpose(0, 2, 1, 3)

        query, key, value = map(split_heads, (query, key, value))
        weights = jax.nn.softmax(
            jnp.einsum("bhid,bhjd->bhij", query, key) / math.sqrt(head_width), axis=-1
        )
        attended = jnp.einsum("bhij,bhjd->bhid", weights, value)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, width)
        x = x + _linear(
            attended,
            self._p(prefix + "self_attn.out_proj.weight"),
            self._p(prefix + "self_attn.out_proj.bias"),
        )
        normalized = _layer_norm(
            x, self._p(prefix + "norm2.weight"), self._p(prefix + "norm2.bias")
        )
        hidden = _linear(normalized, self._p(prefix + "linear1.weight"), self._p(prefix + "linear1.bias"))
        hidden = jax.nn.gelu(hidden, approximate=False)
        return x + _linear(hidden, self._p(prefix + "linear2.weight"), self._p(prefix + "linear2.bias"))

    def predict(self, normalized_history, normalized_command):
        unbatched = normalized_history.ndim == 2
        if unbatched:
            normalized_history = normalized_history[None]
            normalized_command = normalized_command[None]
        x = _linear(normalized_history, self._p("trans/inp.weight"), self._p("trans/inp.bias"))
        x = x + _position_encoding(x.shape[1], x.shape[2], x.dtype)[None]
        for layer in range(self.layers):
            x = self._attention(x, layer)
        context = _layer_norm(x, self._p("trans/nf.weight"), self._p("trans/nf.bias"))[:, -1]
        command = _linear(normalized_command, self._p("trans/cmd.0.weight"), self._p("trans/cmd.0.bias"))
        command = jax.nn.silu(command)
        command = _linear(command, self._p("trans/cmd.2.weight"), self._p("trans/cmd.2.bias"))
        output = _linear(
            jnp.concatenate([context, command], axis=-1),
            self._p("trans/out.0.weight"),
            self._p("trans/out.0.bias"),
        )
        output = jax.nn.silu(output)
        output = _linear(output, self._p("trans/out.2.weight"), self._p("trans/out.2.bias"))
        output = output.reshape(output.shape[0], 8, 16)
        return output[0] if unbatched else output

    def normalize_command(self, command):
        return (command - self.norm["cmean"]) / self.norm["cstd"]

    def predict_next(self, history, raw_command):
        return self.predict(history, self.normalize_command(raw_command))[..., 0, :]

    def log_prob(self, realized, predicted):
        sigma = self.norm["sigma"]
        error = (realized - predicted) / sigma
        return -0.5 * jnp.mean(
            jnp.square(error) + 2 * jnp.log(sigma) + math.log(2 * math.pi), axis=-1
        )


def load_prior(path: str | Path = PRIOR_ASSET) -> FrozenMotionPrior:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run `uv run --extra workshop python -m demo_b.export_jax`"
        )
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata"]))
        if (
            metadata.get("format_version") != 6
            or metadata.get("source_format_version") != 4
            or metadata.get("animal") != "coltrane"
            or metadata.get("feature_dim") != 281
        ):
            raise ValueError(
                "stale Demo E prior: export the format-v6 physical bridge from "
                "the accepted format-v4, 281-D Coltrane Demo B checkpoint"
            )
        arrays = {key: source[key] for key in source.files if key != "metadata"}
    return FrozenMotionPrior(arrays, metadata)
