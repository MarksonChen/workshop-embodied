from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from demo_f.dataset import load_split  # noqa: E402
from demo_f.dataset.contract import DYNAMIC_ROOT  # noqa: E402
from demo_f.artifacts import sha256  # noqa: E402
from demo_f.prior import load_prior as load_torch_prior  # noqa: E402
from demo_g.prior import (  # noqa: E402
    DEFAULT_PRIOR,
    PLANAR_UNSUPPORTED_FEATURES,
    load_prior as load_jax_prior,
)


TORCH_CHECKPOINT = Path(__file__).resolve().parents[2] / "demo_f" / "out" / "prior.pt"


@pytest.mark.skipif(
    not TORCH_CHECKPOINT.is_file()
    or not DEFAULT_PRIOR.is_file()
    or not (DYNAMIC_ROOT / "manifest.json").is_file(),
    reason="local frozen Demo F artifacts are not installed",
)
def test_jax_export_matches_frozen_pytorch_prior():
    torch_prior = load_torch_prior(TORCH_CHECKPOINT)
    checkpoint = torch_prior.checkpoint
    tokenizer = torch_prior.tokenizer
    predictor = torch_prior.predictor
    prior = load_jax_prior(DEFAULT_PRIOR)
    assert prior.metadata["source_checkpoint_sha256"] == sha256(TORCH_CHECKPOINT)

    validation = load_split("validation", DYNAMIC_ROOT)
    raw_features = validation.features[0, :32]
    raw_command = np.asarray(
        (prior.command_scale * prior.source_speed_mps, 0.0, 0.0), np.float32
    )
    np.testing.assert_allclose(prior.target_speed_fetch, 0.9247473545)

    device = next(tokenizer.parameters()).device
    mean = torch.as_tensor(checkpoint["feature_mean"], device=device)
    std = torch.as_tensor(checkpoint["feature_std"], device=device)
    normalized = (torch.as_tensor(raw_features, device=device) - mean) / std
    normalized[:, PLANAR_UNSUPPORTED_FEATURES] = 0.0
    with torch.inference_mode():
        torch_tokens = (
            tokenizer.encode(normalized[None])[0]
            - torch.as_tensor(checkpoint["token_mean"], device=device)
        ) / torch.as_tensor(checkpoint["token_std"], device=device)
        command = (
            torch.as_tensor(raw_command, device=device)
            - torch.as_tensor(checkpoint["command_mean"], device=device)
        ) / torch.as_tensor(checkpoint["command_std"], device=device)
        torch_prediction = predictor.predict(torch_tokens[-5:-1][None], command[None])[0, 0]
        residual = (
            torch_tokens[-1] - torch_prediction
        ) / torch.as_tensor(checkpoint["sigma"], device=device)
        torch_logp = -0.5 * (
            residual.square()
            + 2 * torch.as_tensor(checkpoint["sigma"], device=device).log()
            + np.log(2 * np.pi)
        ).mean()

    jax_tokens = np.asarray(prior.encode(raw_features))
    jax_prediction = np.asarray(prior.predict(jax_tokens[-5:-1], raw_command))
    jax_logp = np.asarray(prior.log_prob(raw_features, raw_command))
    np.testing.assert_allclose(
        jax_tokens, torch_tokens.detach().cpu().numpy(), atol=5e-4, rtol=5e-4
    )
    np.testing.assert_allclose(
        jax_prediction,
        torch_prediction.detach().cpu().numpy(),
        atol=5e-4,
        rtol=5e-4,
    )
    np.testing.assert_allclose(
        jax_logp, torch_logp.detach().cpu().numpy(), atol=5e-4, rtol=5e-4
    )
