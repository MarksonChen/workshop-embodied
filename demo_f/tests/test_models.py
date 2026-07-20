import torch

from demo_f.models import ConditionalTransformer, MotionAutoencoder


def test_tunable_tokenizer_shapes():
    model = MotionAutoencoder(feature_dim=60, hidden=32, latent_dim=8)
    values = torch.randn(3, 64, 60)
    tokens = model.encode(values)
    assert tokens.shape == (3, 16, 8)
    assert model.decode(tokens).shape == values.shape


def test_conditional_gaussian_shapes():
    model = ConditionalTransformer(
        latent_dim=8, future_tokens=8, width=32, layers=1, heads=4
    )
    history = torch.randn(3, 8, 8)
    future = torch.randn(3, 8, 8)
    command = torch.randn(3, 3)
    assert model.predict(history, command).shape == future.shape
    assert model.log_prob(history, future, command, sigma=0.5).shape == (3,)
