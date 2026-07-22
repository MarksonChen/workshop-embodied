import torch

from demo_f.models import MotionAutoencoder
from demo_h.models import FeedbackActionDecoder, tanh_gaussian_nll


def test_tokenizer_history_is_unchanged_by_future_frames():
    torch.manual_seed(0)
    model = MotionAutoencoder(feature_dim=60).eval()
    baseline = torch.randn(2, 64, 60)
    changed = baseline.clone()
    changed[:, 16:] = torch.randn_like(changed[:, 16:])

    with torch.inference_mode():
        baseline_tokens = model.encode(baseline)
        changed_tokens = model.encode(changed)

    # Tokens 0..3 end at frames 3, 7, 11, and 15 respectively. They are the
    # complete causal history available when the action at frame 15 is chosen.
    torch.testing.assert_close(baseline_tokens[:, :4], changed_tokens[:, :4])


def test_action_decoder_starts_at_previous_control_baseline():
    model = FeedbackActionDecoder()
    previous = torch.linspace(-0.5, 0.5, 10).view(1, 10)
    mean = model(
        torch.zeros(1, 60),
        torch.zeros(1, 16),
        previous,
        torch.eye(4)[:1],
        torch.zeros(1, 3),
    )
    torch.testing.assert_close(torch.tanh(mean), previous)


def test_tanh_likelihood_includes_change_of_variables():
    mean = torch.zeros(1, 1)
    log_std = torch.zeros(1)
    center = tanh_gaussian_nll(mean, log_std, torch.zeros(1, 1))
    offset = tanh_gaussian_nll(mean, log_std, torch.full((1, 1), 0.5))
    expected = 0.5 * torch.atanh(torch.tensor(0.5)).square() + torch.log(
        torch.tensor(0.75)
    )
    torch.testing.assert_close(offset - center, expected.view(1, 1))
