import torch

from demo_h.models import FeedbackActionDecoder, tanh_gaussian_nll


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
