import torch

from demo_h.models import FeedbackActionDecoder


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
