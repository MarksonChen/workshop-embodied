import numpy as np
import torch

from demo_c.config import BASE_OBS_SIZE, TASK, WAM_CONTEXT_SIZE
from demo_c.deploy_physics import calibrated_forward_velocity
from demo_c.dream_env import RodentDreamEnv
from demo_c.neural_data import task_pseudo_goal
from demo_c.neural_eval import BLOCK_TOKENS, GAP_TOKENS, split_masks
from demo_c.policy import ActorCritic


class DummyMotor:
    device = torch.device("cpu")

    def sample_histories(self, count, generator):
        return torch.zeros((count, 8, 16))

    def context(self, history):
        return torch.zeros((len(history), WAM_CONTEXT_SIZE))

    def advance(self, history, action, context=None, decode=True):
        delta = torch.stack(
            (torch.full_like(action[:, 0], 0.05), torch.zeros_like(action[:, 0]), 0.1 * action[:, 1]),
            -1,
        )
        return history.clone(), delta, torch.zeros(len(history), dtype=torch.bool)


def test_matched_dream_environment_shapes_and_forward_goals():
    for use_context, expected in ((False, BASE_OBS_SIZE), (True, BASE_OBS_SIZE + WAM_CONTEXT_SIZE)):
        env = RodentDreamEnv(DummyMotor(), 32, use_context, seed=7, auto_reset=False)
        observation, context = env.observe()
        assert observation.shape == (32, expected)
        assert context.shape == (32, WAM_CONTEXT_SIZE)

        direction = torch.stack((torch.cos(env.state.yaw), torch.sin(env.state.yaw)), -1)
        assert torch.all(((env.state.goal - env.state.xy) * direction).sum(-1) >= -1e-6)

        reward, done, info = env.step(torch.zeros((32, 2)), context)
        assert reward.shape == done.shape == (32,)
        assert torch.isfinite(reward).all()
        assert not info["invalid"].any()


def test_squashed_policy_is_bounded_and_log_prob_is_reproducible():
    torch.manual_seed(3)
    model = ActorCritic(BASE_OBS_SIZE)
    observation = torch.randn(17, BASE_OBS_SIZE)
    action, pre_tanh, log_prob, value = model.act(observation)
    checked_log_prob, entropy, checked_value = model.evaluate_actions(observation, pre_tanh)

    assert torch.all(action.abs() < 1)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    torch.testing.assert_close(log_prob, checked_log_prob)
    torch.testing.assert_close(value, checked_value)


def test_recorded_future_direction_becomes_an_in_distribution_food_goal():
    future = np.array([[1, 0], [0, 2], [-1, 0], [0, -3], [0, 0]], np.float32)
    local, distance = task_pseudo_goal(future)
    radius = 0.5 * (TASK.goal_radius_min + TASK.goal_radius_max)
    np.testing.assert_allclose(np.linalg.norm(local, axis=1), radius, rtol=1e-6)
    np.testing.assert_allclose(distance, radius)
    assert np.all(np.abs(np.arctan2(local[:, 1], local[:, 0])) <= TASK.goal_bearing_max + 1e-6)


def test_temporal_split_contains_the_whole_history_and_future_window():
    token_index = np.arange(BLOCK_TOKENS * 10, dtype=np.int32)
    locomotion = token_index % 7 == 0
    data = {"token_index": token_index, "locomotion": locomotion}
    masks = split_masks(data, "loco")

    for kind, mask in masks.items():
        chosen = token_index[mask]
        within = chosen % BLOCK_TOKENS
        fold = (chosen // BLOCK_TOKENS) % 5
        assert np.all(within >= GAP_TOKENS + 8)
        assert np.all(within < BLOCK_TOKENS - GAP_TOKENS - 25)
        expected_folds = {"train": {0, 1, 2}, "val": {3}, "test": {4}}[kind]
        assert set(np.unique(fold)).issubset(expected_folds)

    matched_a = split_masks(data, "all_matched")
    matched_b = split_masks(data, "all_matched")
    for kind in masks:
        np.testing.assert_array_equal(matched_a[kind], matched_b[kind])
        assert matched_a[kind].sum() == masks[kind].sum()


def test_response_curve_bridge_handles_a_low_speed_dead_zone():
    calibration = {
        "rows": [
            {"vx": 0.1, "vyaw": 0.0, "forward": -0.01, "fell": False},
            {"vx": 0.2, "vyaw": 0.0, "forward": 0.0, "fell": False},
            {"vx": 0.3, "vyaw": 0.0, "forward": 0.1, "fell": False},
        ]
    }
    np.testing.assert_allclose(
        calibrated_forward_velocity(0.05, calibration), 0.25, rtol=1e-6
    )
