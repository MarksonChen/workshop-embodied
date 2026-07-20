import numpy as np
import pytest

pytest.importorskip("brax")

try:
    import jax
    import jax.numpy as jnp

    from demo_a.fetch_run import FetchRun
    from demo_a.train_fetch import FetchV2
    from demo_g.env import DemoGFetchRun
    from demo_g.prior import DEFAULT_PRIOR, load_prior
    from demo_g.wrappers import wrap_demo_g_for_training
except ImportError as error:
    pytest.skip(f"Demo G requires its pinned Brax environment: {error}", allow_module_level=True)


@pytest.mark.skipif(not DEFAULT_PRIOR.is_file(), reason="local Demo F prior is absent")
def test_beta_zero_matches_task_and_episode_reset_clears_history():
    prior = load_prior(DEFAULT_PRIOR)
    key = jax.random.PRNGKey(7)
    collector = DemoGFetchRun(prior)
    base = FetchRun(v_target=collector.v_target, sigma=collector.sigma)
    base_state = base.reset(key)
    collector_state = collector.reset(key)
    action = jnp.zeros(base.action_size)

    base_state = base.step(base_state, action)
    collector_state = collector.step(collector_state, action)
    np.testing.assert_allclose(collector_state.obs, base_state.obs, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        collector_state.reward, base_state.reward, atol=1e-6, rtol=1e-6
    )

    wrapped = wrap_demo_g_for_training(
        FetchV2(DemoGFetchRun(prior)),
        episode_length=2,
        prior=prior,
        beta=0.0,
    )
    keys = jax.random.split(key, 2)
    state = wrapped.reset(keys)
    actions = jnp.zeros((2, wrapped.action_size))
    state = wrapped.step(state, actions)
    np.testing.assert_array_equal(state.pipeline_state.info["prior_count"], (1, 1))
    np.testing.assert_allclose(state.reward, state.metrics["task_reward"], atol=1e-7)

    state = wrapped.step(state, actions)
    np.testing.assert_array_equal(state.done, (1.0, 1.0))
    np.testing.assert_array_equal(state.pipeline_state.info["prior_count"], (0, 0))
    np.testing.assert_allclose(state.pipeline_state.info["prior_features"], 0.0)

    state = wrapped.step(state, actions)
    np.testing.assert_array_equal(state.pipeline_state.info["prior_count"], (1, 1))
