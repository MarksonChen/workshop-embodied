import jax.numpy as jnp

from demo_h.policy import diagonal_gaussian_kl, inverse_softplus


def test_diagonal_kl_is_zero_for_identical_parameters():
    std = jnp.asarray([0.1, 0.2])
    raw = inverse_softplus(std - 0.001)
    parameters = jnp.concatenate((jnp.asarray([0.3, -0.2]), raw))
    assert float(diagonal_gaussian_kl(parameters, parameters)) < 1e-7


def test_diagonal_kl_penalizes_mean_shift_per_dimension():
    std = jnp.asarray([0.5, 0.5])
    raw = inverse_softplus(std - 0.001)
    reference = jnp.concatenate((jnp.zeros(2), raw))
    adapted = jnp.concatenate((jnp.ones(2), raw))
    # 0.5 * (1 / 0.5)^2 = 2 in each dimension; the implementation averages.
    assert jnp.isclose(diagonal_gaussian_kl(adapted, reference), 2.0)
