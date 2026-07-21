import jax.numpy as jnp
import numpy as np

from demo_f.kinematics import fetch_feet, fetch_feet_numpy


def test_zero_pose_matches_brax_fetch_default_endpoints():
    expected = np.asarray(
        (
            (1.0, -0.625, -1.375),
            (1.0, 0.625, -1.375),
            (-1.0, -0.625, -1.375),
            (-1.0, 0.625, -1.375),
        )
    )
    np.testing.assert_allclose(np.asarray(fetch_feet(jnp.zeros(10))), expected, atol=1e-6)


def test_kinematics_supports_batched_sequences_and_gradients():
    angles = jnp.zeros((2, 8, 10))
    feet = fetch_feet(angles)
    assert feet.shape == (2, 8, 4, 3)


def test_numpy_validation_kinematics_matches_jax():
    angles = np.linspace(-0.7, 0.7, 60, dtype=np.float32).reshape(2, 3, 10)
    np.testing.assert_allclose(
        fetch_feet_numpy(angles), np.asarray(fetch_feet(angles)), atol=1e-6
    )
