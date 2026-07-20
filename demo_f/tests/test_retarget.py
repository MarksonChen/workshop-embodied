import jax.numpy as jnp
import numpy as np

from demo_f.kinematics import fetch_feet


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
