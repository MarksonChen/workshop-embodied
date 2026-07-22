from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from demo_j.data.dataset import load_reference_set, take_references
from demo_j.control.tracking import ACTION_DIM, FEATURE_DIM, FetchTracking
from demo_j.control.imitation import teacher_forced_sequences


def test_teacher_forcing_matches_environment_reference_observation() -> None:
    reference = take_references(load_reference_set("validation"), [0])
    sequences = teacher_forced_sequences(reference, steps=2)
    environment = FetchTracking(reference, random_start=False)
    tail = environment._reference_observation(
        jnp.asarray(reference.qpos[0, 0]),
        jnp.asarray(reference.qvel[0, 0]),
        jnp.asarray(0),
        jnp.asarray(0),
    )
    np.testing.assert_allclose(
        sequences.observation[0, 0, FEATURE_DIM + ACTION_DIM :],
        np.asarray(tail),
        atol=2e-6,
    )
    np.testing.assert_array_equal(sequences.action[0], reference.teacher_action[0, :2])
