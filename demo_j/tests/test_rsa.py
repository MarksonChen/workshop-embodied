from __future__ import annotations

import numpy as np

from demo_j.analysis.rsa import (
    condition_repeat_means,
    make_condition_design,
    partial_spearman_rsa,
    representational_geometry,
    spearman_rsa,
)


def test_condition_design_keeps_only_patterns_present_in_every_repeat() -> None:
    contacts = np.zeros((8, 12, 4), np.float32)
    speed = np.repeat([1.0, 2.0], 4)
    # Pattern 1 occurs in every repeat; pattern 2 is absent from one repeat.
    contacts[:, 2:5, 0] = 1
    contacts[:, 5:8, 1] = 1
    contacts[3, 5:8] = 0
    design = make_condition_design(
        contacts, speed, warmup_bins=2, minimum_bins_per_repeat=3
    )
    assert set(zip(design.speed, design.contact_pattern, strict=True)) == {
        (1.0, 0),
        (1.0, 1),
        (2.0, 0),
        (2.0, 1),
        (2.0, 2),
    }
    values = np.arange(8 * 12 * 3).reshape(8, 12, 3)
    means = condition_repeat_means(values, design)
    assert means.shape == (4, 5, 3)


def test_crossvalidated_rsa_recovers_shared_geometry() -> None:
    random = np.random.default_rng(4)
    repeats, conditions, latent = 5, 10, 3
    signal = random.normal(size=(conditions, latent))
    left_map = random.normal(size=(latent, 24))
    right_map = random.normal(size=(latent, 18))
    behavior_map = random.normal(size=(latent, 7))
    left = signal[None] @ left_map + random.normal(
        scale=0.15, size=(repeats, conditions, 24)
    )
    right = signal[None] @ right_map + random.normal(
        scale=0.15, size=(repeats, conditions, 18)
    )
    behavior = signal[None] @ behavior_map + random.normal(
        scale=0.40, size=(repeats, conditions, 7)
    )
    left_geometry = representational_geometry(left)
    right_geometry = representational_geometry(right)
    behavior_geometry = representational_geometry(behavior)
    score = spearman_rsa(
        left_geometry.crossvalidated_rdm,
        right_geometry.crossvalidated_rdm,
    )
    shuffled = np.arange(conditions)[::-1]
    shuffled_score = spearman_rsa(
        left_geometry.crossvalidated_rdm,
        right_geometry.crossvalidated_rdm[np.ix_(shuffled, shuffled)],
    )
    partial = partial_spearman_rsa(
        left_geometry.crossvalidated_rdm,
        right_geometry.crossvalidated_rdm,
        behavior_geometry.crossvalidated_rdm,
    )
    assert score > 0.8
    assert score > shuffled_score + 0.3
    assert np.isfinite(partial)
