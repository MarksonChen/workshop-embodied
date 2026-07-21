import numpy as np

from demo_h.gait_metrics import (
    four_limb_contact_metrics,
    four_limb_locomotion_metrics,
)


def regular_trot(frames=250, period=24):
    time = np.arange(frames)
    phase = (time // (period // 2)) % 2
    return np.stack((phase, 1 - phase, 1 - phase, phase), axis=-1)


def test_regular_trot_uses_all_four_feet():
    report = four_limb_contact_metrics(regular_trot())
    assert report["passes_four_limb_gait_gate"]
    assert min(report["switch_count"]) >= 20


def test_permanently_raised_foot_fails_even_when_others_cycle():
    contacts = regular_trot()
    contacts[:, 2] = 0
    report = four_limb_contact_metrics(contacts)
    assert not report["passes_four_limb_gait_gate"]
    assert report["switch_count"][2] == 0
    assert report["contact_entropy_bits"][2] < 0.001


def synthetic_stride(frames=250, period=25):
    time = np.arange(frames)
    phase = 2 * np.pi * time[:, None] / period + np.asarray(
        [[0.0, np.pi, np.pi, 0.0]]
    )
    features = np.zeros((frames, 60), np.float32)
    feet = features[:, 32:44].reshape(frames, 4, 3)
    velocity = features[:, 44:56].reshape(frames, 4, 3)
    feet[..., 0] = 0.2 * np.sin(phase)
    feet[..., 2] = 0.08 * (1.0 + np.cos(phase))
    velocity[1:] = np.diff(feet, axis=0) * 50
    velocity[0] = velocity[1]
    features[:, 56:60] = np.sin(phase) < 0
    return features


def test_regular_stride_passes_offline_stride_gate():
    report = four_limb_locomotion_metrics(synthetic_stride())
    assert report["passes_four_limb_stride_gate"]
    assert min(report["fore_aft_excursion_m"]) > 0.3


def test_vertical_contact_tapping_fails_without_fore_aft_stride():
    features = synthetic_stride()
    features[:, 32:44].reshape(len(features), 4, 3)[..., 0] = 0
    features[:, 44:56].reshape(len(features), 4, 3)[..., 0] = 0
    report = four_limb_locomotion_metrics(features)
    assert not report["passes_four_limb_stride_gate"]
