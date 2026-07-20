import numpy as np

from demo_g.metrics import GAIT_CLIP_FRAMES, GAIT_FIELDS, gait_distance, gait_statistics


def test_gait_statistics_are_finite_and_reference_distance_is_zero():
    time = np.arange(GAIT_CLIP_FRAMES, dtype=np.float32) / 50.0
    features = np.zeros((3, GAIT_CLIP_FRAMES, 60), np.float32)
    for clip in range(3):
        phase = 2 * np.pi * (3.0 + 0.2 * clip) * time
        for foot in range(4):
            features[clip, :, 32 + 3 * foot + 2] = np.sin(phase + foot * np.pi / 2)
            features[clip, :, 44 + 3 * foot] = np.cos(phase + foot * np.pi / 2)
            features[clip, :, 56 + foot] = np.sin(phase + foot * np.pi / 2) < 0
    statistics = gait_statistics(features)
    assert set(statistics) == set(GAIT_FIELDS)
    assert all(np.isfinite(values).all() for values in statistics.values())

    metrics = {f"gait_{name}": float(values.mean()) for name, values in statistics.items()}
    reference = {
        name: {"mean": float(values.mean()), "std": float(values.std())}
        for name, values in statistics.items()
    }
    assert gait_distance(metrics, reference) == 0.0
