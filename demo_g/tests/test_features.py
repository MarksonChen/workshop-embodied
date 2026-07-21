import numpy as np

from demo_f.features import trajectory_features
from demo_f.jax_features import transition_feature


def test_online_transition_matches_offline_feature_contract():
    frames = 9
    time = np.arange(frames, dtype=np.float32) / 50.0
    yaw = 0.3 * time + 0.04 * np.sin(2.0 * time)
    root = np.stack(
        (0.18 * time, 0.03 * np.sin(3.0 * time), 1.38 + 0.01 * np.cos(time)),
        axis=-1,
    ).astype(np.float32)
    quaternion = np.zeros((frames, 4), np.float32)
    quaternion[:, 0] = np.cos(yaw / 2)
    quaternion[:, 3] = np.sin(yaw / 2)
    angles = np.stack(
        [0.2 * np.sin((joint + 1) * time + joint / 7) for joint in range(10)],
        axis=-1,
    ).astype(np.float32)
    nominal_feet = np.asarray(
        ((1, -0.6, -1.3), (1, 0.6, -1.3), (-1, -0.6, -1.3), (-1, 0.6, -1.3)),
        np.float32,
    )
    feet = nominal_feet[None] + 0.03 * np.sin(
        time[:, None, None] * np.arange(1, 13, dtype=np.float32).reshape(1, 4, 3)
    )
    contacts = ((np.arange(frames)[:, None] + np.arange(4)[None]) % 3 == 0).astype(
        np.float32
    )

    offline = trajectory_features(
        root[None], quaternion[None], angles[None], feet[None], contacts[None]
    )[0]
    frame_zero = np.asarray(
        transition_feature(
            root[0] - (root[1] - root[0]),
            root[0],
            quaternion[0],
            quaternion[0],
            angles[0] - (angles[1] - angles[0]),
            angles[0],
            feet[0] - (feet[1] - feet[0]),
            feet[0],
            contacts[0],
        )
    )
    np.testing.assert_allclose(offline[0], frame_zero, atol=1e-6, rtol=1e-6)
    online = np.stack(
        [
            np.asarray(
                transition_feature(
                    root[index - 1],
                    root[index],
                    quaternion[index - 1],
                    quaternion[index],
                    angles[index - 1],
                    angles[index],
                    feet[index - 1],
                    feet[index],
                    contacts[index],
                )
            )
            for index in range(1, frames)
        ]
    )
    np.testing.assert_allclose(online, offline[1:], atol=1e-6, rtol=1e-6)
