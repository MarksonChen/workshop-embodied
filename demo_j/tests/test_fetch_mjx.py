from __future__ import annotations

import mujoco
import numpy as np

from demo_f.config import JOINT_NAMES
from demo_f.kinematics import fetch_feet_numpy
from demo_j.data.physics import FOOT_SITE_NAMES, XML_PATH, host_model, validate_contract


def test_mjcf_matches_demo_f_kinematics() -> None:
    model = host_model()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    for _ in range(20):
        angles = rng.uniform(-0.8, 0.8, len(JOINT_NAMES))
        data.qpos[:] = model.qpos0
        for name, value in zip(JOINT_NAMES, angles, strict=True):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint]] = value
        mujoco.mj_forward(model, data)
        torso = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Torso")]
        feet = np.asarray(
            [
                data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)]
                - torso
                for name in FOOT_SITE_NAMES
            ]
        )
        np.testing.assert_allclose(feet, fetch_feet_numpy(angles), atol=2e-7)


def test_mjcf_contract() -> None:
    report = validate_contract()
    assert XML_PATH.is_file()
    assert np.isclose(report["control_dt"], 0.02)
    assert report["actuator_order"] == list(JOINT_NAMES)
    assert report["actuator_gear"] == [300.0] * 10
