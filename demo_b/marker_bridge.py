"""Forward-kinematic bridge from the MIMIC skeleton to Demo B keypoints.

Demo B's accepted 281-D representation contains the 23 DANNCE landmarks as
well as qpos.  During physics rollouts those landmarks are not observations;
they are deterministic sites on the fitted skeleton.  This module estimates
one fixed local site offset per landmark from Coltrane's training recordings
and validates the reconstruction error before the offsets are exported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import mujoco
import numpy as np

from vnl_playground.tasks.rodent import consts
from vnl_playground.tasks.utils import scale_spec


KEYPOINT_NAMES = (
    "Snout", "EarL", "EarR", "SpineF", "SpineM", "SpineL", "TailBase",
    "ShoulderL", "ElbowL", "WristL", "HandL",
    "ShoulderR", "ElbowR", "WristR", "HandR",
    "HipL", "KneeL", "AnkleL", "FootL",
    "HipR", "KneeR", "AnkleR", "FootR",
)

# This is the keypoint-to-body contract used by STAC for the Aldarondo rodent.
KEYPOINT_BODIES = (
    "skull", "skull", "skull",
    "vertebra_cervical_5", "vertebra_1", "pelvis", "pelvis",
    "scapula_L", "upper_arm_L", "lower_arm_L", "hand_L",
    "scapula_R", "upper_arm_R", "lower_arm_R", "hand_R",
    "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
    "pelvis", "upper_leg_R", "lower_leg_R", "foot_R",
)


@dataclass(frozen=True)
class MarkerCalibration:
    offsets: np.ndarray
    rmse_m: np.ndarray
    samples: int


def build_kinematic_model(*, scale: float = 0.9) -> mujoco.MjModel:
    """Compile the same scaled skeleton as imitation, with no spawn offset."""
    arena = mujoco.MjSpec.from_file(str(consts.ARENA_XML_PATH))
    rodent = mujoco.MjSpec.from_file(str(consts.RODENT_XML_PATH))
    rodent = scale_spec(rodent, scale)
    frame = arena.worldbody.add_frame(pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))
    body = frame.attach_body(rodent.body("walker"), "", suffix="-rodent")
    body.add_freejoint(name="root")
    return arena.compile()


def body_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}-rodent")
        for name in KEYPOINT_BODIES
    ]
    if any(index < 0 for index in ids):
        raise ValueError("the compiled rodent is missing a STAC keypoint body")
    return np.asarray(ids, np.int32)


def _decode(values) -> tuple[str, ...]:
    return tuple(
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    )


def calibrate_offsets(
    data_root: Path,
    sessions: tuple[str, ...],
    *,
    samples_per_session: int = 128,
) -> MarkerCalibration:
    """Robustly estimate body-local marker sites from training sessions."""
    model = build_kinematic_model()
    ids = body_ids(model)
    data = mujoco.MjData(model)
    estimates = []
    observations = []
    for session in sessions:
        path = data_root / "coltrane" / f"{session}.h5"
        with h5py.File(path, "r") as source:
            qpos_ds = source["/pose/qpos"]
            keypoint_ds = source["/pose/keypoints"]
            names = _decode(keypoint_ds.attrs["names"])
            order = np.asarray([names.index(name) for name in KEYPOINT_NAMES])
            indices = np.linspace(
                0, len(qpos_ds) - 1, min(samples_per_session, len(qpos_ds)), dtype=np.int64
            )
            qposes = qpos_ds[indices].astype(np.float64)
            keypoints = keypoint_ds[indices][:, :, order].transpose(0, 2, 1) / 1000.0
        for qpos, keypoint in zip(qposes, keypoints, strict=True):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            position = data.xpos[ids]
            rotation = data.xmat[ids].reshape(-1, 3, 3)
            local = np.einsum("kji,kj->ki", rotation, keypoint - position)
            estimates.append(local)
            observations.append((position.copy(), rotation.copy(), keypoint.copy()))

    estimates_array = np.asarray(estimates)
    offsets = np.median(estimates_array, axis=0)
    squared_errors = []
    for position, rotation, keypoint in observations:
        predicted = position + np.einsum("kij,kj->ki", rotation, offsets)
        squared_errors.append(np.square(predicted - keypoint).sum(axis=-1))
    rmse = np.sqrt(np.mean(squared_errors, axis=0))
    return MarkerCalibration(
        offsets=offsets.astype(np.float32),
        rmse_m=rmse.astype(np.float32),
        samples=len(observations),
    )
