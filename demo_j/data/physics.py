"""Modern MJX compatibility model for the legacy Brax v1 Fetch body."""

from __future__ import annotations

from functools import lru_cache
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.io import mjcf

from demo_f.config import FETCH_FOOT_NAMES, JOINT_NAMES
from demo_j.artifacts import PACKAGE_ROOT


XML_PATH = PACKAGE_ROOT / "assets" / "fetch.xml"
FOOT_SITE_NAMES = (
    "front_right_foot",
    "front_left_foot",
    "back_right_foot",
    "back_left_foot",
)


@lru_cache(maxsize=1)
def host_model() -> mujoco.MjModel:
    """Load the host MuJoCo model used for names, rendering, and validation."""

    return mujoco.MjModel.from_xml_path(str(XML_PATH))


@lru_cache(maxsize=1)
def system():
    """Load the Brax system backed by the same MuJoCo model."""

    return mjcf.load(XML_PATH)


def _named_addresses(
    kind: mujoco.mjtObj, names: tuple[str, ...], field: str
) -> np.ndarray:
    model = host_model()
    values = []
    for name in names:
        index = mujoco.mj_name2id(model, kind, name)
        if index < 0:
            raise ValueError(f"{name!r} is missing from {XML_PATH}")
        values.append(int(getattr(model, field)[index]))
    return np.asarray(values, np.int32)


@lru_cache(maxsize=1)
def joint_qpos_addresses() -> np.ndarray:
    return _named_addresses(mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES, "jnt_qposadr")


@lru_cache(maxsize=1)
def joint_qvel_addresses() -> np.ndarray:
    return _named_addresses(mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES, "jnt_dofadr")


@lru_cache(maxsize=1)
def foot_site_indices() -> np.ndarray:
    model = host_model()
    return np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in FOOT_SITE_NAMES
        ],
        np.int32,
    )


@lru_cache(maxsize=1)
def lower_body_indices() -> np.ndarray:
    model = host_model()
    return np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in FETCH_FOOT_NAMES
        ],
        np.int32,
    )


def joint_angles(qpos: jax.Array) -> jax.Array:
    return jnp.asarray(qpos)[..., jnp.asarray(joint_qpos_addresses())]


def joint_velocities(qvel: jax.Array) -> jax.Array:
    return jnp.asarray(qvel)[..., jnp.asarray(joint_qvel_addresses())]


def set_joint_angles(qpos: jax.Array, angles: jax.Array) -> jax.Array:
    return jnp.asarray(qpos).at[..., jnp.asarray(joint_qpos_addresses())].set(angles)


def set_joint_velocities(qvel: jax.Array, velocities: jax.Array) -> jax.Array:
    return (
        jnp.asarray(qvel).at[..., jnp.asarray(joint_qvel_addresses())].set(velocities)
    )


def validate_contract() -> dict[str, object]:
    """Return fail-fast morphology, ordering, and timing invariants."""

    model = host_model()
    loaded = system()
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    )
    if actuator_names != JOINT_NAMES:
        raise ValueError(f"actuator order {actuator_names!r} != {JOINT_NAMES!r}")
    if model.nq != 17 or model.nv != 16 or model.nu != 10:
        raise ValueError((model.nq, model.nv, model.nu))
    if not np.isclose(float(loaded.opt.timestep) * 4, 0.02):
        raise ValueError("four MJX steps must equal one 20 ms control bin")
    return {
        "xml": str(XML_PATH),
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "simulation_dt": float(loaded.opt.timestep),
        "control_dt": float(loaded.opt.timestep) * 4,
        "joint_qpos_addresses": joint_qpos_addresses().tolist(),
        "joint_qvel_addresses": joint_qvel_addresses().tolist(),
        "actuator_order": list(actuator_names),
        "actuator_gear": model.actuator_gear[:, 0].astype(float).tolist(),
    }
