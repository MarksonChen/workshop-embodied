"""Build the MuJoCo rodent-on-arena model from the bundled XMLs (self-contained; primitives only, no meshes)."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")               # headless render backend (WSL2/servers)
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
RH, RW = 360, 480                                          # render panel height / width


def build_model():
    """arena + rodent walker on a freejoint -> compiled MjModel (camera 'close_profile-rodent')."""
    import mujoco
    arena = mujoco.MjSpec.from_file(str(ASSETS / "arena.xml"))
    rod = mujoco.MjSpec.from_file(str(ASSETS / "rodent.xml"))
    arena.worldbody.add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0)).attach_body(
        rod.body("walker"), "", "-rodent").add_freejoint(name="root")
    return arena.compile()
