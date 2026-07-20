"""Demo D: one-stage hindsight-command imitation RL from scratch."""

import os


# Set the headless backend before any submodule can import Brax/MuJoCo.  Individual
# CLIs may override either value explicitly in their process environment.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
