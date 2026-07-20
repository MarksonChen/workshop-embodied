"""Frozen Demo B/E motion and time contract.

The original standalone Demo B also reconstructed 23 fitted marker sites.  Demo
E deliberately uses only quantities available from both a recorded ``qpos`` and
an MJX state.  That makes the likelihood bridge exact and leaves a compact
feature definition students can inspect in one screen.
"""

import torch

FPS = 50
CLIP = 64
DOWN = 4
NLAT = CLIP // DOWN  # 50 Hz frames -> one 80 ms token -> 16 tokens/crop.
DM = 16
H = 8
K = 8
COMMAND_HORIZON_FRAMES = 31  # 0.62 s, matching the original Demo B command.
NKP = 23

# These are the 38 non-constant qpos coordinates driven by the 38 native
# actuators.  The other 29 fitted coordinates are exactly zero in every MIMIC
# clip yet move passively in physics, so they carry no learnable data signal and
# make a zero-variance likelihood ill-conditioned.
ACTIVE_JOINTS = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16,
    42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56,
    57, 59, 60, 61, 62, 63, 64, 65,
)

# [root-local planar velocity, root height, relative orientation (6D),
#  38 actuated joint angles, 38 finite-difference joint velocities].
SL = {
    "vxy": (0, 2),
    "h": (2, 3),
    "d6": (3, 9),
    "q": (9, 47),
    "qd": (47, 85),
}
FM = 85

# The original, behaviorally validated Demo B representation.  Keep this
# explicit while the 85-D Demo E bridge remains available for exact regression
# comparisons.
FULL_SL = {
    "vxy": (0, 2),
    "h": (2, 3),
    "d6": (3, 9),
    "q": (9, 76),
    "qd": (76, 143),
    "kp": (143, 143 + 3 * NKP),
    "kpd": (143 + 3 * NKP, 143 + 6 * NKP),
}
FULL_FM = 143 + 6 * NKP

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def n_steps_for_seconds(seconds):
    """K-token rollout steps needed to decode at least `seconds` of motion."""
    nframes = int(round(seconds * FPS)); blocks = -(-nframes // CLIP)
    return -(-(blocks * NLAT - H) // K)
