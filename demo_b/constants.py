"""Frozen time grid + feature layout (mirrors CANVAS prepare.py/train.py; kept here so demo_b is self-contained)."""
import torch

FPS = 50; CLIP = 64; DOWN = 4; NLAT = CLIP // DOWN         # 50 Hz frames; 80 ms latent grid; 16 latents per 64-frame crop
NKP = 23                                                   # anatomical keypoints (Snout..FootR)
DM = 16; H = 8; K = 8                                       # motion latent dim; transition history / future tokens

# motion-feature (281-d) channel slices: root planar vel, root height, 6D orient delta, joint qpos/qvel, root-local kp/kpd
SL = {"vxy": (0, 2), "h": (2, 3), "d6": (3, 9), "q": (9, 76), "qd": (76, 143),
      "kp": (143, 143 + 3 * NKP), "kpd": (143 + 3 * NKP, 143 + 6 * NKP)}
FM = 143 + 6 * NKP                                          # 281

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def n_steps_for_seconds(seconds):
    """K-token rollout steps needed to decode at least `seconds` of motion."""
    nframes = int(round(seconds * FPS)); blocks = -(-nframes // CLIP)
    return -(-(blocks * NLAT - H) // K)
