"""Frozen, workshop-scale Demo G comparison settings."""

from __future__ import annotations


# Recalibrated before dynamic Demo G's G1 run.  The accepted retimed validation
# motion has median log p=-0.17, while the matched 0.925-unit/s G0 policy has
# per-window median -21.2 (5th/95th percentiles -34.3/-12.0).  Centering at the
# rounded G0 median gives the sigmoid useful gradient without test-set tuning.
PRIOR_LOGP_CENTER = -20.0
PRIOR_LOGP_SCALE = 5.0
DEFAULT_BETA = 0.10
DEFAULT_SCORE_STRIDE = 4
# The final dynamically aligned arms take 58--70 seconds inside ``ppo.train``.
# Thirty million transitions is therefore the frozen workshop budget: each arm
# remains below two minutes and a matched pair takes about 2.1 minutes.
DEFAULT_TIMESTEPS = 30_000_000
DEFAULT_ENVS = 2_048
DEFAULT_EVALS = 3
