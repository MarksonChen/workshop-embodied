# Demo E research utilities

This directory is outside the workshop-facing package. `audit_prior.py` is the
single offline transition-candidate scorer; it reads a frozen development-token
anchor and never runs physics or PPO. `ARCHIVE_DIRECT_ACTUATOR.md` records the
rejected pre-alignment experiments. `TEN_MINUTE_E1.md` is the tracked audit of
the first pipeline-v6 task-plus-prior run, including the upright-standing
failure mode and exact local-artifact hashes.

Canonical workshop commands remain `demo_e.train`, `demo_e.evaluate`, and
`demo_e.render`.
