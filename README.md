# embodied

Neuromechanical rodent control experiments built on
[MIMIC-MJX](https://mimic-mjx.talmolab.org) (stac-mjx + track-mjx).

Goal: drive the virtual rodent from point A to point B by training a high-level
policy over the published, frozen imitation decoder (the paper's §2.5 transfer),
with an eye toward comparing SNN controller activity against the MC/DLS
recordings from the MIMIC paper.

## Layout

- `rl/` — experiment code. Start with `rl/README.md`.
- `ref/repos/` — upstream repos as submodules (track-mjx, stac-mjx, DART,
  MotionStreamer).
- `ref/papers/` — reference PDFs (gitignored).

## Setup

```bash
uv sync --extra cuda12    # NOT cuda13 on WSL2 — see rl/README.md
```
