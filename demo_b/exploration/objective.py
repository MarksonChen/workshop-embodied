"""FROZEN objective for the transition-simplification autoresearch loop.  ==> DO NOT EDIT during the loop <==

Goal of the loop: find the SIMPLEST transition architecture whose rollout is not meaningfully worse than the
shipped design. So the scalar rewards motion quality and the gates reject degenerate "wins":

  scalar  = jerk of an 8 s constant-forward rollout (lower = smoother; real motion ~570, shipped ~750)
  gates   = walks (net forward displacement in a sane band) · alive (skate in a sane band, i.e. moving but not
            sliding) · sane (finite, jerk not exploded). A gated run is REJECTED regardless of its jerk.

A frozen eval: fixed seed window, fixed constant command, fixed horizon, frozen tokenizer. Every variant is scored
by exactly this function.
"""
import numpy as np, torch
from constants import DEV, CLIP, H, NLAT, DM, FM, n_steps_for_seconds
from geometry import reconstruct_qpos, q0_from
import foot_metrics as met

SECONDS = 8
DISP_BAND = (0.4, 3.0)          # net forward displacement (m) over 8 s: below = frozen, above = runaway
SKATE_BAND = (0.0010, 0.0120)   # foot slide: below = frozen/gliding, above = wildly sliding (real ~0.003)
JERK_CAP = 3000                 # anything above is diverged


@torch.no_grad()
def evaluate(v, mv, data):
    """roll 8 s under a constant forward command from the frozen seed; return jerk + gate status."""
    seed = data["seed"]; zmean, zstd = data["zmean_t"], data["zstd_t"]; cmn, csd = data["cmean_t"], data["cstd_t"]
    mm, ms, fwd = data["mm"], data["ms"], data["fwd"]
    sidx = torch.full((1,), seed["sid"], device=DEV)
    zm0 = mv.encode(torch.tensor((seed["feat"][:CLIP] - mm) / ms, dtype=torch.float32, device=DEV)[None])[0][0]
    stream = [((zm0 - zmean[0]) / zstd[0])[:H]]
    cmdn = (torch.tensor([[fwd, 0.0, 0.0]], device=DEV) - cmn) / csd
    for _ in range(n_steps_for_seconds(SECONDS)):
        stream.append(v.predict(torch.cat(stream, 0)[-H:][None], cmdn, sidx)[0])
    Z = torch.cat(stream, 0) * zstd[0] + zmean[0]; nn_ = (Z.shape[0] // NLAT) * NLAT
    gfeat = mv.decode(Z[:nn_].reshape(-1, NLAT, DM)).reshape(-1, FM).cpu().numpy() * ms + mm
    if not np.isfinite(gfeat).all():
        return dict(jerk=9999.0, disp=0.0, skate=0.0, status="gated:nan")
    gq = reconstruct_qpos(gfeat, q0_from(gfeat[0], seed["xy"][0], seed["yaw"][0]))
    paws = met.paw_positions(gq)
    disp = float(np.linalg.norm(gq[-1, :2] - gq[0, :2])); sk = met.skate(paws); jk = met.jerk(paws)
    status = "ok"
    if not (DISP_BAND[0] <= disp <= DISP_BAND[1]):
        status = f"gated:disp={disp:.2f}"
    elif not (SKATE_BAND[0] <= sk <= SKATE_BAND[1]):
        status = f"gated:skate={sk:.4f}"
    elif jk > JERK_CAP:
        status = "gated:jerk"
    return dict(jerk=jk, disp=disp, skate=sk, status=status)
