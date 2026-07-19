"""FetchRun: the brax v1 'fetch' body + physics, but the task is to RUN at a
constant forward speed, not to reach a target.

Why this and not fetch's reach-a-target reward: reaching a point is a one-shot
problem (a single lunge is optimal, so no rhythm is needed). Holding a constant
forward velocity for the whole episode cannot be done with a lunge -- on legs the
only way to sustain speed is to CYCLE them, so a periodic gait becomes the optimum.
Paired with fall-termination (which fetch lacks entirely), the tumbling scramble
stops being free. This is the textbook setup for eliciting a cyclic gait.

Reward (per step), reusing quantities the fetch body already exposes:
    track   = exp(-(v_x - v_target)^2 / (2 sigma^2))   # hold constant +x speed
    upright = dot(torso_up, world_up)                   # stay on your feet
    reward  = track + upright_w*upright - ctrl_w*sum(action^2)
    done    = (torso_height < h_min) or (upright < 0)   # terminate on fall/flip

Obs is left byte-identical to fetch (101-d): the egocentric obs already carries the
torso's WORLD-frame heading (torso_fwd), so "+x" is observable without any extra
cue; the (now static, un-teleported) target is just an ignored landmark.
"""
from brax.v1 import jumpy as jp
from brax.v1 import math
from brax.v1.envs.fetch import Fetch


class FetchRun(Fetch):
    def __init__(self, v_target=3.0, sigma=1.0, upright_w=0.1, ctrl_w=1e-3,
                 legacy_spring=False, **kwargs):
        super().__init__(legacy_spring=legacy_spring, **kwargs)
        self.v_target = v_target
        self.sigma = sigma
        self.upright_w = upright_w
        self.ctrl_w = ctrl_w
        self._stand_h = float(self.sys.default_qp().pos[self.torso_idx, 2])
        self.h_min = 0.5 * self._stand_h

    def reset(self, rng):
        state = super().reset(rng)  # random target, default (standing) body pose
        zero = state.reward  # scalar 0.0
        metrics = {'speed': zero, 'track': zero, 'upright': zero, 'ctrl_cost': zero}
        return state.replace(metrics=metrics)

    def step(self, state, action):
        qp, info = self.sys.step(state.qp, action)  # target is frozen -> stays put
        obs = self._get_obs(qp, info)

        v = (qp.pos[self.torso_idx] - state.qp.pos[self.torso_idx]) / self.sys.config.dt
        speed = v[0]  # world +x speed
        track = jp.exp(-((speed - self.v_target) ** 2) / (2.0 * self.sigma ** 2))

        up = jp.array([0., 0., 1.])
        torso_up = math.rotate(up, qp.rot[self.torso_idx])
        upright = jp.dot(torso_up, up)
        ctrl_cost = self.ctrl_w * jp.sum(jp.square(action))
        reward = track + self.upright_w * upright - ctrl_cost

        torso_h = qp.pos[self.torso_idx, 2]
        fell = (torso_h < self.h_min) | (upright < 0.0)
        done = jp.where(fell, jp.float32(1), jp.float32(0))

        state.metrics.update(speed=speed, track=track, upright=upright, ctrl_cost=ctrl_cost)
        return state.replace(qp=qp, obs=obs, reward=reward, done=done)


def make_env(name):
    """'fetch' -> reach-a-target (scramble); 'run' -> constant-speed run (gait probe)."""
    if name == "run":
        return FetchRun()
    from brax.v1.envs import fetch as v1fetch
    return v1fetch.Fetch()


def deciles_dir(out, name):
    return out / ("deciles_run" if name == "run" else "deciles")


def out_prefix(name):
    return "decile_run" if name == "run" else "decile"
