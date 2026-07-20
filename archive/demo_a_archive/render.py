"""demo_a/render.py -- render a Demo A (RodentMaintainVelocity) checkpoint and
measure ACTUAL forward velocity.

Demo A is the RL-only from-scratch walker: a raw-torque task policy (NO decoder),
trained by demo_a/train.py. This loads a brax PPO checkpoint, rolls it out in the
task env, measures real root displacement -> forward speed (m/s), renders an mp4,
and (optionally) uploads to wandb. The eval log only exposes the forward-velocity
*reward*; this is the only place we see the policy's velocity in interpretable units
and can watch gait quality (walk vs lunge-and-fall).

Usage:
    uv run python demo_a/render.py --ckpt demo_a/runs/<run>/<step>
    uv run python demo_a/render.py --ckpt rl/runs/RodentMaintainVelocity-.../29491200 --wandb-project embodied-demoa
"""
import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import jax

jax.random.key = jax.random.PRNGKey  # typed-key shim (see rl/train_joystick.py)

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
from brax.training.acme import running_statistics, specs as acme_specs
from brax.training.agents.ppo import networks as ppo_networks
from vnl_playground import registry
from vnl_playground.tasks import wrappers as vnl_wrappers

import sys as _sys

_SCRIPTS = Path(__file__).resolve().parent.parent / "ref" / "repos" / "track-mjx" / "scripts"
_sys.path.insert(0, str(_SCRIPTS))
from utils import apply_env_overrides, parse_env_overrides_str


def build_env(task: str, env_overrides: str | None = None):
    """Mirror scripts/train_task.create_environments (no decoder). env_overrides
    (e.g. "noslip_iterations=5 torque_actuators=False") MUST match the trained
    checkpoint's config or the model/actuators won't line up."""
    env_cfg = registry.get_default_config(task)
    if env_overrides:
        apply_env_overrides(env_cfg, parse_env_overrides_str(env_overrides))
    env = vnl_wrappers.BraxObsWrapper(
        registry.load(task, config=env_cfg, clips=None, flatten_obs=False)
    )
    return env, env_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to a checkpoint step dir")
    ap.add_argument("--task", default="RodentMaintainVelocity")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--wandb-project", default=None,
                    help="If set, upload the video + stats to this wandb project")
    ap.add_argument("--wandb-run", default=None)
    ap.add_argument("--env", default=None,
                    help='Env overrides matching the checkpoint, e.g. '
                         '"noslip_iterations=5 torque_actuators=False reward_config.tracking_sigma=0.1"')
    args = ap.parse_args()

    env, env_cfg = build_env(args.task, args.env)
    inner = env.env  # task env (owns mj_model)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(jax.random.key(0))

    obs_size = jax.tree.map(lambda x: x.shape[-1], state.obs)  # {'state': N}
    network = ppo_networks.make_ppo_networks(
        obs_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(1024, 512, 256),
        value_hidden_layer_sizes=(1024, 512, 256),
        policy_obs_key="state",
        value_obs_key="state",
    )
    norm_target = running_statistics.init_state(
        {"state": acme_specs.Array((obs_size["state"],), jnp.float32)}
    )
    target = (
        norm_target,
        network.policy_network.init(jax.random.key(0)),
        network.value_network.init(jax.random.key(0)),
    )
    params = ocp.PyTreeCheckpointer().restore(str(Path(args.ckpt).resolve()), item=target)
    normalizer, policy_params = params[0], params[1]
    policy = jax.jit(
        ppo_networks.make_inference_fn(network)((normalizer, policy_params), deterministic=True)
    )

    # Roll out; stop at the first termination (the raw env does not auto-reset), so
    # the measured speed is "how fast it walks before it falls".
    rng = jax.random.key(1)
    qpos = []
    s = state
    fell_at = None
    for t in range(args.steps):
        rng, k = jax.random.split(rng)
        action, _ = policy(s.obs, k)
        s = jit_step(s, action)
        qpos.append(np.asarray(s.data.qpos))
        if bool(s.done):
            fell_at = t + 1
            break
    qpos = np.array(qpos)
    dur = len(qpos) * float(inner.dt)
    fwd = float(qpos[-1, 0] - qpos[0, 0])   # +x is the commanded forward axis
    lat = float(qpos[-1, 1] - qpos[0, 1])
    print(f"Rolled out {len(qpos)} steps ({dur:.2f}s)"
          + (f"; FELL at step {fell_at}" if fell_at else "; survived full rollout"))
    print(f"Forward displacement {fwd:+.3f} m -> mean speed {fwd/dur:+.3f} m/s | "
          f"lateral drift {lat:+.3f} m | min root-z {float(qpos[:,2].min()):.3f} m")

    # Render (single rodent; body-tracking camera).
    mj_model = inner.mj_model
    cams = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(mj_model.ncam)]
    camera = next((c for c in cams if c and "close_profile" in c), -1)
    print(f"Cameras: {cams} -> using {camera!r}")
    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    mj_data = mujoco.MjData(mj_model)
    frames = []
    for q in qpos:
        mj_data.qpos = q
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=camera)
        frames.append(renderer.render())

    out = Path(args.out or (Path(args.ckpt).parent / f"demoa_{Path(args.ckpt).name}.mp4"))
    fps = int(1.0 / float(inner.dt))
    with imageio.get_writer(out, fps=fps) as w:
        for f in frames:
            w.append_data(f)
    print(f"Wrote {out}  ({len(frames)} frames @ {fps} fps)")

    if args.wandb_project:
        import wandb

        step = None
        try:
            step = int(Path(args.ckpt).name)
        except ValueError:
            pass
        run = wandb.init(
            project=args.wandb_project, id=args.wandb_run,
            resume="allow" if args.wandb_run else None, job_type="render",
            config={"ckpt": str(args.ckpt), "task": args.task, "steps": args.steps},
        )
        log = {
            "render/video": wandb.Video(str(out), format="mp4"),
            "render/fwd_speed_mps": fwd / dur,
            "render/fwd_displacement_m": fwd,
            "render/lateral_drift_m": lat,
            "render/survived_steps": len(qpos),
        }
        if step is not None:
            log["render/ckpt_step"] = step
        wandb.log(log)
        wandb.finish()
        print(f"Uploaded to wandb project {args.wandb_project} ({run.url})")


if __name__ == "__main__":
    main()
