"""Render either held-out imitation (with ghost) or a direct Demo B command."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp

from demo_d.config import OUT
from demo_d.env import replace_command
from demo_d.runtime import load_policy, load_validation_environment, resolve_run, resolve_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--mode", choices=("imitation", "command"), default="imitation")
    parser.add_argument("--clip", type=int, default=0, help="local validation clip index")
    parser.add_argument("--command", type=float, nargs=3, default=(0.08, 0.0, 0.0))
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    jax.random.key = jax.random.PRNGKey
    run = resolve_run(args.run)
    step_path = resolve_step(run, args.step)
    env = load_validation_environment()
    policy, _ = load_policy(env, step_path)
    reset = jax.jit(env.reset)
    step_fn = jax.jit(env.step if args.mode == "imitation" else env.command_step)
    state = reset(jax.random.PRNGKey(0), clip_idx=args.clip, start_frame=0)
    if args.mode == "command":
        state = replace_command(state, jnp.asarray(args.command))
    states = [state]
    rng = jax.random.PRNGKey(1)
    for _ in range(args.steps):
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        if args.mode == "command":
            state = step_fn(state, action, jnp.asarray(args.command))
        else:
            state = step_fn(state, action)
        states.append(state)
        if float(state.done) > 0.5:
            break

    # Physics controls run at 100 Hz; workshop videos use the 50 Hz mocap grid.
    video_states = states[::2]
    if args.mode == "command":
        frames = env.render_commands(video_states, width=640, height=480)
    else:
        frames = env.render(
            video_states,
            width=640,
            height=480,
            render_ghost=True,
            termination_extra_frames=25,
        )
    default = OUT / "videos" / f"{args.mode}-{run.name}-{step_path.name}.mp4"
    out = Path(args.out) if args.out else default
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=50) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
