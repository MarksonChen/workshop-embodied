"""Train the matched goal-only or WAM-context PPO policy in short dreams.

Examples:
    uv run --extra cuda12 --extra workshop python -m demo_c.train --variant goal_only --seed 0
    uv run --extra cuda12 --extra workshop python -m demo_c.train --variant wam --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from demo_c.config import EVAL_SEED, POLICY, PPO, TASK, VARIANTS, resolved_config
from demo_c.dream_env import RodentDreamEnv
from demo_c.motor import FrozenMotor
from demo_c.policy import ActorCritic, heuristic_action, save_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"


def provenance():
    def run(*args):
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
    return {
        "git_commit": run("git", "rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(run("git", "status", "--porcelain")),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@torch.no_grad()
def evaluate_controller(motor, controller, variant, episodes, seed=EVAL_SEED):
    env = RodentDreamEnv(motor, episodes, variant == "wam", seed=seed, auto_reset=False)
    returns = torch.zeros(episodes, device=motor.device)
    active = torch.ones(episodes, dtype=torch.bool, device=motor.device)
    success = torch.zeros_like(active)
    invalid = torch.zeros_like(active)
    final_distance = torch.zeros(episodes, device=motor.device)
    for _ in range(TASK.horizon):
        observation, context = env.observe()
        action = controller(observation, env.base_observation())
        reward, done, info = env.step(action, context)
        returns += reward * active
        success |= info["success"] & active
        invalid |= info["invalid"] & active
        final_distance = torch.where(active, info["distance"], final_distance)
        active &= ~done
    return {
        "success_rate": float(success.float().mean()),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "final_distance_mean": float(final_distance.mean()),
        "invalid_rate": float(invalid.float().mean()),
    }


def evaluate_policy(motor, model, variant, episodes, seed=EVAL_SEED):
    return evaluate_controller(
        motor,
        lambda observation, _: model.act(observation, deterministic=True)[0],
        variant,
        episodes,
        seed,
    )


def evaluate_references(motor, episodes):
    heuristic = evaluate_controller(
        motor,
        lambda _observation, base: heuristic_action(base),
        "goal_only",
        episodes,
        EVAL_SEED,
    )
    generator = torch.Generator(device=motor.device).manual_seed(EVAL_SEED + 1)
    random = evaluate_controller(
        motor,
        lambda observation, _base: torch.rand(
            (len(observation), 2), generator=generator, device=motor.device
        ) * 2 - 1,
        "goal_only",
        episodes,
        EVAL_SEED,
    )
    return {"heuristic": heuristic, "random": random}


def append_result(row):
    path = OUT / "results.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row)
    exists = path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fields, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_curve(curve, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [x["steps"] for x in curve]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=140)
    axes[0].plot(steps, [x["episode_return"] for x in curve], color="#ee8b3a")
    axes[0].set(xlabel="dream environment steps", ylabel="mean episodic return")
    axes[1].plot(steps, [x["success"] for x in curve], color="#3b9f8c")
    axes[1].set(xlabel="dream environment steps", ylabel="training success", ylim=(-0.02, 1.02))
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path); plt.close(fig)


def train(variant: str, seed: int, cfg, device: str, tag: str = ""):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    motor = FrozenMotor(device)
    env = RodentDreamEnv(motor, cfg.num_envs, variant == "wam", seed=seed)
    model = ActorCritic(env.observation_size).to(motor.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)
    batch_size = cfg.num_envs * cfg.rollout_steps
    updates = max(1, cfg.total_env_steps // batch_size)
    if batch_size % cfg.minibatch_size:
        raise ValueError("num_envs * rollout_steps must be divisible by minibatch_size")

    observation, context = env.observe()
    curve = []
    completed_return, completed_success = [], []
    start = time.perf_counter()
    max_memory = 0.0

    for update in range(updates):
        obs_buf = torch.empty((cfg.rollout_steps, cfg.num_envs, env.observation_size), device=motor.device)
        pre_buf = torch.empty((cfg.rollout_steps, cfg.num_envs, 2), device=motor.device)
        logp_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), device=motor.device)
        value_buf = torch.empty_like(logp_buf); reward_buf = torch.empty_like(logp_buf)
        done_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), dtype=torch.bool, device=motor.device)

        for step in range(cfg.rollout_steps):
            with torch.no_grad():
                action, pre_tanh, log_prob, value = model.act(observation)
            obs_buf[step].copy_(observation); pre_buf[step].copy_(pre_tanh)
            logp_buf[step].copy_(log_prob); value_buf[step].copy_(value)
            reward, done, info = env.step(action, context)
            reward_buf[step].copy_(reward); done_buf[step].copy_(done)
            if done.any():
                completed_return.extend(info["episode_return"][done].cpu().tolist())
                completed_success.extend(info["success"][done].float().cpu().tolist())
            observation, context = env.observe()

        with torch.no_grad():
            next_value = model(observation)[1]
            advantage = torch.zeros_like(reward_buf)
            gae = torch.zeros(cfg.num_envs, device=motor.device)
            for step in reversed(range(cfg.rollout_steps)):
                nonterminal = (~done_buf[step]).float()
                following = next_value if step == cfg.rollout_steps - 1 else value_buf[step + 1]
                delta = reward_buf[step] + cfg.gamma * following * nonterminal - value_buf[step]
                gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
                advantage[step] = gae
            returns = advantage + value_buf

        flat_obs = obs_buf.flatten(0, 1); flat_pre = pre_buf.flatten(0, 1)
        flat_logp = logp_buf.flatten(); flat_adv = advantage.flatten(); flat_returns = returns.flatten()
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        indices = torch.arange(batch_size, device=motor.device)
        update_losses, grad_norms, approx_kls = [], [], []
        for _ in range(cfg.update_epochs):
            indices = indices[torch.randperm(batch_size, device=motor.device)]
            for begin in range(0, batch_size, cfg.minibatch_size):
                idx = indices[begin:begin + cfg.minibatch_size]
                new_logp, entropy, new_value = model.evaluate_actions(flat_obs[idx], flat_pre[idx])
                log_ratio = new_logp - flat_logp[idx]
                ratio = log_ratio.exp()
                unclipped = ratio * flat_adv[idx]
                clipped = ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * flat_adv[idx]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (new_value - flat_returns[idx]).square().mean()
                loss = policy_loss + cfg.value_coefficient * value_loss - cfg.entropy_coefficient * entropy.mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                update_losses.append(float(loss.detach())); grad_norms.append(float(grad_norm))
                approx_kls.append(float(((ratio - 1) - log_ratio).mean().detach()))

        total_steps = (update + 1) * batch_size
        recent = min(len(completed_return), max(cfg.num_envs, 1))
        row = {
            "steps": total_steps,
            "episode_return": float(np.mean(completed_return[-recent:])) if recent else float("nan"),
            "success": float(np.mean(completed_success[-recent:])) if recent else float("nan"),
            "loss": float(np.mean(update_losses)),
            "grad_norm": float(np.mean(grad_norms)),
            "approx_kl": float(np.mean(approx_kls)),
        }
        curve.append(row)
        if torch.cuda.is_available():
            max_memory = max(max_memory, torch.cuda.max_memory_allocated() / 2**30)
        if update == 0 or (update + 1) % max(1, updates // 10) == 0 or update + 1 == updates:
            elapsed = time.perf_counter() - start
            print(
                f"[{variant} s{seed}] {total_steps:>8,d}/{updates * batch_size:,d} "
                f"return={row['episode_return']:+.3f} success={row['success']:.3f} "
                f"loss={row['loss']:.3f} {total_steps / elapsed:,.0f} steps/s",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    evaluation = evaluate_policy(motor, model, variant, cfg.eval_episodes)
    references = evaluate_references(motor, cfg.eval_episodes) if seed == 0 else {}
    metrics = {
        **evaluation,
        "elapsed_seconds": elapsed,
        "steps_per_second": updates * batch_size / elapsed,
        "peak_memory_gib": max_memory,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "world_checkpoint": str(motor.checkpoint),
        "world_validation_skill": motor.world_training.get("best_validation_skill"),
        "references": references,
    }
    prov = provenance()
    safe_tag = "".join(c for c in tag if c.isalnum() or c in ("-", "_"))
    stem = f"{variant}_seed{seed}" + (f"_{safe_tag}" if safe_tag else "")
    checkpoint = OUT / "checkpoints" / f"{stem}.pt"
    save_policy(checkpoint, model, variant, seed, metrics, prov)
    plot_curve(curve, OUT / "viz" / f"{stem}.png")
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    with (OUT / "metrics" / f"{stem}.json").open("w") as file:
        json.dump({"metrics": metrics, "curve": curve, "config": resolved_config(), "provenance": prov}, file, indent=2)
    if safe_tag.startswith("smoke"):
        status = "smoke"
    else:
        status = "keep" if evaluation["invalid_rate"] == 0 and np.isfinite(evaluation["return_mean"]) else "gated"
    append_result({
        "variant": variant,
        "seed": seed,
        "steps": updates * batch_size,
        "success": f"{evaluation['success_rate']:.6f}",
        "return": f"{evaluation['return_mean']:.6f}",
        "final_distance": f"{evaluation['final_distance_mean']:.6f}",
        "invalid_rate": f"{evaluation['invalid_rate']:.6f}",
        "steps_per_second": f"{metrics['steps_per_second']:.1f}",
        "peak_memory_gib": f"{max_memory:.3f}",
        "status": status,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
    })
    print(json.dumps(metrics, indent=2), flush=True)
    return checkpoint, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total-steps", type=int, default=PPO.total_env_steps)
    parser.add_argument("--num-envs", type=int, default=PPO.num_envs)
    parser.add_argument("--eval-episodes", type=int, default=PPO.eval_episodes)
    parser.add_argument("--tag", default="", help="artifact suffix for a probe; keeps the frozen anchor intact")
    parser.add_argument("--smoke", action="store_true", help="2 updates + 64 eval episodes; never a reportable run")
    args = parser.parse_args()
    reportable_world = OUT / "world" / "world_seed0.pt"
    if not args.smoke and not reportable_world.exists():
        raise SystemExit(
            f"missing {reportable_world}; run python -m demo_c.train_world before a reportable PPO run"
        )
    custom_run = (
        args.total_steps != PPO.total_env_steps
        or args.num_envs != PPO.num_envs
        or args.eval_episodes != PPO.eval_episodes
    )
    if custom_run and not args.smoke and not args.tag:
        raise SystemExit("custom budgets require --tag so they cannot overwrite a frozen checkpoint")
    cfg = replace(
        PPO,
        total_env_steps=(2 * args.num_envs * PPO.rollout_steps if args.smoke else args.total_steps),
        num_envs=args.num_envs,
        eval_episodes=(64 if args.smoke else args.eval_episodes),
    )
    print(json.dumps({"variant": args.variant, "seed": args.seed, "ppo": asdict(cfg), "policy": asdict(POLICY)}, indent=2))
    tag = f"smoke_{args.tag}".rstrip("_") if args.smoke else args.tag
    train(args.variant, args.seed, cfg, args.device, tag)


if __name__ == "__main__":
    main()
