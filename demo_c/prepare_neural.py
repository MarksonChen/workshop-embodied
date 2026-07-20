"""Build temporally aligned neural/representation caches from Aldarondo sessions."""
import argparse

import torch

from demo_c.motor import FrozenMotor
from demo_c.neural_data import DEFAULT_SESSIONS, build_cache, default_checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="first session, first 60k frames")
    args = parser.parse_args()
    sessions = args.sessions[:1] if args.smoke else args.sessions
    max_frames = 60_000 if args.smoke else args.max_frames
    checkpoints = default_checkpoints()
    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        raise SystemExit(f"missing trained policies: {missing}")
    motor = FrozenMotor(args.device)
    for session in sessions:
        build_cache(session, motor, checkpoints, max_frames=max_frames, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
