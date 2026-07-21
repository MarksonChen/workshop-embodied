"""CLI for the modern-MJX reference replay cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from demo_j.projection import DEFAULT_ROOT, build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(build(args.output_root, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
