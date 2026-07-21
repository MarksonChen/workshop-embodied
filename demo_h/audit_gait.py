"""Apply the four-limb contact gate to saved Demo H rollout traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from demo_h.gait_metrics import four_limb_contact_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = []
    for path in args.traces:
        with np.load(path) as trace:
            report = four_limb_contact_metrics(trace["contacts"])
        report = {"trace": str(path), **report}
        reports.append(report)
        print(json.dumps(report), flush=True)
    output = {
        "schema": "demo-h-four-limb-gait-audit-v1",
        "all_pass": all(row["passes_four_limb_gait_gate"] for row in reports),
        "traces": reports,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
