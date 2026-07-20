"""Aggregate the predeclared three-seed Demo G comparison.

The summary retains the failed full-composite and cyclicity diagnostics instead
of reducing the result to learned likelihood alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import GAIT_FIELDS

OUT = Path(__file__).resolve().parent / "out"
DEFAULT_REPORTS = tuple(
    OUT / f"evaluation_dynamic_seed{seed}.json" for seed in range(3)
)


DIRECT_FIELDS = GAIT_FIELDS


def scalar_summary(values) -> dict:
    values = np.asarray(values, np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "positive_seeds": int((values > 0).sum()),
        "count": len(values),
    }


def training_metadata(checkpoint: str) -> dict:
    path = Path(checkpoint).with_suffix(".json")
    if not path.is_file():
        return {}
    report = json.loads(path.read_text())
    metadata = {
        name: report.get(name)
        for name in (
            "arm",
            "beta",
            "seed",
            "num_timesteps",
            "num_envs",
            "training_seconds",
        )
    }
    # The original accepted runs used the already-frozen stride-4 wrapper but
    # predate logging this field. Preserve that provenance instead of emitting
    # an unexplained null in the aggregate report.
    if "score_stride" in report:
        metadata["score_stride"] = report["score_stride"]
        metadata["score_stride_source"] = "checkpoint_report"
    else:
        metadata["score_stride"] = 4
        metadata["score_stride_source"] = "frozen_default_not_logged"
    return metadata


def summarize(paths: list[Path]) -> dict:
    reports = [json.loads(path.read_text()) for path in paths]
    rows, raw, composite, track, survival = [], [], [], [], []
    direct = {name: [] for name in DIRECT_FIELDS}
    task_ratio, action_reduction = [], []
    runtimes = []
    for path, report in zip(paths, reports, strict=True):
        g0, g1 = report["arms"]["g0"], report["arms"]["g1"]
        comparison = report["comparison"]
        raw.append(comparison["raw_logp"]["mean_improvement"])
        composite.append(comparison["direct_gait_distance"]["mean_improvement"])
        track.append(comparison["track_retention_ratio"])
        survival.append(comparison["survival_retention_ratio"])
        task_ratio.append(
            g1["aggregate"]["task_return"]["mean"]
            / max(g0["aggregate"]["task_return"]["mean"], 1e-8)
        )
        action_reduction.append(
            g0["aggregate"]["action_energy"]["mean"]
            - g1["aggregate"]["action_energy"]["mean"]
        )
        for name in DIRECT_FIELDS:
            target = report["test_gait_reference"][name]["mean"]
            g0_distance = abs(g0["aggregate"][f"gait_{name}"]["mean"] - target)
            g1_distance = abs(g1["aggregate"][f"gait_{name}"]["mean"] - target)
            direct[name].append(g0_distance - g1_distance)
        metadata = {
            arm: training_metadata(report["arms"][arm]["checkpoint"])
            for arm in ("g0", "g1")
        }
        runtimes.extend(
            item.get("training_seconds")
            for item in metadata.values()
            if item.get("training_seconds") is not None
        )
        rows.append(
            {
                "evaluation": str(path),
                "training": metadata,
                "raw_logp_improvement": raw[-1],
                "direct_composite_improvement": composite[-1],
                "track_retention_ratio": track[-1],
                "survival_retention_ratio": survival[-1],
                "direct_field_improvement": {
                    name: direct[name][-1] for name in DIRECT_FIELDS
                },
            }
        )

    direct_summary = {name: scalar_summary(values) for name, values in direct.items()}
    raw_summary = scalar_summary(raw)
    robust_direct = [
        name
        for name, values in direct_summary.items()
        if values["positive_seeds"] == len(paths) and values["mean"] > 0
    ]
    gates = {
        "every_arm_under_two_minutes": bool(runtimes and max(runtimes) < 120),
        "tracking_retained_each_seed": bool(min(track) >= 0.95),
        "survival_retained_each_seed": bool(min(survival) >= 0.95),
        "raw_logp_improves_each_seed": bool(raw_summary["positive_seeds"] == len(paths)),
        "raw_logp_exceeds_two_sigma": bool(raw_summary["mean"] > 2 * raw_summary["std"]),
        "at_least_one_direct_measure_improves_each_seed": bool(robust_direct),
        "direct_composite_improves_each_seed": bool(min(composite) > 0),
    }
    limited_claim_gates = {
        key: value
        for key, value in gates.items()
        if key != "direct_composite_improves_each_seed"
    }
    return {
        "schema": "demo-g-multiseed-summary-v1",
        "reports": rows,
        "aggregate": {
            "raw_logp_improvement": raw_summary,
            "direct_composite_improvement": scalar_summary(composite),
            "track_retention_delta_fraction": scalar_summary(np.asarray(track) - 1.0),
            "survival_retention_delta_fraction": scalar_summary(
                np.asarray(survival) - 1.0
            ),
            "task_return_delta_fraction": scalar_summary(
                np.asarray(task_ratio) - 1.0
            ),
            "action_energy_reduction": scalar_summary(action_reduction),
            "direct_field_improvement": direct_summary,
        },
        "robustly_improved_direct_fields": robust_direct,
        "gates": gates,
        "accepted_limited_claim": all(limited_claim_gates.values()),
        "limited_claim": (
            "At beta=0.1, the frozen motion prior improves held-out learned likelihood "
            "while retaining function. Airborne fraction, stance-foot speed, approximate "
            "stance-world foot speed, and joint-speed RMS move toward held-out motion in "
            "all three training seeds, but the complete gait composite improves in only "
            "two seeds and cyclicity improves in none."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="*", default=DEFAULT_REPORTS)
    parser.add_argument(
        "--output", type=Path, default=OUT / "evaluation_dynamic_multiseed.json"
    )
    args = parser.parse_args()
    report = summarize(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"aggregate": report["aggregate"], "gates": report["gates"]}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
