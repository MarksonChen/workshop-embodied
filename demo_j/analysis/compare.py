"""Compare Demo H hidden states with Demo J spikes using RSM/RSA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from demo_j.artifacts import OUTPUT_ROOT, sha256
from demo_f.features import SL
from demo_j.analysis.rsa import (
    causal_shift,
    condition_repeat_means,
    make_condition_design,
    partial_spearman_rsa,
    permutation_control,
    representational_geometry,
    spearman_rsa,
)


PRIMARY_LAYER = "hidden_2"
ACTIVATION_FIELDS = (
    "hidden_1",
    "hidden_2",
    "residual_mean_correction",
    "raw_scale_correction",
    "policy_mean",
)
WARMUP_BINS = 8
MINIMUM_BINS_PER_REPEAT = 5
SHIFT_BINS = 10


def _metadata(path: Path) -> dict:
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    return json.loads(sidecar.read_text())


def _split_scores(left, right, behavior) -> dict[str, object]:
    ordinary = []
    partial = []
    for left_rdm, right_rdm, behavior_rdm in zip(
        left.split_rdm,
        right.split_rdm,
        behavior.split_rdm,
        strict=True,
    ):
        ordinary.append(spearman_rsa(left_rdm, right_rdm))
        partial.append(partial_spearman_rsa(left_rdm, right_rdm, behavior_rdm))
    ordinary = np.asarray(ordinary)
    partial = np.asarray(partial)
    return {
        "crossvalidated_rdm_spearman": spearman_rsa(
            left.crossvalidated_rdm, right.crossvalidated_rdm
        ),
        "behavior_partial_crossvalidated_rdm_spearman": partial_spearman_rsa(
            left.crossvalidated_rdm,
            right.crossvalidated_rdm,
            behavior.crossvalidated_rdm,
        ),
        "split_spearman": ordinary.tolist(),
        "split_spearman_standard_error": float(
            ordinary.std(ddof=1) / np.sqrt(len(ordinary))
        ),
        "split_partial_spearman": partial.tolist(),
        "split_partial_spearman_standard_error": float(
            partial.std(ddof=1) / np.sqrt(len(partial))
        ),
        "correlation_rsm_spearman": spearman_rsa(
            left.correlation_rdm, right.correlation_rdm
        ),
    }


def compare(
    recordings: tuple[Path, ...],
    trace: Path,
    activations: tuple[Path, ...],
    output: Path,
    *,
    layer: str = PRIMARY_LAYER,
    permutations: int = 1_000,
    exclude_input_quartile: bool = False,
) -> dict[str, object]:
    """Run matched-condition RSM/RSA across SNN and Demo H seeds."""

    if layer not in ACTIVATION_FIELDS:
        raise ValueError(layer)
    trace_metadata = _metadata(trace)
    trace_hash = sha256(trace)
    with np.load(trace) as archive:
        feature = np.asarray(archive["feature"], np.float32)
        target_speed = np.asarray(archive["target_speed"], np.float32)
    contact_start, contact_stop = SL["contacts"]

    recording_rows = []
    recording_counts = []
    recording_behavior = []
    recording_steps = set()
    for path in recordings:
        metadata = _metadata(path)
        if metadata["trace_sha256"] != trace_hash:
            raise ValueError(f"recording/trace mismatch for {path}")
        with np.load(path) as archive:
            counts = np.asarray(archive["spike_counts_20ms"], np.float64)
            behavior = (
                np.asarray(archive["behavior"], np.float64)
                if "behavior" in archive.files
                else feature[:, : counts.shape[1]].astype(np.float64)
            )
            retained_neurons = np.ones(counts.shape[-1], bool)
            if exclude_input_quartile:
                if "input_weight_norm" not in archive.files:
                    raise ValueError(
                        f"{path} has no input_weight_norm for the sensitivity analysis"
                    )
                norm = np.asarray(archive["input_weight_norm"], np.float64)
                retained_neurons = norm <= np.quantile(norm, 0.75)
                counts = counts[..., retained_neurons]
        recording_steps.add(counts.shape[1])
        recording_counts.append(counts)
        recording_behavior.append(behavior)
        recording_rows.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "checkpoint": metadata["checkpoint"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "snn_seed": int(metadata["snn_seed"]),
                "neurons_retained": int(retained_neurons.sum()),
                "neurons_total": int(len(retained_neurons)),
            }
        )
    if len(recording_steps) != 1:
        raise ValueError(f"recording lengths disagree: {recording_steps}")
    steps = recording_steps.pop()
    design = make_condition_design(
        feature[:, :steps, contact_start:contact_stop],
        target_speed,
        warmup_bins=WARMUP_BINS,
        minimum_bins_per_repeat=MINIMUM_BINS_PER_REPEAT,
    )
    behavior_geometry = [
        representational_geometry(condition_repeat_means(behavior[:, :steps], design))
        for behavior in recording_behavior
    ]
    # Complete construction only after the shared design is frozen.
    snn_geometry = [
        representational_geometry(condition_repeat_means(counts, design))
        for counts in recording_counts
    ]

    activation_rows = []
    h_geometry = []
    delayed_geometry = []
    for path in activations:
        metadata = _metadata(path)
        if metadata["trace_sha256"] != trace_hash:
            raise ValueError(f"activation/trace mismatch for {path}")
        with np.load(path) as archive:
            values = np.asarray(archive[layer], np.float64)[:, :steps]
        h_geometry.append(
            representational_geometry(condition_repeat_means(values, design))
        )
        delayed_geometry.append(
            representational_geometry(
                condition_repeat_means(causal_shift(values, SHIFT_BINS), design)
            )
        )
        activation_rows.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "checkpoint": metadata["checkpoint"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "beta": float(metadata["beta"]),
                "seed": int(metadata["seed"]),
            }
        )

    comparisons = []
    for snn_index, (snn_row, snn) in enumerate(
        zip(recording_rows, snn_geometry, strict=True)
    ):
        for h_index, (h_row, h, delayed) in enumerate(
            zip(activation_rows, h_geometry, delayed_geometry, strict=True)
        ):
            score = _split_scores(snn, h, behavior_geometry[snn_index])
            delay_score = _split_scores(snn, delayed, behavior_geometry[snn_index])
            permutation = permutation_control(
                snn.crossvalidated_rdm,
                h.crossvalidated_rdm,
                permutations=permutations,
                seed=10_000 * snn_index + h_index,
            )
            row = {
                "snn_seed": snn_row["snn_seed"],
                "snn_checkpoint_sha256": snn_row["checkpoint_sha256"],
                "beta": h_row["beta"],
                "h_seed": h_row["seed"],
                "h_checkpoint_sha256": h_row["checkpoint_sha256"],
                **score,
                "delay_control_crossvalidated_rdm_spearman": delay_score[
                    "crossvalidated_rdm_spearman"
                ],
                "delay_control_partial_spearman": delay_score[
                    "behavior_partial_crossvalidated_rdm_spearman"
                ],
                "condition_permutation": permutation,
            }
            comparisons.append(row)
            print(
                json.dumps(
                    {
                        "snn_seed": row["snn_seed"],
                        "beta": row["beta"],
                        "h_seed": row["h_seed"],
                        "rsa": row["crossvalidated_rdm_spearman"],
                        "partial": row["behavior_partial_crossvalidated_rdm_spearman"],
                        "delay": row["delay_control_crossvalidated_rdm_spearman"],
                    }
                ),
                flush=True,
            )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix_path = output.with_suffix(".npz")
    np.savez_compressed(
        matrix_path,
        condition_speed=design.speed.astype(np.float32),
        condition_contact_pattern=design.contact_pattern,
        condition_sample_count=design.sample_count,
        behavior_crossvalidated_rdm=np.stack(
            [geometry.crossvalidated_rdm for geometry in behavior_geometry]
        ).astype(np.float32),
        behavior_rsm=np.stack([geometry.rsm for geometry in behavior_geometry]).astype(
            np.float32
        ),
        snn_crossvalidated_rdm=np.stack(
            [geometry.crossvalidated_rdm for geometry in snn_geometry]
        ).astype(np.float32),
        snn_rsm=np.stack([geometry.rsm for geometry in snn_geometry]).astype(
            np.float32
        ),
        h_crossvalidated_rdm=np.stack(
            [geometry.crossvalidated_rdm for geometry in h_geometry]
        ).astype(np.float32),
        h_rsm=np.stack([geometry.rsm for geometry in h_geometry]).astype(np.float32),
        delayed_h_crossvalidated_rdm=np.stack(
            [geometry.crossvalidated_rdm for geometry in delayed_geometry]
        ).astype(np.float32),
        snn_seed=np.asarray([row["snn_seed"] for row in recording_rows]),
        h_beta=np.asarray([row["beta"] for row in activation_rows], np.float32),
        h_seed=np.asarray([row["seed"] for row in activation_rows]),
    )
    report = {
        "schema": "demo-j-rsm-rsa-v1",
        "status": "secondary fixed-input representational analysis",
        "trace": str(trace),
        "trace_sha256": trace_hash,
        "trace_schema": trace_metadata.get("schema"),
        "recordings": recording_rows,
        "activations": activation_rows,
        "layer": layer,
        "warmup_bins_discarded": WARMUP_BINS,
        "delay_control_bins": SHIFT_BINS,
        "condition_definition": "target speed x exact four-foot contact pattern",
        "minimum_bins_per_condition_repeat": MINIMUM_BINS_PER_REPEAT,
        "conditions": design.conditions,
        "repeats": design.repeats,
        "condition_rows": [
            {"speed_m_s": float(speed), "contact_pattern": int(pattern)}
            for speed, pattern in zip(design.speed, design.contact_pattern, strict=True)
        ],
        "distance": (
            "diagonally noise-normalized crossvalidated squared Euclidean "
            "distance, averaged over all disjoint 2-vs-2 repeat splits"
        ),
        "rsm": "Pearson correlation between all-repeat condition means",
        "rsa": "Spearman correlation of unique RDM entries",
        "behavior_control": (
            "partial Spearman RSA controlling each SNN recording's exact raw "
            "state, previous-action, future-token, token-validity-mask, and command input"
        ),
        "input_proximity_sensitivity": (
            "excluded the top quartile of neurons by input-weight norm"
            if exclude_input_quartile
            else "all recurrent neurons retained (primary analysis)"
        ),
        "permutations": permutations,
        "matrix_artifact": str(matrix_path),
        "matrix_artifact_sha256": sha256(matrix_path),
        "comparisons": comparisons,
        "interpretation_limit": (
            "Conditions and behavior are fixed simulated trajectory measurements; "
            "RSA tests shared geometry, not unit alignment or causal equivalence."
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output} and {matrix_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, nargs="+", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--activation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "beta_rsa.json")
    parser.add_argument("--layer", choices=ACTIVATION_FIELDS, default=PRIMARY_LAYER)
    parser.add_argument("--permutations", type=int, default=1_000)
    parser.add_argument("--exclude-input-quartile", action="store_true")
    args = parser.parse_args()
    compare(
        tuple(args.recording),
        args.trace,
        tuple(args.activation),
        args.output,
        layer=args.layer,
        permutations=args.permutations,
        exclude_input_quartile=args.exclude_input_quartile,
    )


if __name__ == "__main__":
    main()
