"""Plot the crossed-seed Demo J RSM/RSA result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from demo_j.artifacts import OUTPUT_ROOT, write_json


BOOTSTRAPS = 20_000
COLORS = {"rsa": "#007C91", "partial": "#7B4AB5", "delay": "#777777"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "white",
        }
    )


def _arrays(report: dict) -> tuple[dict[str, np.ndarray], np.ndarray, tuple, tuple]:
    betas = np.asarray(sorted({float(row["beta"]) for row in report["comparisons"]}))
    snn_seeds = tuple(sorted({int(row["snn_seed"]) for row in report["comparisons"]}))
    h_seeds = tuple(sorted({int(row["h_seed"]) for row in report["comparisons"]}))
    shape = (len(snn_seeds), len(h_seeds), len(betas))
    fields = {
        "rsa": "crossvalidated_rdm_spearman",
        "partial": "behavior_partial_crossvalidated_rdm_spearman",
        "delay": "delay_control_crossvalidated_rdm_spearman",
        "delay_partial": "delay_control_partial_spearman",
        "rsm": "correlation_rsm_spearman",
    }
    arrays = {name: np.full(shape, np.nan) for name in fields}
    for row in report["comparisons"]:
        beta_index = int(np.flatnonzero(np.isclose(betas, row["beta"]))[0])
        index = (
            snn_seeds.index(int(row["snn_seed"])),
            h_seeds.index(int(row["h_seed"])),
            beta_index,
        )
        for name, field in fields.items():
            if not np.isnan(arrays[name][index]):
                raise ValueError(f"duplicate comparison {index}")
            arrays[name][index] = row[field]
    if any(np.isnan(value).any() for value in arrays.values()):
        raise ValueError("incomplete crossed-seed RSA")
    return arrays, betas, snn_seeds, h_seeds


def _crossed_bootstrap(values: np.ndarray, random: np.random.Generator) -> np.ndarray:
    snn = random.integers(0, values.shape[0], size=(BOOTSTRAPS, values.shape[0]))
    policy = random.integers(0, values.shape[1], size=(BOOTSTRAPS, values.shape[1]))
    samples = np.empty((BOOTSTRAPS, values.shape[-1]))
    for index in range(BOOTSTRAPS):
        samples[index] = values[snn[index]][:, policy[index]].mean(axis=(0, 1))
    return samples


def _summary(values: np.ndarray, random: np.random.Generator) -> dict[str, object]:
    bootstrap = _crossed_bootstrap(values, random)
    return {
        "mean": values.mean(axis=(0, 1)).tolist(),
        "crossed_seed_95_ci_low": np.quantile(bootstrap, 0.025, axis=0).tolist(),
        "crossed_seed_95_ci_high": np.quantile(bootstrap, 0.975, axis=0).tolist(),
        "raw_crossed_seed_values": values.tolist(),
    }


def _draw_curve(ax, values, delayed, *, betas, color, title, ylabel, random):
    for array, line_color, label, marker in (
        (values, color, "aligned", "o"),
        (delayed, COLORS["delay"], "Demo H delayed by 200 ms", "s"),
    ):
        bootstrap = _crossed_bootstrap(array, random)
        low, high = np.quantile(bootstrap, (0.025, 0.975), axis=0)
        ax.fill_between(betas, low, high, color=line_color, alpha=0.16, linewidth=0)
        ax.plot(
            betas,
            array.mean(axis=(0, 1)),
            color=line_color,
            marker=marker,
            linewidth=2,
            label=label,
        )
        jitter = np.linspace(-0.0025, 0.0025, values.shape[0] * values.shape[1])
        for beta_index, beta in enumerate(betas):
            ax.scatter(
                beta + jitter,
                array[..., beta_index].reshape(-1),
                color=line_color,
                alpha=0.25,
                s=13,
                linewidth=0,
            )
    ax.set_title(title, loc="left")
    ax.set_xlabel(r"prior strength $\beta$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(betas)
    ax.set_xticklabels([f"{value:g}" for value in betas])


def plot(report_path: Path, output_dir: Path) -> dict[str, object]:
    report = json.loads(Path(report_path).read_text())
    if report.get("schema") != "demo-j-rsm-rsa-v2":
        raise ValueError(report.get("schema"))
    arrays, betas, snn_seeds, h_seeds = _arrays(report)
    exact_input_control = "exact raw" in report.get("behavior_control", "")
    random = np.random.default_rng(20_260_721)
    summary = {
        "schema": "demo-j-rsm-rsa-summary-v2",
        "source": str(report_path),
        "betas": betas.tolist(),
        "snn_seeds": list(snn_seeds),
        "demo_h_seeds": list(h_seeds),
        "uncertainty": (
            "descriptive 95% crossed hierarchical bootstrap; SNN and Demo H "
            "seeds resampled independently"
            if len(snn_seeds) > 1 and len(h_seeds) > 1
            else "single crossed seed pair; intervals are degenerate"
        ),
        **{name: _summary(value, random) for name, value in arrays.items()},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(output_dir / "beta_rsa_summary.json", summary)

    _style()
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    _draw_curve(
        axes[0],
        arrays["rsa"],
        arrays["delay"],
        betas=betas,
        color=COLORS["rsa"],
        title="A  Crossvalidated representational geometry",
        ylabel="RDM Spearman correlation",
        random=random,
    )
    _draw_curve(
        axes[1],
        arrays["partial"],
        arrays["delay_partial"],
        betas=betas,
        color=COLORS["partial"],
        title=(
            "B  Geometry beyond the exact SNN input"
            if exact_input_control
            else "B  Geometry beyond the measured motion"
        ),
        ylabel=(
            "input-partial RDM correlation"
            if exact_input_control
            else "behavior-partial RDM correlation"
        ),
        random=random,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Demo J 64-frame RSM/RSA on matched speed × contact conditions",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    figure.subplots_adjust(top=0.76, bottom=0.16, left=0.08, right=0.98, wspace=0.29)
    curve_png = output_dir / "beta_rsa.png"
    figure.savefig(curve_png, dpi=220)
    figure.savefig(curve_png.with_suffix(".svg"))
    plt.close(figure)

    matrix_path = Path(report["matrix_artifact"])
    with np.load(matrix_path) as archive:
        speeds = np.asarray(archive["condition_speed"])
        snn_seed = np.asarray(archive["snn_seed"])
        h_beta = np.asarray(archive["h_beta"])
        h_seed = np.asarray(archive["h_seed"])
        snn_rsm = np.asarray(archive["snn_rsm"])
        h_rsm = np.asarray(archive["h_rsm"])
    display_snn_seed = 1 if np.any(snn_seed == 1) else int(snn_seed[0])
    display_beta = 0.10 if np.any(np.isclose(h_beta, 0.10)) else float(h_beta[0])
    display_h_seed = 0 if np.any(h_seed == 0) else int(h_seed[0])
    snn_index = int(np.flatnonzero(snn_seed == display_snn_seed)[0])
    h_index = int(
        np.flatnonzero(np.isclose(h_beta, display_beta) & (h_seed == display_h_seed))[0]
    )
    boundaries = np.flatnonzero(np.diff(speeds)) + 0.5
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)
    for ax, matrix, title in (
        (
            axes[0],
            snn_rsm[snn_index],
            f"A  Demo J SNN spikes (seed {display_snn_seed})",
        ),
        (
            axes[1],
            h_rsm[h_index],
            f"B  Demo H hidden state ($\\beta={display_beta:g}$, seed {display_h_seed})",
        ),
    ):
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", origin="lower")
        for boundary in boundaries:
            ax.axhline(boundary, color="white", linewidth=0.6, alpha=0.8)
            ax.axvline(boundary, color="white", linewidth=0.6, alpha=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("speed × contact condition")
        ax.set_ylabel("speed × contact condition")
        ax.grid(False)
    figure.colorbar(image, ax=axes, shrink=0.82, label="population-pattern correlation")
    rsm_png = output_dir / "rsm_examples.png"
    figure.savefig(rsm_png, dpi=220)
    figure.savefig(rsm_png.with_suffix(".svg"))
    plt.close(figure)
    print(
        json.dumps(
            {"summary": str(summary_path), "charts": [str(curve_png), str(rsm_png)]},
            indent=2,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    plot(args.report, args.output_dir)


if __name__ == "__main__":
    main()
