"""Leak-resistant Poisson encoding and crossvalidated RSA for Demo C.

Primary question: at a fixed 16-dimensional bottleneck, which frozen representation
best explains held-out real neural population activity? The result is empirical; this
script contains no rule that assumes Demo C must win.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy.special import gammaln
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from demo_c.neural_data import CACHE, CACHE_VERSION, DEFAULT_SESSIONS, GOAL_TOKENS, H, build_cache, default_checkpoints, load_cache
from demo_c.motor import FrozenMotor

OUT = Path(__file__).resolve().parent / "out" / "neural"
TOKEN_SECONDS = 0.08
BLOCK_TOKENS = round(60 / TOKEN_SECONDS)
GAP_TOKENS = round(5 / TOKEN_SECONDS)
PCA_DIM = 16
ALPHA = 0.01
POISSON_SOLVER = "newton-cholesky"  # exact convex GLM solver; much faster for n_rows >> 16 PCs
REPRESENTATION_PREFIXES = (
    "kinematics", "motion_latent", "predictive_context", "goal_only_policy_s", "wam_policy_s"
)


def split_masks(data, scope="loco"):
    """60-s blocks, 5-s boundary gaps, and whole-input-window containment."""
    index = data["token_index"].astype(np.int64)
    block = index // BLOCK_TOKENS; within = index % BLOCK_TOKENS
    fold = block % 5
    safe = (within >= GAP_TOKENS + H) & (within < BLOCK_TOKENS - GAP_TOKENS - GOAL_TOKENS)
    split_kind = {
        "train": fold <= 2,
        "val": fold == 3,
        "test": fold == 4,
    }
    if scope == "loco":
        safe &= data["locomotion"].astype(bool)
    elif scope == "all_matched":
        # Fixed n-matched control recommended by canvas/misc/autoresearch.md: draw
        # from the full behavioral range but match locomotion's sample count in
        # every split, so a difference cannot be blamed on subset size alone.
        loco = data["locomotion"].astype(bool)
        matched = np.zeros_like(safe)
        rng = np.random.default_rng(20260719)
        for kind in split_kind.values():
            candidates = np.flatnonzero(safe & kind)
            count = int(np.sum(safe & kind & loco))
            matched[rng.choice(candidates, count, replace=False)] = True
        safe &= matched
    elif scope != "all":
        raise ValueError(scope)
    return {kind: safe & mask for kind, mask in split_kind.items()}


def fit_projection(x, train_mask, dimension=PCA_DIM):
    scaler = StandardScaler().fit(x[train_mask])
    scaled = scaler.transform(x)
    n = min(dimension, scaled.shape[1], int(train_mask.sum()) - 1)
    pca = PCA(n_components=n, random_state=0).fit(scaled[train_mask])
    return pca.transform(scaled).astype(np.float32), {"explained_variance": float(pca.explained_variance_ratio_.sum())}


def circular_shift_within_blocks(x, token_index, shift_tokens=250):
    """20-s alignment null that never moves a feature across a split block."""
    output = np.empty_like(x)
    blocks = token_index.astype(np.int64) // BLOCK_TOKENS
    for block in np.unique(blocks):
        rows = np.flatnonzero(blocks == block)
        shift = min(shift_tokens, max(1, len(rows) // 3))
        output[rows] = x[np.roll(rows, shift)]
    return output


def poisson_log_likelihood(y, rate):
    rate = np.clip(rate, 1e-8, 1e6)
    return y * np.log(rate) - rate - gammaln(y + 1)


def score_poisson(projected, spikes, masks):
    train, val, test = masks["train"], masks["val"], masks["test"]
    unit_bps, val_bps = [], []
    aligned_ll = null_ll = aligned_val_ll = null_val_ll = 0.0
    test_spikes = val_spikes = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for unit in range(spikes.shape[1]):
            y = spikes[:, unit].astype(np.float64)
            mean_rate = max(float(y[train].mean()), 1e-8)
            model = PoissonRegressor(alpha=ALPHA, solver=POISSON_SOLVER, max_iter=100, tol=1e-6)
            model.fit(projected[train], y[train])
            for mask, collection, is_test in ((test, unit_bps, True), (val, val_bps, False)):
                ll_model = float(poisson_log_likelihood(y[mask], model.predict(projected[mask])).sum())
                ll_null = float(poisson_log_likelihood(y[mask], np.full(mask.sum(), mean_rate)).sum())
                count = float(y[mask].sum())
                collection.append((ll_model - ll_null) / (max(count, 1.0) * math.log(2)))
                if is_test:
                    aligned_ll += ll_model; null_ll += ll_null; test_spikes += count
                else:
                    aligned_val_ll += ll_model; null_val_ll += ll_null; val_spikes += count
    return {
        "population_bps": (aligned_ll - null_ll) / (max(test_spikes, 1.0) * math.log(2)),
        "validation_population_bps": (aligned_val_ll - null_val_ll) / (max(val_spikes, 1.0) * math.log(2)),
        "unit_bps_median": float(np.median(unit_bps)),
        "unit_bps_mean": float(np.mean(unit_bps)),
        "positive_unit_fraction": float(np.mean(np.asarray(unit_bps) > 0)),
        "test_spikes": int(test_spikes),
        "unit_bps": unit_bps,
    }


def _condition_labels(data, train_mask):
    speed, turn = data["speed"].astype(float), data["turn_rate"].astype(float)
    speed_edges = np.quantile(speed[train_mask], [1 / 3, 2 / 3])
    turn_edges = np.quantile(turn[train_mask], [1 / 3, 2 / 3])
    return np.digitize(speed, speed_edges) * 3 + np.digitize(turn, turn_edges)


def _crossnobis(x, condition, partition, min_count=10):
    conditions = [c for c in range(9) if all(np.sum((condition == c) & (partition == p)) >= min_count for p in (0, 1))]
    if len(conditions) < 4:
        return None, conditions, float("nan")
    means = np.stack([[x[(condition == c) & (partition == p)].mean(0) for p in (0, 1)] for c in conditions])
    grand = np.stack([x[condition == c].mean(0) for c in conditions])
    residual = x.copy()
    for i, c in enumerate(conditions):
        residual[condition == c] -= grand[i]
    precision = LedoitWolf().fit(residual).precision_
    rdm, rdm_a, rdm_b = [], [], []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            da, db = means[i, 0] - means[j, 0], means[i, 1] - means[j, 1]
            rdm.append(float(da @ precision @ db / x.shape[1]))
            rdm_a.append(float(da @ precision @ da / x.shape[1]))
            rdm_b.append(float(db @ precision @ db / x.shape[1]))
    reliability = float(spearmanr(rdm_a, rdm_b).statistic)
    return np.asarray(rdm), conditions, reliability


def _permute_rdm_test(model_rdm, neural_rdm, n_conditions, seed=20260719, permutations=1000):
    """Condition-label permutation null for RDM correspondence."""
    matrix = np.zeros((n_conditions, n_conditions), float); upper = np.triu_indices(n_conditions, 1)
    matrix[upper] = model_rdm; matrix[(upper[1], upper[0])] = model_rdm
    observed = float(spearmanr(model_rdm, neural_rdm).statistic)
    rng = np.random.default_rng(seed); null = np.empty(permutations)
    for i in range(permutations):
        order = rng.permutation(n_conditions)
        permuted = matrix[np.ix_(order, order)][upper]
        null[i] = spearmanr(permuted, neural_rdm).statistic
    p = (1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1)
    return observed, float(p), float(np.nanmean(null)), float(np.nanstd(null))


def rsa_score(projected, neural_projected, data, masks):
    test_rows = np.flatnonzero(masks["test"])
    absolute_blocks = data["token_index"][test_rows].astype(np.int64) // BLOCK_TOKENS
    partition = absolute_blocks % 2
    condition_all = _condition_labels(data, masks["train"])
    condition = condition_all[test_rows]
    model_rdm, conditions, model_reliability = _crossnobis(projected[test_rows], condition, partition)
    neural_rdm, neural_conditions, neural_reliability = _crossnobis(neural_projected[test_rows], condition, partition)
    if model_rdm is None or neural_rdm is None or conditions != neural_conditions:
        return {"rsa_spearman": float("nan"), "conditions": len(conditions)}
    score, p, null_mean, null_std = _permute_rdm_test(model_rdm, neural_rdm, len(conditions))
    return {
        "rsa_spearman": score,
        "rsa_permutation_p": p,
        "rsa_permutation_null_mean": null_mean,
        "rsa_permutation_null_std": null_std,
        "model_rdm_reliability": model_reliability,
        "neural_rdm_reliability": neural_reliability,
        "conditions": len(conditions),
    }


def representation_names(data):
    return [name for name in data if any(name == p or name.startswith(p) for p in REPRESENTATION_PREFIXES)]


def evaluate_session(cache_path: Path, scope: str):
    data = load_cache(cache_path); metadata = json.loads(str(data["metadata"]))
    masks = split_masks(data, scope)
    counts = {key: int(mask.sum()) for key, mask in masks.items()}
    if min(counts.values()) < 100:
        raise ValueError(f"too few {scope} rows after strict split for {metadata['session']}: {counts}")
    spikes = data["spikes"].astype(np.float32)
    neural_projected, _ = fit_projection(np.sqrt(spikes), masks["train"])
    rows = []
    for name in representation_names(data):
        raw = data[name].astype(np.float32)
        projected, projection = fit_projection(raw, masks["train"])
        encoding = score_poisson(projected, spikes, masks)
        shifted = circular_shift_within_blocks(raw, data["token_index"])
        shifted_projected, _ = fit_projection(shifted, masks["train"])
        shuffle = score_poisson(shifted_projected, spikes, masks)
        rsa = rsa_score(projected, neural_projected, data, masks)
        shifted_rsa = rsa_score(shifted_projected, neural_projected, data, masks)
        row = {
            "session": metadata["session"], "animal": metadata["animal"], "region": metadata["region"],
            "scope": scope, "representation": name, "split_counts": counts,
            "pca_explained_variance": projection["explained_variance"],
            **encoding,
            "shuffle_population_bps": shuffle["population_bps"],
            "corrected_population_bps": encoding["population_bps"] - max(shuffle["population_bps"], 0.0),
            **rsa,
            "shuffle_rsa_spearman": shifted_rsa["rsa_spearman"],
        }
        rows.append(row)
        print(
            f"{metadata['session']:>28} {name:>24}: "
            f"bps={row['population_bps']:+.4f} shuffle={row['shuffle_population_bps']:+.4f} "
            f"RSA={row['rsa_spearman']:+.3f}", flush=True,
        )
    return rows


def family(name):
    if name.startswith("goal_only_policy_s"):
        return "RL-only policy"
    if name.startswith("wam_policy_s"):
        return "Demo C WAM+RL policy"
    return {"kinematics": "kinematics", "motion_latent": "Demo B autoencoder", "predictive_context": "Demo B predictor"}.get(name, name)


def summarize(rows):
    summary = {}
    for family_name in sorted({family(row["representation"]) for row in rows}):
        chosen = [row for row in rows if family(row["representation"]) == family_name]
        # First average policy seeds within a session, then balance sessions.
        per_session = []
        for session in sorted({row["session"] for row in chosen}):
            group = [row for row in chosen if row["session"] == session]
            per_session.append({
                "bps": float(np.mean([row["population_bps"] for row in group])),
                "corrected_bps": float(np.mean([row["corrected_population_bps"] for row in group])),
                "rsa": float(np.nanmean([row["rsa_spearman"] for row in group])),
                "rsa_p": float(np.nanmean([row.get("rsa_permutation_p", float("nan")) for row in group])),
            })
        summary[family_name] = {
            "population_bps_mean": float(np.mean([x["bps"] for x in per_session])),
            "population_bps_session_sd": float(np.std([x["bps"] for x in per_session], ddof=1)) if len(per_session) > 1 else 0.0,
            "corrected_population_bps_mean": float(np.mean([x["corrected_bps"] for x in per_session])),
            "rsa_spearman_mean": float(np.nanmean([x["rsa"] for x in per_session])),
            "rsa_permutation_p_session_mean": float(np.nanmean([x["rsa_p"] for x in per_session])),
            "sessions": len(per_session),
        }
    return summary


def paired_contrasts(rows):
    """Exploratory paired session effects; exact sign test reflects the small n."""
    family_rows = {}
    for row in rows:
        key = (row["session"], family(row["representation"]))
        family_rows.setdefault(key, []).append(row)
    sessions = sorted({row["session"] for row in rows})

    def values(family_name, metric):
        return np.array([
            np.mean([row[metric] for row in family_rows[(session, family_name)]]) for session in sessions
        ])

    def exact_sign_p(difference):
        observed = abs(float(difference.mean())); n = len(difference)
        means = []
        for bits in range(1 << n):
            signs = np.array([1 if bits & (1 << i) else -1 for i in range(n)])
            means.append(abs(float(np.mean(difference * signs))))
        return float(np.mean(np.asarray(means) >= observed - 1e-12))

    rng = np.random.default_rng(20260719); output = {}
    for reference in ("RL-only policy", "Demo B autoencoder", "Demo B predictor", "kinematics"):
        output[reference] = {}
        for metric in ("population_bps", "corrected_population_bps", "rsa_spearman"):
            difference = values("Demo C WAM+RL policy", metric) - values(reference, metric)
            bootstrap = np.array([
                difference[rng.integers(0, len(difference), len(difference))].mean() for _ in range(10_000)
            ])
            output[reference][metric] = {
                "demo_c_minus_reference_mean": float(difference.mean()),
                "bootstrap_95_interval": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
                "exact_sign_permutation_p": exact_sign_p(difference),
                "session_differences": difference.tolist(),
            }
    return output


def plot_summary(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(summary); values = [summary[x]["population_bps_mean"] for x in labels]
    errors = [summary[x]["population_bps_session_sd"] for x in labels]
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    colors = ["#888888", "#4b9cc8", "#5b7db1", "#e78b39", "#57a773"][:len(labels)]
    ax.bar(np.arange(len(labels)), values, yerr=errors, color=colors, capsize=3)
    ax.axhline(0, color="black", lw=.8); ax.set_ylabel("held-out population bits/spike")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS))
    parser.add_argument("--scope", choices=("loco", "all_matched", "all"), default="loco")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    sessions = args.sessions[:1] if args.smoke else args.sessions
    max_frames = 60_000 if args.smoke else None
    motor = None; paths = []
    for session in sessions:
        suffix = f"_n{max_frames}" if max_frames else ""
        path = CACHE / f"v{CACHE_VERSION}_{session.replace('/', '_')}{suffix}.npz"
        if not path.exists() or args.rebuild:
            motor = motor or FrozenMotor("cuda")
            path = build_cache(session, motor, default_checkpoints(), max_frames=max_frames, rebuild=args.rebuild)
        paths.append(path)
    rows = []
    for path in paths:
        rows.extend(evaluate_session(path, args.scope))
    summary = summarize(rows)
    contrasts = paired_contrasts(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scope}" + ("_smoke" if args.smoke else "")
    (OUT / f"{stem}.json").write_text(json.dumps({"frozen": {
        "block_seconds": 60, "gap_seconds": 5, "pca_dimension": PCA_DIM, "poisson_alpha": ALPHA,
        "poisson_solver": POISSON_SOLVER,
        "scope": args.scope, "sessions": sessions,
    }, "summary": summary, "contrasts": contrasts, "rows": rows}, indent=2) + "\n")
    plot_summary(summary, OUT / f"{stem}.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
