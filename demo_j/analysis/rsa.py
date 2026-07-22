"""Cross-validated representational similarity analysis for Demo J.

The primary neural score remains held-out Poisson encoding.  This module adds
the familiar population-geometry view: a descriptive correlation RSM and a
noise-normalized, cross-validated distance matrix for inferential RSA.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import rankdata, spearmanr


CONTACT_BITS = 4
CONTACT_PATTERNS = 1 << CONTACT_BITS


def causal_shift(values: np.ndarray, bins: int = 10) -> np.ndarray:
    """Delay activity within each episode without circular wraparound."""

    values = np.asarray(values)
    if values.ndim < 2 or not 1 <= bins < values.shape[1]:
        raise ValueError((values.shape, bins))
    shifted = np.zeros_like(values)
    shifted[:, bins:] = values[:, :-bins]
    return shifted


@dataclass(frozen=True)
class ConditionDesign:
    """Locked speed-by-contact conditions with five independent repeats."""

    speed: np.ndarray
    contact_pattern: np.ndarray
    episode_index: np.ndarray
    time_mask: np.ndarray
    sample_count: np.ndarray
    warmup_bins: int

    @property
    def repeats(self) -> int:
        return int(self.episode_index.shape[0])

    @property
    def conditions(self) -> int:
        return int(self.episode_index.shape[1])

    @property
    def bins(self) -> int:
        return int(self.time_mask.shape[-1])


@dataclass(frozen=True)
class Geometry:
    """Condition patterns and their descriptive and cross-validated geometry."""

    repeat_mean: np.ndarray
    rsm: np.ndarray
    correlation_rdm: np.ndarray
    crossvalidated_rdm: np.ndarray
    split_rdm: np.ndarray
    noise_scale: np.ndarray


def contact_pattern(contacts: np.ndarray) -> np.ndarray:
    """Encode four binary foot contacts as integers in ``[0, 15]``."""

    contacts = np.asarray(contacts)
    if contacts.ndim != 3 or contacts.shape[-1] != CONTACT_BITS:
        raise ValueError(contacts.shape)
    binary = contacts >= 0.5
    return np.sum(binary * (1 << np.arange(CONTACT_BITS)), axis=-1).astype(np.int16)


def make_condition_design(
    contacts: np.ndarray,
    target_speed: np.ndarray,
    *,
    warmup_bins: int = 32,
    minimum_bins_per_repeat: int = 5,
) -> ConditionDesign:
    """Select speed/contact conditions represented in every repeat.

    Episodes must form a balanced target-speed bank.  Their within-speed order
    defines independent repeats; no temporal bins are exchanged across them.
    """

    contacts = np.asarray(contacts)
    target_speed = np.asarray(target_speed, np.float64)
    if contacts.shape[0] != len(target_speed):
        raise ValueError((contacts.shape, target_speed.shape))
    if not 0 <= warmup_bins < contacts.shape[1]:
        raise ValueError(warmup_bins)
    if minimum_bins_per_repeat < 1:
        raise ValueError(minimum_bins_per_repeat)
    patterns = contact_pattern(contacts)[:, warmup_bins:]
    speeds = np.unique(target_speed)
    groups = [np.flatnonzero(np.isclose(target_speed, speed)) for speed in speeds]
    repeat_counts = {len(group) for group in groups}
    if len(repeat_counts) != 1 or next(iter(repeat_counts)) < 4:
        raise ValueError(f"unbalanced or insufficient repeats: {repeat_counts}")
    condition_speed: list[float] = []
    condition_pattern: list[int] = []
    episode_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    count_rows: list[np.ndarray] = []
    for speed, group in zip(speeds, groups, strict=True):
        for pattern in range(CONTACT_PATTERNS):
            masks = patterns[group] == pattern
            counts = masks.sum(axis=1)
            if np.all(counts >= minimum_bins_per_repeat):
                condition_speed.append(float(speed))
                condition_pattern.append(pattern)
                episode_rows.append(group)
                mask_rows.append(masks)
                count_rows.append(counts)
    if len(condition_speed) < 3:
        raise ValueError("fewer than three estimable speed/contact conditions")
    # Lists are condition-major; public arrays are repeat-major.
    return ConditionDesign(
        speed=np.asarray(condition_speed, np.float64),
        contact_pattern=np.asarray(condition_pattern, np.int16),
        episode_index=np.stack(episode_rows, axis=1).astype(np.int32),
        time_mask=np.stack(mask_rows, axis=1).astype(bool),
        sample_count=np.stack(count_rows, axis=1).astype(np.int32),
        warmup_bins=int(warmup_bins),
    )


def condition_repeat_means(
    values: np.ndarray,
    design: ConditionDesign,
) -> np.ndarray:
    """Average population vectors within each condition and repeat."""

    values = np.asarray(values, np.float64)
    if values.ndim != 3:
        raise ValueError(values.shape)
    stop = design.warmup_bins + design.bins
    if values.shape[1] < stop:
        raise ValueError((values.shape, stop))
    values = values[:, design.warmup_bins : stop]
    means = np.empty((design.repeats, design.conditions, values.shape[-1]))
    for repeat in range(design.repeats):
        for condition in range(design.conditions):
            episode = design.episode_index[repeat, condition]
            mask = design.time_mask[repeat, condition]
            means[repeat, condition] = values[episode, mask].mean(axis=0)
    if not np.all(np.isfinite(means)):
        raise ValueError("non-finite condition mean")
    return means


def _noise_scale(repeat_mean: np.ndarray) -> np.ndarray:
    residual = repeat_mean - repeat_mean.mean(axis=0, keepdims=True)
    variance = np.sum(np.square(residual), axis=(0, 1))
    variance /= repeat_mean.shape[1] * max(repeat_mean.shape[0] - 1, 1)
    positive = variance[variance > 0]
    floor = max(float(np.median(positive)) * 1e-3, 1e-12) if len(positive) else 1.0
    return np.sqrt(np.maximum(variance, floor))


def _repeat_splits(repeats: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return unique disjoint 2-vs-2 partitions (15 when repeats is five)."""

    if repeats < 4:
        raise ValueError("cross-validation needs at least four repeats")
    indices = tuple(range(repeats))
    partitions = []
    seen = set()
    for left_tuple in combinations(indices, 2):
        remaining = tuple(index for index in indices if index not in left_tuple)
        for right_tuple in combinations(remaining, 2):
            canonical = tuple(sorted((tuple(left_tuple), tuple(right_tuple))))
            if canonical in seen:
                continue
            seen.add(canonical)
            partitions.append(
                (
                    np.asarray(canonical[0], np.int32),
                    np.asarray(canonical[1], np.int32),
                )
            )
    return tuple(partitions)


def _cross_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Crossvalidated squared Euclidean distances between condition patterns."""

    left_gram = left @ right.T
    right_gram = right @ left.T
    diagonal = np.diag(left_gram)
    # Average both orientations to make numerical symmetry explicit.
    distance = diagonal[:, None] + diagonal[None, :] - left_gram - right_gram
    distance /= left.shape[-1]
    np.fill_diagonal(distance, 0.0)
    return distance


def correlation_rsm(pattern: np.ndarray) -> np.ndarray:
    """Pearson correlation RSM over condition population patterns."""

    pattern = np.asarray(pattern, np.float64)
    centered = pattern - pattern.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norm, 1e-12)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def representational_geometry(repeat_mean: np.ndarray) -> Geometry:
    """Estimate descriptive RSM and diagonally whitened crossnobis-like RDM."""

    repeat_mean = np.asarray(repeat_mean, np.float64)
    if repeat_mean.ndim != 3 or repeat_mean.shape[1] < 3:
        raise ValueError(repeat_mean.shape)
    scale = _noise_scale(repeat_mean)
    normalized = repeat_mean / scale
    split_rdm = []
    for left, right in _repeat_splits(len(normalized)):
        split_rdm.append(
            _cross_distance(
                normalized[left].mean(axis=0), normalized[right].mean(axis=0)
            )
        )
    split_rdm = np.stack(split_rdm)
    rsm = correlation_rsm(repeat_mean.mean(axis=0))
    return Geometry(
        repeat_mean=repeat_mean,
        rsm=rsm,
        correlation_rdm=1.0 - rsm,
        crossvalidated_rdm=split_rdm.mean(axis=0),
        split_rdm=split_rdm,
        noise_scale=scale,
    )


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(matrix.shape)
    return matrix[np.triu_indices(len(matrix), 1)]


def spearman_rsa(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation between the unique entries of two RDMs."""

    left_vector = upper_triangle(left)
    right_vector = upper_triangle(right)
    value = spearmanr(left_vector, right_vector).statistic
    return float(value)


def partial_spearman_rsa(
    left: np.ndarray,
    right: np.ndarray,
    nuisance: np.ndarray,
) -> float:
    """Rank RSA after linearly partialling one behavior RDM from both sides."""

    left_rank = rankdata(upper_triangle(left))
    right_rank = rankdata(upper_triangle(right))
    nuisance_rank = rankdata(upper_triangle(nuisance))
    design = np.column_stack((np.ones_like(nuisance_rank), nuisance_rank))
    left_residual = (
        left_rank - design @ np.linalg.lstsq(design, left_rank, rcond=None)[0]
    )
    right_residual = (
        right_rank - design @ np.linalg.lstsq(design, right_rank, rcond=None)[0]
    )
    denominator = np.linalg.norm(left_residual) * np.linalg.norm(right_residual)
    return float(left_residual @ right_residual / max(denominator, 1e-12))


def permutation_control(
    left: np.ndarray,
    right: np.ndarray,
    *,
    permutations: int = 1_000,
    seed: int = 0,
) -> dict[str, float]:
    """Condition-label permutation control for a positive-direction RSA score."""

    if permutations < 1:
        raise ValueError(permutations)
    observed = spearman_rsa(left, right)
    random = np.random.default_rng(seed)
    null = np.empty(permutations)
    for index in range(permutations):
        order = random.permutation(len(right))
        null[index] = spearman_rsa(left, right[np.ix_(order, order)])
    return {
        "observed": observed,
        "null_median": float(np.median(null)),
        "null_95_percentile": float(np.quantile(null, 0.95)),
        "one_sided_p": float((1 + np.sum(null >= observed)) / (permutations + 1)),
    }
