"""Stable notebook-facing API for the supported Demo J workflows."""

from demo_j.control.aligned import (
    MotionTokenizer,
    build_clip_sequences,
    clip_observations,
    fit_tokenizer,
    load_tokenizer,
)
from demo_j.experiments.aligned import load_aligned_checkpoint
from demo_j.experiments.aligned_rollout import (
    evaluate as evaluate_aligned,
    record as record_aligned_spikes,
)
from demo_j.analysis.compare import compare as compare_rsa
from demo_j.experiments.evaluate_imitation import evaluate as evaluate_imitation
from demo_j.data.projection import build as build_reference_cache
from demo_j.experiments.render import render_comparison as render_imitation
from demo_j.experiments.render_aligned import render as render_aligned

__all__ = [
    "MotionTokenizer",
    "build_clip_sequences",
    "build_reference_cache",
    "clip_observations",
    "compare_rsa",
    "evaluate_aligned",
    "evaluate_imitation",
    "fit_tokenizer",
    "load_aligned_checkpoint",
    "load_tokenizer",
    "record_aligned_spikes",
    "render_aligned",
    "render_imitation",
]
