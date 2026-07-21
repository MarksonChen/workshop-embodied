"""Notebook-facing Demo H data/prior API with lazy pinned-Brax operations."""

from pathlib import Path

from .dataset import BodyActionSet, load_manifest, load_split
from .evaluate_prior import evaluate as evaluate_prior
from .prior import DemoHPrior, load_prior


def rollout_speeds(*args, **kwargs):
    from .visualize import rollout_speeds as implementation

    return implementation(*args, **kwargs)


def render_sweeps(metrics_paths, output: Path, **kwargs) -> None:
    from .render_speed_comparison import render_sweeps as implementation

    implementation(metrics_paths, output, **kwargs)


__all__ = [
    "BodyActionSet",
    "DemoHPrior",
    "evaluate_prior",
    "load_manifest",
    "load_prior",
    "load_split",
    "render_sweeps",
    "rollout_speeds",
]
