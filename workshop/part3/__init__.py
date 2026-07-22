__all__ = [
    "BodyActionPrior",
    "BodyActionSet",
    "evaluate_prior",
    "load_manifest",
    "load_prior",
    "load_split",
    "render_sweeps",
    "rollout_speeds",
]


def __getattr__(name):
    if name in {"BodyActionSet", "load_manifest", "load_split"}:
        from . import data

        return getattr(data, name)
    if name in {"BodyActionPrior", "load_prior"}:
        from .core import prior

        return getattr(prior, name)
    if name == "evaluate_prior":
        from .evaluate_prior import evaluate

        return evaluate
    if name == "rollout_speeds":
        from .visualize import rollout_speeds

        return rollout_speeds
    if name == "render_sweeps":
        from .render import render_sweeps

        return render_sweeps
    raise AttributeError(name)
