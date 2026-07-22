__all__ = [
    "FetchMotionSet",
    "MotionPrior",
    "evaluate_checkpoint",
    "generate_rollouts",
    "load_manifest",
    "load_prior",
    "load_split",
]


def __getattr__(name):
    if name in {"FetchMotionSet", "load_manifest", "load_split"}:
        from . import data

        return getattr(data, name)
    if name in {"MotionPrior", "load_prior"}:
        from .core import model

        return getattr(model, name)
    if name == "evaluate_checkpoint":
        from .evaluate import evaluate_checkpoint

        return evaluate_checkpoint
    if name == "generate_rollouts":
        from .generate import generate_rollouts

        return generate_rollouts
    raise AttributeError(name)
