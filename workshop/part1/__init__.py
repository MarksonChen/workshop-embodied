from .environment import FetchRun


def train(*args, **kwargs):
    from .train import train as implementation

    return implementation(*args, **kwargs)


def render(*args, **kwargs):
    from .visualize import render as implementation

    return implementation(*args, **kwargs)


__all__ = ["FetchRun", "render", "train"]
