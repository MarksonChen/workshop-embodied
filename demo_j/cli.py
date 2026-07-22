"""One discoverable command entry point for every supported Demo J workflow."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    module: str
    description: str
    prefix: tuple[str, ...] = ()


COMMANDS = {
    "smoke": Command("demo_j.experiments.smoke", "verify the BrainPy/JAX/SNN runtime"),
    "build-cache": Command(
        "demo_j.data.projection", "replay the accepted controls in modern MJX"
    ),
    "train-imitation": Command(
        "demo_j.experiments.train_imitation", "train the short-clip SNN controller"
    ),
    "evaluate-imitation": Command(
        "demo_j.experiments.evaluate_imitation",
        "evaluate short-clip imitation on a held-out split",
    ),
    "render-imitation": Command(
        "demo_j.experiments.render", "render the short-clip comparison video"
    ),
    "fit-tokenizer": Command(
        "demo_j.experiments.aligned",
        "fit the aligned train-only PCA tokenizer",
        ("fit-tokenizer",),
    ),
    "train-aligned": Command(
        "demo_j.experiments.aligned",
        "train the finite native-clip recurrent SNN",
        ("train",),
    ),
    "evaluate-aligned": Command(
        "demo_j.experiments.aligned_rollout",
        "audit all test clips and retain six examples",
        ("evaluate",),
    ),
    "record-aligned": Command(
        "demo_j.experiments.aligned_rollout",
        "record aligned spikes on a fixed Demo H input bank",
        ("record",),
    ),
    "render-aligned": Command(
        "demo_j.experiments.render_aligned",
        "render native targets and SNN rollouts",
    ),
    "export-h-trace": Command(
        "demo_j.analysis.bridge",
        "export the fixed Demo H trajectory bank (legacy runtime)",
        ("trace",),
    ),
    "export-h-activations": Command(
        "demo_j.analysis.bridge",
        "export Demo H activations on that bank (legacy runtime)",
        ("activations",),
    ),
    "compare-rsa": Command(
        "demo_j.analysis.compare", "run the crossed-seed fixed-input RSM/RSA"
    ),
    "plot-rsa": Command(
        "demo_j.analysis.plot", "plot the final RSA curves and example RSMs"
    ),
}


def _print_help() -> None:
    print("usage: python -m demo_j.cli <command> [arguments]\n")
    print("commands:")
    width = max(map(len, COMMANDS))
    for name, command in COMMANDS.items():
        print(f"  {name:<{width}}  {command.description}")
    print("\nRun a command with --help to see its arguments.")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return
    name, *arguments = argv
    try:
        command = COMMANDS[name]
    except KeyError:
        _print_help()
        raise SystemExit(f"\nunknown Demo J command: {name}") from None
    module = importlib.import_module(command.module)
    sys.argv = [f"demo-j {name}", *command.prefix, *arguments]
    module.main()


if __name__ == "__main__":
    main()
