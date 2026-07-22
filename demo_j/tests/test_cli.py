from __future__ import annotations

from types import SimpleNamespace

from demo_j import cli


def test_cli_exposes_only_supported_workflows() -> None:
    assert set(cli.COMMANDS) == {
        "smoke",
        "build-cache",
        "train-imitation",
        "evaluate-imitation",
        "render-imitation",
        "fit-tokenizer",
        "train-aligned",
        "train-ppo",
        "evaluate-aligned",
        "record-aligned",
        "render-aligned",
        "export-h-trace",
        "export-h-activations",
        "compare-rsa",
        "plot-rsa",
    }


def test_cli_lazily_dispatches_with_internal_prefix(monkeypatch) -> None:
    invoked = []

    def import_module(name):
        invoked.append(name)
        return SimpleNamespace(main=lambda: invoked.append(tuple(cli.sys.argv)))

    monkeypatch.setattr(cli.importlib, "import_module", import_module)
    cli.main(["fit-tokenizer", "--output", "tokenizer.npz"])
    assert invoked == [
        "demo_j.experiments.aligned",
        ("demo-j fit-tokenizer", "fit-tokenizer", "--output", "tokenizer.npz"),
    ]
