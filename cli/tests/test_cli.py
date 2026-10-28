from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Annotated

import pytest
import typer
from typer.testing import CliRunner

import tdp_cli
from tdp_cli.main import build_app
from tdp_cli.registry import Command, CommandRegistry, load_registry

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_version() -> None:
    result = runner.invoke(build_app(), ["--version"])
    assert result.exit_code == 0
    assert result.stdout == f"entropiq {tdp_cli.__version__}\n"


def test_help_lists_subcommands() -> None:
    result = runner.invoke(build_app(), ["--help"])
    assert result.exit_code == 0
    text = ANSI.sub("", result.stdout)  # Typer forces colour on GitHub Actions
    for name in ("analyze", "history", "model", "--dev-model", "--model-dir"):
        assert name in text


def test_analyze_formats_and_reuse(dev_home: Path, sample_repo: Path) -> None:
    app = build_app()
    first = runner.invoke(app, ["analyze", str(sample_repo), "--dev-model", "--format", "json"])
    assert first.exit_code == 0, first.stderr
    data = json.loads(first.stdout)
    assert data["schema"] == "entropiq.analysis.v1"
    assert data["sha"] == _head(sample_repo)
    assert data["demo_model"] is True
    assert data["summary"]["files"] == len(data["files"]) > 5
    assert "[ 25%] Mined" in first.stderr
    assert "Demo model in use" in first.stderr

    md = runner.invoke(app, ["--dev-model", "analyze", str(sample_repo), "-f", "md", "--top", "2"])
    assert md.exit_code == 0, md.stderr
    assert "Using the saved analysis" in md.stderr
    assert md.stdout.startswith("## Entropiq: sample at ")
    assert "| # | File | Priority | Risk | Cost | Top reason |" in md.stdout
    assert f"Showing 2 of {data['summary']['files']} files." in md.stdout

    table = runner.invoke(app, ["analyze", str(sample_repo), "--dev-model", "--top", "3"])
    assert table.exit_code == 0, table.stderr
    assert f"Showing 3 of {data['summary']['files']} files." in table.stdout
    assert "Estimated Critical debt" in table.stdout

    history = runner.invoke(app, ["history", "--format", "json"])
    assert history.exit_code == 0, history.stderr
    entries = json.loads(history.stdout)
    assert len(entries) == 1 and entries[0]["number"] == 1 and entries[0]["sha"] == data["sha"]
    listed = runner.invoke(app, ["history"])
    assert "sample" in listed.stdout


def test_history_empty() -> None:
    result = runner.invoke(build_app(), ["history"])
    assert result.exit_code == 0
    assert "No saved analyses yet" in result.stdout


def test_model_json(dev_home: Path) -> None:
    result = runner.invoke(build_app(), ["model", "--dev-model", "--format", "json"])
    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert (data["n_features"], data["demo"], data["source"]) == (89, True, "dev")
    table = runner.invoke(build_app(), ["model", "--dev-model"])
    assert "89 inputs" in table.stdout


def test_errors_exit_with_codes(dev_home: Path, tmp_path: Path, sample_repo: Path) -> None:
    app = build_app()
    plain = tmp_path / "plain"
    plain.mkdir()
    cases = [
        (["analyze", str(plain), "--dev-model"], 2, "is not a git repository"),
        (["analyze", "https://example.com/a/b", "--dev-model"], 2, "Cannot analyse this URL"),
        (["analyze", str(sample_repo), "--model-dir", str(tmp_path / "none")], 2, "is not a model bundle"),
        (["analyze", str(sample_repo), "--dev-model", "--model-dir", str(tmp_path)], 2, "either --model-dir"),
        (["analyze", str(sample_repo), "--dev-model", "--ref", "nope"], 2, "does not name a commit"),
        (["--model-dir", str(tmp_path / "none")], 2, "is not a model bundle"),
    ]
    for args, code, message in cases:
        result = runner.invoke(app, args)
        assert result.exit_code == code, (args, result.stdout, result.stderr)
        assert message in result.stderr, args
    bad_format = runner.invoke(app, ["analyze", str(sample_repo), "--format", "xml"])
    assert bad_format.exit_code == 2


def test_no_subcommand_opens_the_shell(dev_home: Path) -> None:
    result = runner.invoke(build_app(), ["--dev-model"], input="/status\n/quit\n")
    assert result.exit_code == 0, result.stderr
    assert "Rank technical debt by future maintenance pain" in result.stdout
    assert "Session model" in result.stdout
    assert "Bye." in result.stdout


def test_plugins_add_subcommands() -> None:
    registry = load_registry()

    def hello(name: Annotated[str, typer.Argument()] = "world") -> None:
        """Say hello."""
        typer.echo(f"hello {name}")

    registry.add(
        Command(name="hello", help="Say hello", handler=lambda s, a: None, group="Report", cli=hello)
    )
    registry.add_cli("ping", lambda: typer.echo("pong"), help="Answer pong")
    app = build_app(registry)
    assert runner.invoke(app, ["hello", "you"]).stdout == "hello you\n"
    assert runner.invoke(app, ["ping"]).stdout == "pong\n"
    assert "hello" in ANSI.sub("", runner.invoke(app, ["--help"]).stdout)


def test_custom_registry_without_plugins() -> None:
    app = build_app(CommandRegistry())
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code == 2


@pytest.mark.network
def test_analyze_a_github_url(dev_home: Path) -> None:
    result = runner.invoke(
        build_app(),
        ["analyze", "https://github.com/pallets/click", "--dev-model", "--format", "json", "--top", "3"],
    )
    assert result.exit_code == 0, result.stderr
    assert len(json.loads(result.stdout)["files"]) == 3
