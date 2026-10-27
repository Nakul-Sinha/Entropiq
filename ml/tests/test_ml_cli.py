from typer.testing import CliRunner

from tdp_ml.cli import app, load_commands


def test_cli_loads_and_shows_help() -> None:
    load_commands()
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Entropiq offline pipeline" in result.output
