from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tdp_cli.args import parse_args
from tdp_cli.commands import whatif
from tdp_cli.home import EntropiqHome
from tdp_cli.main import build_app
from tdp_cli.session import Session
from tdp_cli.shell import dispatch
from tdp_cli.store import AnalysisStore, LoadedAnalysis

runner = CliRunner()


@pytest.fixture
def saved_home(dev_home: Path, sample_loaded: LoadedAnalysis) -> Path:
    """The test home with ``sample_loaded`` saved, so ``tdp whatif ... <sample_repo>`` reuses it."""
    AnalysisStore(EntropiqHome.from_env().analyses).save(sample_loaded)
    return dev_home


def _out(session: Session) -> str:
    return session.console.file.getvalue()  # type: ignore[attr-defined]


def test_tables_cover_every_action_and_column() -> None:
    from tdp_core.contracts import WHATIF_ACTIONS
    from tdp_core.planning.whatif import ACTION_COLUMNS

    assert tuple(whatif.ACTION_FLAGS.values()) == WHATIF_ACTIONS
    assert set(whatif.ACTION_LABELS) == set(WHATIF_ACTIONS) == set(whatif.ACTION_HINTS)
    columns = {c for cols in ACTION_COLUMNS.values() for c in cols}
    assert columns <= set(whatif.FEATURE_LABELS)


def test_actions_from_flags_keep_canonical_order() -> None:
    flags = (*whatif.ACTION_FLAGS, whatif.ALL_FLAG)
    parsed = parse_args(["--dedupe", "--add-tests"], flags=flags)
    assert whatif.actions_from_flags(parsed) == ["add_tests", "remove_duplication"]
    assert whatif.actions_from_flags(parse_args(["--all"], flags=flags)) == list(whatif.ACTION_FLAGS.values())
    assert whatif.selected_actions(break_cycle=True, split=True) == ["split_longest_function", "break_cycle"]
    assert whatif.selected_actions(all_actions=True) == list(whatif.ACTION_FLAGS.values())
    assert whatif.selected_actions() == []


def test_format_feature_value() -> None:
    assert whatif.format_feature_value(3.0) == "3"
    assert whatif.format_feature_value(0.5) == "0.50"
    assert whatif.format_feature_value(12.34) == "12.3"
    assert whatif.format_feature_value(123.4) == "123"
    assert whatif.format_feature_value(math.nan) == "nan"
    assert whatif.format_feature_value(math.inf) == "inf"


def test_whatif_add_tests(loaded_session: Session, sample_loaded: LoadedAnalysis) -> None:
    from tdp_core.planning.whatif import what_if

    dispatch(loaded_session, "/whatif pkg/mu.py --add-tests")
    text = _out(loaded_session)
    assert "What if: pkg/mu.py" in text
    assert "Actions: Add tests" in text
    assert "has_test_file" in text
    expected = what_if(sample_loaded.result.features, "pkg/mu.py", ["add_tests"], loaded_session.bundle())
    before, after = expected.before, expected.after
    assert f"{before[1]} {before[0]}" in text
    assert f"{after[1]} {after[0]}" in text
    assert whatif.WHATIF_CAPTION in text


def test_whatif_by_rank(loaded_session: Session) -> None:
    dispatch(loaded_session, "/whatif 1 --split")
    text = _out(loaded_session)
    assert "What if: pkg/mu.py" in text
    assert "Split the longest function" in text


def test_whatif_without_change(loaded_session: Session, sample_loaded: LoadedAnalysis) -> None:
    from tdp_core.planning.whatif import what_if

    bundle = loaded_session.bundle()
    features = sample_loaded.result.features
    idle = [
        (flag, action)
        for flag, action in whatif.ACTION_FLAGS.items()
        if not what_if(features, "pkg/mu.py", [action], bundle).changed_features
    ]
    assert idle, "expected an action that changes nothing for pkg/mu.py"
    dispatch(loaded_session, f"/whatif pkg/mu.py {idle[0][0]}")
    text = _out(loaded_session)
    assert whatif.NO_CHANGE in text
    assert "no change" in text


def test_whatif_usage_errors(loaded_session: Session) -> None:
    dispatch(loaded_session, "/whatif")
    text = _out(loaded_session)
    assert "say which file to simulate" in text
    assert "Usage: /whatif <path>" in text
    dispatch(loaded_session, "/whatif pkg/mu.py")
    text = _out(loaded_session)
    assert whatif.NO_ACTION in text
    assert whatif.NO_ACTION_HINT in text
    dispatch(loaded_session, "/whatif pkg/nothing.py --add-tests")
    assert "pkg/nothing.py is not a ranked file" in _out(loaded_session)


@pytest.mark.parametrize(
    ("picked", "expected"),
    [(["add_tests"], "Actions: Add tests"), (None, "Cancelled."), ([], "No action chosen")],
)
def test_whatif_picker(
    loaded_session: Session, monkeypatch: pytest.MonkeyPatch, picked: list[str] | None, expected: str
) -> None:
    seen: list[str] = []

    def fake_pick(session: Session, path: str) -> list[str] | None:
        seen.append(path)
        return picked

    monkeypatch.setattr(whatif, "can_pick", lambda session: True)
    monkeypatch.setattr(whatif, "_pick_actions", fake_pick)
    dispatch(loaded_session, "/what-if mu.py")
    assert seen == ["pkg/mu.py"]
    assert expected in _out(loaded_session)


def test_can_pick_is_false_for_a_recording_console(loaded_session: Session) -> None:
    assert whatif.can_pick(loaded_session) is False


def test_model_version_notice(loaded_session: Session, sample_loaded: LoadedAnalysis) -> None:
    other = LoadedAnalysis(
        dataclasses.replace(sample_loaded.meta, model_version="other"), sample_loaded.result
    )
    loaded_session.current = other
    dispatch(loaded_session, "/whatif pkg/lam.py --dedupe")
    text = _out(loaded_session)
    assert "This analysis used model other" in text
    assert "What if: pkg/lam.py" in text


def test_whatif_json(sample_loaded: LoadedAnalysis, loaded_session: Session) -> None:
    result = whatif.simulate(loaded_session, sample_loaded, "pkg/mu.py", ["add_tests"])
    data = whatif.whatif_json(sample_loaded, "pkg/mu.py", ["add_tests"], result, "v1")
    assert data["schema"] == "entropiq.whatif.v1"
    assert data["actions"] == ["add_tests"]
    json.dumps(data)


def test_whatif_cli(saved_home: Path, sample_repo: Path) -> None:
    app = build_app()
    args = ["whatif", "pkg/mu.py", str(sample_repo), "--dev-model"]
    as_json = runner.invoke(app, [*args, "--add-tests", "--format", "json"])
    assert as_json.exit_code == 0, as_json.stderr
    data = json.loads(as_json.stdout)
    assert data["schema"] == "entropiq.whatif.v1"
    assert data["path"] == "pkg/mu.py"
    assert {"before", "after", "changed_features"} <= set(data)
    assert "has_test_file" in data["changed_features"]

    table = runner.invoke(app, [*args, "--all"])
    assert table.exit_code == 0, table.stderr
    assert "What if: pkg/mu.py" in table.stdout

    none = runner.invoke(app, args)
    assert none.exit_code == 2
    assert whatif.NO_ACTION in none.stderr

    unknown = runner.invoke(app, ["whatif", "nope.py", str(sample_repo), "--dev-model", "--split"])
    assert unknown.exit_code == 2
    assert "nope.py is not a ranked file" in unknown.stderr
