from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tdp_cli.errors import CliError, UsageError
from tdp_cli.home import EntropiqHome, Prefs, load_prefs, save_prefs, set_pref
from tdp_cli.model import (
    ModelChoice,
    clear_model_cache,
    find_repo_root,
    load_model,
    read_model_info,
    resolve_model,
)


def test_home_from_env(tmp_path: Path) -> None:
    home = EntropiqHome.from_env({"ENTROPIQ_HOME": str(tmp_path / "h")})
    assert home.root == (tmp_path / "h").resolve()
    assert home.history_file == home.root / "history"
    assert home.settings_file.name == "settings.json"
    assert home.dev_model == home.root / "models" / "dev"
    assert (
        home.mirror_path("github.com", "psf", "requests") == home.root / "mirrors/github.com/psf/requests.git"
    )
    assert home.szz_cache("x") == home.root / "cache" / "szz" / "x"
    assert home.analyses == home.root / "analyses"
    default = EntropiqHome.from_env({})
    assert default.root == (Path.home() / ".entropiq").resolve()
    home.ensure()
    assert home.root.is_dir()


def test_home_ensure_fails_cleanly(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(CliError, match="Cannot create"):
        EntropiqHome(blocker / "home").ensure()


def test_prefs_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    assert load_prefs(path) == Prefs()
    prefs = set_pref(Prefs(), "budget-hours", "12.5")
    prefs = set_pref(prefs, "TOP", "7")
    save_prefs(path, prefs)
    assert load_prefs(path) == Prefs(budget_hours=12.5, top=7, jobs=4)
    assert set_pref(Prefs(), "budget_hours", "0").budget_hours == 0


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("colour", "1", "unknown setting"),
        ("top", "0", "between 1 and 500"),
        ("top", "1.5", "between 1 and 500"),
        ("jobs", "abc", "between 1 and 64"),
        ("budget_hours", "nan", "finite"),
        ("budget_hours", "3000", "between 0 and 2000"),
    ],
)
def test_set_pref_rejects(key: str, value: str, message: str) -> None:
    with pytest.raises(UsageError, match=message):
        set_pref(Prefs(), key, value)


def test_prefs_ignore_bad_values(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"top": 0, "jobs": 8, "budget_hours": "x", "extra": 1}), encoding="utf-8")
    assert load_prefs(path) == Prefs(jobs=8)
    path.write_text("{not json", encoding="utf-8")
    assert load_prefs(path) == Prefs()
    path.write_text("[1]", encoding="utf-8")
    assert load_prefs(path) == Prefs()


def _bundle(folder: Path, version: str, *, synthetic: bool = False) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"version": version, "metrics": {"ndcg": {"at10": 0.5}, "flag": True}, "training_rows": 10}
    if synthetic:
        meta["synthetic"] = True
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    columns = [f"c{i}" for i in range(5)]
    schema = {"extractor_version": "1", "columns": columns}
    (folder / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return folder


def _repo(tmp_path: Path, production: object) -> Path:
    root = tmp_path / "repo"
    (root / "models").mkdir(parents=True, exist_ok=True)
    text = production if isinstance(production, str) else json.dumps(production)
    (root / "models" / "production.json").write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def home(tmp_path: Path) -> EntropiqHome:
    return EntropiqHome(tmp_path / "home")


def test_dev_flag_and_conflict(home: EntropiqHome, tmp_path: Path) -> None:
    choice = resolve_model(dev=True, home=home, environ={}, repo_root=None)
    assert (choice.source, choice.demo, choice.version, choice.path) == ("dev", True, None, home.dev_model)
    assert choice.kind_label == "demo model"
    assert choice.notice is not None and "--dev-model" in choice.notice
    with pytest.raises(UsageError, match="either --model-dir or --dev-model"):
        resolve_model(tmp_path, dev=True, home=home, environ={}, repo_root=None)


def test_option_beats_env_beats_production(home: EntropiqHome, tmp_path: Path) -> None:
    option = _bundle(tmp_path / "opt", "ranker-opt")
    env = _bundle(tmp_path / "env", "ranker-env")
    prod = _bundle(tmp_path / "prod", "ranker-2026.10.1")
    root = _repo(tmp_path, {"version": "ranker-2026.10.1", "uri": str(prod), "sha256": None})
    environ = {"TDP_MODEL_DIR": str(env)}
    first = resolve_model(option, home=home, environ=environ, repo_root=root)
    assert (first.source, first.version, first.kind_label, first.notice) == (
        "option",
        "ranker-opt",
        "custom model",
        None,
    )
    second = resolve_model(home=home, environ=environ, repo_root=root)
    assert (second.source, second.version) == ("env", "ranker-env")
    third = resolve_model(home=home, environ={}, repo_root=root)
    assert (third.source, third.version, third.kind_label) == (
        "production",
        "ranker-2026.10.1",
        "production model",
    )
    assert third.path == prod.resolve()


def test_production_uri_forms(home: EntropiqHome, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    bundle = _bundle(root / "bundles" / "v1", "v1")
    for uri in ("bundles/v1", bundle.as_uri(), str(bundle)):
        _repo(tmp_path, {"version": "v1", "uri": uri})
        choice = resolve_model(home=home, environ={}, repo_root=root)
        assert (choice.source, choice.path) == ("production", bundle.resolve()), uri


def test_production_folder_lookup(home: EntropiqHome, tmp_path: Path) -> None:
    root = _repo(tmp_path, {"version": "v2", "uri": "https://example.com/v2.tar.gz"})
    models_dir = tmp_path / "shared"
    shared = _bundle(models_dir / "v2", "v2")
    choice = resolve_model(home=home, environ={"TDP_MODELS_DIR": str(models_dir)}, repo_root=root)
    assert choice.path == shared.resolve()
    in_home = _bundle(home.models / "v2", "v2")
    assert resolve_model(home=home, environ={}, repo_root=root).path == in_home.resolve()
    shutil.rmtree(in_home)
    in_repo = _bundle(root / "models" / "v2", "v2", synthetic=True)
    choice = resolve_model(home=home, environ={}, repo_root=root)
    assert (choice.path, choice.demo, choice.kind_label) == (in_repo.resolve(), True, "demo model")


def test_production_version_mismatch_and_fallbacks(home: EntropiqHome, tmp_path: Path) -> None:
    root = _repo(tmp_path, {"version": "v3", "uri": None})
    _bundle(home.models / "v3", "something-else")
    choice = resolve_model(home=home, environ={}, repo_root=root)
    assert (choice.source, choice.reason) == ("dev", "production model v3 is not on this machine")
    _repo(tmp_path, {"version": None, "uri": None, "sha256": None})
    assert (
        resolve_model(home=home, environ={}, repo_root=root).reason
        == "no production model has been published yet"
    )
    _repo(tmp_path, "{broken")
    assert (
        resolve_model(home=home, environ={}, repo_root=root).reason == "models/production.json is unreadable"
    )
    assert resolve_model(home=home, environ={}, repo_root=None).reason == "no models/production.json found"


def test_explicit_folder_must_be_a_bundle(home: EntropiqHome, tmp_path: Path) -> None:
    with pytest.raises(CliError, match=r"metadata\.json is missing") as info:
        resolve_model(tmp_path / "missing", home=home, environ={}, repo_root=None)
    assert info.value.exit_code == 2
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CliError, match="no version"):
        resolve_model(home=home, environ={"TDP_MODEL_DIR": str(bad)}, repo_root=None)
    synthetic = _bundle(tmp_path / "syn", "dev-1", synthetic=True)
    choice = resolve_model(synthetic, home=home, environ={}, repo_root=None)
    assert choice.demo and choice.kind_label == "demo model"


def test_find_repo_root_from_the_package() -> None:
    root = find_repo_root()
    assert root is not None
    assert (root / "models" / "production.json").is_file()
    assert (root / "cli" / "tdp_cli").is_dir()


def test_find_repo_root_none(tmp_path: Path) -> None:
    assert find_repo_root(tmp_path) is None


def test_read_model_info_flattens_metrics(tmp_path: Path) -> None:
    folder = _bundle(tmp_path / "b", "v9")
    info = read_model_info(ModelChoice(folder, "option", "v9", False, "--model-dir"))
    assert info.metrics == {"ndcg.at10": 0.5}
    assert (info.n_features, info.training_rows, info.extractor_version) == (5, 10, "1")
    with pytest.raises(CliError):
        read_model_info(ModelChoice(tmp_path / "none", "option", "x", False, "x"))


def test_load_model_builds_the_demo_bundle_once(home: EntropiqHome) -> None:
    choice = resolve_model(dev=True, home=home, environ={}, repo_root=None)
    bundle = load_model(choice)
    assert bundle.metadata.get("synthetic") is True
    assert (home.dev_model / "metadata.json").is_file()
    assert load_model(choice) is bundle
    info = read_model_info(resolve_model(dev=True, home=home, environ={}, repo_root=None))
    assert (info.n_features, info.demo, info.version) == (89, True, bundle.version)

    clear_model_cache()
    (home.dev_model / "metadata.json").write_text("{corrupt", encoding="utf-8")
    rebuilt = load_model(choice)
    assert rebuilt.metadata.get("synthetic") is True
    assert json.loads((home.dev_model / "metadata.json").read_text(encoding="utf-8"))["synthetic"] is True


def test_load_model_refuses_foreign_files(home: EntropiqHome) -> None:
    home.dev_model.mkdir(parents=True)
    (home.dev_model / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(CliError, match="holds other files"):
        load_model(resolve_model(dev=True, home=home, environ={}, repo_root=None))


def test_load_model_bad_explicit_bundle(tmp_path: Path) -> None:
    folder = _bundle(tmp_path / "b", "v1")
    with pytest.raises(CliError, match="Cannot load the model") as info:
        load_model(ModelChoice(folder, "option", "v1", False, "--model-dir"))
    assert info.value.exit_code == 2
