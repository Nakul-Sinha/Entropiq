from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from tdp_platform.settings import (
    DEFAULT_ALLOWED_GIT_HOSTS,
    DEFAULT_DATABASE_URL,
    DEFAULT_MAX_REPO_KB,
    DEFAULT_REDIS_URL,
    PlatformSettings,
    get_settings,
)


def _settings(**kwargs: Any) -> PlatformSettings:
    """Build settings; keyword arguments are untyped here so tests can pass raw or invalid values."""
    return PlatformSettings(**kwargs)


@pytest.fixture
def isolated(clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """No settings variables and a working directory without a ``.env`` file."""
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_cache() -> Iterator[None]:
    yield
    get_settings.cache_clear()


def test_defaults(isolated: Path) -> None:
    s = _settings(_env_file=None)
    assert s.database_url == DEFAULT_DATABASE_URL
    assert s.database_schema is None
    assert s.redis_url == DEFAULT_REDIS_URL
    assert s.repo_cache_dir == isolated / "data" / "mirrors"
    assert s.model_dir == isolated / "models"
    assert s.github_token is None
    assert s.anthropic_api_key is None
    assert s.sentry_dsn is None
    assert s.allowed_git_hosts == DEFAULT_ALLOWED_GIT_HOSTS == ("github.com", "gitlab.com")
    assert s.max_repo_kb == DEFAULT_MAX_REPO_KB == 1_000_000
    assert s.advisor_enabled is False


def test_folders_follow_shared_data_variables(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TDP_DATA_DIR", str(isolated / "shared-data"))
    monkeypatch.setenv("TDP_MODELS_DIR", str(isolated / "shared-models"))
    s = _settings(_env_file=None)
    assert s.repo_cache_dir == isolated / "shared-data" / "mirrors"
    assert s.model_dir == isolated / "shared-models"


def test_every_variable_is_read(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "DATABASE_URL": "postgresql://u:p@db:5432/x",
        "DATABASE_SCHEMA": "tdp_ci",
        "REDIS_URL": "redis://cache:6379/1",
        "REPO_CACHE_DIR": str(isolated / "mirrors"),
        "MODEL_DIR": str(isolated / "bundles"),
        "GITHUB_TOKEN": "ghp_example",
        "ANTHROPIC_API_KEY": "sk-example",
        "SENTRY_DSN": "https://abc@sentry.example/1",
        "ALLOWED_GIT_HOSTS": " GitHub.com, ,gitlab.com,github.com ",
        "MAX_REPO_KB": "2048",
        "ADVISOR_ENABLED": "false",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    s = _settings(_env_file=None)
    assert s.database_url == "postgresql://u:p@db:5432/x"
    assert s.database_schema == "tdp_ci"
    assert s.redis_url == "redis://cache:6379/1"
    assert s.repo_cache_dir == isolated / "mirrors"
    assert s.model_dir == isolated / "bundles"
    assert s.github_token is not None and s.github_token.get_secret_value() == "ghp_example"
    assert s.anthropic_api_key is not None and s.anthropic_api_key.get_secret_value() == "sk-example"
    assert s.sentry_dsn is not None and s.sentry_dsn.get_secret_value() == "https://abc@sentry.example/1"
    assert s.allowed_git_hosts == ("github.com", "gitlab.com")
    assert s.max_repo_kb == 2048
    assert s.advisor_enabled is False


def test_empty_values_fall_back_to_defaults(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GITHUB_TOKEN", "ANTHROPIC_API_KEY", "ALLOWED_GIT_HOSTS", "MAX_REPO_KB", "ADVISOR_ENABLED"):
        monkeypatch.setenv(name, "")
    s = _settings(_env_file=None)
    assert s.github_token is None
    assert s.anthropic_api_key is None
    assert s.allowed_git_hosts == DEFAULT_ALLOWED_GIT_HOSTS
    assert s.max_repo_kb == DEFAULT_MAX_REPO_KB
    assert s.advisor_enabled is False


@pytest.mark.parametrize(
    ("key", "flag", "expected"),
    [
        ("sk-test", None, True),
        ("sk-test", "false", False),
        (None, "true", True),
        (None, None, False),
        ("", None, False),
    ],
)
def test_advisor_enabled_defaults_to_key_presence(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, key: str | None, flag: str | None, expected: bool
) -> None:
    if key is not None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    if flag is not None:
        monkeypatch.setenv("ADVISOR_ENABLED", flag)
    assert _settings(_env_file=None).advisor_enabled is expected


def test_advisor_enabled_from_keyword_arguments(isolated: Path) -> None:
    assert _settings(_env_file=None, anthropic_api_key=SecretStr("k")).advisor_enabled is True
    assert _settings(_env_file=None, anthropic_api_key="k", advisor_enabled=False).advisor_enabled is False


def test_hosts_accept_any_iterable(isolated: Path) -> None:
    s = _settings(_env_file=None, allowed_git_hosts=["Example.org", "example.org", " git.example.com "])
    assert s.allowed_git_hosts == ("example.org", "git.example.com")


def test_repr_hides_passwords_and_secrets(isolated: Path) -> None:
    s = _settings(
        _env_file=None,
        database_url="postgresql+asyncpg://tdp:db-password@db:5432/tdp",
        redis_url="redis://:redis-password@cache:6379/0",
        github_token="ghp-secret",
        anthropic_api_key="sk-ant-secret",
        sentry_dsn="https://dsn-secret@sentry.example/1",
    )
    for text in (repr(s), str(s)):
        for secret in ("db-password", "redis-password", "ghp-secret", "sk-ant-secret", "dsn-secret"):
            assert secret not in text


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MAX_REPO_KB", "0"),
        ("MAX_REPO_KB", "lots"),
        ("ALLOWED_GIT_HOSTS", " , "),
        ("DATABASE_SCHEMA", "Bad-Name"),
        ("DATABASE_SCHEMA", "1abc"),
        ("ADVISOR_ENABLED", "perhaps"),
    ],
)
def test_invalid_values_are_rejected(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        _settings(_env_file=None)


def test_invalid_host_list_types_are_rejected(isolated: Path) -> None:
    with pytest.raises(ValidationError):
        _settings(_env_file=None, allowed_git_hosts=5)
    with pytest.raises(ValidationError):
        _settings(_env_file=None, allowed_git_hosts=["github.com", 7])


def test_env_file_is_read_and_environment_wins(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = isolated / "custom.env"
    env_file.write_text(
        "MAX_REPO_KB=123\nGITHUB_TOKEN=\nALLOWED_GIT_HOSTS=example.org\nPOSTGRES_PASSWORD=ignored\n",
        encoding="utf-8",
    )
    s = _settings(_env_file=env_file)
    assert s.max_repo_kb == 123
    assert s.github_token is None
    assert s.allowed_git_hosts == ("example.org",)
    monkeypatch.setenv("MAX_REPO_KB", "456")
    assert _settings(_env_file=env_file).max_repo_kb == 456


def test_dot_env_in_working_directory_is_read(isolated: Path) -> None:
    (isolated / ".env").write_text("REDIS_URL=redis://from-file:6379/2\n", encoding="utf-8")
    assert _settings().redis_url == "redis://from-file:6379/2"


def test_settings_are_frozen_and_hashable(isolated: Path) -> None:
    s = _settings(_env_file=None)
    with pytest.raises(ValidationError):
        s.max_repo_kb = 5
    assert hash(s) == hash(_settings(_env_file=None))


def test_get_settings_is_cached_until_cleared(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    assert get_settings() is first
    monkeypatch.setenv("MAX_REPO_KB", "7")
    assert get_settings().max_repo_kb == DEFAULT_MAX_REPO_KB
    get_settings.cache_clear()
    assert get_settings().max_repo_kb == 7
