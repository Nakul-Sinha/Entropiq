from __future__ import annotations

import io
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tdp_cli.errors import CliError
from tdp_cli.home import EntropiqHome
from tdp_cli.repos import (
    Target,
    local_target,
    looks_like_url,
    prepare_mirror,
    resolve_commit,
    resolve_local,
    resolve_target,
    resolve_url,
)
from tdp_core.mining.git import TransientGitError, run_git
from tdp_core.security.urls import HostLookupError

GITHUB_IP = "140.82.112.3"


def _console() -> Console:
    return Console(file=io.StringIO(), width=200, color_system=None)


def _out(console: Console) -> str:
    return console.file.getvalue()  # type: ignore[attr-defined, no-any-return]


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(host: str, port: object, *args: object, **kwargs: object) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (GITHUB_IP, 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)


@pytest.fixture
def no_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(*args: object, **kwargs: object) -> list[Any]:
        raise AssertionError("no DNS lookup expected")

    monkeypatch.setattr(socket, "getaddrinfo", fake)


def test_resolve_local_root_and_subfolder(sample_repo: Path) -> None:
    root = resolve_local(str(sample_repo))
    assert (root.kind, root.repo_dir, root.ref, root.display) == (
        "local",
        sample_repo.resolve(),
        "HEAD",
        "sample",
    )
    assert root.slug.startswith("sample-") and len(root.slug) == len("sample-") + 8
    sub = resolve_local(str(sample_repo / "pkg"), "HEAD~2")
    assert (sub.repo_dir, sub.slug, sub.ref) == (root.repo_dir, root.slug, "HEAD~2")
    inside_git = resolve_local(str(sample_repo / ".git"))
    assert inside_git.repo_dir == root.repo_dir
    assert local_target(sample_repo).slug == root.slug


def test_resolve_local_bare_clone(sample_repo: Path, tmp_path: Path) -> None:
    bare = tmp_path / "My Repo.git"
    run_git(["clone", "-q", "--bare", "--", str(sample_repo), str(bare)], tmp_path, allow_file_protocol=True)
    target = resolve_local(str(bare))
    assert target.repo_dir == bare.resolve()
    assert target.display == "My Repo"
    assert target.slug.startswith("my-repo-")


def test_resolve_local_errors(tmp_path: Path, sample_repo: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    a_file = tmp_path / "f.txt"
    a_file.write_text("x", encoding="utf-8")
    for path, message in [
        (plain, "is not a git repository"),
        (a_file, "is not a folder"),
        (tmp_path / "missing", "does not exist"),
    ]:
        with pytest.raises(CliError, match=message) as info:
            resolve_local(str(path))
        assert info.value.exit_code == 2
    with pytest.raises(CliError, match="not a valid revision") as info:
        resolve_local(str(sample_repo), "-x")
    assert info.value.exit_code == 2


def test_resolve_commit(sample_repo: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(sample_repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert resolve_commit(local_target(sample_repo)) == head
    assert resolve_commit(local_target(sample_repo, head[:8])) == head
    with pytest.raises(CliError, match="does not name a commit") as info:
        resolve_commit(local_target(sample_repo, "no-such-branch"))
    assert info.value.exit_code == 2


def test_looks_like_url(tmp_path: Path) -> None:
    assert looks_like_url("https://github.com/psf/requests")
    assert looks_like_url("github.com/psf/requests")
    assert looks_like_url("git@github.com:psf/requests.git")
    assert looks_like_url("http://example.com/a/b")
    assert not looks_like_url(".")
    assert not looks_like_url("")
    assert not looks_like_url("some/folder")
    folder = tmp_path / "github.com"
    folder.mkdir()
    assert not looks_like_url(str(folder))


def test_resolve_url(no_dns: None, tmp_path: Path) -> None:
    home = EntropiqHome(tmp_path / "home")
    target, repo = resolve_url("https://github.com/PSF/Requests.git", None, home)
    assert repo.canonical_url == "https://github.com/psf/requests"
    assert (target.kind, target.slug, target.display, target.ref) == (
        "url",
        "psf__requests",
        "psf/requests",
        "HEAD",
    )
    assert target.repo_dir == home.root / "mirrors" / "github.com" / "psf" / "requests.git"
    assert target.repo_dir.is_absolute()
    short, _ = resolve_url("github.com/psf/requests", "v2.31.0", home)
    assert (short.source, short.ref) == ("https://github.com/psf/requests", "v2.31.0")
    gitlab, _ = resolve_url("https://gitlab.com/Group/Project", None, home)
    assert gitlab.slug == "gitlab.com__group__project"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/a/b",
        "http://github.com/psf/requests",
        "git@github.com:psf/requests.git",
        "ssh://git@github.com/psf/requests",
        "https://github.com/psf",
        "https://user:token@github.com/psf/requests",
    ],
)
def test_resolve_url_rejects_without_dns(no_dns: None, tmp_path: Path, url: str) -> None:
    with pytest.raises(CliError, match="Cannot analyse this URL") as info:
        resolve_url(url, None, EntropiqHome(tmp_path))
    assert info.value.exit_code == 2
    assert "token" not in info.value.message


def test_resolve_url_bad_ref(no_dns: None, tmp_path: Path) -> None:
    with pytest.raises(CliError, match="Cannot use this ref") as info:
        resolve_url("https://github.com/psf/requests", "-x", EntropiqHome(tmp_path))
    assert info.value.exit_code == 2


@pytest.fixture
def fake_clone(monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> list[Path]:
    calls: list[Path] = []

    def ensure(url: str, dest: Path, *, allow_file_protocol: bool = False) -> Path:
        calls.append(dest)
        if (dest / "HEAD").is_file():
            run_git(["--git-dir", str(dest), "fetch", "-q", str(sample_repo)], dest, allow_file_protocol=True)
        else:
            run_git(
                ["clone", "-q", "--bare", "--", str(sample_repo), str(dest)],
                dest.parent,
                allow_file_protocol=True,
            )
        return dest

    monkeypatch.setattr("tdp_core.mining.mirror.ensure_mirror", ensure)
    return calls


def test_prepare_mirror_clone_update_and_offline(
    public_dns: None, fake_clone: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = EntropiqHome(tmp_path / "home")
    console = _console()
    target = resolve_target("https://github.com/psf/requests", None, home=home, console=console)
    assert (target.repo_dir / "HEAD").is_file()
    assert "Cloning https://github.com/psf/requests" in _out(console)
    assert resolve_commit(target)

    prepare_mirror(target, resolve_url(target.source, None, home)[1], console)
    assert "Updating the mirror of psf/requests" in _out(console)
    assert len(fake_clone) == 2

    def failing(url: str, dest: Path, *, allow_file_protocol: bool = False) -> Path:
        raise TransientGitError("network down", stderr="fatal: unable to access\n")

    monkeypatch.setattr("tdp_core.mining.mirror.ensure_mirror", failing)
    prepare_mirror(target, resolve_url(target.source, None, home)[1], console)
    assert "Could not update the mirror (fatal: unable to access)" in _out(console)

    def no_host(*args: object, **kwargs: object) -> list[Any]:
        raise OSError("lookup failed")

    monkeypatch.setattr(socket, "getaddrinfo", no_host)
    prepare_mirror(target, resolve_url(target.source, None, home)[1], console)
    assert "Offline" in _out(console)


def test_prepare_mirror_fresh_failure_cleans_up(
    public_dns: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(url: str, dest: Path, *, allow_file_protocol: bool = False) -> Path:
        dest.mkdir(parents=True)
        (dest / "partial").write_text("x", encoding="utf-8")
        raise TransientGitError("network down", stderr="fatal: could not resolve host\n")

    monkeypatch.setattr("tdp_core.mining.mirror.ensure_mirror", failing)
    home = EntropiqHome(tmp_path / "home")
    target, repo = resolve_url("https://github.com/psf/requests", None, home)
    with pytest.raises(CliError, match="Cannot clone https://github") as info:
        prepare_mirror(target, repo, _console())
    assert info.value.exit_code == 1
    assert info.value.hint == "fatal: could not resolve host"
    assert not target.repo_dir.exists()

    def interrupted(url: str, dest: Path, *, allow_file_protocol: bool = False) -> Path:
        dest.mkdir(parents=True)
        raise KeyboardInterrupt

    monkeypatch.setattr("tdp_core.mining.mirror.ensure_mirror", interrupted)
    with pytest.raises(KeyboardInterrupt):
        prepare_mirror(target, repo, _console())
    assert not target.repo_dir.exists()


def test_prepare_mirror_without_dns_or_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_host(*args: object, **kwargs: object) -> list[Any]:
        raise OSError("lookup failed")

    monkeypatch.setattr(socket, "getaddrinfo", no_host)
    home = EntropiqHome(tmp_path / "home")
    target, repo = resolve_url("https://github.com/psf/requests", None, home)
    with pytest.raises(CliError, match="Cannot reach github") as info:
        prepare_mirror(target, repo, _console())
    assert info.value.exit_code == 1


def test_prepare_mirror_private_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def private(*args: object, **kwargs: object) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", private)
    home = EntropiqHome(tmp_path / "home")
    target, repo = resolve_url("https://github.com/psf/requests", None, home)
    with pytest.raises(CliError, match="Cannot analyse this URL") as info:
        prepare_mirror(target, repo, _console())
    assert info.value.exit_code == 2


def test_target_is_frozen(sample_repo: Path) -> None:
    target = local_target(sample_repo)
    assert isinstance(target, Target)
    with pytest.raises(AttributeError):
        target.ref = "x"  # type: ignore[misc]
    assert HostLookupError.__mro__[1].__name__ == "UnsafeRepoError"
