"""Tests of ``action/run.py``, the helper of the Entropiq GitHub Action (no network, no real ``gh``)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[3]
VERSION = "ranker-2026.10.1"
REPO = "Nakul-Sinha/Entropiq"


def load_action() -> ModuleType:
    """``action/run.py`` as a module (registered before running it, which dataclasses need)."""
    name = "entropiq_action_run"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "action" / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


act = load_action()


def make_tarball(
    where: Path,
    version: str = VERSION,
    *,
    extra: dict[str, bytes] | None = None,
    meta_version: str | None = None,
) -> tuple[Path, str]:
    """A bundle tarball with ``<version>/metadata.json`` and ``ranker.json``; its path and sha256."""
    src = where / "src" / version
    src.mkdir(parents=True)
    meta = {"version": meta_version or version, "synthetic": False}
    (src / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (src / "ranker.json").write_text("{}", encoding="utf-8")
    tarball = where / f"{version}.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(src, arcname=version)
        for name, data in (extra or {}).items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return tarball, hashlib.sha256(tarball.read_bytes()).hexdigest()


class FakeDownload:
    """A downloader that copies a local tarball (and checksum text) and can fail the first tries."""

    def __init__(self, tarball: Path | None = None, checksum: str | None = None, fail: int = 0) -> None:
        self.tarball, self.checksum, self.fail = tarball, checksum, fail
        self.calls: list[tuple[str, str, bool]] = []

    def __call__(self, spec: Any, dest: Path, with_checksum: bool) -> None:
        self.calls.append((spec.repo, spec.tag, with_checksum))
        if self.fail > 0:
            self.fail -= 1
            raise act.ActionError("gh release download failed: HTTP 502")
        if self.tarball is not None:
            shutil.copy(self.tarball, dest / spec.asset)
        if self.checksum is not None and with_checksum:
            (dest / f"{spec.asset}.sha256").write_text(self.checksum, encoding="utf-8")


class FakeApi(act.GhApi):  # type: ignore[misc,name-defined]
    """Records the comment calls; flags make listing, updating or creating fail."""

    def __init__(
        self,
        comments: tuple[Any, ...] = (),
        viewer: str | None = None,
        *,
        fail_list: bool = False,
        fail_update: bool = False,
        fail_create: bool = False,
    ) -> None:
        self.comments = list(comments)
        self.viewer = viewer
        self.fail_list, self.fail_update, self.fail_create = fail_list, fail_update, fail_create
        self.calls: list[tuple[Any, ...]] = []

    def viewer_login(self) -> str | None:
        self.calls.append(("viewer",))
        return self.viewer

    def report_comments(self, repo: str, number: int) -> list[Any]:
        self.calls.append(("list", repo, number))
        if self.fail_list:
            raise act.ActionError(
                "gh api --paginate failed: HTTP 403: Resource not accessible by integration"
            )
        return list(self.comments)

    def create_comment(self, repo: str, number: int, body: str) -> int:
        self.calls.append(("create", repo, number, body))
        if self.fail_create:
            raise act.ActionError("gh api --method failed: HTTP 403")
        return 9001

    def update_comment(self, repo: str, comment_id: int, body: str) -> int:
        self.calls.append(("update", repo, comment_id, body))
        if self.fail_update:
            raise act.ActionError("gh api --method failed: HTTP 404")
        return comment_id


def row(path: str, band: str, priority: int = 50, risk: float = 0.5, reason: str = "High churn") -> dict:
    return {"path": path, "band": band, "priority": priority, "risk": risk, "top_reason": reason}


def make_diff(
    files: list[dict], unranked: tuple[str, ...] = (), *, demo: bool = False, base: str = "HEAD~10"
) -> dict[str, Any]:
    """A ``tdp diff --format json`` document whose summary is computed from ``files``."""
    counts = {band.lower(): sum(1 for f in files if f["band"] == band) for band in act.BAND_ORDER}
    return {
        "schema": act.DIFF_SCHEMA,
        "repo": "local/sample",
        "sha": "a" * 40,
        "model_version": "dev-0" if demo else VERSION,
        "demo_model": demo,
        "base": base,
        "base_sha": "b" * 40,
        "merge_base": "c" * 40,
        "head_sha": "a" * 40,
        "summary": {"changed": len(files) + len(unranked), "ranked": len(files), **counts},
        "files": [{"rank": i + 1, **f, "change": "modified", "old_path": None} for i, f in enumerate(files)],
        "unranked": [{"path": p, "change": "deleted", "old_path": None} for p in unranked],
    }


def make_env(tmp_path: Path, **values: str) -> dict[str, str]:
    """The environment of one action step; outputs and the summary go into ``tmp_path``."""
    env = {
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "ENTROPIQ_WORK_DIR": str(tmp_path / "work"),
        "ENTROPIQ_POINTER": str(ROOT / "models" / "production.json"),
        "ENTROPIQ_DEFAULT_REPO": REPO,
        "INPUT_BASE": "b" * 40,
        "INPUT_COMMENT": "true",
        "INPUT_FAIL_ON": "none",
        "INPUT_SARIF": "",
        "INPUT_MODEL_VERSION": "",
    }
    env.update(values)
    return env


def read_outputs(tmp_path: Path) -> dict[str, str]:
    text = (tmp_path / "output.txt").read_text(encoding="utf-8")
    return dict(line.split("=", 1) for line in text.splitlines())


# Inputs


def test_parse_bool_and_fail_on() -> None:
    assert act.parse_bool(" True ", "comment") is True
    assert act.parse_bool("on", "comment") is True
    assert act.parse_bool("0", "comment") is False
    assert act.parse_bool("NO", "comment") is False
    with pytest.raises(act.ActionError) as caught:
        act.parse_bool("maybe", "comment")
    assert caught.value.exit_code == 2 and "comment" in caught.value.message
    assert act.parse_fail_on("") == "none"
    assert act.parse_fail_on(" HIGH ") == "high"
    assert act.parse_fail_on("critical") == "critical"
    with pytest.raises(act.ActionError) as bad:
        act.parse_fail_on("medium")
    assert bad.value.exit_code == 2


@pytest.mark.parametrize(
    ("key", "value"),
    [("INPUT_BASE", "  "), ("INPUT_COMMENT", "sure"), ("INPUT_FAIL_ON", "low"), ("INPUT_SARIF", " - ")],
)
def test_check_inputs_rejects(tmp_path: Path, key: str, value: str) -> None:
    with pytest.raises(act.ActionError) as caught:
        act.check_inputs(make_env(tmp_path, **{key: value}))
    assert caught.value.exit_code == 2
    if key == "INPUT_BASE":
        assert "github.event.pull_request.base.sha" in caught.value.message


def test_check_inputs_accepts_defaults(tmp_path: Path) -> None:
    act.check_inputs(make_env(tmp_path, INPUT_SARIF="entropiq.sarif", INPUT_FAIL_ON="Critical"))


# Model choice


def test_tracked_pointer_names_the_release_asset() -> None:
    pointer = act.read_pointer(ROOT / "models" / "production.json")
    assert pointer is not None and pointer.version is not None
    assert pointer.version.startswith("ranker-")
    match = act.RELEASE_URL_RE.fullmatch(pointer.uri or "")
    assert match, f"models/production.json uri is not a release asset URL: {pointer.uri}"
    assert match["tag"] == f"model-{pointer.version}" and match["asset"] == f"{pointer.version}.tar.gz"
    assert act.SHA256_RE.fullmatch(pointer.sha256 or "")
    spec = act.model_spec(pointer, "", "someone/else")
    assert spec.repo == REPO and spec.sha256 == pointer.sha256 and spec.tag == f"model-{pointer.version}"


def test_read_pointer_edge_cases(tmp_path: Path) -> None:
    assert act.read_pointer(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert act.read_pointer(tmp_path / "bad.json") is None
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    assert act.read_pointer(tmp_path / "list.json") is None
    (tmp_path / "null.json").write_text('{"version": null, "uri": " ", "sha256": 5}', encoding="utf-8")
    assert act.read_pointer(tmp_path / "null.json") == act.Pointer(None, None, None)


def test_model_spec_variants() -> None:
    sha = "ab" * 32
    assert act.model_spec(None, "", REPO) is None
    assert act.model_spec(act.Pointer(None, None, None), " ", REPO) is None
    local = act.Pointer(VERSION, "C:/models/ranker-2026.10.1", sha.upper())
    spec = act.model_spec(local, "", REPO)
    assert spec == act.ModelSpec(VERSION, REPO, f"model-{VERSION}", f"{VERSION}.tar.gz", sha)
    other = act.model_spec(local, "ranker-2026.11.1", REPO)
    assert other == act.ModelSpec(
        "ranker-2026.11.1", REPO, "model-ranker-2026.11.1", "ranker-2026.11.1.tar.gz", None
    )
    url = f"https://github.com/acme/models/releases/download/model%2Dx/{VERSION}.tar.gz"
    moved = act.model_spec(act.Pointer(VERSION, url, "not-a-sha"), "", REPO)
    assert moved == act.ModelSpec(VERSION, "acme/models", "model-x", f"{VERSION}.tar.gz", None)
    assert act.cache_key(spec) == f"tdp-model-{VERSION}"
    for version in ("../x", "a b", "-x", ""):
        if version:
            with pytest.raises(act.ActionError) as caught:
                act.model_spec(local, version, REPO)
            assert caught.value.exit_code == 2
    bad_asset = f"https://github.com/acme/models/releases/download/t/{'%20'}x.tar.gz"
    with pytest.raises(act.ActionError):
        act.model_spec(act.Pointer(VERSION, bad_asset, sha), "", REPO)
    with pytest.raises(act.ActionError) as empty:
        act.model_spec(local, "", "")
    assert empty.value.exit_code == 2


def test_no_model_reason() -> None:
    assert "could not be read" in act.no_model_reason(None)
    assert act.no_model_reason(act.Pointer(None, None, None)) == "no production model has been published yet"


def test_read_checksum(tmp_path: Path) -> None:
    sha = "0f" * 32
    path = tmp_path / "x.sha256"
    assert act.read_checksum(path) is None
    path.write_text(f"\n{sha.upper()}  x.tar.gz\n", encoding="utf-8")
    assert act.read_checksum(path) == sha
    path.write_text(f"{sha} *x.tar.gz\n", encoding="utf-8")
    assert act.read_checksum(path) == sha
    path.write_text("nothing here\n", encoding="utf-8")
    assert act.read_checksum(path) is None


def spec_for(sha: str | None, version: str = VERSION) -> Any:
    return act.ModelSpec(version, REPO, f"model-{version}", f"{version}.tar.gz", sha)


def test_fetch_downloads_verifies_caches_and_extracts(tmp_path: Path) -> None:
    tarball, sha = make_tarball(tmp_path / "build")
    work = tmp_path / "work"
    download = FakeDownload(tarball)
    sleeps: list[float] = []
    outcome = act.fetch_model(spec_for(sha), work, download=download, sleep=sleeps.append)
    assert outcome.source == "release" and outcome.reason == "" and outcome.downloaded
    assert outcome.model_dir == work / "model" / VERSION
    meta = json.loads((outcome.model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["version"] == VERSION
    assert download.calls == [(REPO, f"model-{VERSION}", False)] and sleeps == []
    assert (work / "model-cache" / VERSION / f"{VERSION}.tar.gz").is_file()
    assert not (work / "download").exists()
    again = act.fetch_model(spec_for(sha), work, download=download, sleep=sleeps.append)
    assert again.source == "release" and not again.downloaded and len(download.calls) == 1


def test_fetch_rejects_a_wrong_sha(tmp_path: Path) -> None:
    tarball, _ = make_tarball(tmp_path / "build")
    outcome = act.fetch_model(spec_for("1" * 64), tmp_path / "work", download=FakeDownload(tarball))
    assert outcome.source == "dev" and outcome.model_dir is None
    assert outcome.reason.startswith(
        "the downloaded model failed its sha256 check (expected 111111111111, got "
    )
    assert not (tmp_path / "work" / "model-cache" / VERSION / f"{VERSION}.tar.gz").exists()


def test_fetch_retries_then_falls_back(tmp_path: Path) -> None:
    download = FakeDownload(None, fail=5)
    sleeps: list[float] = []
    outcome = act.fetch_model(spec_for("1" * 64), tmp_path / "work", download=download, sleep=sleeps.append)
    assert outcome.source == "dev" and len(download.calls) == 3 and sleeps == [5.0, 15.0]
    assert outcome.reason == (
        f"the model release model-{VERSION} could not be downloaded (gh release download failed: HTTP 502)"
    )
    recovered = FakeDownload(make_tarball(tmp_path / "build")[0], fail=1)
    sha = hashlib.sha256((tmp_path / "build" / f"{VERSION}.tar.gz").read_bytes()).hexdigest()
    fine = act.fetch_model(spec_for(sha), tmp_path / "work2", download=recovered, sleep=sleeps.append)
    assert fine.source == "release" and len(recovered.calls) == 2


def test_fetch_missing_asset(tmp_path: Path) -> None:
    outcome = act.fetch_model(spec_for("1" * 64), tmp_path / "work", download=FakeDownload(None))
    assert outcome.reason == f"the release model-{VERSION} has no asset {VERSION}.tar.gz"


def test_fetch_other_version_uses_the_checksum_file(tmp_path: Path) -> None:
    version = "ranker-2026.11.1"
    tarball, sha = make_tarball(tmp_path / "build", version)
    work = tmp_path / "work"
    download = FakeDownload(tarball, checksum=f"{sha}  {version}.tar.gz\n")
    outcome = act.fetch_model(spec_for(None, version), work, download=download)
    assert outcome.source == "release" and download.calls == [(REPO, f"model-{version}", True)]
    cached = work / "model-cache" / version / f"{version}.tar.gz.sha256"
    assert act.read_checksum(cached) == sha
    again = act.fetch_model(spec_for(None, version), work, download=download)
    assert again.source == "release" and len(download.calls) == 1
    missing = act.fetch_model(spec_for(None, version), tmp_path / "other", download=FakeDownload(tarball))
    assert missing.source == "dev"
    assert missing.reason == f"the release model-{version} has no sha256 file to verify the model"


@pytest.mark.parametrize("kind", ["foreign", "meta", "corrupt", "parent"])
def test_fetch_rejects_bad_archives(tmp_path: Path, kind: str) -> None:
    build = tmp_path / "build"
    if kind == "foreign":
        tarball, sha = make_tarball(build, extra={"evil.sh": b"echo hi\n"})
    elif kind == "parent":
        tarball, sha = make_tarball(build, extra={f"{VERSION}/../escape.txt": b"x"})
    elif kind == "meta":
        tarball, sha = make_tarball(build, meta_version="ranker-1999.01.1")
    else:
        build.mkdir()
        tarball = build / f"{VERSION}.tar.gz"
        tarball.write_bytes(b"\x1f\x8b\x08\x00 not really gzip")
        sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
    outcome = act.fetch_model(spec_for(sha), tmp_path / "work", download=FakeDownload(tarball))
    assert outcome.source == "dev" and outcome.model_dir is None
    assert outcome.reason.startswith("the model archive could not be unpacked (")
    assert (
        not (tmp_path / "work" / "escape.txt").exists()
        and not (tmp_path / "work" / "model" / "evil.sh").exists()
    )


def test_fetch_without_a_spec(tmp_path: Path) -> None:
    outcome = act.fetch_model(None, tmp_path, no_model="the pointer is gone", download=FakeDownload())
    assert outcome == act.ModelOutcome("dev", None, "the pointer is gone")


# GitHub CLI


def test_gh_download_arguments(tmp_path: Path) -> None:
    seen: list[tuple[list[str], str | None]] = []

    def run(args: Any, stdin: str | None) -> str:
        seen.append((list(args), stdin))
        return ""

    act.gh_download(spec_for(None), tmp_path, True, run=run)
    act.gh_download(spec_for("1" * 64), tmp_path, False, run=run)
    base = ["release", "download", f"model-{VERSION}", "--repo", REPO, "--dir", str(tmp_path), "--clobber"]
    assert seen[0] == (
        [*base, "--pattern", f"{VERSION}.tar.gz", "--pattern", f"{VERSION}.tar.gz.sha256"],
        None,
    )
    assert seen[1] == ([*base, "--pattern", f"{VERSION}.tar.gz"], None)


def test_run_gh_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(act.shutil, "which", lambda name: None)
    with pytest.raises(act.ActionError, match="not installed"):
        act.run_gh(["api", "user"])
    monkeypatch.setattr(act.shutil, "which", lambda name: "/usr/bin/gh")
    calls: list[dict[str, Any]] = []

    def failing(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"cmd": cmd, **kwargs})
        return subprocess.CompletedProcess(
            cmd, 1, "", "\n  HTTP 404: Not Found (https://api.github.com/x)\nmore\n"
        )

    monkeypatch.setattr(act.subprocess, "run", failing)
    with pytest.raises(act.ActionError) as caught:
        act.run_gh(["release", "download", "x"], "in", timeout=7)
    assert (
        caught.value.message == "gh release download failed: HTTP 404: Not Found (https://api.github.com/x)"
    )
    assert calls[0]["cmd"] == ["/usr/bin/gh", "release", "download", "x"]
    assert calls[0]["input"] == "in" and calls[0]["timeout"] == 7 and calls[0]["check"] is False

    def slow(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd, 7)

    monkeypatch.setattr(act.subprocess, "run", slow)
    with pytest.raises(act.ActionError, match="timed out"):
        act.run_gh(["api", "user"], timeout=7)

    def fine(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, "octocat\n", "")

    monkeypatch.setattr(act.subprocess, "run", fine)
    assert act.run_gh(["api", "user"]) == "octocat\n"


def test_gh_api_calls() -> None:
    seen: list[tuple[list[str], str | None]] = []
    replies = iter(['[11,"github-actions[bot]"]\n\n[12,null]\n', '{"id": 77}', '{"id": 11}'])

    def run(args: Any, stdin: str | None) -> str:
        seen.append((list(args), stdin))
        return next(replies)

    api = act.GhApi(run)
    assert api.report_comments(REPO, 7) == [act.Comment(11, "github-actions[bot]"), act.Comment(12, "")]
    args = seen[0][0]
    assert args[:3] == ["api", "--paginate", f"repos/{REPO}/issues/7/comments?per_page=100"]
    assert act.MARKER in args[4]
    assert api.create_comment(REPO, 7, "body") == 77
    assert seen[1] == (
        ["api", "--method", "POST", f"repos/{REPO}/issues/7/comments", "--input", "-"],
        '{"body": "body"}',
    )
    assert api.update_comment(REPO, 11, "new") == 11
    assert seen[2] == (
        ["api", "--method", "PATCH", f"repos/{REPO}/issues/comments/11", "--input", "-"],
        '{"body": "new"}',
    )

    def forbidden(args: Any, stdin: str | None) -> str:
        raise act.ActionError("gh api user failed: HTTP 403")

    assert act.GhApi(forbidden).viewer_login() is None
    assert act.GhApi(lambda args, stdin: "nakul\n").viewer_login() == "nakul"


def test_upsert_comment() -> None:
    bot, other_bot = act.Comment(30, "github-actions[bot]"), act.Comment(20, "github-actions[bot]")
    human = act.Comment(10, "someone")
    empty = FakeApi()
    assert act.upsert_comment(empty, REPO, 7, "b") == ("created", 9001)
    assert [c[0] for c in empty.calls] == ["list", "create"]
    two = FakeApi((bot, other_bot))
    assert act.upsert_comment(two, REPO, 7, "b") == ("updated", 20)
    assert [c[0] for c in two.calls] == ["list", "update"]
    quoted = FakeApi((human,), viewer=None)
    assert act.upsert_comment(quoted, REPO, 7, "b") == ("created", 9001)
    assert [c[0] for c in quoted.calls] == ["list", "viewer", "create"]
    pat = FakeApi((human,), viewer="someone")
    assert act.upsert_comment(pat, REPO, 7, "b") == ("updated", 10)
    broken = FakeApi((bot,), fail_update=True)
    assert act.upsert_comment(broken, REPO, 7, "b") == ("created", 9001)
    assert [c[0] for c in broken.calls] == ["list", "update", "create"]
    assert not act.is_own(act.Comment(1, ""), "")


# Comment


def test_render_comment_exact() -> None:
    files = [
        row("pkg/a|b.py", "Critical", 91, 0.64, "Many <b>changes</b> | churn"),
        row("pkg/c.py", "Low", 12, 0.1, ""),
    ]
    body = act.render_comment(make_diff(files, ("tests/test_x.py",)), fail_on="none")
    assert body.splitlines() == [
        act.MARKER,
        "### Entropiq report",
        "",
        "2 of 3 changed files are ranked: 1 Critical, 0 High, 0 Medium, 1 Low.",
        "",
        "| File | Band | Priority | Risk | Top reason |",
        "| --- | --- | ---: | ---: | --- |",
        "| `pkg/a\\|b.py` | Critical | 91 | 64% | Many \\<b\\>changes\\</b\\> \\| churn |",
        "| `pkg/c.py` | Low | 12 | 10% | - |",
        "",
        "1 changed file is not ranked (deleted, tests, skipped or not Python).",
        "",
        "Commit `aaaaaaaaaaaa` compared with `HEAD~10` (merge base `cccccccccccc`), "
        f"model `{VERSION}`. Fail-on is off.",
    ]
    assert body.endswith(".\n") and not body.endswith("\n\n")
    hexed = act.render_comment(make_diff(files, base="b" * 40), fail_on="none")
    assert "compared with `bbbbbbbbbbbb` (merge base" in hexed and "`" + "b" * 13 not in hexed
    named = act.render_comment(make_diff(files, base="main"), fail_on="none")
    assert "compared with `main` (merge base" in named


def test_render_comment_edge_cases() -> None:
    nothing = act.render_comment(make_diff([]), fail_on="none")
    assert "No files changed compared with `HEAD~10`." in nothing and "| File |" not in nothing
    one = act.render_comment(make_diff([], ("README.md",)), fail_on="none")
    assert "The changed file is not ranked." in one and "not ranked (deleted" not in one
    many = act.render_comment(make_diff([], ("a.md", "b.md")), fail_on="none")
    assert "None of the 2 changed files are ranked." in many
    single = act.render_comment(make_diff([row("x.py", "High")]), fail_on="none")
    assert "1 of 1 changed file is ranked: 0 Critical, 1 High, 0 Medium, 0 Low." in single
    files = [row(f"m{i}.py", "Medium") for i in range(3)]
    cut = act.render_comment(make_diff(files), fail_on="none", max_rows=1)
    assert cut.count("| `m") == 1 and "Showing the top 1 of 3 ranked files." in cut
    odd = {"path": "z.py", "band": "Low", "priority": None, "risk": "high"}
    weird = act.render_comment({**make_diff([]), "files": [odd], "summary": {}}, fail_on="none")
    assert "| `z.py` | Low | n/a | n/a | - |" in weird
    demo = act.render_comment(make_diff(files, demo=True), fail_on="high", reason="no model <yet>.")
    assert (
        "> [!WARNING]\n> Demo model in use: no model \\<yet\\>. These rankings show how Entropiq works"
        in demo
    )
    assert demo.rstrip().endswith("Fail-on high is not applied because the demo model is in use.")
    plain = act.render_comment(make_diff(files, demo=True), fail_on="none")
    assert "> Demo model in use. These rankings show how Entropiq works, not real risk." in plain


def test_render_comment_body_cap() -> None:
    files = [row(f"f{i}.py", "High", reason="x" * 9_000) for i in range(20)]
    body = act.render_comment(make_diff(files), fail_on="none")
    assert len(body) <= act.MAX_BODY_CHARS
    assert body.startswith(act.MARKER) and "Showing the top" in body and body.count("| `f") < 20


@pytest.mark.parametrize(
    ("bands", "fail_on", "fails", "sentence"),
    [
        (["Critical"], "none", False, "Fail-on is off."),
        (
            ["High", "Low"],
            "high",
            True,
            "Fail-on high: 1 changed file is High or Critical, so this check fails.",
        ),
        (
            ["Critical", "High"],
            "high",
            True,
            "Fail-on high: 2 changed files are High or Critical, so this check fails.",
        ),
        (["Medium", "Low"], "high", False, "Fail-on high: no changed file is High or Critical."),
        (["High"], "critical", False, "Fail-on critical: no changed file is Critical."),
        (
            ["Critical"],
            "critical",
            True,
            "Fail-on critical: 1 changed file is Critical, so this check fails.",
        ),
    ],
)
def test_gate(bands: list[str], fail_on: str, fails: bool, sentence: str) -> None:
    diff = make_diff([row(f"f{i}.py", band) for i, band in enumerate(bands)])
    assert act.gate(diff, fail_on) == (fails, sentence)
    demo = make_diff([row(f"f{i}.py", band) for i, band in enumerate(bands)], demo=True)
    assert act.gate(demo, fail_on)[0] is False


def test_highest_band() -> None:
    assert act.highest_band({"critical": 0, "high": 2, "low": 1}) == "High"
    assert act.highest_band({"critical": 1}) == "Critical"
    assert act.highest_band({"low": 0}) == "none"
    assert act.highest_band({}) == "none"


def test_pr_number() -> None:
    assert act.pr_number({"pull_request": {"number": 7}}) == 7
    assert act.pr_number({"pull_request": {"number": True}}) is None
    assert act.pr_number({"push": {}}) is None


# Subcommands


def report_env(
    tmp_path: Path, diff: dict[str, Any] | None, *, event: bool = True, **values: str
) -> dict[str, str]:
    diff_path = tmp_path / "diff.json"
    if diff is not None:
        diff_path.write_text(json.dumps(diff), encoding="utf-8")
    env = make_env(tmp_path, ENTROPIQ_DIFF_JSON=str(diff_path), GITHUB_REPOSITORY=REPO, **values)
    if event:
        (tmp_path / "event.json").write_text(json.dumps({"pull_request": {"number": 7}}), encoding="utf-8")
        env["GITHUB_EVENT_PATH"] = str(tmp_path / "event.json")
    return env


def test_cmd_report_comments_and_gates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    diff = make_diff([row("a.py", "High", 80), row("b.py", "Low", 5)])
    api = FakeApi((act.Comment(5, "github-actions[bot]"),))
    env = report_env(tmp_path, diff, INPUT_FAIL_ON="high")
    assert act.cmd_report(env, api=api) == 1
    outputs = read_outputs(tmp_path)
    assert outputs == {"highest-band": "High", "comment-id": "5", "model-version": VERSION}
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith(act.MARKER) and "| `a.py` | High | 80 | 50% | High churn |" in summary
    assert api.calls[-1][0] == "update" and api.calls[-1][3] == summary
    out = capsys.readouterr().out
    assert "Updated comment 5." in out
    assert "::error::Fail-on high: 1 changed file is High or Critical, so this check fails." in out


def test_cmd_report_without_comment_or_event(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    diff = make_diff([row("a.py", "Critical")])
    api = FakeApi()
    assert act.cmd_report(report_env(tmp_path, diff, INPUT_COMMENT="false"), api=api) == 0
    assert api.calls == [] and read_outputs(tmp_path)["comment-id"] == ""
    other = tmp_path / "other"
    other.mkdir()
    assert act.cmd_report(report_env(other, diff, event=False), api=api) == 0
    assert api.calls == []
    assert (
        "::notice::No pull request in this event, so the report is only in the job summary."
        in capsys.readouterr().out
    )


def test_cmd_report_comment_failure_is_a_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    diff = make_diff([row("a.py", "Low")], demo=True)
    env = report_env(tmp_path, diff, INPUT_FAIL_ON="critical")
    assert act.cmd_report(env, api=FakeApi(fail_list=True)) == 0
    out = capsys.readouterr().out
    assert "::warning::Could not write the pull request comment: gh api --paginate failed: HTTP 403" in out
    assert "::warning::Fail-on critical is not applied because the demo model is in use." in out
    assert read_outputs(tmp_path)["comment-id"] == ""


def test_main_report_rejects_bad_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert act.main(["report"], report_env(tmp_path, None)) == 1
    assert "could not be read" in capsys.readouterr().out
    assert act.main(["report"], report_env(tmp_path, {"schema": "entropiq.other.v1"})) == 1
    assert f"is not an {act.DIFF_SCHEMA} document" in capsys.readouterr().out


def test_cmd_resolve_and_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = make_env(tmp_path)
    assert act.main(["resolve"], env) == 0
    outputs = read_outputs(tmp_path)
    assert outputs["version"] == VERSION and outputs["cache-key"] == f"tdp-model-{VERSION}"
    assert outputs["cache-dir"] == str(tmp_path / "work" / "model-cache" / VERSION)
    assert act.main(["resolve"], make_env(tmp_path, INPUT_BASE="")) == 2
    assert act.main(["resolve"], make_env(tmp_path, INPUT_MODEL_VERSION="../up")) == 2
    assert act.main(["nope"], env) == 2 and act.main([], env) == 2 and act.main(["fetch", "x"], env) == 2
    del env["ENTROPIQ_WORK_DIR"]
    assert act.main(["resolve"], env) == 2
    assert "::error::internal: ENTROPIQ_WORK_DIR is not set" in capsys.readouterr().out
    none = tmp_path / "none"
    none.mkdir()
    empty_pointer = tmp_path / "pointer.json"
    empty_pointer.write_text('{"version": null}', encoding="utf-8")
    quiet = make_env(none, ENTROPIQ_POINTER=str(empty_pointer))
    assert act.cmd_resolve(quiet) == 0
    assert read_outputs(none) == {"version": "", "cache-dir": "", "cache-key": ""}


def test_cmd_fetch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build = tmp_path / "build"
    tarball, sha = make_tarball(build)
    pointer = tmp_path / "pointer.json"
    url = f"https://github.com/{REPO}/releases/download/model-{VERSION}/{VERSION}.tar.gz"
    pointer.write_text(json.dumps({"version": VERSION, "uri": url, "sha256": sha}), encoding="utf-8")
    env = make_env(tmp_path, ENTROPIQ_POINTER=str(pointer))
    assert act.cmd_fetch(env, download=FakeDownload(tarball)) == 0
    outputs = read_outputs(tmp_path)
    assert outputs == {
        "source": "release",
        "model-dir": str(tmp_path / "work" / "model" / VERSION),
        "reason": "",
        "downloaded": "true",
    }
    assert "(sha256 verified)" in capsys.readouterr().out
    failing = tmp_path / "failing"
    failing.mkdir()
    env = make_env(failing, ENTROPIQ_POINTER=str(pointer))
    assert act.cmd_fetch(env, download=FakeDownload(fail=3), sleep=lambda s: None) == 0
    assert read_outputs(failing)["source"] == "dev"
    assert "::warning::Falling back to the demo model: the model release" in capsys.readouterr().out


def test_annotate_and_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    act.annotate("warning", "50% done\r\nnext")
    assert capsys.readouterr().out == "::warning::50%25 done%0D%0Anext\n"
    act.write_outputs({"GITHUB_OUTPUT": str(tmp_path / "output.txt")}, {"reason": "one\ntwo\rthree"})
    assert (tmp_path / "output.txt").read_bytes() == b"reason=one two three\n"
    act.write_outputs({}, {"a": "b"})
    act.append_summary({}, "text\n")
    assert capsys.readouterr().out == "a=b\ntext\n"


# The CLI contract: the real tdp diff JSON goes through the comment and the gate


def test_tdp_diff_json_feeds_the_comment(dev_home: Path, sample_loaded: Any, sample_repo: Path) -> None:
    from tdp_cli.home import EntropiqHome
    from tdp_cli.main import build_app
    from tdp_cli.store import AnalysisStore

    AnalysisStore(EntropiqHome.from_env().analyses).save(sample_loaded)
    result = CliRunner().invoke(
        build_app(), ["diff", str(sample_repo), "--base", "HEAD~10", "--dev-model", "--format", "json"]
    )
    assert result.exit_code == 0, result.stderr
    diff = json.loads(result.stdout)
    assert diff["schema"] == act.DIFF_SCHEMA and diff["demo_model"] is True
    body = act.render_comment(diff, fail_on="critical", reason="no production model has been published yet")
    rows = [line for line in body.splitlines() if line.startswith("| `")]
    assert len(rows) == min(diff["summary"]["ranked"], act.MAX_ROWS) > 0
    assert "> Demo model in use: no production model has been published yet." in body
    assert act.gate(diff, "critical") == (
        False,
        "Fail-on critical is not applied because the demo model is in use.",
    )
    assert act.highest_band(diff["summary"]) in act.BAND_ORDER
