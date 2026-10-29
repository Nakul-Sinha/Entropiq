"""ClaudeCLIClient against a fake ``claude`` program, and find_claude_cli on a made-up PATH."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tdp_core.advisor.client import ENV_CLAUDE_CLI, ClaudeCLIClient, LLMError, find_claude_cli

MODEL = "claude-sonnet-5-5"

# A stand-in for `claude -p`: records what it was given, then behaves as FAKE_MODE says.
FAKE_CLI = r"""
import json, os, sys, time
argv = sys.argv[1:]
prompt_file = argv[argv.index("--system-prompt-file") + 1]
record = {
    "argv": argv,
    "stdin": sys.stdin.read(),
    "system": open(prompt_file, encoding="utf-8").read(),
    "cwd": os.getcwd(),
    "has_key": "ANTHROPIC_API_KEY" in os.environ,
    "max_tokens": os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"),
}
with open(os.environ["FAKE_RECORD"], "w", encoding="utf-8") as out:
    json.dump(record, out)
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "ok":
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "pong"}))
elif mode == "error":
    failure = {"type": "result", "subtype": "success", "is_error": True, "result": os.environ["FAKE_TEXT"]}
    print(json.dumps(failure))
    sys.exit(1)
elif mode == "max_turns":
    print(json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": False}))
elif mode == "garbage":
    sys.stderr.write("boom: something broke")
    sys.exit(3)
elif mode == "sleep":
    time.sleep(5)
"""


@pytest.fixture
def fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], Path]:
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLI, encoding="utf-8")
    record = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_RECORD", str(record))
    monkeypatch.delenv("FAKE_MODE", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", raising=False)
    return [sys.executable, str(script)], record


def test_reply_and_isolation(fake_cli: tuple[list[str], Path], monkeypatch: pytest.MonkeyPatch) -> None:
    command, record = fake_cli
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    client = ClaudeCLIClient(MODEL, command)
    assert client.complete("SYSTEM PROMPT", "USER MESSAGE", max_tokens=1234) == "pong"
    seen = json.loads(record.read_text(encoding="utf-8"))
    argv = seen["argv"]
    assert argv[0] == "-p"
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--model") + 1] == MODEL
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in argv
    assert seen["system"] == "SYSTEM PROMPT"
    assert seen["stdin"] == "USER MESSAGE"
    assert seen["has_key"] is False
    assert seen["max_tokens"] == "1234"
    assert Path(seen["cwd"]).name.startswith("entropiq-advice-")
    assert not Path(seen["cwd"]).exists()  # the temporary folder is removed after the call


def test_an_existing_output_cap_is_kept(
    fake_cli: tuple[list[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    command, record = fake_cli
    monkeypatch.setenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "999")
    ClaudeCLIClient(MODEL, command).complete("s", "u", max_tokens=1234)
    assert json.loads(record.read_text(encoding="utf-8"))["max_tokens"] == "999"


def test_a_logged_out_cli_is_not_retryable(
    fake_cli: tuple[list[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    command, _ = fake_cli
    monkeypatch.setenv("FAKE_MODE", "error")
    monkeypatch.setenv("FAKE_TEXT", "Not logged in · Please run /login")
    with pytest.raises(LLMError, match="Please run /login") as info:
        ClaudeCLIClient(MODEL, command).complete("s", "u", max_tokens=10)
    assert info.value.retryable is False


def test_a_rate_limit_is_retryable(fake_cli: tuple[list[str], Path], monkeypatch: pytest.MonkeyPatch) -> None:
    command, _ = fake_cli
    monkeypatch.setenv("FAKE_MODE", "error")
    monkeypatch.setenv("FAKE_TEXT", "API Error: 429 rate limit exceeded")
    with pytest.raises(LLMError, match="rate limit") as info:
        ClaudeCLIClient(MODEL, command).complete("s", "u", max_tokens=10)
    assert info.value.retryable is True


def test_a_non_success_subtype_fails(
    fake_cli: tuple[list[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    command, _ = fake_cli
    monkeypatch.setenv("FAKE_MODE", "max_turns")
    with pytest.raises(LLMError, match="error_max_turns"):
        ClaudeCLIClient(MODEL, command).complete("s", "u", max_tokens=10)


def test_output_that_is_not_json_fails_with_stderr(
    fake_cli: tuple[list[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    command, _ = fake_cli
    monkeypatch.setenv("FAKE_MODE", "garbage")
    with pytest.raises(LLMError, match=r"exit code 3\): boom: something broke") as info:
        ClaudeCLIClient(MODEL, command).complete("s", "u", max_tokens=10)
    assert info.value.retryable is False


def test_a_slow_cli_times_out_and_is_retryable(
    fake_cli: tuple[list[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    command, _ = fake_cli
    monkeypatch.setenv("FAKE_MODE", "sleep")
    with pytest.raises(LLMError, match="did not answer within") as info:
        ClaudeCLIClient(MODEL, command, timeout_s=1).complete("s", "u", max_tokens=10)
    assert info.value.retryable is True


def test_a_missing_executable(tmp_path: Path) -> None:
    with pytest.raises(LLMError, match="not found"):
        ClaudeCLIClient(MODEL, [str(tmp_path / "no-such-claude.exe")]).complete("s", "u", max_tokens=10)


@pytest.mark.parametrize(
    ("model", "command", "timeout", "message"),
    [(" ", ["claude"], 1.0, "model"), (MODEL, [], 1.0, "command"), (MODEL, ["claude"], 0.0, "timeout_s")],
)
def test_constructor_checks(model: str, command: list[str], timeout: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ClaudeCLIClient(model, command, timeout_s=timeout)


def _executable(folder: Path, stem: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (f"{stem}.exe" if os.name == "nt" else stem)
    path.write_bytes(b"")
    path.chmod(0o755)
    return path


def test_find_uses_the_explicit_variable(tmp_path: Path) -> None:
    exe = _executable(tmp_path / "tools", "my-claude")
    assert find_claude_cli({ENV_CLAUDE_CLI: str(exe), "PATH": ""}) == [str(exe)]
    assert find_claude_cli({ENV_CLAUDE_CLI: str(tmp_path / "missing.exe"), "PATH": ""}) is None


def test_find_looks_on_path(tmp_path: Path) -> None:
    exe = _executable(tmp_path / "bin", "claude")
    found = find_claude_cli({"PATH": str(tmp_path / "bin")})
    assert found is not None
    assert Path(found[0]).resolve() == exe.resolve()


def test_find_returns_none_when_missing(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert find_claude_cli({"PATH": str(tmp_path / "empty")}) is None


@pytest.mark.skipif(os.name != "nt", reason="the npm claude.cmd shim exists on Windows only")
def test_find_skips_the_npm_shim_for_the_native_exe(tmp_path: Path) -> None:
    npm = tmp_path / "npm"
    npm.mkdir()
    (npm / "claude.cmd").write_text("@echo off\r\n", encoding="utf-8")
    native = npm / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"")
    assert find_claude_cli({"PATH": str(npm)}) == [str(native)]
