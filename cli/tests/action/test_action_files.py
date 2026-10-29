"""Checks of the GitHub Action definition, its self-test workflow and its README."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
ACTION = ROOT / "action" / "action.yml"
WORKFLOW = ROOT / ".github" / "workflows" / "action-selftest.yml"
README = ROOT / "action" / "README.md"
INPUTS = {"base", "comment", "sarif", "fail-on", "model-version", "github-token"}
OUTPUTS = {"model-source", "model-version", "highest-band", "comment-id", "report-json"}
PINNED = re.compile(r"^[\w.-]+/[\w./-]+@(v\d+|[0-9a-f]{40})$")


def load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def steps() -> list[dict[str, Any]]:
    return load(ACTION)["runs"]["steps"]


def test_action_inputs_and_outputs() -> None:
    action = load(ACTION)
    assert action["runs"]["using"] == "composite"
    inputs = action["inputs"]
    assert set(inputs) == INPUTS
    assert inputs["base"]["required"] is True and "default" not in inputs["base"]
    defaults = {name: inputs[name].get("default") for name in INPUTS - {"base"}}
    assert defaults == {
        "comment": "true",
        "sarif": "",
        "fail-on": "none",
        "model-version": "",
        "github-token": "${{ github.token }}",
    }
    for name, spec in inputs.items():
        assert "${{" not in spec["description"], name
    assert set(action["outputs"]) == OUTPUTS


def test_run_steps_are_injection_safe() -> None:
    runs = [step for step in steps() if "run" in step]
    assert runs
    for step in runs:
        assert step["shell"] == "bash", step["name"]
        assert "${{" not in step["run"], step["name"]


def test_step_references_exist() -> None:
    ids = {step["id"] for step in steps() if "id" in step}
    text = ACTION.read_text(encoding="utf-8")
    referenced = set(re.findall(r"steps\.([A-Za-z0-9_-]+)\.", text))
    assert referenced and referenced <= ids
    for name, spec in load(ACTION)["outputs"].items():
        match = re.fullmatch(r"\$\{\{ steps\.([a-z]+)\.outputs\.[a-z-]+ \}\}", spec["value"])
        assert match and match[1] in ids, name


def test_sarif_upload_never_fails_the_job() -> None:
    (upload,) = [
        s for s in steps() if str(s.get("uses", "")).startswith("github/codeql-action/upload-sarif@")
    ]
    assert upload["continue-on-error"] is True and "inputs.sarif" in upload["if"]
    names = [s["name"] for s in steps()]
    assert names[-1] == "Comment and apply fail-on" and names.index(upload["name"]) < len(names) - 1


def test_actions_are_pinned_and_the_install_is_locked() -> None:
    uses = [str(s["uses"]) for s in steps() if "uses" in s]
    assert uses
    for ref in uses:
        assert PINNED.fullmatch(ref), ref
    (install,) = [s for s in steps() if s["name"] == "Install the Entropiq CLI"]
    assert "uv sync --frozen" in install["run"] and "--package tdp-cli" in install["run"]


def test_selftest_workflow() -> None:
    workflow = load(WORKFLOW)
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"pull_request"}
    assert triggers["pull_request"]["paths"] == ["action/**", ".github/workflows/action-selftest.yml"]
    assert workflow["permissions"]["pull-requests"] == "write"
    assert workflow["permissions"]["contents"] == "read"
    (job,) = workflow["jobs"].values()
    job_steps = job["steps"]
    assert [s.get("uses") for s in job_steps].count("./action") == 2
    (checkout,) = [s for s in job_steps if str(s.get("uses", "")).startswith("actions/checkout@")]
    assert checkout["with"]["fetch-depth"] == 0
    for step in job_steps:
        if "uses" in step and step["uses"] != "./action":
            assert PINNED.fullmatch(step["uses"]), step["uses"]


def test_readme_documents_inputs_outputs_and_marker() -> None:
    text = README.read_text(encoding="utf-8")
    for name in INPUTS | OUTPUTS:
        assert f"`{name}`" in text, name
    assert "<!-- tdp-report -->" in text


@pytest.mark.parametrize(
    "path",
    [ACTION, WORKFLOW, README, ROOT / "action" / "run.py", ROOT / "handoff" / "P7.3.md"],
    ids=lambda p: p.name,
)
def test_no_long_dashes(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} is not written yet")
    text = path.read_text(encoding="utf-8")
    assert chr(0x2013) not in text and chr(0x2014) not in text
