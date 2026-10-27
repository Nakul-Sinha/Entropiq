"""``tdp-ml sample-bugfix`` and ``tdp-ml score-bugfix``: measure the bug-fix detector by hand labelling."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import pyarrow.parquet as pq
import typer

from tdp_core.labeling.bugfix import classify_bugfix
from tdp_core.labeling.github_issues import GitHubError, fetch_issues
from tdp_core.mining import mine_history
from tdp_core.paths import data_dir, results_dir
from tdp_ml.tools.github import github_token, split_slug
from tdp_ml.tools.sample_bugfix import STRATA, key_path, sample_bugfix, write_sample
from tdp_ml.tools.score_bugfix import read_key, read_labels, score_bugfix


def _fmt(value: float | None) -> str:
    return "n/a" if value is None or value != value else f"{value:.3f}"


def sample_bugfix_command(
    repo: Annotated[str, typer.Option("--repo", help="Repository slug, for example psf__requests.")],
    size: Annotated[int, typer.Option("--size", min=1, help="Commits to sample.")] = 200,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    repo_dir: Annotated[
        Path | None, typer.Option("--repo-dir", help="Git repository (default: the data folder's mirror).")
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Sample CSV (default: results/bugfix_eval/<slug>.sample.csv)."),
    ] = None,
    issues: Annotated[
        bool, typer.Option("--issues/--no-issues", help="Link GitHub bug issues (cached under data/issues).")
    ] = True,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing sample (and its labels).")
    ] = False,
) -> None:
    """Write a stratified, blind sample of commits for hand-labelling bug fixes, plus its key file."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        owner, name = split_slug(repo)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--repo") from exc
    git_dir = repo_dir if repo_dir is not None else data_dir() / "mirrors" / f"{repo}.git"
    if not git_dir.exists():
        typer.echo(f"Repository not found: {git_dir}", err=True)
        raise typer.Exit(2)
    out_csv = out if out is not None else results_dir() / "bugfix_eval" / f"{repo}.sample.csv"
    if out_csv.exists() and not force:
        typer.echo(f"{out_csv} exists (it may hold hand labels); pass --force to overwrite it", err=True)
        raise typer.Exit(1)

    hist = mine_history(git_dir)
    issue_table = None
    if issues:
        cache = data_dir() / "issues" / f"{repo}.parquet"
        token = None if cache.exists() else github_token()
        try:
            issue_table = fetch_issues(owner, name, token, cache_path=cache)
        except GitHubError as exc:
            typer.echo(f"Could not fetch issues for {owner}/{name}: {exc}", err=True)
            raise typer.Exit(1) from exc
    bugfix = classify_bugfix(hist.commits, hist.file_changes, issue_table, repo=f"{owner}/{name}")
    sample = sample_bugfix(
        hist.commits,
        hist.file_changes,
        bugfix,
        n=size,
        seed=seed,
        url_prefix=f"https://github.com/{owner}/{name}",
    )
    try:
        sample_csv, key_csv = write_sample(sample, out_csv, force=force)
    except FileExistsError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    key = sample.key
    for stratum in STRATA:
        part = key[key["stratum"] == stratum]
        population = int(part["stratum_size"].iloc[0]) if len(part) else 0
        typer.echo(f"{stratum:8s} population {population:6d}  sampled {len(part):4d}")
    typer.echo(f"Sample: {sample_csv}")
    typer.echo(f"Key:    {key_csv}")


def score_bugfix_command(
    sample: Annotated[Path, typer.Argument(help="Labelled sample CSV.", exists=True, dir_okay=False)],
    key: Annotated[
        Path | None, typer.Option("--key", help="Key CSV (default: <sample stem>.key.csv).")
    ] = None,
    bugfix: Annotated[
        Path | None,
        typer.Option("--bugfix", help="Parquet with sha and is_bugfix to re-score instead of the key."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="JSON output (default: <sample stem>.score.json).")
    ] = None,
) -> None:
    """Weighted precision, recall and F1 of the detector from the hand labels in SAMPLE."""
    key_csv = key if key is not None else key_path(sample)
    if not key_csv.exists():
        typer.echo(f"Key file not found: {key_csv}", err=True)
        raise typer.Exit(2)
    predicted = None
    if bugfix is not None:
        table = pq.read_table(bugfix, columns=["sha", "is_bugfix"])
        predicted = dict(
            zip(table.column("sha").to_pylist(), table.column("is_bugfix").to_pylist(), strict=True)
        )
    try:
        score = score_bugfix(read_labels(sample), read_key(key_csv), predicted)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    out_json = out if out is not None else sample.with_name(f"{sample.stem}.score.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(score.to_dict(), indent=2) + "\n", encoding="utf-8")
    typer.echo(
        f"labelled {score.labelled}: precision {_fmt(score.precision)}, recall {_fmt(score.recall)}, "
        f"F1 {_fmt(score.f1)}"
    )
    for s in score.strata:
        typer.echo(
            f"{s.stratum:8s} population {s.population:6d}  labelled {s.labelled:4d}  bug fixes {s.bugs:4d}  "
            f"rate {_fmt(s.rate)} [{_fmt(s.ci_low)}, {_fmt(s.ci_high)}]"
        )
    typer.echo(f"Written: {out_json}")


def register(app: typer.Typer) -> None:
    """Add ``sample-bugfix`` and ``score-bugfix`` to ``app``."""
    app.command("sample-bugfix")(sample_bugfix_command)
    app.command("score-bugfix")(score_bugfix_command)
