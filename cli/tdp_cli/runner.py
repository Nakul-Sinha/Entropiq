"""Run an analysis with live progress, reuse saved ones, and save the result."""

from __future__ import annotations

import time
from types import TracebackType
from typing import TYPE_CHECKING, Final

from rich.console import Console
from rich.text import Text

from tdp_cli.errors import CliError
from tdp_cli.repos import resolve_commit, resolve_target
from tdp_cli.theme import CYAN, activity, clean, console_glyphs, notice

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

    from tdp_cli.session import Session
    from tdp_cli.store import LoadedAnalysis

STAGE_LABELS: Final[dict[str, str]] = {
    "mine": "Mining history",
    "parse": "Parsing source files",
    "graphs": "Building the import and co-change graphs",
    "features": "Computing file features",
    "score": "Scoring files",
    "explain": "Explaining the scores",
    "plan": "Planning fixes",
}
"""What the progress line says while each stage of ``tdp_core.analyze`` runs."""

_ORDER: Final = tuple(STAGE_LABELS)


class AnalysisProgress:
    """A ``tdp_core.analyze`` progress callback that draws a live bar (terminals) or plain lines.

    Use it as a context manager around the analysis. Each finished stage prints a permanent line with its
    message and duration; the bar shows the overall percentage, the running stage and the elapsed time.
    """

    def __init__(self, console: Console, *, title: str) -> None:
        self.console = console
        self.title = title
        self._progress: Progress | None = None
        self._task: TaskID | None = None
        self._started = 0.0
        self._last = 0.0

    def __enter__(self) -> AnalysisProgress:
        self._started = self._last = time.perf_counter()
        self.console.print(Text(clean(self.title), style="bold"))
        if self.console.is_terminal:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            self._progress = Progress(
                SpinnerColumn(style=CYAN),
                TextColumn("{task.description}", markup=False),
                BarColumn(complete_style=CYAN, finished_style=CYAN),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            )
            self._progress.start()
            self._task = self._progress.add_task(STAGE_LABELS["mine"], total=100)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        if exc_type is None:
            total = time.perf_counter() - self._started
            self.console.print(Text(f"Analysis finished in {total:.1f} s", style="dim"))

    def __call__(self, stage: str, pct: int, message: str) -> None:
        """Record that ``stage`` finished at ``pct`` percent with ``message``."""
        try:
            now = time.perf_counter()
            took = now - self._last
            self._last = now
            text = clean(message)
            if self._progress is not None and self._task is not None:
                check = console_glyphs(self.console).check
                self._progress.console.print(Text(f"{check} {text} ({took:.1f} s)", style="green"))
                index = _ORDER.index(stage) if stage in _ORDER else len(_ORDER) - 1
                following = STAGE_LABELS[_ORDER[index + 1]] if index + 1 < len(_ORDER) else "Finishing"
                self._progress.update(self._task, completed=pct, description=following)
            else:
                self.console.print(Text(f"[{pct:3d}%] {text} ({took:.1f} s)"))
        except KeyboardInterrupt:
            raise
        except Exception:  # a display problem must never stop the analysis
            return


def run_analysis(
    session: Session, target_text: str | None, *, ref: str | None = None, fresh: bool = False
) -> tuple[LoadedAnalysis, bool]:
    """Analyse a folder or URL (or reuse the saved analysis of the same commit and model).

    Returns ``(loaded, reused)`` and makes ``loaded`` the session's current analysis. Progress and notices
    go to ``session.err_console``. Raises :class:`CliError` for bad input (exit code 2) and git or disk
    failures (exit code 1); ``KeyboardInterrupt`` propagates and nothing is saved.
    """
    from tdp_cli.store import LoadedAnalysis, make_meta

    console = session.err_console
    target = resolve_target(target_text, ref, home=session.home, console=console)
    sha = resolve_commit(target)
    if not fresh and session.model.version is not None:
        meta = session.store.get(target.slug, sha)
        if meta is not None and meta.model_version == session.model.version:
            with activity(console, "Loading the saved analysis"):
                loaded = session.store.load(meta)
            session.current = loaded
            return loaded, True

    bundle = session.bundle()
    from tdp_core.analyze import analyze
    from tdp_core.mining.git import GitError, TransientGitError

    title = f"Analysing {target.display} @ {sha[:7]}"
    try:
        with AnalysisProgress(console, title=title) as progress:
            result = analyze(
                target.repo_dir,
                sha,
                bundle,
                progress=progress,
                budget_hours=session.prefs.budget_hours,
                jobs=session.prefs.jobs,
                szz_cache_dir=session.home.szz_cache(target.slug),
                cache=None,
            )
    except TransientGitError as exc:
        raise CliError(
            "git timed out or the network failed", hint=clean(exc.stderr.strip())[:300] or None
        ) from exc
    except GitError as exc:
        detail = next((clean(line) for line in exc.stderr.splitlines() if line.strip()), "")
        raise CliError(f"git failed: {detail or clean(str(exc))}"[:400]) from exc
    except FileNotFoundError as exc:
        raise CliError(f"A file the analysis needs is missing: {clean(str(exc))}") from exc
    except ValueError as exc:
        raise CliError(f"Cannot analyse this commit: {clean(str(exc))}", exit_code=2) from exc

    loaded = LoadedAnalysis(make_meta(target, result, demo_model=session.model.demo), result)
    try:
        session.store.save(loaded)
    except (OSError, ValueError, TypeError) as exc:
        notice(console, f"Could not save the analysis: {clean(str(exc))}")
    session.current = loaded
    return loaded, False
