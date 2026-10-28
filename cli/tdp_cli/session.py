"""The state one run of the app works with: consoles, home folder, settings, model, registry, analysis."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rich.console import Console

from tdp_cli.errors import CliError
from tdp_cli.home import EntropiqHome, Prefs, load_prefs, save_prefs
from tdp_cli.model import ModelChoice, load_model, resolve_model
from tdp_cli.registry import CommandRegistry, load_registry
from tdp_cli.store import AnalysisStore
from tdp_cli.theme import Glyphs, activity, console_glyphs

if TYPE_CHECKING:
    from tdp_cli.store import LoadedAnalysis
    from tdp_core.model.bundle import Bundle

NO_ANALYSIS: str = "No analysis is loaded. Run /analyze <path or URL> or /open <number> first."


class Session:
    """Everything a command handler needs.

    ``console`` receives the command's output and ``err_console`` progress and notices (the same console in
    the interactive shell, stderr for the non-interactive commands). ``current`` is the loaded analysis.
    """

    def __init__(
        self,
        *,
        console: Console,
        err_console: Console,
        home: EntropiqHome,
        prefs: Prefs,
        model: ModelChoice,
        registry: CommandRegistry,
        store: AnalysisStore,
        interactive: bool,
    ) -> None:
        self.console = console
        self.err_console = err_console
        self.home = home
        self.prefs = prefs
        self.model = model
        self.registry = registry
        self.store = store
        self.interactive = interactive
        self.current: LoadedAnalysis | None = None
        self._bundle: Bundle | None = None

    @classmethod
    def create(
        cls,
        *,
        console: Console | None = None,
        err_console: Console | None = None,
        model_dir: Path | None = None,
        dev_model: bool = False,
        interactive: bool = True,
        registry: CommandRegistry | None = None,
        home: EntropiqHome | None = None,
        environ: Mapping[str, str] | None = None,
        repo_root: Path | Literal["auto"] | None = "auto",
    ) -> Session:
        """Build a session: create the home folder, read settings, pick the model (JSON only), load commands.

        Raises :class:`CliError` for an unusable home folder or model choice.
        """
        home = home or EntropiqHome.from_env(environ)
        home.ensure()
        out = console or Console()
        err = err_console or (out if interactive else Console(stderr=True))
        model = resolve_model(model_dir, dev=dev_model, home=home, environ=environ, repo_root=repo_root)
        return cls(
            console=out,
            err_console=err,
            home=home,
            prefs=load_prefs(home.settings_file),
            model=model,
            registry=registry if registry is not None else load_registry(),
            store=AnalysisStore(home.analyses),
            interactive=interactive,
        )

    @property
    def glyphs(self) -> Glyphs:
        """Glyphs the output console can show."""
        return console_glyphs(self.console)

    def bundle(self) -> Bundle:
        """The loaded model bundle (loaded or built once, with a spinner)."""
        if self._bundle is None:
            label = (
                "Building the demo model (first run only)"
                if self.model.source == "dev" and self.model.version is None
                else f"Loading model {self.model.version}"
            )
            with activity(self.err_console, label):
                bundle = load_model(self.model)
            self._bundle = bundle
            self.model = dataclasses.replace(self.model, version=bundle.version)
        return self._bundle

    def require_analysis(self) -> LoadedAnalysis:
        """The loaded analysis. Raises :class:`CliError` when there is none."""
        if self.current is None:
            raise CliError(NO_ANALYSIS)
        return self.current

    def set_prefs(self, prefs: Prefs) -> None:
        """Save ``prefs`` to the settings file and use them from now on."""
        save_prefs(self.home.settings_file, prefs)
        self.prefs = prefs
