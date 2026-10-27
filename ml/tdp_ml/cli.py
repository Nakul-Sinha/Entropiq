"""Entry point of the ``tdp-ml`` command (offline dataset, training and evaluation pipeline).

Commands live in :mod:`tdp_ml.commands`; each module there exposes ``register(app)``. Modules are loaded in
alphabetical order, so parallel work can add commands without touching this file.
"""

from __future__ import annotations

import importlib
import pkgutil

import typer

import tdp_ml.commands as commands_pkg

app = typer.Typer(help="Entropiq offline pipeline: dataset, training, evaluation.", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """Entropiq offline pipeline."""


def load_commands(target: typer.Typer = app) -> list[str]:
    """Import every module in ``tdp_ml.commands`` and call its ``register`` hook. Returns module names."""
    loaded: list[str] = []
    for info in sorted(pkgutil.iter_modules(commands_pkg.__path__), key=lambda m: m.name):
        module = importlib.import_module(f"{commands_pkg.__name__}.{info.name}")
        register = getattr(module, "register", None)
        if callable(register):
            register(target)
            loaded.append(info.name)
    return loaded


def main() -> None:
    load_commands()
    app()


if __name__ == "__main__":
    main()
