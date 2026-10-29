"""``/tree`` (plus ``tdp tree``): the folder hierarchy of the ranked files.

Each folder shows how many ranked files it holds, their lines of code and the highest band among them, and
is coloured by that band (the band word is always printed too). ``--files`` adds the files with their rank,
``--depth`` limits the levels and a folder argument shows only that part of the tree.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

import typer
from rich.console import Group
from rich.text import Text
from rich.tree import Tree

from tdp_cli.args import parse_args
from tdp_cli.commands._views import (
    NO_FILES,
    RankedFile,
    file_slocs,
    folder_suggestions,
    json_header,
    normalise_path,
    parse_count,
    ranked_files,
)
from tdp_cli.completion import options_completer
from tdp_cli.errors import CliError
from tdp_cli.options import (
    DevModelOption,
    FreshOption,
    ModelDirOption,
    RefOption,
    TargetArgument,
    cli_analysis,
    cli_guard,
    cli_session,
)
from tdp_cli.registry import Command, CommandRegistry, Suggestion
from tdp_cli.theme import UNICODE_GLYPHS, Glyphs, band_style, band_text, clean

if TYPE_CHECKING:
    from tdp_cli.session import Session
    from tdp_cli.store import LoadedAnalysis
    from tdp_core.contracts import AnalysisResult

TREE_SCHEMA: Final = "entropiq.tree.v1"
FOLDERS_ONLY_NOTE: Final = "Folders only; add --files to list the files with their rank."
_OPTIONS = options_completer("--depth", "--files")


@dataclass(slots=True)
class Folder:
    """One folder of the tree. ``path`` is ``""`` for the root, else ``"a/b/"``.

    ``n_files`` and ``sloc`` add up every ranked file below the folder (subfolders included) and ``top`` is
    the one with the lowest rank (the highest priority), None only for an empty root.
    """

    name: str
    path: str
    folders: dict[str, Folder] = field(default_factory=dict)
    files: list[RankedFile] = field(default_factory=list)
    n_files: int = 0
    sloc: int = 0
    top: RankedFile | None = None


def build_tree(result: AnalysisResult, folder: str = "") -> Folder:
    """The folder tree of ``result``'s ranked files, or of the part below ``folder`` (``""`` is the root).

    ``folder`` matches whole path components (``app`` matches ``app/core.py``, not ``apps/x.py``). Raises
    :class:`CliError` (exit code 2) when ``folder`` is a ranked file or holds no ranked file.
    """
    prefix = normalise_path(folder).rstrip("/")
    rows = ranked_files(result)
    slocs = file_slocs(result)
    if prefix:
        if any(item.path == prefix for item in rows):
            raise CliError(
                f"{clean(prefix)} is a file", exit_code=2, hint=f"Use /file {clean(prefix)} to see it."
            )
        rows = [item for item in rows if item.path.startswith(prefix + "/")]
        if not rows:
            raise CliError(
                f"No ranked files under {clean(prefix)}/", exit_code=2, hint="Type /tree to see every folder."
            )
    root = Folder(name=prefix.rsplit("/", 1)[-1], path=f"{prefix}/" if prefix else "")
    for item in rows:
        parts = (item.path[len(prefix) + 1 :] if prefix else item.path).split("/")
        chain = [root]
        node = root
        for part in parts[:-1]:
            child = node.folders.get(part)
            if child is None:
                child = Folder(name=part, path=f"{node.path}{part}/")
                node.folders[part] = child
            node = child
            chain.append(node)
        node.files.append(item)
        for owner in chain:
            owner.n_files += 1
            owner.sloc += slocs.get(item.path, 0)
            if owner.top is None or item.rank < owner.top.rank:
                owner.top = item
    return root


def _plural(count: int, singular: str) -> str:
    return f"{count:,} {singular if count == 1 else singular + 's'}"


def _stats(folder: Folder, glyphs: Glyphs) -> Text:
    if folder.top is None:
        return Text("  no ranked files", style="dim")
    sep = f" {glyphs.sep} "
    stats = Text(
        f"  {_plural(folder.n_files, 'file')}{sep}{_plural(folder.sloc, 'line')}{sep}highest ", style="dim"
    )
    stats.append_text(band_text(folder.top.score.band, folder.top.score.priority, glyphs=glyphs))
    return stats


def _file_leaf(item: RankedFile, name: str, sloc: int, glyphs: Glyphs) -> Text:
    leaf = Text(clean(name), style=band_style(item.score.band))
    leaf.append(f"  {_plural(sloc, 'line')}  ", style="dim")
    leaf.append_text(band_text(item.score.band, item.score.priority, glyphs=glyphs))
    leaf.append(f"  #{item.rank}", style="dim")
    return leaf


def _file_name(item: RankedFile) -> str:
    return item.path.rsplit("/", 1)[-1]


def _fill(
    node: Tree,
    folder: Folder,
    *,
    level: int,
    slocs: Mapping[str, int],
    depth: int,
    files: bool,
    glyphs: Glyphs,
) -> None:
    if depth and level >= depth:
        return
    for name in sorted(folder.folders):
        child = folder.folders[name]
        label = Text(clean(name) + "/", style=band_style(child.top.score.band) if child.top else "")
        label.append_text(_stats(child, glyphs))
        _fill(node.add(label), child, level=level + 1, slocs=slocs, depth=depth, files=files, glyphs=glyphs)
    if files:
        for item in sorted(folder.files, key=_file_name):
            node.add(_file_leaf(item, _file_name(item), slocs.get(item.path, 0), glyphs))


def tree_view(
    root: Folder,
    *,
    title: str,
    slocs: Mapping[str, int],
    depth: int = 0,
    files: bool = False,
    glyphs: Glyphs = UNICODE_GLYPHS,
) -> Tree:
    """``root`` as a rich tree titled ``title``: subfolders by name, then (with ``files``) files by name.

    The root is level 0; a folder's children are shown when ``depth`` is 0 (no limit) or its level is below
    ``depth``. ``slocs`` gives each file's lines of code.
    """
    label = Text(clean(title), style="bold")
    label.append_text(_stats(root, glyphs))
    tree = Tree(label, guide_style="dim")
    _fill(tree, root, level=0, slocs=slocs, depth=depth, files=files, glyphs=glyphs)
    return tree


def _node_json(
    folder: Folder, *, level: int, slocs: Mapping[str, int], depth: int, files: bool
) -> dict[str, Any]:
    collapsed = bool(depth) and level >= depth
    children: list[dict[str, Any]] = []
    if not collapsed:
        children = [
            _node_json(folder.folders[name], level=level + 1, slocs=slocs, depth=depth, files=files)
            for name in sorted(folder.folders)
        ]
        if files:
            children += [
                {
                    "name": _file_name(item),
                    "path": item.path,
                    "rank": item.rank,
                    "sloc": slocs.get(item.path, 0),
                    "priority": item.score.priority,
                    "band": item.score.band,
                }
                for item in sorted(folder.files, key=_file_name)
            ]
    top = folder.top
    return {
        "name": folder.name,
        "path": folder.path,
        "files": folder.n_files,
        "sloc": folder.sloc,
        "priority": top.score.priority if top else None,
        "band": top.score.band if top else None,
        "top_file": top.path if top else None,
        "collapsed": collapsed,
        "children": children,
    }


def tree_json(
    root: Folder, *, slocs: Mapping[str, int], depth: int = 0, files: bool = False
) -> dict[str, Any]:
    """The ``root`` node as JSON: folders with ``files``, ``sloc``, ``priority``, ``band``, ``top_file``.

    Folders below the ``depth`` limit have ``"collapsed": true`` and no children; file nodes (``rank``,
    ``sloc``, ``priority``, ``band``) appear only with ``files``. Priority, band and top_file are null for
    an empty root.
    """
    return _node_json(root, level=0, slocs=slocs, depth=depth, files=files)


def tree_handler(session: Session, args: Sequence[str]) -> None:
    """``/tree [FOLDER] [--depth N] [--files]``: the folder tree of the loaded analysis."""
    parsed = parse_args(args, values=("--depth",), flags=("--files",), max_positionals=1)
    raw_depth = parsed.value("--depth")
    depth = 0 if raw_depth is None else parse_count(raw_depth, "--depth")
    files = parsed.flag("--files")
    loaded = session.require_analysis()
    folder = parsed.positionals[0] if parsed.positionals else ""
    _print_tree(session, loaded, folder, depth=depth, files=files)


def _print_tree(session: Session, loaded: LoadedAnalysis, folder: str, *, depth: int, files: bool) -> None:
    result = loaded.result
    if not result.files:
        session.console.print(Text(NO_FILES, style="dim"))
        return
    root = build_tree(result, folder)
    title = root.path or loaded.meta.repo
    view = tree_view(
        root, title=title, slocs=file_slocs(result), depth=depth, files=files, glyphs=session.glyphs
    )
    if files:
        session.console.print(view)
    else:
        session.console.print(Group(view, Text(FOLDERS_ONLY_NOTE, style="dim")))


def complete_tree(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    """Completion for ``/tree``: option names, and folders of the loaded analysis for the first argument."""
    if args and args[-1] == "--depth":
        return
    if word.startswith("-"):
        yield from _OPTIONS(session, args, word)
        return
    positionals = 0
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg == "--depth":
            skip = True
        elif not arg.startswith("-"):
            positionals += 1
    if positionals == 0:
        yield from folder_suggestions(session, word)


def tree_cli(
    ctx: typer.Context,
    target: TargetArgument = ".",
    path: Annotated[
        str | None, typer.Option("--path", help="Show only this folder.", show_default=False)
    ] = None,
    ref: RefOption = None,
    depth: Annotated[int, typer.Option("--depth", min=0, help="Folder levels to show (0 = all).")] = 0,
    files: Annotated[bool, typer.Option("--files", help="List the files too.")] = False,
    output_format: Annotated[
        Literal["table", "json"], typer.Option("--format", "-f", help="Output format.")
    ] = "table",
    fresh: FreshOption = False,
    model_dir: ModelDirOption = None,
    dev_model: DevModelOption = False,
) -> None:
    """Show the folder tree with lines of code and the highest band."""
    session = cli_session(ctx, model_dir=model_dir, dev_model=dev_model)
    loaded = cli_analysis(session, target, ref=ref, fresh=fresh)
    with cli_guard(session.err_console):
        if output_format == "json":
            root = build_tree(loaded.result, path or "")
            node = tree_json(root, slocs=file_slocs(loaded.result), depth=depth, files=files)
            sys.stdout.write(json.dumps({**json_header(loaded, TREE_SCHEMA), "root": node}, indent=2) + "\n")
            return
        _print_tree(session, loaded, path or "", depth=depth, files=files)


def register(registry: CommandRegistry) -> None:
    """Register ``/tree`` and ``tdp tree``."""
    registry.add(
        Command(
            name="tree",
            group="Explore",
            help="Show the folder tree with lines of code and the highest band",
            usage="/tree [FOLDER] [--depth N] [--files]",
            details=(
                "Shows every folder that holds ranked files with how many it holds, their lines of code and "
                "the highest band among them; each folder is coloured by that band. Name a folder to see "
                "only that part, add --depth N to show N levels (0 = all) and --files to list the files "
                "with their priority and rank (open one with /file and the rank)."
            ),
            handler=tree_handler,
            complete=complete_tree,
            needs_analysis=True,
            cli=tree_cli,
        )
    )
