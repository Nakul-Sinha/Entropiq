"""``/graph`` (plus ``tdp graph``): the import graph of the ranked files.

Without a file it shows the most central files (how many files reach each one through imports, and its
PageRank), the import cycles and the hidden couplings (files that change together without importing each
other). With a file it shows a tree of what imports it and what it imports. Only ranked, non-test files take
part, like ``AnalysisResult.graph``; tests that import a file are listed separately.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

import typer
from rich.console import Group, RenderableType
from rich.text import Text
from rich.tree import Tree

from tdp_cli.args import parse_args
from tdp_cli.commands._views import (
    NO_FILES,
    RankedFile,
    file_label,
    find_file,
    json_header,
    limited,
    parse_count,
    ranked_by_path,
    ranked_files,
    section,
)
from tdp_cli.completion import complete_analysis_paths, options_completer
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
from tdp_cli.theme import UNICODE_GLYPHS, Glyphs, band_text, clean, format_risk, make_table

if TYPE_CHECKING:
    from tdp_cli.session import Session
    from tdp_cli.store import LoadedAnalysis
    from tdp_core.contracts import AnalysisResult, GraphEdge

GRAPH_SCHEMA: Final = "entropiq.graph.v1"
GRAPH_FILE_SCHEMA: Final = "entropiq.graph.file.v1"
DEFAULT_TOP: Final = 10
DEFAULT_DEPTH: Final = 2
MAX_DEPTH: Final = 6
MAX_CHILDREN: Final = 12
MAX_CYCLE_MEMBERS: Final = 8
GRAPH_OPTIONS: Final = ("--depth", "--top")

DEPENDENTS_NOTE: Final = "Dependents counts every ranked file that reaches this one through imports."
COUPLING_NOTE: Final = (
    "Confidence is the share of the first file's commits that also changed the second; "
    "neither imports the other."
)
_OPTIONS = options_completer(*GRAPH_OPTIONS)


@dataclass(frozen=True, slots=True)
class ImportGraph:
    """Import edges between ranked files: ``imports[a]`` are the files ``a`` imports, ``importers[a]`` the
    files importing ``a``. Every ranked path is a key of both; neighbours are in rank order."""

    imports: dict[str, tuple[str, ...]]
    importers: dict[str, tuple[str, ...]]
    n_edges: int


@dataclass(frozen=True, slots=True)
class CentralFile:
    """A file other files import: transitive dependents, direct importers and imports, and its PageRank."""

    item: RankedFile
    dependents: int
    importers: int
    imports: int
    pagerank: float


def _dim(text: str) -> Text:
    return Text(text, style="dim")


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def import_graph(result: AnalysisResult) -> ImportGraph:
    """The import edges of ``result.graph`` between ranked files (self loops and duplicates dropped)."""
    rank = {f.path: i for i, f in enumerate(result.files)}
    imports: dict[str, set[str]] = {path: set() for path in rank}
    importers: dict[str, set[str]] = {path: set() for path in rank}
    for edge in result.graph.edges:
        if edge.kind != "import" or edge.source == edge.target:
            continue
        if edge.source not in rank or edge.target not in rank:
            continue
        imports[edge.source].add(edge.target)
        importers[edge.target].add(edge.source)

    def ordered(paths: set[str]) -> tuple[str, ...]:
        return tuple(sorted(paths, key=rank.__getitem__))

    return ImportGraph(
        imports={p: ordered(v) for p, v in imports.items()},
        importers={p: ordered(v) for p, v in importers.items()},
        n_edges=sum(len(v) for v in imports.values()),
    )


def reachable(start: str, neighbours: Mapping[str, Sequence[str]]) -> list[str]:
    """Every path reachable from ``start`` along ``neighbours`` (``start`` excluded), sorted by path."""
    seen = {start}
    queue = deque([start])
    while queue:
        for nxt in neighbours.get(queue.popleft(), ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    seen.discard(start)
    return sorted(seen)


def import_cycles(graph: ImportGraph) -> list[tuple[str, ...]]:
    """The import cycles: strongly connected groups of two or more files, members sorted by path.

    The list is sorted by size (largest first), then by members.
    """
    import networkx as nx

    digraph: nx.DiGraph[str] = nx.DiGraph()
    digraph.add_nodes_from(graph.imports)
    digraph.add_edges_from((a, b) for a, targets in graph.imports.items() for b in targets)
    groups = [tuple(sorted(c)) for c in nx.strongly_connected_components(digraph) if len(c) >= 2]
    return sorted(groups, key=lambda g: (-len(g), g))


def _pagerank(result: AnalysisResult, path: str) -> float:
    features = result.features
    if "pagerank" not in features.columns or path not in features.index:
        return 0.0
    value: Any = features.at[path, "pagerank"]
    return float(value)


def central_files(result: AnalysisResult, graph: ImportGraph) -> list[CentralFile]:
    """Files imported by at least one ranked file, most dependents first (then PageRank, then rank).

    Dependents are transitive (``GraphNode.dependents``, counted from the edges when a node is missing);
    PageRank is the ``pagerank`` feature (0.0 when absent).
    """
    nodes = {node.id: node for node in result.graph.nodes}
    out: list[CentralFile] = []
    for item in ranked_files(result):
        direct = graph.importers.get(item.path, ())
        if not direct:
            continue
        node = nodes.get(item.path)
        dependents = node.dependents if node is not None else len(reachable(item.path, graph.importers))
        out.append(
            CentralFile(
                item=item,
                dependents=dependents,
                importers=len(direct),
                imports=len(graph.imports.get(item.path, ())),
                pagerank=_pagerank(result, item.path),
            )
        )
    out.sort(key=lambda c: (-c.dependents, -c.pagerank, c.item.rank))
    return out


def hidden_couplings(result: AnalysisResult) -> list[GraphEdge]:
    """Hidden coupling edges between ranked files, strongest first (then by source and target).

    ``weight`` is the share of the source's commits that also changed the target; neither imports the other.
    """
    ranked = {f.path for f in result.files}
    edges = [
        e
        for e in result.graph.edges
        if e.kind == "hidden_coupling" and e.source in ranked and e.target in ranked and e.source != e.target
    ]
    return sorted(edges, key=lambda e: (-e.weight, e.source, e.target))


def _cut[T](rows: Sequence[T], top: int) -> list[T]:
    return list(rows[:top]) if top else list(rows)


def _showing(shown: int, total: int) -> Text | None:
    if shown >= total:
        return None
    return _dim(f"Showing {shown} of {total}; add --top 0 to see all.")


def _members(paths: Sequence[str]) -> Text:
    shown, more = limited(paths, MAX_CYCLE_MEMBERS)
    text = Text(clean(", ".join(shown)))
    if more:
        text.append(f" and {more} more", style="dim")
    return text


def graph_overview(
    loaded: LoadedAnalysis, *, top: int = DEFAULT_TOP, glyphs: Glyphs = UNICODE_GLYPHS
) -> Group:
    """The ``/graph`` screen: a summary line, then the central files, import cycles and hidden couplings.

    Each list keeps its first ``top`` rows (0 keeps every row) and says how many were left out.
    """
    result = loaded.result
    graph = import_graph(result)
    by_path = ranked_by_path(result)
    central = central_files(result, graph)
    cycles = import_cycles(graph)
    couplings = hidden_couplings(result)
    parts: list[RenderableType] = [
        _dim(
            f"{_plural(len(result.files), 'ranked file')}, {_plural(graph.n_edges, 'import edge')}, "
            f"{_plural(len(cycles), 'import cycle')}, {_plural(len(couplings), 'hidden coupling')}"
        )
    ]

    def add(title: str, body: Sequence[RenderableType | None]) -> None:
        parts.append(Text(""))
        parts.append(section(title))
        parts.extend(b for b in body if b is not None)

    if central:
        table = make_table()
        table.add_column("#", justify="right", style="dim", no_wrap=True)
        table.add_column("File", overflow="fold")
        table.add_column("Priority", no_wrap=True)
        for name in ("Dependents", "Importers", "Imports", "PageRank"):
            table.add_column(name, justify="right", no_wrap=True)
        shown_central = _cut(central, top)
        for c in shown_central:
            score = c.item.score
            table.add_row(
                str(c.item.rank),
                Text(clean(c.item.path)),
                band_text(score.band, score.priority, glyphs=glyphs),
                str(c.dependents),
                str(c.importers),
                str(c.imports),
                f"{c.pagerank:.3f}",
            )
        add("Most central files", [table, _showing(len(shown_central), len(central)), _dim(DEPENDENTS_NOTE)])
    else:
        add("Most central files", [_dim("No ranked file imports another ranked file.")])

    if cycles:
        table = make_table()
        table.add_column("Files", justify="right", no_wrap=True)
        table.add_column("Highest", no_wrap=True)
        table.add_column("Members", overflow="fold")
        shown_cycles = _cut(cycles, top)
        for group in shown_cycles:
            highest = min((by_path[p] for p in group), key=lambda r: r.rank)
            table.add_row(
                str(len(group)),
                band_text(highest.score.band, highest.score.priority, glyphs=glyphs),
                _members(group),
            )
        add("Import cycles", [table, _showing(len(shown_cycles), len(cycles))])
    else:
        add("Import cycles", [_dim("No import cycles.")])

    if couplings:
        table = make_table()
        table.add_column("File", overflow="fold")
        table.add_column("Changes with", overflow="fold")
        table.add_column("Confidence", justify="right", no_wrap=True)
        shown_couplings = _cut(couplings, top)
        for edge in shown_couplings:
            table.add_row(
                file_label(by_path[edge.source], glyphs=glyphs),
                file_label(by_path[edge.target], glyphs=glyphs),
                format_risk(edge.weight),
            )
        add(
            "Hidden couplings",
            [table, _showing(len(shown_couplings), len(couplings)), _dim(COUPLING_NOTE)],
        )
    else:
        add("Hidden couplings", [_dim("No hidden couplings found.")])
    return Group(*parts)


def _expand(
    node: Tree,
    path: str,
    neighbours: Mapping[str, Sequence[str]],
    by_path: Mapping[str, RankedFile],
    *,
    depth: int,
    level: int,
    trail: tuple[str, ...],
    expanded: set[str],
    glyphs: Glyphs,
) -> None:
    """Add the neighbours of ``path`` (at ``level``) below ``node``, depth first, up to ``depth`` levels."""
    children = neighbours.get(path, ())
    for child in children[:MAX_CHILDREN]:
        label = file_label(by_path[child], glyphs=glyphs)
        if child in trail:
            label.append("  (import cycle)", style="dim")
            node.add(label)
            continue
        if child in expanded:
            label.append("  (shown above)", style="dim")
            node.add(label)
            continue
        if level + 1 >= depth:
            hidden = [g for g in neighbours.get(child, ()) if g not in trail and g != child]
            if hidden:
                label.append(f"  (+{len(hidden)} more)", style="dim")
            node.add(label)
            continue
        expanded.add(child)
        branch = node.add(label)
        _expand(
            branch,
            child,
            neighbours,
            by_path,
            depth=depth,
            level=level + 1,
            trail=(*trail, child),
            expanded=expanded,
            glyphs=glyphs,
        )
    if len(children) > MAX_CHILDREN:
        node.add(_dim(f"and {len(children) - MAX_CHILDREN} more"))


def _couplings_of(result: AnalysisResult, path: str) -> list[tuple[str, float]]:
    """The hidden coupling partners of ``path`` in either direction, strongest first (one entry each)."""
    best: dict[str, float] = {}
    for edge in hidden_couplings(result):
        if edge.source == path:
            partner = edge.target
        elif edge.target == path:
            partner = edge.source
        else:
            continue
        best[partner] = max(best.get(partner, 0.0), edge.weight)
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


def _cycle_of(graph: ImportGraph, path: str) -> tuple[str, ...]:
    for group in import_cycles(graph):
        if path in group:
            return group
    return ()


def dependency_tree(
    loaded: LoadedAnalysis,
    item: RankedFile,
    *,
    depth: int = DEFAULT_DEPTH,
    glyphs: Glyphs = UNICODE_GLYPHS,
) -> Group:
    """The ``/graph <file>`` screen: a tree of the file's importers and imports, ``depth`` levels deep.

    Branches: files importing it, files it imports, tests importing it, and (when there are any) files that
    change together with it. A file already on the path from the root is marked ``(import cycle)``, one
    expanded earlier in the same branch ``(shown above)``, and one at the depth limit with neighbours of its
    own ``(+K more)``; each node lists at most :data:`MAX_CHILDREN` children.
    """
    result = loaded.result
    graph = import_graph(result)
    by_path = ranked_by_path(result)
    path = item.path
    tree = Tree(file_label(item, glyphs=glyphs), guide_style="dim")
    for title, neighbours in (("Imported by", graph.importers), ("Imports", graph.imports)):
        direct = neighbours.get(path, ())
        total = len(reachable(path, neighbours))
        branch = tree.add(Text(f"{title}: {len(direct)} directly, {total} in total", style="bold"))
        _expand(
            branch,
            path,
            neighbours,
            by_path,
            depth=depth,
            level=0,
            trail=(path,),
            expanded=set(),
            glyphs=glyphs,
        )
    tests = item.extra.test_importers if item.extra is not None else ()
    branch = tree.add(Text(f"Tests that import it: {len(tests)}", style="bold"))
    shown, more = limited(tests, MAX_CHILDREN)
    for test in shown:
        branch.add(Text(clean(test)))
    if more:
        branch.add(_dim(f"and {more} more"))
    partners = _couplings_of(result, path)
    if partners:
        branch = tree.add(Text(f"Changes together with: {len(partners)}", style="bold"))
        for partner, weight in partners[:MAX_CHILDREN]:
            label = file_label(by_path[partner], glyphs=glyphs)
            label.append(f"  {format_risk(weight)}", style="dim")
            branch.add(label)
        if len(partners) > MAX_CHILDREN:
            branch.add(_dim(f"and {len(partners) - MAX_CHILDREN} more"))
    parts: list[RenderableType] = [tree]
    cycle = [p for p in _cycle_of(graph, path) if p != path]
    if cycle:
        parts.append(_dim(f"In an import cycle with: {clean(', '.join(cycle))}"))
    return Group(*parts)


def graph_json(loaded: LoadedAnalysis, *, top: int = DEFAULT_TOP) -> dict[str, Any]:
    """The ``tdp graph --format json`` document (schema ``entropiq.graph.v1``).

    Lists keep their first ``top`` rows (0 = all); ``summary`` holds the totals; ``confidence`` is a
    share from 0 to 1.
    """
    result = loaded.result
    graph = import_graph(result)
    central = central_files(result, graph)
    cycles = import_cycles(graph)
    couplings = hidden_couplings(result)
    return {
        **json_header(loaded, GRAPH_SCHEMA),
        "summary": {
            "files": len(result.files),
            "import_edges": graph.n_edges,
            "cycles": len(cycles),
            "hidden_couplings": len(couplings),
        },
        "central": [
            {
                "rank": c.item.rank,
                "path": c.item.path,
                "priority": c.item.score.priority,
                "band": c.item.score.band,
                "dependents": c.dependents,
                "importers": c.importers,
                "imports": c.imports,
                "pagerank": c.pagerank,
            }
            for c in _cut(central, top)
        ],
        "cycles": [{"size": len(g), "files": list(g)} for g in _cut(cycles, top)],
        "hidden_couplings": [
            {"source": e.source, "target": e.target, "confidence": e.weight} for e in _cut(couplings, top)
        ],
    }


def file_graph_json(loaded: LoadedAnalysis, item: RankedFile) -> dict[str, Any]:
    """The ``tdp graph --file F --format json`` document (schema ``entropiq.graph.file.v1``).

    ``dependents`` and ``dependencies`` are direct (rank order); ``all_*`` are transitive (path order);
    ``cycle`` lists the members of its import cycle (itself included) or is empty.
    """
    result = loaded.result
    graph = import_graph(result)
    path = item.path
    return {
        **json_header(loaded, GRAPH_FILE_SCHEMA),
        "file": {"rank": item.rank, "path": path, "priority": item.score.priority, "band": item.score.band},
        "dependents": list(graph.importers.get(path, ())),
        "dependencies": list(graph.imports.get(path, ())),
        "all_dependents": reachable(path, graph.importers),
        "all_dependencies": reachable(path, graph.imports),
        "test_importers": list(item.extra.test_importers) if item.extra is not None else [],
        "cycle": list(_cycle_of(graph, path)),
        "hidden_couplings": [{"path": p, "confidence": w} for p, w in _couplings_of(result, path)],
    }


def graph_handler(session: Session, args: Sequence[str]) -> None:
    """``/graph [PATH or RANK] [--depth N] [--top N]``: the overview, or one file's dependency tree."""
    parsed = parse_args(args, values=GRAPH_OPTIONS, max_positionals=1)
    raw_depth, raw_top = parsed.value("--depth"), parsed.value("--top")
    depth = (
        DEFAULT_DEPTH
        if raw_depth is None
        else parse_count(raw_depth, "--depth", minimum=1, maximum=MAX_DEPTH)
    )
    top = DEFAULT_TOP if raw_top is None else parse_count(raw_top, "--top")
    loaded = session.require_analysis()
    if not loaded.result.files:
        session.console.print(_dim(NO_FILES))
        return
    if parsed.positionals:
        item = find_file(loaded.result, parsed.positionals[0])
        session.console.print(dependency_tree(loaded, item, depth=depth, glyphs=session.glyphs))
    else:
        session.console.print(graph_overview(loaded, top=top, glyphs=session.glyphs))


def complete_graph(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    """Completion for ``/graph``: option names, and the analysis' paths for the file argument."""
    if args and args[-1] in GRAPH_OPTIONS:
        return
    if word.startswith("-"):
        yield from _OPTIONS(session, args, word)
        return
    positionals = 0
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg in GRAPH_OPTIONS:
            skip = True
        elif not arg.startswith("-"):
            positionals += 1
    if positionals == 0:
        yield from complete_analysis_paths(session, args, word)


def graph_cli(
    ctx: typer.Context,
    target: TargetArgument = ".",
    file: Annotated[
        str | None,
        typer.Option(
            "--file",
            help="Show this file's dependency tree (path, unique ending or rank).",
            show_default=False,
        ),
    ] = None,
    ref: RefOption = None,
    depth: Annotated[
        int, typer.Option("--depth", min=1, max=MAX_DEPTH, help="Levels of the dependency tree.")
    ] = DEFAULT_DEPTH,
    top: Annotated[int, typer.Option("--top", min=0, help="Rows per list (0 = all).")] = DEFAULT_TOP,
    output_format: Annotated[
        Literal["table", "json"], typer.Option("--format", "-f", help="Output format.")
    ] = "table",
    fresh: FreshOption = False,
    model_dir: ModelDirOption = None,
    dev_model: DevModelOption = False,
) -> None:
    """Show central files, import cycles and hidden couplings, or one file's dependencies."""
    session = cli_session(ctx, model_dir=model_dir, dev_model=dev_model)
    loaded = cli_analysis(session, target, ref=ref, fresh=fresh)
    with cli_guard(session.err_console):
        item = find_file(loaded.result, file) if file is not None else None
        if output_format == "json":
            document = file_graph_json(loaded, item) if item is not None else graph_json(loaded, top=top)
            sys.stdout.write(json.dumps(document, indent=2) + "\n")
            return
        if not loaded.result.files:
            session.console.print(_dim(NO_FILES))
        elif item is not None:
            session.console.print(dependency_tree(loaded, item, depth=depth, glyphs=session.glyphs))
        else:
            session.console.print(graph_overview(loaded, top=top, glyphs=session.glyphs))


def register(registry: CommandRegistry) -> None:
    """Register ``/graph`` and ``tdp graph``."""
    registry.add(
        Command(
            name="graph",
            group="Explore",
            help="Show central files, import cycles and hidden couplings, or one file's dependencies",
            usage="/graph [PATH or RANK] [--depth N] [--top N]",
            details=(
                "Without a file, lists the files most other files depend on (transitive dependents and "
                "PageRank), the import cycles, and the hidden couplings: pairs of files that usually change "
                "together although neither imports the other. With a path, a unique ending or a rank, shows "
                "a tree of the files that import it and the files it imports, --depth levels deep (default "
                "2, at most 6), plus the tests that import it. --top sets the rows per list (0 = all)."
            ),
            handler=graph_handler,
            complete=complete_graph,
            needs_analysis=True,
            cli=graph_cli,
        )
    )
