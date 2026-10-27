"""Python plugin: Tree-sitter parse into ``FileSyntax`` (IMPLEMENTATION_PLAN.md step P1.2).

The metric rules follow the step's numbered implementation notes exactly. Where the notes leave a choice open:

- A ``string`` node is one leaf for SLOC and tokens (all lines it spans count; one ``"LIT"`` token).
- Zero-width leaves (tree-sitter's MISSING nodes from error recovery) are not code: no SLOC, no token.
- ``line_continuation`` (a trailing backslash) counts for SLOC but is not a token, like comments it is layout.
- ``FunctionInfo.start_line`` is the ``def`` line (decorators excluded); ``end_line`` is the last line holding
  code, so trailing comments that tree-sitter attaches to the body do not stretch the function.
- Cognitive complexity descends into nested functions (one extra nesting level) and lambdas (no extra level);
  cyclomatic and max nesting stop at nested functions and classes, which report their own values.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from fnmatch import fnmatchcase

import tree_sitter_python
from tree_sitter import Language, Node, Parser, Point

from tdp_core.contracts import CommentInfo, FileSyntax, FunctionInfo, RawImport

LANGUAGE_NAME = "python"

_LANGUAGE = Language(tree_sitter_python.language())
_THREAD_STATE = threading.local()

_SCOPES = frozenset({"function_definition", "class_definition"})
# Rule 4: each adds one to cyclomatic complexity.
_DECISIONS = frozenset(
    {
        "if_statement",
        "elif_clause",
        "for_statement",
        "while_statement",
        "except_clause",
        "case_clause",
        "conditional_expression",
        "boolean_operator",
        "for_in_clause",
        "if_clause",
    }
)
# Rule 5: cognitive increments, and the nodes whose ``block`` children sit one nesting level deeper.
_COG_NESTED = frozenset(
    {
        "if_statement",
        "for_statement",
        "while_statement",
        "except_clause",
        "conditional_expression",
        "match_statement",
    }
)
_COG_FLAT = frozenset({"elif_clause", "else_clause", "boolean_operator"})
_COG_NESTING = frozenset(
    {
        "if_statement",
        "elif_clause",
        "else_clause",
        "for_statement",
        "while_statement",
        "except_clause",
        "with_statement",
        "match_statement",
        "function_definition",
    }
)
# Rule 6: blocks that count towards max_nesting.
_BLOCKS = frozenset(
    {"if_statement", "for_statement", "while_statement", "try_statement", "with_statement", "match_statement"}
)
# Rule 10: normalised token per leaf type; any other leaf keeps its node type.
_TOKEN_NAMES = {"identifier": "ID"} | {
    kind: "LIT" for kind in ("string", "integer", "float", "true", "false", "none")
}
_IMPORTS = frozenset({"import_statement", "import_from_statement", "future_import_statement"})
_NOT_PARAMS = frozenset({"keyword_separator", "positional_separator", "comment", "ERROR"})
_NAMED_PARAMS = frozenset({"typed_parameter", "default_parameter", "typed_default_parameter"})
_IMPLICIT_FIRST = frozenset({"self", "cls"})
_TEST_DIRS = frozenset({"tests", "test"})


def _parser() -> Parser:
    """The calling thread's parser, created on first use (a tree-sitter parser is not thread safe)."""
    parser: Parser | None = getattr(_THREAD_STATE, "parser", None)
    if parser is None:
        parser = Parser(_LANGUAGE)
        _THREAD_STATE.parser = parser
    return parser


def _text(node: Node) -> str:
    return (node.text or b"").decode("utf-8", errors="replace")


# Never read ``Point.row`` / ``Point.column``: in tree-sitter 0.26.0 those properties return a dangling
# reference for values above 256 and corrupt the heap. Indexing or unpacking the point is safe.
def _start_row(node: Node) -> int:
    """0-based first row of ``node``."""
    return node.start_point[0]


def _end_row(node: Node) -> int:
    """0-based last row the node occupies (a node ending at column 0 does not occupy that row)."""
    row, column = node.end_point
    return row - 1 if column == 0 and row > _start_row(node) else row


@dataclass(slots=True)
class _Scan:
    """Everything collected in one depth-first pass over the tree."""

    rows: set[int] = field(default_factory=set)  # 0-based rows holding code (SLOC)
    tokens: list[str] = field(default_factory=list)
    token_lines: list[int] = field(default_factory=list)
    comments: list[CommentInfo] = field(default_factory=list)
    imports: list[RawImport] = field(default_factory=list)
    functions: list[Node] = field(default_factory=list)  # document order
    classes: int = 0

    def run(self, root: Node) -> None:
        cursor = root.walk()
        if not cursor.goto_first_child():
            return
        while True:
            node = cursor.node
            if node is not None and self._visit(node) and cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    return

    def _visit(self, node: Node) -> bool:
        """Record ``node``; return True if its children should be visited."""
        kind = node.type
        if kind == "comment":
            text = _text(node)
            self.comments.append(CommentInfo(_start_row(node) + 1, text.removeprefix("#").strip()))
            return False
        if kind == "string" or node.child_count == 0:
            start, end = node.start_point, node.end_point
            if start != end:  # zero width means a MISSING node invented by error recovery
                self._leaf(kind, start[0], end)
            return False
        if kind in _IMPORTS:
            self.imports.extend(_raw_imports(node))
        elif kind == "function_definition":
            self.functions.append(node)
        elif kind == "class_definition":
            self.classes += 1
        return True

    def _leaf(self, kind: str, first: int, end: Point) -> None:
        last, column = end
        if column == 0 and last > first:
            last -= 1
        if last == first:
            self.rows.add(first)
        else:
            self.rows.update(range(first, last + 1))
        if kind != "line_continuation":
            self.tokens.append(_TOKEN_NAMES.get(kind, kind))
            self.token_lines.append(first + 1)


def _raw_imports(node: Node) -> list[RawImport]:
    """RawImports of one import statement (rule 9); ``import a, b`` yields one per module."""
    line = _start_row(node) + 1
    targets = [_import_target(child) for child in node.children_by_field_name("name")]
    if node.type == "import_statement":
        return [RawImport(target, (), 0, line) for target in targets if target]
    if node.type == "future_import_statement":
        module, level = "__future__", 0
    else:
        source = node.child_by_field_name("module_name")
        if source is None:
            return []
        module, level = _import_source(source)
    if any(child.type == "wildcard_import" for child in node.children):
        names: tuple[str, ...] = ("*",)
    else:
        names = tuple(target for target in targets if target)
    return [RawImport(module, names, level, line)]


def _import_target(node: Node) -> str:
    """Imported dotted name, without any ``as`` alias."""
    if node.type == "aliased_import":
        name = node.child_by_field_name("name")
        return _dotted(name) if name is not None else ""
    return _dotted(node)


def _dotted(node: Node) -> str:
    if node.type == "dotted_name":
        return ".".join(_text(child) for child in node.named_children if child.type == "identifier")
    return _text(node)


def _import_source(node: Node) -> tuple[str, int]:
    """(module, level) of the ``from`` part: ``..a.b`` -> ("a.b", 2), ``.`` -> ("", 1)."""
    if node.type != "relative_import":
        return _dotted(node), 0
    module, level = "", 0
    for child in node.named_children:
        if child.type == "import_prefix":
            level = _text(child).count(".")
        elif child.type == "dotted_name":
            module = _dotted(child)
    return module, level


def _scope_walk(root: Node) -> tuple[int, int]:
    """(decision nodes, deepest block nesting) under ``root``, not entering nested functions or classes."""
    decisions = 0
    deepest = 0
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        for child in node.children:
            kind = child.type
            if kind in _SCOPES:
                continue
            if kind in _DECISIONS:
                decisions += 1
            child_depth = depth
            if kind in _BLOCKS:
                child_depth += 1
                deepest = max(deepest, child_depth)
            if child.child_count:
                stack.append((child, child_depth))
    return decisions, deepest


def _cognitive(body: Node) -> int:
    """Cognitive complexity of a function body (rule 5), nested functions included."""
    total = 0
    stack = [(body, 0)]
    while stack:
        node, nesting = stack.pop()
        kind = node.type
        if kind in _COG_NESTED:
            total += 1 + nesting
        elif kind in _COG_FLAT:
            total += 1
        inner = nesting + 1 if kind in _COG_NESTING else nesting
        for child in node.children:
            if child.child_count:
                stack.append((child, inner if child.type == "block" else nesting))
    return total


def _qualified_name(node: Node) -> str:
    parts = [_scope_name(node)]
    parent = node.parent
    while parent is not None:
        if parent.type in _SCOPES:
            parts.append(_scope_name(parent))
        parent = parent.parent
    return ".".join(reversed(parts))


def _scope_name(node: Node) -> str:
    name = node.child_by_field_name("name")
    return _text(name) if name is not None else "<unknown>"


def _param_count(parameters: Node | None) -> int:
    """Parameters excluding a leading ``self``/``cls``; ``*args`` and ``**kwargs`` count one each."""
    if parameters is None:
        return 0
    params = [child for child in parameters.named_children if child.type not in _NOT_PARAMS]
    if params and _param_name(params[0]) in _IMPLICIT_FIRST:
        return len(params) - 1
    return len(params)


def _param_name(param: Node) -> str | None:
    if param.type == "identifier":
        return _text(param)
    if param.type in _NAMED_PARAMS and param.named_child_count:
        first = param.named_children[0]
        if first.type == "identifier":
            return _text(first)
    return None


def _last_code_row(node: Node) -> int:
    """0-based end row of the last non-comment, non-empty leaf under ``node``."""
    current = node
    while current.child_count and current.type != "string":
        code = [c for c in current.children if c.type != "comment" and c.end_byte > c.start_byte]
        if not code:
            break
        current = code[-1]
    return _end_row(current)


def _function_info(node: Node) -> FunctionInfo:
    body = node.child_by_field_name("body")
    decisions, nesting = _scope_walk(body) if body is not None else (0, 0)
    return FunctionInfo(
        name=_qualified_name(node),
        start_line=_start_row(node) + 1,
        end_line=_last_code_row(node) + 1,
        cyclomatic=1 + decisions,
        cognitive=_cognitive(body) if body is not None else 0,
        max_nesting=nesting,
        params=_param_count(node.child_by_field_name("parameters")),
    )


def parse_python(path: str, source: bytes) -> FileSyntax:
    """Parse Python ``source`` bytes (UTF-8) of repo-relative ``path`` into a ``FileSyntax``.

    Lines are 1-based. Syntax errors never raise: ``parse_error`` is set and every metric is computed on
    tree-sitter's error-recovered tree. An empty file gives ``sloc == 0``, ``module_cc == 1`` and no tokens.
    """
    root = _parser().parse(source).root_node
    scan = _Scan()
    scan.run(root)
    module_decisions, _ = _scope_walk(root)
    return FileSyntax(
        path=path,
        language=LANGUAGE_NAME,
        sloc=len(scan.rows),
        module_cc=1 + module_decisions,
        functions=tuple(_function_info(node) for node in scan.functions),
        classes=scan.classes,
        comments=tuple(scan.comments),
        imports=tuple(scan.imports),
        tokens=tuple(scan.tokens),
        token_lines=tuple(scan.token_lines),
        parse_error=root.has_error,
    )


def is_test_file(path: str) -> bool:
    """True for ``test_*.py``, ``*_test.py``, ``conftest.py``, or a file under a ``tests``/``test`` directory.

    ``path`` is repo-relative POSIX; matching is case sensitive.
    """
    *dirs, base = path.split("/")
    if base == "conftest.py" or fnmatchcase(base, "test_*.py") or fnmatchcase(base, "*_test.py"):
        return True
    return any(part in _TEST_DIRS for part in dirs)


class PythonPlugin:
    """``LanguagePlugin`` for Python source files (``.py``)."""

    name: str = LANGUAGE_NAME
    extensions: tuple[str, ...] = (".py",)

    def parse(self, path: str, source: bytes) -> FileSyntax:
        """See ``parse_python``."""
        return parse_python(path, source)

    def is_test_file(self, path: str) -> bool:
        """See ``is_test_file``."""
        return is_test_file(path)
