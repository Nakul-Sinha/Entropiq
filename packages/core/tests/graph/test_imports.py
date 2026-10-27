import pytest

from tdp_core.contracts import FileSyntax, RawImport
from tdp_core.graph.imports import ModuleIndex, build_import_graph, resolve


def imp(module: str, *names: str, level: int = 0, line: int = 1) -> RawImport:
    return RawImport(module=module, names=tuple(names), level=level, line=line)


def syntax(path: str, *imports: RawImport) -> FileSyntax:
    return FileSyntax(
        path=path,
        language="python",
        sloc=1,
        module_cc=1,
        functions=(),
        classes=0,
        comments=(),
        imports=imports,
        tokens=(),
        token_lines=(),
        parse_error=False,
    )


def is_test(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return name.startswith("test_") or name == "conftest.py" or "tests" in parts[:-1]


PATHS = (
    "app.py",
    "setup.py",
    "README.md",
    "pkg/__init__.py",
    "pkg/core.py",
    "pkg/util.py",
    "pkg/sub/__init__.py",
    "pkg/sub/deep.py",
    "pkg/sub/leaf.py",
    "pkg/other/__init__.py",
    "pkg/other/mod.py",
    "src/srcpkg/__init__.py",
    "src/srcpkg/a.py",
    "src/srcpkg/b.py",
    "lib/libmod.py",
    "nsp/inner/x.py",
)


@pytest.fixture
def index() -> ModuleIndex:
    return ModuleIndex.build(PATHS)


# ---------- ModuleIndex ----------


@pytest.mark.parametrize(
    ("dotted", "path"),
    [
        ("app", "app.py"),
        ("pkg", "pkg/__init__.py"),
        ("pkg.core", "pkg/core.py"),
        ("pkg.sub", "pkg/sub/__init__.py"),
        ("pkg.sub.deep", "pkg/sub/deep.py"),
        ("src.srcpkg", "src/srcpkg/__init__.py"),
        ("srcpkg", "src/srcpkg/__init__.py"),
        ("srcpkg.a", "src/srcpkg/a.py"),
        ("src.srcpkg.a", "src/srcpkg/a.py"),
        ("libmod", "lib/libmod.py"),
        ("lib.libmod", "lib/libmod.py"),
        ("nsp.inner.x", "nsp/inner/x.py"),
    ],
)
def test_module_names(index: ModuleIndex, dotted: str, path: str) -> None:
    assert index.lookup(dotted) == path


@pytest.mark.parametrize("dotted", ["", "README", "nsp", "nsp.inner", "numpy", "pkg.missing", "src"])
def test_unknown_names(index: ModuleIndex, dotted: str) -> None:
    assert index.lookup(dotted) is None


def test_collisions_prefer_shorter_path() -> None:
    index = ModuleIndex.build(
        ["a/b/__init__.py", "a/b.py", "src/pkg/m.py", "pkg/m.py", "lib/x.py", "src/x.py"]
    )
    assert index.lookup("a.b") == "a/b.py"
    assert index.lookup("pkg.m") == "pkg/m.py"
    assert index.lookup("src.pkg.m") == "src/pkg/m.py"
    # same length: lexicographically smaller path wins, independent of input order
    assert index.lookup("x") == "lib/x.py"
    assert ModuleIndex.build(["src/x.py", "lib/x.py"]).lookup("x") == "lib/x.py"


def test_non_importable_paths_are_skipped() -> None:
    index = ModuleIndex.build(
        ["__init__.py", "src/__init__.py", "scripts/my-tool.py", ".github/ci.py", "a/b.c.py", "x.pyi"]
    )
    assert len(index) == 1
    assert index.lookup("src") == "src/__init__.py"
    assert index.lookup("a.b.c") is None


# ---------- resolve: absolute ----------


def test_import_exact_module(index: ModuleIndex) -> None:
    assert resolve(imp("pkg.sub.deep"), "app.py", index) == ["pkg/sub/deep.py"]
    assert resolve(imp("pkg"), "app.py", index) == ["pkg/__init__.py"]


def test_import_falls_back_to_longest_prefix(index: ModuleIndex) -> None:
    assert resolve(imp("pkg.sub.missing"), "app.py", index) == ["pkg/sub/__init__.py"]
    assert resolve(imp("pkg.missing.deeper"), "app.py", index) == ["pkg/__init__.py"]
    assert resolve(imp("pkg.core.attr"), "app.py", index) == ["pkg/core.py"]


@pytest.mark.parametrize("module", ["os", "os.path", "numpy.linalg", "requests"])
def test_third_party_imports_are_dropped(index: ModuleIndex, module: str) -> None:
    assert resolve(imp(module), "app.py", index) == []
    assert resolve(imp(module, "thing"), "app.py", index) == []


def test_from_import_submodule_rule(index: ModuleIndex) -> None:
    # `deep` is a submodule, `helper` is a name defined in pkg/sub/__init__.py
    got = resolve(imp("pkg.sub", "deep", "helper", "leaf"), "app.py", index)
    assert got == ["pkg/sub/deep.py", "pkg/sub/__init__.py", "pkg/sub/leaf.py"]


def test_from_import_names_deduplicate(index: ModuleIndex) -> None:
    assert resolve(imp("pkg.core", "A", "B", "C"), "app.py", index) == ["pkg/core.py"]
    assert resolve(imp("pkg", "*"), "app.py", index) == ["pkg/__init__.py"]


def test_from_import_of_missing_module_uses_prefix(index: ModuleIndex) -> None:
    assert resolve(imp("pkg.sub.gone", "x"), "app.py", index) == ["pkg/sub/__init__.py"]


def test_namespace_package_submodule(index: ModuleIndex) -> None:
    # nsp/inner has no __init__.py, but its submodule is still importable
    assert resolve(imp("nsp.inner", "x"), "app.py", index) == ["nsp/inner/x.py"]
    assert resolve(imp("nsp.inner", "y"), "app.py", index) == []


def test_src_layout_absolute(index: ModuleIndex) -> None:
    assert resolve(imp("srcpkg.a"), "app.py", index) == ["src/srcpkg/a.py"]
    assert resolve(imp("srcpkg", "b", "func"), "app.py", index) == [
        "src/srcpkg/b.py",
        "src/srcpkg/__init__.py",
    ]


# ---------- resolve: relative ----------


def test_relative_sibling(index: ModuleIndex) -> None:
    assert resolve(imp("", "util", level=1), "pkg/core.py", index) == ["pkg/util.py"]
    assert resolve(imp("util", "helper", level=1), "pkg/core.py", index) == ["pkg/util.py"]


def test_relative_name_from_package_init(index: ModuleIndex) -> None:
    assert resolve(imp("", "something", level=1), "pkg/core.py", index) == ["pkg/__init__.py"]


def test_relative_from_init_uses_own_package(index: ModuleIndex) -> None:
    assert resolve(imp("", "core", level=1), "pkg/__init__.py", index) == ["pkg/core.py"]
    assert resolve(imp("core", "Thing", level=1), "pkg/__init__.py", index) == ["pkg/core.py"]
    assert resolve(imp("", "deep", level=1), "pkg/sub/__init__.py", index) == ["pkg/sub/deep.py"]


def test_relative_parent_levels(index: ModuleIndex) -> None:
    assert resolve(imp("", "core", level=2), "pkg/sub/deep.py", index) == ["pkg/core.py"]
    assert resolve(imp("other.mod", "f", level=2), "pkg/sub/deep.py", index) == ["pkg/other/mod.py"]
    assert resolve(imp("other", "mod", level=2), "pkg/sub/deep.py", index) == ["pkg/other/mod.py"]
    assert resolve(imp("leaf", "x", level=1), "pkg/sub/deep.py", index) == ["pkg/sub/leaf.py"]


def test_relative_to_repo_root_but_not_beyond(index: ModuleIndex) -> None:
    # level 3 from pkg/sub/deep.py lands on the repository root (still inside the repo)
    assert resolve(imp("", "app", level=3), "pkg/sub/deep.py", index) == ["app.py"]
    assert resolve(imp("", "app", level=4), "pkg/sub/deep.py", index) == []
    assert resolve(imp("", "app", level=2), "setup.py", index) == []
    assert resolve(imp("", "nothing", level=1), "setup.py", index) == []


def test_relative_src_layout(index: ModuleIndex) -> None:
    assert resolve(imp("b", "f", level=1), "src/srcpkg/a.py", index) == ["src/srcpkg/b.py"]
    assert resolve(imp("", "b", level=1), "src/srcpkg/a.py", index) == ["src/srcpkg/b.py"]


def test_relative_missing_module_uses_prefix(index: ModuleIndex) -> None:
    assert resolve(imp("nope", "x", level=1), "pkg/sub/deep.py", index) == ["pkg/sub/__init__.py"]


# ---------- build_import_graph ----------


@pytest.fixture
def project() -> dict[str, FileSyntax]:
    files = [
        syntax("app.py", imp("pkg", "Thing"), imp("os"), imp("numpy"), imp("pkg.util"), imp("pkg.util")),
        syntax("pkg/__init__.py", imp("core", "Thing", level=1), imp("", "helper_fn", level=1)),
        syntax("pkg/core.py", imp("", "util", level=1), imp("typing", "Any")),
        syntax("pkg/util.py"),
        syntax("src/srcpkg/__init__.py"),
        syntax("src/srcpkg/a.py", imp("srcpkg", "b"), imp("pkg.core")),
        syntax("src/srcpkg/b.py", imp("a", "x", level=1)),
        syntax("tests/test_core.py", imp("pkg.core", "Thing"), imp("pytest"), imp("conftest", "fixture")),
        syntax("tests/conftest.py", imp("srcpkg.a")),
        syntax("isolated.py", imp("requests")),
    ]
    return {f.path: f for f in files}


def test_graph_nodes_and_test_attribute(project: dict[str, FileSyntax]) -> None:
    graph = build_import_graph(project, is_test)
    assert list(graph.nodes) == sorted(project)
    assert {n for n, t in graph.nodes(data="is_test") if t} == {"tests/test_core.py", "tests/conftest.py"}
    assert "isolated.py" in graph
    assert graph.degree("isolated.py") == 0


def test_graph_edges(project: dict[str, FileSyntax]) -> None:
    graph = build_import_graph(project, is_test)
    assert set(graph.edges) == {
        ("app.py", "pkg/__init__.py"),
        ("app.py", "pkg/util.py"),
        ("pkg/__init__.py", "pkg/core.py"),
        ("pkg/core.py", "pkg/util.py"),
        ("src/srcpkg/a.py", "src/srcpkg/b.py"),
        ("src/srcpkg/a.py", "pkg/core.py"),
        ("src/srcpkg/b.py", "src/srcpkg/a.py"),
        ("tests/test_core.py", "pkg/core.py"),
        ("tests/conftest.py", "src/srcpkg/a.py"),
    }
    # `from . import helper_fn` inside pkg/__init__.py resolves to itself: no self loop
    assert not any(u == v for u, v in graph.edges)


def test_graph_empty_input() -> None:
    graph = build_import_graph({}, is_test)
    assert graph.number_of_nodes() == 0
