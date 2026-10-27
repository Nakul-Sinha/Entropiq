import io
import keyword
import random
import time
import tokenize
import zlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tdp_core.config import DEFAULT_SETTINGS, with_overrides
from tdp_core.contracts import CloneStats, FileSyntax
from tdp_core.parsing.clones import BASE, MOD, _kgram_hashes, clone_stats

K = DEFAULT_SETTINGS.clone_k
W = DEFAULT_SETTINGS.clone_window
ZERO = CloneStats(dup_ratio=0.0, dup_tokens=0, clone_groups=0, dup_lines=0)

VOCAB = (
    "ID", "LIT", "(", ")", "[", "]", "{", "}", ":", ",", ".", "=", "==", "<", ">", "+", "-", "*",
    "def", "return", "if", "else", "for", "in", "while", "not", "and", "or", "class", "import",
    "from", "try", "except", "with", "as", "yield", "lambda", "is",
)  # fmt: skip


def syntax(path: str, tokens: Sequence[str], lines: Sequence[int] | None = None) -> FileSyntax:
    toks = tuple(tokens)
    token_lines = tuple(lines) if lines is not None else tuple(i // 8 + 1 for i in range(len(toks)))
    return FileSyntax(
        path=path,
        language="python",
        sloc=len(set(token_lines)),
        module_cc=1,
        functions=(),
        classes=0,
        comments=(),
        imports=(),
        tokens=toks,
        token_lines=token_lines,
        parse_error=False,
    )


def random_tokens(seed: int, n: int) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(VOCAB) for _ in range(n)]


def normalise(source: str) -> tuple[list[str], list[int]]:
    """Normalised tokens like the parser emits (identifiers -> ID, literals -> LIT), via stdlib tokenize."""
    tokens: list[str] = []
    lines: list[int] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.NAME:
            if tok.string in ("True", "False", "None"):
                text = "LIT"
            else:
                text = tok.string if keyword.iskeyword(tok.string) else "ID"
        elif tok.type in (tokenize.NUMBER, tokenize.STRING):
            text = "LIT"
        elif tok.type == tokenize.OP:
            text = tok.string
        else:
            continue
        tokens.append(text)
        lines.append(tok.start[0])
    return tokens, lines


def reference_clone_stats(syntax_by_path: Mapping[str, FileSyntax], k: int, w: int) -> dict[str, CloneStats]:
    """Straight per-file transcription of the plan's algorithm with a true rolling hash."""
    occurrences: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for path in sorted(syntax_by_path):
        ids = [zlib.crc32(t.encode()) for t in syntax_by_path[path].tokens]
        if len(ids) < k:
            continue
        top = pow(BASE, k - 1, MOD)
        h = 0
        for t in ids[:k]:
            h = (h * BASE + t) % MOD
        hashes = [h]
        for i in range(len(ids) - k):
            h = ((h - ids[i] * top) * BASE + ids[i + k]) % MOD
            hashes.append(h)
        win = min(w, len(hashes))
        selected = set()
        for s in range(len(hashes) - win + 1):
            window = hashes[s : s + win]
            low = min(window)
            selected.add(s + max(j for j, v in enumerate(window) if v == low))
        for pos in selected:
            occurrences[hashes[pos]].append((path, pos))

    marked: dict[str, set[int]] = defaultdict(set)
    groups: dict[str, set[int]] = defaultdict(set)
    for h, occ in occurrences.items():
        if any(a[0] != b[0] or abs(a[1] - b[1]) >= k for a, b in combinations(occ, 2)):
            for path, pos in occ:
                marked[path].update(range(pos, pos + k))
                groups[path].add(h)

    out = {}
    for path in sorted(syntax_by_path):
        fs = syntax_by_path[path]
        dup = len(marked[path])
        out[path] = CloneStats(
            dup_ratio=dup / len(fs.tokens) if dup else 0.0,
            dup_tokens=dup,
            clone_groups=len(groups[path]),
            dup_lines=len({fs.token_lines[i] for i in marked[path]}),
        )
    return out


ORIGINAL = '''
def merge_intervals(intervals, tolerance=0):
    """Merge overlapping intervals."""
    ordered = sorted(intervals, key=lambda pair: pair[0])
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + tolerance:
            last_start, last_end = merged.pop()
            merged.append((last_start, max(last_end, end)))
        else:
            merged.append((start, end))
    total = 0
    for start, end in merged:
        total += end - start
    return merged, total
'''

RENAMED = """
def combine_ranges(spans, slack=5):
    "Combine touching ranges."
    # same logic, different names, literals and comments
    spans_sorted = sorted(spans, key=lambda item: item[0])

    out = []
    for lo, hi in spans_sorted:
        if out and lo <= out[-1][1] + slack:
            prev_lo, prev_hi = out.pop()
            out.append((prev_lo, max(prev_hi, hi)))
        else:
            out.append((lo, hi))
    covered = 0
    for lo, hi in out:
        covered += hi - lo
    return out, covered
"""

UNRELATED = """
class Cache:
    def __init__(self, size):
        self.size = size
        self.items = {}

    def get(self, key, default=None):
        try:
            return self.items[key]
        except KeyError:
            return default

    def put(self, key, value):
        while len(self.items) >= self.size:
            self.items.pop(next(iter(self.items)))
        self.items[key] = value
"""


def test_identical_files_ratio_near_one() -> None:
    tokens = random_tokens(1, 500)
    stats = clone_stats({"a.py": syntax("a.py", tokens), "b.py": syntax("b.py", tokens)})
    assert stats["a.py"] == stats["b.py"]
    # Winnowing may leave at most w - 1 tokens unmarked at each end.
    assert stats["a.py"].dup_ratio >= 1 - 2 * (W - 1) / 500
    assert stats["a.py"].dup_tokens == round(stats["a.py"].dup_ratio * 500)
    assert stats["a.py"].clone_groups > 0
    assert stats["a.py"].dup_lines > 0


def test_renamed_identifiers_still_detected() -> None:
    orig_tokens, orig_lines = normalise(ORIGINAL)
    ren_tokens, ren_lines = normalise(RENAMED)
    assert orig_tokens == ren_tokens  # sanity: normalisation erases the renames
    assert len(orig_tokens) >= K + W - 1
    other_tokens, other_lines = normalise(UNRELATED)
    stats = clone_stats(
        {
            "pkg/orig.py": syntax("pkg/orig.py", orig_tokens, orig_lines),
            "pkg/copy.py": syntax("pkg/copy.py", ren_tokens, ren_lines),
            "pkg/cache.py": syntax("pkg/cache.py", other_tokens, other_lines),
        }
    )
    for path in ("pkg/orig.py", "pkg/copy.py"):
        assert stats[path].dup_ratio >= 1 - 2 * (W - 1) / len(orig_tokens)
        assert stats[path].clone_groups > 0
    # Blank and comment lines hold no tokens, so the copy covers as many distinct lines as the original.
    assert stats["pkg/copy.py"] == stats["pkg/orig.py"]
    assert stats["pkg/cache.py"] == ZERO


def test_unrelated_files_have_no_clones() -> None:
    files = {f"m{i}.py": syntax(f"m{i}.py", random_tokens(100 + i, 400)) for i in range(4)}
    assert all(s == ZERO for s in clone_stats(files).values())


def test_short_file_gets_zeros() -> None:
    short = random_tokens(2, K - 1)
    files = {"a.py": syntax("a.py", short), "b.py": syntax("b.py", short), "e.py": syntax("e.py", [])}
    stats = clone_stats(files)
    assert stats == {"a.py": ZERO, "b.py": ZERO, "e.py": ZERO}


def test_file_of_exactly_k_tokens_can_match() -> None:
    tokens = random_tokens(3, K)
    lines = [i // 5 + 1 for i in range(K)]
    stats = clone_stats({"a.py": syntax("a.py", tokens, lines), "b.py": syntax("b.py", tokens, lines)})
    assert stats["a.py"] == CloneStats(dup_ratio=1.0, dup_tokens=K, clone_groups=1, dup_lines=K // 5)


def test_shared_block_between_files() -> None:
    block = random_tokens(4, 150)
    a = random_tokens(5, 120) + block + random_tokens(6, 80)
    b = random_tokens(7, 60) + block + random_tokens(8, 200)
    stats = clone_stats({"a.py": syntax("a.py", a), "b.py": syntax("b.py", b)})
    for path, total in (("a.py", len(a)), ("b.py", len(b))):
        assert len(block) - 2 * (W - 1) <= stats[path].dup_tokens <= len(block)
        assert stats[path].dup_ratio == stats[path].dup_tokens / total
    assert stats["a.py"].clone_groups == stats["b.py"].clone_groups > 0


def test_duplicate_block_inside_one_file() -> None:
    block = random_tokens(9, 120)
    tokens = block + random_tokens(10, 200) + block
    lines = [i // 10 + 1 for i in range(len(tokens))]  # copies on lines 1-12 and 33-44
    s = clone_stats({"a.py": syntax("a.py", tokens, lines)})["a.py"]
    assert 2 * (len(block) - 2 * (W - 1)) <= s.dup_tokens <= 2 * len(block)
    assert s.clone_groups > 0
    assert 0 < s.dup_lines <= 24  # nothing from the middle section


def test_overlapping_repeats_in_one_file_are_not_clones() -> None:
    periodic = ["ID", ","] * 35  # 70 tokens: its repeated k-grams all overlap each other
    tokens = random_tokens(11, 100) + periodic + random_tokens(12, 100)
    assert clone_stats({"a.py": syntax("a.py", tokens)})["a.py"] == ZERO

    long_periodic = ["ID", ","] * 100  # 200 tokens: repeats far enough apart not to overlap
    tokens = random_tokens(11, 100) + long_periodic + random_tokens(12, 100)
    assert clone_stats({"a.py": syntax("a.py", tokens)})["a.py"].dup_tokens > 0


def test_deterministic_output_order() -> None:
    block = random_tokens(13, 100)
    files = [
        syntax("z.py", block + random_tokens(14, 50)),
        syntax("a/b.py", random_tokens(15, 30) + block),
        syntax("m.py", random_tokens(16, 10)),
        syntax("a.py", random_tokens(17, 300)),
    ]
    forward = clone_stats({fs.path: fs for fs in files})
    backward = clone_stats({fs.path: fs for fs in reversed(files)})
    assert list(forward) == list(backward) == ["a.py", "a/b.py", "m.py", "z.py"]
    assert forward == backward
    assert forward == clone_stats({fs.path: fs for fs in files})


def test_empty_input() -> None:
    assert clone_stats({}) == {}


def test_respects_settings() -> None:
    tokens = random_tokens(18, 30)
    files = {"a.py": syntax("a.py", tokens), "b.py": syntax("b.py", tokens)}
    assert clone_stats(files)["a.py"] == ZERO  # shorter than the default k
    small = with_overrides(DEFAULT_SETTINGS, clone_k=10, clone_window=4)
    assert clone_stats(files, small)["a.py"].dup_ratio >= 1 - 2 * 3 / 30


def test_rejects_bad_input() -> None:
    bad = FileSyntax("a.py", "python", 1, 1, (), 0, (), (), ("ID", "="), (1,), False)
    with pytest.raises(ValueError, match="token_lines"):
        clone_stats({"a.py": bad})
    with pytest.raises(ValueError, match="clone_k"):
        clone_stats({}, with_overrides(DEFAULT_SETTINGS, clone_k=0))
    with pytest.raises(ValueError, match="clone_window"):
        clone_stats({}, with_overrides(DEFAULT_SETTINGS, clone_window=0))


@pytest.mark.parametrize("k", [1, 2, 7, 50, 200])
def test_kgram_hashes_match_rolling_hash(k: int) -> None:
    rng = random.Random(k)
    ids = [rng.randrange(1 << 32) for _ in range(400)] + [0, (1 << 32) - 1] * 20
    top = pow(BASE, k - 1, MOD)
    h = 0
    for t in ids[:k]:
        h = (h * BASE + t) % MOD
    expected = [h]
    for i in range(len(ids) - k):
        h = ((h - ids[i] * top) * BASE + ids[i + k]) % MOD
        expected.append(h)
    got = _kgram_hashes(np.array(ids, dtype=np.uint64), k)
    assert got.tolist() == expected


@st.composite
def corpora(draw: st.DrawFn) -> tuple[dict[str, FileSyntax], int, int]:
    k = draw(st.integers(1, 8))
    w = draw(st.integers(1, 6))
    alphabet = st.sampled_from(VOCAB[: draw(st.integers(1, 4))])
    files = {}
    for i in range(draw(st.integers(1, 5))):
        tokens = draw(st.lists(alphabet, max_size=40))
        steps = draw(st.lists(st.integers(0, 2), min_size=len(tokens), max_size=len(tokens)))
        lines = np.cumsum([1, *steps])[1:].tolist()
        files[f"f{i}.py"] = syntax(f"f{i}.py", tokens, lines)
    return files, k, w


@settings(max_examples=300, deadline=None, derandomize=True)
@given(corpora())
def test_matches_reference_implementation(case: tuple[dict[str, FileSyntax], int, int]) -> None:
    files, k, w = case
    got = clone_stats(files, with_overrides(DEFAULT_SETTINGS, clone_k=k, clone_window=w))
    assert got == reference_clone_stats(files, k, w)


def test_performance_2000_files_x_500_tokens() -> None:
    rng = np.random.default_rng(DEFAULT_SETTINGS.seed)
    vocab = np.array(VOCAB)
    streams = [vocab[rng.integers(0, len(VOCAB), size=500)].tolist() for _ in range(2000)]
    for i in range(0, 2000, 10):  # plant a 200-token clone in every tenth file
        streams[i][100:300] = streams[(i + 7) % 2000][:200]
    lines = [i // 8 + 1 for i in range(500)]
    files = {f"pkg/m{i:04d}.py": syntax(f"pkg/m{i:04d}.py", s, lines) for i, s in enumerate(streams)}

    started = time.perf_counter()
    stats = clone_stats(files)
    elapsed = time.perf_counter() - started

    assert elapsed < 10.0
    assert len(stats) == 2000
    assert stats["pkg/m0000.py"].dup_tokens >= 200 - 2 * (W - 1)
    assert stats["pkg/m0001.py"] == ZERO
