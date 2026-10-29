from __future__ import annotations

from collections.abc import Iterator

import pytest
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

from tdp_cli.commands._picker import PickerState, pick_many, picker_fragments
from tdp_cli.theme import ASCII_GLYPHS

OPTIONS = [("a", "First", "one"), ("b", "Second", ""), ("c", "Third", "three")]


@pytest.fixture
def pipe() -> Iterator[PipeInput]:
    with create_pipe_input() as pipe_input:
        yield pipe_input


def test_picker_state() -> None:
    state = PickerState(3)
    assert state.checked == [False, False, False]
    state.move(-1)
    assert state.index == 2
    state.move(2)
    assert state.index == 1
    state.toggle()
    state.toggle(7)
    assert state.chosen(["a", "b", "c"]) == ["b"]
    state.toggle_all()
    assert state.chosen(["a", "b", "c"]) == ["a", "b", "c"]
    state.toggle_all()
    assert state.chosen(["a", "b", "c"]) == []
    empty = PickerState(0)
    empty.move(1)
    assert empty.index == 0


def test_picker_fragments() -> None:
    state = PickerState(3)
    state.toggle(1)
    text = "".join(part[1] for part in picker_fragments("Pick", OPTIONS, state, ASCII_GLYPHS))
    assert text.startswith("Pick\n")
    assert "> [ ] 1. First  one\n" in text
    assert "  [x] 2. Second\n" in text
    assert "Space toggles" in text


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ("j \r", ["b"]),
        ("a\r", ["a", "b", "c"]),
        ("3\r", ["c"]),
        ("jjk \r", ["b"]),
        ("x\r", ["a"]),
        ("\r", []),
        ("q", None),
        ("\x03", None),
    ],
)
def test_pick_many(pipe: PipeInput, keys: str, expected: list[str] | None) -> None:
    pipe.send_text(keys)
    assert pick_many("Pick", OPTIONS, input=pipe, output=DummyOutput()) == expected
