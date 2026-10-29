"""A small inline checkbox picker for the interactive shell (Up/Down, Space, Enter, Esc).

prompt_toolkit's ``CheckboxList`` toggles on Enter, so this is a tiny custom application instead: one line per
option, drawn below the prompt and erased when done. Only use it on a real terminal; callers fall back to
command-line flags otherwise. Not a plug-in (leading ``_``), and imported only when a picker is shown.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from tdp_cli.theme import CYAN, DIM, UNICODE_GLYPHS, VIOLET, Glyphs

if TYPE_CHECKING:
    from prompt_toolkit.formatted_text import StyleAndTextTuples
    from prompt_toolkit.input import Input
    from prompt_toolkit.key_binding import KeyPressEvent
    from prompt_toolkit.output import Output

type PickerOption = tuple[str, str, str]
"""``(value, label, hint)``: the value returned when ticked, the label shown, and a dim explanation."""

PICKER_HELP: Final = "Up/Down move, Space toggles, a toggles all, Enter runs, Esc cancels"

PICKER_STYLE: Final = Style.from_dict(
    {
        "picker.title": f"bold {VIOLET}",
        "picker.pointer": f"bold {CYAN}",
        "picker.box": CYAN,
        "picker.current": "bold",
        "picker.hint": DIM,
        "picker.help": f"italic {DIM}",
    }
)


@dataclass(slots=True)
class PickerState:
    """Which row is highlighted and which rows are ticked."""

    count: int
    index: int = 0
    checked: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.checked:
            self.checked = [False] * self.count

    def move(self, delta: int) -> None:
        """Move the highlight by ``delta`` rows, wrapping around."""
        if self.count:
            self.index = (self.index + delta) % self.count

    def toggle(self, i: int | None = None) -> None:
        """Tick or untick row ``i`` (the highlighted row when None); out-of-range rows are ignored."""
        row = self.index if i is None else i
        if 0 <= row < self.count:
            self.checked[row] = not self.checked[row]

    def toggle_all(self) -> None:
        """Untick every row when all are ticked, else tick every row."""
        value = not all(self.checked)
        self.checked = [value] * self.count

    def chosen(self, values: Sequence[str]) -> list[str]:
        """The values of the ticked rows, in option order."""
        return [v for v, on in zip(values, self.checked, strict=True) if on]


def picker_fragments(
    title: str, options: Sequence[PickerOption], state: PickerState, glyphs: Glyphs = UNICODE_GLYPHS
) -> StyleAndTextTuples:
    """The formatted text of the picker: a title line, one line per option, then the key help."""
    out: StyleAndTextTuples = [("class:picker.title", title), ("", "\n")]
    for i, (_, label, hint) in enumerate(options):
        current = i == state.index
        out.append(("class:picker.pointer", f"{glyphs.chevron} " if current else "  "))
        out.append(("class:picker.box", "[x] " if state.checked[i] else "[ ] "))
        out.append(("class:picker.current" if current else "", f"{i + 1}. {label}"))
        if hint:
            out.append(("class:picker.hint", f"  {hint}"))
        out.append(("", "\n"))
    out.append(("class:picker.help", PICKER_HELP))
    return out


def pick_many(
    title: str,
    options: Sequence[PickerOption],
    *,
    glyphs: Glyphs = UNICODE_GLYPHS,
    input: Input | None = None,
    output: Output | None = None,
) -> list[str] | None:
    """Show the picker and return the ticked values (``[]`` when none), or None when cancelled.

    Keys: Up/Down (or k/j) move, Space or x toggles, the digits 1 to 9 toggle that row, ``a`` toggles all,
    Enter accepts, Esc, q or Ctrl+C cancel. ``input``/``output`` inject prompt_toolkit I/O for tests.
    """
    state = PickerState(len(options))
    values = [value for value, _, _ in options]
    bindings = KeyBindings()

    @bindings.add("up")
    @bindings.add("k")
    def _up(event: KeyPressEvent) -> None:
        state.move(-1)

    @bindings.add("down")
    @bindings.add("j")
    def _down(event: KeyPressEvent) -> None:
        state.move(1)

    @bindings.add(" ")
    @bindings.add("x")
    def _toggle(event: KeyPressEvent) -> None:
        state.toggle()

    @bindings.add("a")
    def _toggle_all(event: KeyPressEvent) -> None:
        state.toggle_all()

    @bindings.add("enter")
    def _accept(event: KeyPressEvent) -> None:
        event.app.exit(result=state.chosen(values))

    @bindings.add("escape", eager=True)
    @bindings.add("q")
    @bindings.add("c-c")
    def _cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    for number in range(1, min(len(options), 9) + 1):

        def _digit(event: KeyPressEvent, row: int = number - 1) -> None:
            state.toggle(row)
            state.index = row

        bindings.add(str(number))(_digit)

    control = FormattedTextControl(
        lambda: picker_fragments(title, options, state, glyphs), focusable=True, show_cursor=False
    )
    app: Application[list[str] | None] = Application(
        layout=Layout(Window(control, dont_extend_height=True, always_hide_cursor=True)),
        key_bindings=bindings,
        style=PICKER_STYLE,
        full_screen=False,
        erase_when_done=True,
        mouse_support=False,
        input=input,
        output=output,
    )
    try:
        return app.run()
    except EOFError:
        return None
