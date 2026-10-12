"""Central GUI display-name policy for layout-bearing text.

These helpers are intentionally Qt-free so policy tests can run without a GUI
runtime.  Qt widgets should use these functions before putting user/project
labels into labels, tabs, list rows, status/footer messages, or plot titles.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import re

__all__ = [
    "CompactText",
    "DATASET_LIST_LABEL_MAX_CHARS",
    "DIAGNOSTIC_LABEL_MAX_CHARS",
    "INLINE_ERROR_MAX_CHARS",
    "PLOT_TITLE_LABEL_MAX_CHARS",
    "TAB_LABEL_MAX_CHARS",
    "compact_dataset_label",
    "compact_diagnostic_text",
    "compact_plot_title_label",
    "compact_text",
    "dataset_alias",
    "literal_tooltip",
    "scientific_symbol_label",
    "scientific_symbol_html",
    "scientific_symbol_spoken",
]

_SCIENTIFIC_SYMBOL_LABELS = {
    "h_bar": "ħ",
    "kappa": "κ",
    "dG_act": "ΔG‡",
    "dG_eq": "ΔG°",
    "C0": "C°",
}


def scientific_symbol_label(symbol: str) -> str:
    """Project one exact scientific base symbol into its visible label."""
    return _SCIENTIFIC_SYMBOL_LABELS.get(symbol, symbol)


def literal_tooltip(text: str) -> str:
    """Prevent Qt's automatic rich-text detection from interpreting authored text."""
    return "<qt>" + escape(text).replace("\n", "<br>") + "</qt>" if "<" in text else text


def _scientific_parts(symbol: str, family: str, role: str) -> tuple[str, str] | None:
    if family == "time" and role in {"t_start", "t_end"} and symbol == role:
        return "t", role[2:]
    if role in {"k", "kf", "kr", "Keq"} and family in {
        "Constant", "Arrhenius", "Eyring", "reverse", "equilibrium"
    } and re.fullmatch(re.escape(role) + r"[0-9]+", symbol):
        return ("K", "eq" + symbol[3:]) if role == "Keq" else ("k", symbol[1:])
    if family == "Arrhenius" and role in {"A", "Ea"} and re.fullmatch(re.escape(role) + r"[0-9]+", symbol):
        index = symbol[len(role):]
        return ("E", "a" + index) if role == "Ea" else ("A", index)
    if (family, role) in {
        ("Eyring", "dG_act"), ("Eyring", "kappa"),
        ("equilibrium", "dG_eq"),
    } and re.fullmatch(re.escape(role) + r"[0-9]*", symbol):
        return scientific_symbol_label(role), symbol[len(role):]
    return None


def scientific_symbol_html(symbol: str, *, family: str, role: str) -> str:
    """Project only an explicitly classified scientific coordinate to rich text."""
    parts = _scientific_parts(symbol, family, role)
    if parts is None:
        return escape(symbol)
    base, subscript = parts
    return f"<i>{escape(base)}</i>" + (f"<sub>{escape(subscript)}</sub>" if subscript else "")


def scientific_symbol_spoken(symbol: str, *, family: str, role: str) -> str:
    parts = _scientific_parts(symbol, family, role)
    if parts is None:
        return symbol
    base, subscript = parts
    return base + (f" subscript {subscript}" if subscript else "")


ELLIPSIS = "…"
DATASET_LIST_LABEL_MAX_CHARS = 48
TAB_LABEL_MAX_CHARS = 24
PLOT_TITLE_LABEL_MAX_CHARS = 18
INLINE_ERROR_MAX_CHARS = 120
DIAGNOSTIC_LABEL_MAX_CHARS = 160


@dataclass(frozen=True, slots=True)
class CompactText:
    """A compact visible value plus the original full text for tooltip/details."""

    display: str
    full: str
    was_elided: bool

    @property
    def tooltip(self) -> str:
        # Tooltip owns full/canonical text.  Show it whenever the visible
        # layout-bearing string differs from the canonical value, including
        # whitespace-normalized diagnostics that were not character-elided.
        return self.full if self.was_elided or self.display != self.full else ""


def _raw_full_text(text: object, *, empty_text: str = "") -> str:
    if text is None:
        return str(empty_text or "")
    value = str(text)
    if not value:
        return str(empty_text or "")
    return value


def _normalize_text(text: object, *, empty_text: str = "") -> str:
    value = _raw_full_text(text, empty_text=empty_text)
    value = " ".join(value.split())
    if not value:
        value = str(empty_text or "")
    return value


def compact_text(
    text: object,
    *,
    max_chars: int,
    empty_text: str = "",
    mode: str = "middle",
) -> CompactText:
    """Return text bounded by ``max_chars`` including the ellipsis.

    Display text is normalized for one-line GUI surfaces, but ``CompactText.full``
    preserves the caller's original string for tooltip/details/copy channels.
    """

    full = _raw_full_text(text, empty_text=empty_text)
    display_source = _normalize_text(text, empty_text=empty_text)
    limit = max(1, int(max_chars))
    if len(display_source) <= limit:
        return CompactText(display=display_source, full=full, was_elided=(display_source != full))
    if limit == 1:
        return CompactText(display=ELLIPSIS, full=full, was_elided=True)
    if limit <= 4:
        return CompactText(display=display_source[: limit - 1] + ELLIPSIS, full=full, was_elided=True)

    mode_norm = str(mode or "middle").strip().lower()
    if mode_norm == "right":
        return CompactText(display=display_source[: limit - 1].rstrip() + ELLIPSIS, full=full, was_elided=True)

    available = limit - 1
    head = max(1, available // 2)
    tail = max(1, available - head)
    return CompactText(
        display=f"{display_source[:head].rstrip()}{ELLIPSIS}{display_source[-tail:].lstrip()}",
        full=full,
        was_elided=True,
    )


def compact_dataset_label(text: object, *, max_chars: int = DATASET_LIST_LABEL_MAX_CHARS) -> CompactText:
    return compact_text(text, max_chars=max_chars, empty_text="dataset", mode="middle")


def compact_plot_title_label(text: object, *, max_chars: int = PLOT_TITLE_LABEL_MAX_CHARS) -> CompactText:
    return compact_text(text, max_chars=max_chars, empty_text="dataset", mode="middle")


def compact_diagnostic_text(text: object, *, max_chars: int = DIAGNOSTIC_LABEL_MAX_CHARS) -> CompactText:
    return compact_text(text, max_chars=max_chars, empty_text="", mode="right")


def dataset_alias(index: int, *, prefix: str = "D") -> str:
    try:
        idx = int(index)
    except (TypeError, ValueError):
        idx = 0
    return f"{str(prefix or 'D')}{max(0, idx) + 1}"


def reaction_text(context: str) -> str:
    """Authored reaction text with display arrows: → and ⇌."""

    return context.replace("<=>", "⇌").replace("<->", "⇌").replace("->", "→")


_SMALL_COUNTS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def count_text(count: int, noun: str) -> str:
    """``count`` and ``noun`` in prose: counts of ten or fewer are spelled out."""

    word = _SMALL_COUNTS[count] if 0 <= count <= 10 else str(count)
    return f"{word} {noun if count == 1 else noun + 's'}"


