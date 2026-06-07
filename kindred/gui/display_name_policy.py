"""Central GUI display-name policy for layout-bearing text.

These helpers are intentionally Qt-free so policy tests can run without a GUI
runtime.  Qt widgets should use these functions before putting user/project
labels into labels, tabs, list rows, status/footer messages, or plot titles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "CompactText",
    "NamedCountSummary",
    "DATASET_LIST_LABEL_MAX_CHARS",
    "DIAGNOSTIC_LABEL_MAX_CHARS",
    "FOOTER_SUMMARY_MAX_CHARS",
    "INLINE_ERROR_MAX_CHARS",
    "PLOT_TITLE_LABEL_MAX_CHARS",
    "SLIDER_TARGET_LABEL_MAX_CHARS",
    "STATUS_ITEM_LABEL_MAX_CHARS",
    "TAB_LABEL_MAX_CHARS",
    "compact_dataset_label",
    "compact_diagnostic_text",
    "compact_plot_title_label",
    "compact_set_label",
    "compact_text",
    "dataset_alias",
    "format_named_count_summary",
    "summarize_named_count",
    "tooltip_for_compact_text",
]

ELLIPSIS = "…"
DATASET_LIST_LABEL_MAX_CHARS = 48
TAB_LABEL_MAX_CHARS = 24
PLOT_TITLE_LABEL_MAX_CHARS = 18
STATUS_ITEM_LABEL_MAX_CHARS = 26
SLIDER_TARGET_LABEL_MAX_CHARS = 30
INLINE_ERROR_MAX_CHARS = 120
DIAGNOSTIC_LABEL_MAX_CHARS = 160
FOOTER_SUMMARY_MAX_CHARS = 140


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


def tooltip_for_compact_text(compact: CompactText) -> str:
    return compact.tooltip


def compact_dataset_label(text: object, *, max_chars: int = DATASET_LIST_LABEL_MAX_CHARS) -> CompactText:
    return compact_text(text, max_chars=max_chars, empty_text="dataset", mode="middle")


def compact_set_label(text: object, *, max_chars: int = STATUS_ITEM_LABEL_MAX_CHARS) -> CompactText:
    return compact_text(text, max_chars=max_chars, empty_text="set", mode="middle")


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



@dataclass(frozen=True, slots=True)
class NamedCountSummary:
    """Compact count summary plus full item truth for tooltip/details."""

    display: str
    full: str
    count: int

    @property
    def tooltip(self) -> str:
        return self.full if self.full and self.full != self.display else ""

def _plural(noun: str, plural: str | None = None) -> str:
    noun_s = str(noun or "item")
    return str(plural) if plural else f"{noun_s}s"


def summarize_named_count(
    names: Sequence[object] | Iterable[object],
    *,
    singular: str = "dataset",
    plural: str | None = None,
    max_items: int = 3,
    item_max_chars: int = STATUS_ITEM_LABEL_MAX_CHARS,
    empty_text: str = "none",
) -> NamedCountSummary:
    """Summarize item names without taking ownership of their full truth.

    The visible ``display`` is compact and count-first.  The ``full`` field
    keeps every non-empty input item, including duplicates, for tooltips or
    message-box details.  Callers that have stable ids should include them in
    the input values before calling this helper.
    """

    items: list[tuple[str, str]] = []
    for name in names or ():
        full = _raw_full_text(name).strip()
        display_source = _normalize_text(name)
        if not display_source:
            continue
        items.append((full or display_source, display_source))
    count = len(items)
    if count <= 0:
        empty = str(empty_text)
        return NamedCountSummary(display=empty, full=empty, count=0)
    noun = str(singular or "item") if count == 1 else _plural(singular, plural)
    limit = max(0, int(max_items))
    examples = [
        compact_text(display_source, max_chars=item_max_chars, mode="middle").display
        for _, display_source in items[:limit]
    ]
    if examples:
        suffix = "" if count <= len(examples) else f", +{count - len(examples)} more"
        display = f"{count} {noun}: {', '.join(examples)}{suffix}"
    else:
        display = f"{count} {noun}"
    full_lines = "\n".join(f"- {full}" for full, _ in items)
    full = f"{count} {noun}:\n{full_lines}" if full_lines else display
    return NamedCountSummary(display=display, full=full, count=count)


def format_named_count_summary(
    names: Sequence[object] | Iterable[object],
    *,
    singular: str = "dataset",
    plural: str | None = None,
    max_items: int = 3,
    item_max_chars: int = STATUS_ITEM_LABEL_MAX_CHARS,
    empty_text: str = "none",
) -> str:
    """Return the compact display string for ``summarize_named_count``."""

    return summarize_named_count(
        names,
        singular=singular,
        plural=plural,
        max_items=max_items,
        item_max_chars=item_max_chars,
        empty_text=empty_text,
    ).display
