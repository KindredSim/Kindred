"""Central GUI display-name policy for layout-bearing text.

These helpers are intentionally Qt-free so policy tests can run without a GUI
runtime.  Qt widgets should use these functions before putting user/project
labels into labels, tabs, list rows, status/footer messages, or plot titles.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CompactText",
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
    "compact_text",
    "dataset_alias",
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
