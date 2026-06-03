from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def normalize_preview_target_set_ids(values: object) -> tuple[str, ...]:
    """Canonical preview target identity normalizer.

    Preview ownership, replay state, runtime launch intent, controller currentness,
    and runtime prewarm checks must all compare target set identities through this
    function.  It strips string-like ids, drops empties, preserves first-seen
    order, and de-duplicates by normalized id.
    """
    if not values:
        return ()
    if isinstance(values, (str, bytes)):
        values = (values,)
    elif isinstance(values, Mapping):
        values = values.keys()
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:  # type: ignore[union-attr]
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class PreviewTargetIdentity:
    set_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "set_ids", normalize_preview_target_set_ids(self.set_ids))

    @classmethod
    def from_raw(cls, values: object) -> "PreviewTargetIdentity":
        return cls(normalize_preview_target_set_ids(values))

    def matches(self, values: object) -> bool:
        return self.set_ids == normalize_preview_target_set_ids(values)
