"""Centralized GUI color and style ownership for species plots."""

from __future__ import annotations

from typing import Iterable, Sequence

from PySide6 import QtCore, QtGui

__all__ = ["ColorManager"]


class ColorManager:
    """
    Singleton source of truth for live GUI species colors and experiment styles.

    Species colors are owned globally and assigned deterministically from a
    24-slot Okabe-Ito-derived palette:
    - 8 base hues (black excluded from the cycle)
    - 8 darker variants
    - 8 lighter variants

    Overflow beyond 24 species is deterministic and wraps by slot index modulo
    24 after the first 24 assigned canonical species.
    """

    _instance: "ColorManager | None" = None

    _SPECIES_SUFFIXES: tuple[str, ...] = ("_conc", "_concentration")
    _DATASET_SYMBOLS: tuple[str, ...] = ("o", "t", "s", "d", "+", "x", "p", "h")
    _DATASET_LINE_STYLES: tuple[QtCore.Qt.PenStyle, ...] = (
        QtCore.Qt.PenStyle.DashLine,
        QtCore.Qt.PenStyle.DotLine,
        QtCore.Qt.PenStyle.DashDotLine,
        QtCore.Qt.PenStyle.DashDotDotLine,
    )
    _NON_SPECIES_PALETTE: tuple[tuple[int, int, int], ...] = (
        (80, 80, 80),
        (110, 110, 110),
        (90, 105, 120),
        (120, 105, 90),
        (95, 120, 105),
        (120, 95, 105),
    )
    _BASE_SPECIES_RGB: tuple[tuple[int, int, int], ...] = (
        (230, 159, 0),    # Okabe-Ito orange
        (86, 180, 233),   # Okabe-Ito sky blue
        (0, 158, 115),    # Okabe-Ito bluish green
        (240, 228, 66),   # Okabe-Ito yellow
        (0, 114, 178),    # Okabe-Ito blue
        (213, 94, 0),     # Okabe-Ito vermillion
        (204, 121, 167),  # Okabe-Ito reddish purple
        (0, 134, 149),    # Okabe-Ito-derived teal (fills the reserved black slot)
    )

    def __new__(cls) -> "ColorManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._species_slots: dict[str, int] = {}
        self._registered_species_roster: tuple[str, ...] = ()
        self._current_roster_preview_slots: dict[str, int] = {}
        self._species_palette: tuple[QtGui.QColor, ...] = self._build_species_palette()

    @classmethod
    def instance(cls) -> "ColorManager":
        return cls()

    @classmethod
    def reset_for_tests(cls) -> None:
        """Reset singleton state for deterministic regression tests."""
        cls._instance = None

    def species_palette(self) -> tuple[QtGui.QColor, ...]:
        return tuple(QtGui.QColor(color) for color in self._species_palette)

    def registered_species_names(self) -> tuple[str, ...]:
        """Return the current authoritative real-species roster."""
        return tuple(self._registered_species_roster)

    def set_species_roster(self, species_names: Iterable[str]) -> None:
        """
        Replace the authoritative real-species roster and persist missing slot owners.

        Exact names always win. Alias handling is only applied at lookup time against
        this registered roster; registration itself never strips supported suffixes.
        """
        cleaned = self._set_registered_species_roster(species_names)
        self._commit_registered_species_slots(cleaned)
        self._refresh_current_roster_preview_slots(cleaned)

    def set_current_species_roster(self, species_names: Iterable[str]) -> None:
        """
        Replace the authoritative real-species roster without consuming new slots.

        This is used for live draft mechanism edits where alias resolution must track
        the current parseable roster, but abandoned drafts must not permanently affect
        long-lived color ownership.
        """
        cleaned = self._set_registered_species_roster(species_names)
        self._refresh_current_roster_preview_slots(cleaned)

    def _set_registered_species_roster(self, species_names: Iterable[str]) -> tuple[str, ...]:
        cleaned = tuple(sorted(self._clean_species_names(species_names), key=lambda item: item.casefold()))
        self._registered_species_roster = cleaned
        return cleaned

    def resolve_species_key(
        self,
        species_name: str,
        *,
        known_species: Sequence[str] | None = None,
    ) -> str:
        raw = str(species_name or "").strip()
        if not raw:
            return ""
        canonical = self.resolve_current_species_key(raw, known_species=known_species)
        return canonical if canonical is not None else raw

    def resolve_current_species_key(
        self,
        species_name: str,
        *,
        known_species: Sequence[str] | None = None,
    ) -> str | None:
        raw = str(species_name or "").strip()
        if not raw:
            return None

        roster = self._resolution_roster(known_species)
        if not roster:
            return None

        return self.resolve_known_species_key(raw, roster)

    def resolve_known_species_key(
        self,
        species_name: str,
        known_species: Sequence[str],
    ) -> str | None:
        """Resolve a name against only the supplied roster, ignoring global state."""
        raw = str(species_name or "").strip()
        if not raw:
            return None

        known = [str(name).strip() for name in (known_species or ()) if str(name).strip()]
        if not known:
            return None
        exact = {name: name for name in known}
        if raw in exact:
            return exact[raw]

        lower_map = {name.casefold(): name for name in known}
        raw_lower = raw.casefold()
        if raw_lower in lower_map:
            return lower_map[raw_lower]

        stripped = self._strip_species_suffix(raw)
        if stripped is not None:
            stripped_lower = stripped.casefold()
            if stripped_lower in lower_map:
                return lower_map[stripped_lower]

        return None

    def seed_species(self, species_names: Iterable[str]) -> None:
        """Append exact species names without changing existing slot ownership."""
        for name in sorted(self._clean_species_names(species_names), key=lambda item: item.casefold()):
            if name in self._species_slots:
                continue
            self._species_slots[name] = len(self._species_slots)

    def get_species_color(
        self,
        species_name: str,
        *,
        known_species: Sequence[str] | None = None,
    ) -> QtGui.QColor:
        canonical = self.resolve_species_key(species_name, known_species=known_species)
        if not canonical:
            return QtGui.QColor(*self._NON_SPECIES_PALETTE[0])
        current_slot = self._registered_roster_slot(canonical)
        if current_slot is not None:
            return self._species_color_for_slot(current_slot)
        return self._species_color_for_key(canonical)

    def get_current_species_color(
        self,
        species_name: str,
        *,
        known_species: Sequence[str] | None = None,
    ) -> QtGui.QColor | None:
        canonical = self.resolve_current_species_key(species_name, known_species=known_species)
        if not canonical:
            return None
        current_slot = self._registered_roster_slot(canonical)
        if current_slot is not None:
            return self._species_color_for_slot(current_slot)
        return self._species_color_for_key(canonical)

    def get_species_rgb(
        self,
        species_name: str,
        *,
        known_species: Sequence[str] | None = None,
    ) -> tuple[int, int, int]:
        color = self.get_species_color(species_name, known_species=known_species)
        return (int(color.red()), int(color.green()), int(color.blue()))

    def get_non_species_color(self, series_name: str) -> QtGui.QColor:
        key = str(series_name or "").strip()
        if not key:
            return QtGui.QColor(*self._NON_SPECIES_PALETTE[0])
        idx = sum(ord(ch) for ch in key) % len(self._NON_SPECIES_PALETTE)
        return QtGui.QColor(*self._NON_SPECIES_PALETTE[idx])

    def get_dataset_symbol(self, dataset_index: int) -> str:
        return self._DATASET_SYMBOLS[int(dataset_index) % len(self._DATASET_SYMBOLS)]

    def get_dataset_line_style(self, dataset_index: int) -> QtCore.Qt.PenStyle:
        return self._DATASET_LINE_STYLES[int(dataset_index) % len(self._DATASET_LINE_STYLES)]

    @staticmethod
    def _clean_species_names(species_names: Iterable[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw_name in species_names or []:
            name = str(raw_name or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            cleaned.append(name)
        return cleaned

    def _resolution_roster(self, known_species: Sequence[str] | None) -> tuple[str, ...]:
        if self._registered_species_roster:
            return tuple(self._registered_species_roster)
        return tuple(self._clean_species_names(known_species or ()))

    @staticmethod
    def _fallback_palette_slot(species_name: str) -> int:
        raw = str(species_name or "").strip()
        if not raw:
            return 0
        return sum(ord(ch) for ch in raw) % 24

    def _species_color_for_key(self, canonical: str) -> QtGui.QColor:
        slot = self._species_slots.get(canonical)
        if slot is None:
            slot = self._fallback_palette_slot(canonical)
        return self._species_color_for_slot(slot)

    def _registered_roster_slot(self, canonical: str) -> int | None:
        if canonical not in self._registered_species_roster:
            return None
        slot = self._current_roster_preview_slots.get(canonical)
        if slot is not None:
            return slot
        return self._species_slots.get(canonical)

    def _refresh_current_roster_preview_slots(self, roster: Sequence[str]) -> None:
        preview_slots: dict[str, int] = {}
        reserved_slots = set(self._species_slots.values())
        reserved_visible_slots = self._visible_slots(reserved_slots)

        for name in roster:
            committed_slot = self._species_slots.get(name)
            if committed_slot is None:
                continue
            preview_slots[name] = committed_slot

        preview_visible_slots = set(reserved_visible_slots)
        for name in roster:
            if name in preview_slots:
                continue
            preview_slot = self._current_roster_preview_slots.get(name)
            if preview_slot is None:
                continue
            used_slots = reserved_slots | set(preview_slots.values())
            if not self._can_use_preview_slot(
                preview_slot,
                used_slots=used_slots,
                used_visible_slots=preview_visible_slots,
            ):
                continue
            preview_slots[name] = preview_slot
            preview_visible_slots.add(self._visible_slot(preview_slot))

        for name in roster:
            if name in preview_slots:
                continue
            used_slots = reserved_slots | set(preview_slots.values())
            slot = self._next_distinct_visible_slot(
                used_slots=used_slots,
                used_visible_slots=preview_visible_slots,
            )
            if slot is None:
                slot = self._next_free_slot(used_slots)
            preview_slots[name] = slot
            preview_visible_slots.add(self._visible_slot(slot))

        self._current_roster_preview_slots = preview_slots

    def _commit_registered_species_slots(self, roster: Sequence[str]) -> None:
        used_slots = set(self._species_slots.values())
        used_visible_slots = self._visible_slots(used_slots)
        for name in roster:
            if name in self._species_slots:
                continue
            preview_slot = self._current_roster_preview_slots.get(name)
            if preview_slot is None:
                slot = self._next_distinct_visible_slot(
                    used_slots=used_slots,
                    used_visible_slots=used_visible_slots,
                )
                if slot is None:
                    slot = self._next_free_slot(used_slots)
            else:
                slot = self._next_commit_slot(
                    preview_slot,
                    used_slots=used_slots,
                    used_visible_slots=used_visible_slots,
                )
            self._species_slots[name] = slot
            used_slots.add(slot)
            used_visible_slots.add(self._visible_slot(slot))

    @staticmethod
    def _next_free_slot(used_slots: set[int]) -> int:
        slot = 0
        while slot in used_slots:
            slot += 1
        return slot

    @classmethod
    def _next_commit_slot(
        cls,
        preferred_slot: int,
        *,
        used_slots: set[int],
        used_visible_slots: set[int],
    ) -> int:
        preferred = int(preferred_slot)
        preferred_visible_slot = cls._visible_slot(preferred)
        if preferred not in used_slots and preferred_visible_slot not in used_visible_slots:
            return preferred

        distinct_slot = cls._next_distinct_visible_slot(
            used_slots=used_slots,
            used_visible_slots=used_visible_slots,
            preferred_slot=preferred,
        )
        if distinct_slot is not None:
            return distinct_slot
        return cls._next_compatible_slot(preferred, used_slots)

    @classmethod
    def _next_compatible_slot(cls, preferred_slot: int, used_slots: set[int]) -> int:
        slot = int(preferred_slot)
        while slot in used_slots:
            slot += cls._palette_size()
        return slot

    def _species_color_for_slot(self, slot: int) -> QtGui.QColor:
        return QtGui.QColor(self._species_palette[int(slot) % len(self._species_palette)])

    @classmethod
    def _palette_size(cls) -> int:
        return len(cls._BASE_SPECIES_RGB) * 3

    @classmethod
    def _visible_slot(cls, slot: int) -> int:
        return int(slot) % cls._palette_size()

    @classmethod
    def _visible_slots(cls, slots: Iterable[int]) -> set[int]:
        return {cls._visible_slot(slot) for slot in slots}

    @classmethod
    def _can_use_preview_slot(
        cls,
        slot: int,
        *,
        used_slots: set[int],
        used_visible_slots: set[int],
    ) -> bool:
        visible_slot = cls._visible_slot(slot)
        if int(slot) in used_slots:
            return False
        if visible_slot not in used_visible_slots:
            return True
        return len(used_visible_slots) >= cls._palette_size()

    @classmethod
    def _next_distinct_visible_slot(
        cls,
        *,
        used_slots: set[int],
        used_visible_slots: set[int],
        preferred_slot: int | None = None,
    ) -> int | None:
        if len(used_visible_slots) >= cls._palette_size():
            return None

        start = cls._visible_slot(preferred_slot or 0)
        for offset in range(cls._palette_size()):
            slot = (start + offset) % cls._palette_size()
            if slot in used_slots or slot in used_visible_slots:
                continue
            return slot
        return None

    @classmethod
    def _strip_species_suffix(cls, species_name: str) -> str | None:
        raw = str(species_name or "").strip()
        raw_lower = raw.casefold()
        for suffix in cls._SPECIES_SUFFIXES:
            if raw_lower.endswith(suffix) and len(raw) > len(suffix):
                return raw[: -len(suffix)]
        return None

    @classmethod
    def _build_species_palette(cls) -> tuple[QtGui.QColor, ...]:
        base = [QtGui.QColor(*rgb) for rgb in cls._BASE_SPECIES_RGB]
        darker = [cls._mix(color, factor=0.22, toward_white=False) for color in base]
        lighter = [cls._mix(color, factor=0.18, toward_white=True) for color in base]
        return tuple(base + darker + lighter)

    @staticmethod
    def _mix(color: QtGui.QColor, *, factor: float, toward_white: bool) -> QtGui.QColor:
        factor = max(0.0, min(float(factor), 1.0))
        target = 255 if toward_white else 0

        def _channel(value: int) -> int:
            if toward_white:
                mixed = value + (target - value) * factor
            else:
                mixed = value * (1.0 - factor)
            return max(0, min(int(round(mixed)), 255))

        return QtGui.QColor(_channel(color.red()), _channel(color.green()), _channel(color.blue()))
