"""Headless batch initial conditions helpers and storage."""

from __future__ import annotations

import math
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .validation import validate_name

__all__ = [
    "INITIAL_CONC_STUB_LINE",
    "BatchInitialConditionsStore",
    "dataset_base_label",
    "migrate_reaction_dsl_initial_concentration_sets",
    "strip_named_reaction_dsl_initial_concentration_sets",
    "strip_reaction_dsl_initial_concentrations",
    "resolve_run_scope",
    "seed_batch_set_from_dataset_first_row",
]


INITIAL_CONC_STUB_LINE = "# Initial concentrations moved to Batch Initial Conditions table ({set_name}). Edit there."


def _new_batch_set_id() -> str:
    return f"batch-{uuid.uuid4().hex}"


def _parse_float_token(value: str) -> float:
    try:
        f = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Initial condition value must be numeric, got: {value!r}") from exc
    if not math.isfinite(f):
        raise ValueError(f"Initial condition value must be finite, got: {value!r}")
    return f


def _extract_initials_from_bracket_line(line: str) -> Dict[str, float]:
    initials: Dict[str, float] = {}
    for item in str(line).split(","):
        item = item.strip()
        if not item:
            continue
        if not (item.startswith("[") and "]" in item and "=" in item):
            continue
        bracket_end = item.index("]")
        species = item[1:bracket_end].strip()
        if not species:
            continue
        eq_pos = item.index("=", bracket_end)
        value_str = item[eq_pos + 1 :].strip()
        initials[species] = _parse_float_token(value_str)
    return initials


def _extract_initials_from_init_line(line: str) -> Dict[str, float]:
    lowered = str(line).strip().lower()
    if not (lowered.startswith("init:") or lowered.startswith("initial:")):
        return {}
    _, rest = str(line).split(":", 1)
    rest = rest.strip()
    if not rest:
        return {}
    initials: Dict[str, float] = {}
    parts = [p.strip() for p in rest.split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        species = key.strip()
        if not species:
            continue
        initials[species] = _parse_float_token(val.strip())
    return initials


def _render_removed_block_stubs(
    lines: Sequence[str],
    *,
    removed: set[int],
    stubs_by_index: Dict[int, str],
) -> str:
    out: List[str] = []
    for idx, raw in enumerate(lines):
        stub = stubs_by_index.get(idx)
        if stub is not None:
            out.append(str(stub))
        if idx in removed:
            continue
        out.append(raw)
    return "\n".join(out)


_NAMED_INITIAL_SET_HEADER_RE = re.compile(
    r"^(?P<name>[^#=][^=]*?)\s*=\s*\{\s*(?:#.*)?$",
    flags=re.IGNORECASE,
)

_UNSUPPORTED_NAMED_INITIAL_SET_NAME_RE = re.compile(
    r"^(?:let|param)\s+\S",
    flags=re.IGNORECASE,
)


def _is_unsupported_named_initial_set_name(name: str) -> bool:
    return bool(_UNSUPPORTED_NAMED_INITIAL_SET_NAME_RE.match(str(name or "")))


def _match_named_initial_set_header(line: str) -> str | None:
    match = _NAMED_INITIAL_SET_HEADER_RE.match(str(line or "").strip())
    if match is None:
        return None
    name = validate_name(str(match.group("name") or ""))
    if _is_unsupported_named_initial_set_name(name):
        return None
    return name


def _match_unsupported_named_initial_set_header(line: str) -> str | None:
    match = _NAMED_INITIAL_SET_HEADER_RE.match(str(line or "").strip())
    if match is None:
        return None
    name = validate_name(str(match.group("name") or ""))
    if not _is_unsupported_named_initial_set_name(name):
        return None
    return name


def _parse_named_initial_concentration_block(
    lines: Sequence[str],
    *,
    start_idx: int,
) -> tuple[str, Dict[str, float], int] | None:
    set_name = _match_named_initial_set_header(lines[start_idx])
    if set_name is None:
        return None

    block_seed: Dict[str, float] = {}
    saw_initial_assignment = False
    cursor = int(start_idx) + 1
    while cursor < len(lines):
        block_raw = lines[cursor]
        block_stripped = block_raw.strip()
        if not block_stripped or block_stripped.startswith("#"):
            cursor += 1
            continue
        if block_stripped == "}":
            if not saw_initial_assignment:
                return set_name, block_seed, cursor
            return set_name, block_seed, cursor
        if block_stripped.startswith("[") and "=" in block_stripped:
            block_seed.update(_extract_initials_from_bracket_line(block_stripped))
            saw_initial_assignment = True
            cursor += 1
            continue
        lowered = block_stripped.lower()
        if lowered.startswith("init:") or lowered.startswith("initial:"):
            block_seed.update(_extract_initials_from_init_line(block_stripped))
            saw_initial_assignment = True
            cursor += 1
            continue
        if not saw_initial_assignment:
            return None
        raise ValueError(
            "Unsupported inline initial-concentration line in "
            f"{set_name!r}: {block_raw!r}"
        )

    raise ValueError(
        f"Inline initial-concentration set {set_name!r} is missing a closing '}}' line"
    )


def _find_named_initial_brace_block_end(lines: Sequence[str], *, start_idx: int) -> int:
    depth = 1
    cursor = int(start_idx) + 1
    while cursor < len(lines):
        stripped = str(lines[cursor] or "").strip()
        if not stripped or stripped.startswith("#"):
            cursor += 1
            continue
        if _NAMED_INITIAL_SET_HEADER_RE.match(stripped):
            depth += 1
            cursor += 1
            continue
        if stripped == "}":
            depth -= 1
            if depth == 0:
                return cursor
        cursor += 1
    return max(int(start_idx), len(lines) - 1)


def _migrate_named_initial_concentration_sets(
    text: str,
) -> Tuple[Dict[str, Dict[str, float]], str]:
    lines = str(text or "").splitlines()
    if not lines:
        return {}, str(text or "")

    removed: set[int] = set()
    stubs_by_index: Dict[int, str] = {}
    seeds_by_name: Dict[str, Dict[str, float]] = {}
    idx = 0

    while idx < len(lines):
        if _match_unsupported_named_initial_set_header(lines[idx]) is not None:
            idx = _find_named_initial_brace_block_end(lines, start_idx=idx) + 1
            continue
        parsed_block = _parse_named_initial_concentration_block(lines, start_idx=idx)
        if parsed_block is None:
            idx += 1
            continue
        set_name, block_seed, cursor = parsed_block

        if set_name in seeds_by_name:
            raise ValueError(f"Duplicate inline initial-concentration set name: {set_name!r}")

        removed.update(range(int(idx), int(cursor) + 1))

        seeds_by_name[set_name] = block_seed
        stubs_by_index[idx] = INITIAL_CONC_STUB_LINE.format(set_name=set_name)
        idx = cursor + 1

    if not seeds_by_name:
        return {}, str(text or "")
    return seeds_by_name, _render_removed_block_stubs(lines, removed=removed, stubs_by_index=stubs_by_index)


def _migrate_legacy_initial_concentrations(
    reaction_text: str,
    *,
    set_name: str,
) -> Tuple[Dict[str, float], str]:
    text = str(reaction_text or "")
    if not text.strip():
        return {}, text

    lines = text.splitlines()

    removed: set[int] = set()
    seed: Dict[str, float] = {}

    header_re = re.compile(
        r"^#\s*initial concentrations(?:\s*\([^)]*\))?\s*$",
        flags=re.IGNORECASE,
    )

    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        stripped = raw.strip()
        if not stripped:
            idx += 1
            continue
        if _match_unsupported_named_initial_set_header(raw) is not None:
            idx = _find_named_initial_brace_block_end(lines, start_idx=idx) + 1
            continue
        if stripped.startswith("#"):
            if header_re.match(stripped):
                removed.add(idx)
            idx += 1
            continue

        if stripped.startswith("[") and "=" in stripped:
            removed.add(idx)
            seed.update(_extract_initials_from_bracket_line(stripped))
            idx += 1
            continue

        lowered = stripped.lower()
        if lowered.startswith("init:") or lowered.startswith("initial:"):
            removed.add(idx)
            seed.update(_extract_initials_from_init_line(stripped))
            idx += 1
            continue
        idx += 1

    if not removed or not seed:
        return {}, text

    rewritten = _render_removed_block_stubs(
        lines,
        removed=removed,
        stubs_by_index={min(removed): INITIAL_CONC_STUB_LINE.format(set_name=set_name)},
    )
    return seed, rewritten


def _merge_legacy_seed_into_named_seeds(
    seeds_by_name: Dict[str, Dict[str, float]],
    *,
    default_set_name: str,
    legacy_seed: Dict[str, float],
) -> None:
    normalized_default = validate_name(str(default_set_name))
    existing_seed = seeds_by_name.get(normalized_default)
    if existing_seed is None:
        seeds_by_name[normalized_default] = dict(legacy_seed)
        return
    if existing_seed:
        raise ValueError(
            f"Duplicate inline initial-concentration set name while importing anonymous inline initials: {normalized_default!r}"
        )
    merged_seed = dict(existing_seed)
    merged_seed.update(dict(legacy_seed))
    seeds_by_name[normalized_default] = merged_seed


def migrate_reaction_dsl_initial_concentration_sets(
    reaction_text: str,
    *,
    default_set_name: str = "set1",
) -> Tuple[Dict[str, Dict[str, float]], str]:
    """
    Import inline initial concentrations from reaction text into named batch-set seeds.

    Supports both:
    - anonymous inline initials (`# Initial concentrations`, `[A]=...`, `initial: ...`)
    - named import-only blocks (`set_name = { ... }`)

    Returned mappings are import-only seeds; concentrations remain authoritative in
    the Batch Initial Conditions table after helper-based materialization or stripping.
    """
    text = str(reaction_text or "")
    if not text.strip():
        return {}, text

    seeds_by_name, rewritten = _migrate_named_initial_concentration_sets(text)
    legacy_seed, rewritten = _migrate_legacy_initial_concentrations(
        rewritten,
        set_name=str(default_set_name),
    )
    if legacy_seed:
        _merge_legacy_seed_into_named_seeds(
            seeds_by_name,
            default_set_name=str(default_set_name),
            legacy_seed=legacy_seed,
        )
    return seeds_by_name, rewritten


def _strip_named_initial_concentration_sets(text: str) -> str:
    lines = str(text or "").splitlines()
    if not lines:
        return str(text or "")
    removed: set[int] = set()
    idx = 0
    while idx < len(lines):
        if _match_unsupported_named_initial_set_header(lines[idx]) is not None:
            idx = _find_named_initial_brace_block_end(lines, start_idx=idx) + 1
            continue
        parsed_block = _parse_named_initial_concentration_block(lines, start_idx=idx)
        if parsed_block is None:
            idx += 1
            continue
        _set_name, _seed, cursor = parsed_block
        removed.update(range(int(idx), int(cursor) + 1))
        idx = cursor + 1
    if not removed:
        return str(text or "")
    return "\n".join(raw for idx, raw in enumerate(lines) if idx not in removed)


def strip_named_reaction_dsl_initial_concentration_sets(reaction_text: str) -> str:
    """Return reaction DSL with only supported named initial-concentration blocks removed."""
    text = str(reaction_text or "")
    if not text.strip():
        return text
    return _strip_named_initial_concentration_sets(text)


def strip_reaction_dsl_initial_concentrations(reaction_text: str) -> str:
    """
    Return reaction DSL with any inline initial-concentration lines removed.

    This is used to ensure the Batch Initial Conditions table is the source-of-truth
    for simulations after migration. The editor rewrite is handled separately.
    """
    text = str(reaction_text or "")
    if not text.strip():
        return text
    text = _strip_named_initial_concentration_sets(text)
    lines = text.splitlines()
    removed: set[int] = set()

    header_re = re.compile(
        r"^#\s*initial concentrations(?:\s*\([^)]*\))?\s*$",
        flags=re.IGNORECASE,
    )
    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        stripped = raw.strip()
        if not stripped:
            idx += 1
            continue
        if _match_unsupported_named_initial_set_header(raw) is not None:
            idx = _find_named_initial_brace_block_end(lines, start_idx=idx) + 1
            continue
        if stripped.startswith("#"):
            if header_re.match(stripped):
                removed.add(idx)
            idx += 1
            continue
        if stripped.startswith("[") and "=" in stripped:
            removed.add(idx)
            idx += 1
            continue
        lowered = stripped.lower()
        if lowered.startswith("init:") or lowered.startswith("initial:"):
            removed.add(idx)
            idx += 1
            continue
        idx += 1

    if not removed:
        return text
    return "\n".join(raw for idx, raw in enumerate(lines) if idx not in removed)


@dataclass
class BatchSet:
    name: str
    values: Dict[str, str] = field(default_factory=dict)
    set_id: str = field(default_factory=_new_batch_set_id)
    requested_show: bool = True


class BatchInitialConditionsStore:
    """
    Pure-python store backing the Batch Initial Conditions table.

    Stores raw strings per cell to allow invalid values to exist until runtime validation.
    """

    def __init__(self, sets: Iterable[BatchSet] | None = None) -> None:
        if sets is None:
            self._sets: List[BatchSet] = [BatchSet("set1")]
        else:
            self._sets = [
                BatchSet(
                    str(s.name),
                    dict(s.values),
                    str(getattr(s, "set_id", "") or _new_batch_set_id()),
                    bool(getattr(s, "requested_show", True)),
                )
                for s in sets
            ]
            if not self._sets:
                self._sets = [BatchSet("set1")]
        self._normalize_set_ids()
        self._visible_species: List[str] = []

    # ---------------- public API ----------------
    def row_count(self) -> int:
        return len(self._sets)

    def column_count(self) -> int:
        return 1 + len(self._visible_species)

    def set_species(self, species_names: Sequence[str]) -> None:
        visible = [str(s) for s in (species_names or []) if str(s)]
        self._visible_species = visible
        for batch in self._sets:
            for sp in visible:
                batch.values.setdefault(sp, "0.0")

    def visible_species(self) -> List[str]:
        return list(self._visible_species)

    def ensure_set(self, name: str) -> int:
        name = str(name or "").strip() or "set"
        for idx, batch in enumerate(self._sets):
            if batch.name == name:
                return idx
        # Include any hidden species keys that were previously present so new rows
        # do not lose the ability to round-trip values if species are removed and
        # later reintroduced.
        all_species: set[str] = set(self._visible_species)
        for existing in self._sets:
            values = getattr(existing, "values", None)
            if isinstance(values, dict):
                all_species.update(map(str, values.keys()))
        new = BatchSet(name)
        for sp in sorted(all_species):
            if not sp:
                continue
            new.values.setdefault(sp, "0.0")
        self._sets.append(new)
        return len(self._sets) - 1

    def row_for_set_id(self, set_id: str) -> int | None:
        target = str(set_id or "").strip()
        if not target:
            return None
        for idx, batch in enumerate(self._sets):
            if str(batch.set_id) == target:
                return idx
        return None

    def set_id_for_row(self, row: int) -> str:
        if not (0 <= int(row) < len(self._sets)):
            raise IndexError("row out of range")
        return str(self._sets[int(row)].set_id)

    def is_requested_show(self, row: int) -> bool:
        if not (0 <= int(row) < len(self._sets)):
            raise IndexError("row out of range")
        return bool(self._sets[int(row)].requested_show)

    def set_requested_show(self, row: int, requested_show: bool) -> None:
        if not (0 <= int(row) < len(self._sets)):
            raise IndexError("row out of range")
        self._sets[int(row)].requested_show = bool(requested_show)

    def requested_show_set_ids(self) -> List[str]:
        return [str(batch.set_id) for batch in self._sets if bool(batch.requested_show)]

    def set_name_for_row(self, row: int) -> str:
        if not (0 <= int(row) < len(self._sets)):
            raise IndexError("row out of range")
        return str(self._sets[int(row)].name)

    def set_name_for_set_id(self, set_id: str) -> str | None:
        row = self.row_for_set_id(set_id)
        if row is None:
            return None
        return self.set_name_for_row(int(row))

    def set_ids(self) -> List[str]:
        return [str(s.set_id) for s in self._sets]

    def delete_sets_by_ids(self, set_ids: Sequence[str]) -> List[str]:
        targets = {str(v) for v in (set_ids or []) if str(v)}
        if not targets:
            return []
        removed = [str(s.set_id) for s in self._sets if str(s.set_id) in targets]
        if not removed:
            return []
        self._sets = [s for s in self._sets if str(s.set_id) not in targets]
        return removed

    def move_rows(self, rows: Sequence[int], delta: int) -> List[int]:
        """
        Move the specified row indices up/down by `delta` positions.

        Returns the new row indices for the moved selection.
        """
        try:
            delta = int(delta)
        except Exception:
            return [int(r) for r in (rows or [])]
        if delta == 0 or self.row_count() <= 1:
            return [int(r) for r in (rows or [])]

        n = int(self.row_count())
        selected = sorted({int(r) for r in (rows or []) if 0 <= int(r) < n})
        if not selected:
            return []

        steps = abs(delta)
        direction = -1 if delta < 0 else 1
        active: set[int] = set(selected)

        for _ in range(steps):
            if direction < 0:
                # Move up: iterate ascending so earlier rows shift first.
                for r in sorted(active):
                    if r <= 0:
                        continue
                    if (r - 1) in active:
                        continue
                    self._sets[r - 1], self._sets[r] = self._sets[r], self._sets[r - 1]
                    active.remove(r)
                    active.add(r - 1)
            else:
                # Move down: iterate descending so later rows shift first.
                for r in sorted(active, reverse=True):
                    if r >= n - 1:
                        continue
                    if (r + 1) in active:
                        continue
                    self._sets[r + 1], self._sets[r] = self._sets[r], self._sets[r + 1]
                    active.remove(r)
                    active.add(r + 1)

        return sorted(active)

    def set_value(self, row: int, species: str, value: str) -> None:
        if not (0 <= int(row) < len(self._sets)):
            raise IndexError("row out of range")
        sp = str(species)
        self._sets[int(row)].values[sp] = str(value)

    def get_value(self, row: int, species: str) -> str:
        if not (0 <= int(row) < len(self._sets)):
            raise IndexError("row out of range")
        sp = str(species)
        return str(self._sets[int(row)].values.get(sp, "0.0"))

    def set_set_name(self, row: int, name: str) -> None:
        if not (0 <= int(row) < len(self._sets)):
            raise IndexError("row out of range")
        self._sets[int(row)].name = str(name or "").strip()

    def set_names(self) -> List[str]:
        return [s.name for s in self._sets]

    def row_for_set(self, name: str) -> int | None:
        target = str(name or "").strip()
        if not target:
            return None
        for idx, batch in enumerate(self._sets):
            if batch.name == target:
                return idx
        return None

    def values_for_set(self, name: str) -> Dict[str, str]:
        idx = self.row_for_set(name)
        if idx is None:
            return {}
        return dict(self._sets[idx].values)

    def apply_paste_block(self, *, start_row: int, start_col: int, text: str) -> List[Tuple[int, int]]:
        """
        Apply a tab/newline-delimited block paste at (start_row, start_col).

        Columns are: 0=SetName, 1..N species (in current visible order).

        Raises
        ------
        ValueError
            If the paste would exceed current table bounds.
        """
        start_row = int(start_row)
        start_col = int(start_col)
        if start_row < 0 or start_col < 0:
            raise ValueError("start_row/start_col must be non-negative")

        raw_text = str(text or "")
        rows_raw = raw_text.splitlines()
        if not rows_raw:
            return []

        block = [row.split("\t") for row in rows_raw]
        block_rows = len(block)
        block_cols = max((len(r) for r in block), default=0)
        if block_rows <= 0 or block_cols <= 0:
            return []

        if start_row + block_rows > self.row_count() or start_col + block_cols > self.column_count():
            raise ValueError("paste exceeds table bounds")

        changed: List[Tuple[int, int]] = []
        for r in range(block_rows):
            for c in range(block_cols):
                val = block[r][c] if c < len(block[r]) else ""
                target_row = start_row + r
                target_col = start_col + c
                if target_col == 0:
                    next_name = str(val or "").strip()
                    if self._sets[target_row].name == next_name:
                        continue
                    self.set_set_name(target_row, val)
                else:
                    sp = self._visible_species[target_col - 1]
                    if self.get_value(target_row, sp) == str(val):
                        continue
                    self.set_value(target_row, sp, val)
                changed.append((target_row, target_col))
        return changed

    def validate_numeric_cells(self, *, rows: Sequence[int]) -> set[Tuple[int, str]]:
        invalid: set[Tuple[int, str]] = set()
        for row in rows:
            r = int(row)
            if not (0 <= r < len(self._sets)):
                continue
            batch = self._sets[r]
            for sp in self._visible_species:
                raw = str(batch.values.get(sp, "")).strip()
                try:
                    f = float(raw)
                except (TypeError, ValueError):
                    invalid.add((r, sp))
                    continue
                if not math.isfinite(f):
                    invalid.add((r, sp))
        return invalid

    def as_serializable(self) -> Dict[str, object]:
        return {
            "sets": [
                {
                    "set_id": str(s.set_id),
                    "name": s.name,
                    "values": dict(s.values),
                    "requested_show": bool(s.requested_show),
                }
                for s in self._sets
            ],
            "visible_species": list(self._visible_species),
        }

    @classmethod
    def from_serializable(cls, payload: Dict[str, object]) -> "BatchInitialConditionsStore":
        sets_in = payload.get("sets") if isinstance(payload, dict) else None
        sets: List[BatchSet] = []
        if isinstance(sets_in, list):
            for entry in sets_in:
                if not isinstance(entry, dict):
                    continue
                set_id = str(entry.get("set_id") or "").strip() or _new_batch_set_id()
                name = str(entry.get("name") or "").strip() or "set"
                values = entry.get("values") or {}
                if not isinstance(values, dict):
                    values = {}
                sets.append(
                    BatchSet(
                        name=name,
                        values={str(k): str(v) for k, v in values.items()},
                        set_id=set_id,
                        requested_show=bool(entry.get("requested_show", True)),
                    )
                )
        store = cls(sets=sets)
        visible = payload.get("visible_species") if isinstance(payload, dict) else None
        if isinstance(visible, list):
            store.set_species([str(s) for s in visible])
        return store

    def _normalize_set_ids(self) -> None:
        used: set[str] = set()
        for batch in self._sets:
            sid = str(getattr(batch, "set_id", "") or "").strip()
            if not sid or sid in used:
                sid = _new_batch_set_id()
            batch.set_id = sid
            used.add(sid)


def resolve_run_scope(
    *,
    selected_rows: Sequence[int],
    total_rows: int,
    mode: str,
) -> List[int]:
    total_rows = int(total_rows)
    if total_rows <= 0:
        return []
    mode = str(mode or "").strip().lower()
    if mode == "all":
        return list(range(total_rows))
    # selected mode
    sel = [int(r) for r in (selected_rows or []) if 0 <= int(r) < total_rows]
    return list(sel)


# DataManagerPanel appends internal duplicate suffixes as "_1", "_2", ... (no leading
# zeros). Avoid stripping labels like "dataset_01.csv" where the underscore digits
# are part of the real filename.
_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_(?P<idx>[1-9]\d*)$")


def dataset_base_label(name: str) -> str:
    """Return dataset label before any internal _N suffixing."""
    label = str(name or "").strip()
    if not label:
        return ""
    base, _ext = os.path.splitext(label)
    if not base:
        base = label
    while True:
        m = _SUFFIX_RE.match(base)
        if not m:
            break
        base = str(m.group("base"))
    return base


def seed_batch_set_from_dataset_first_row(
    dataset: Dict[str, object],
    mechanism_species: Sequence[str],
    *,
    tol: float,
) -> Dict[str, float]:
    """
    If dataset starts at ~t=0 (abs(t0)<=tol), seed species present from row 0 and
    missing species to 0. Otherwise return empty dict.
    """
    try:
        t = np.asarray((dataset or {}).get("t", []), dtype=float).reshape(-1)
    except Exception:
        t = np.asarray([], dtype=float)
    if t.size == 0:
        return {}
    t0 = float(t[0])
    if not (abs(t0) <= float(tol)):
        return {}
    species_map = (dataset or {}).get("species") or {}
    if not isinstance(species_map, dict):
        species_map = {}

    seeded: Dict[str, float] = {}
    for sp in mechanism_species:
        sp_name = str(sp)
        if sp_name in species_map:
            try:
                y = np.asarray(species_map[sp_name], dtype=float).reshape(-1)
                seeded[sp_name] = float(y[0]) if y.size else 0.0
            except Exception:
                seeded[sp_name] = 0.0
        else:
            seeded[sp_name] = 0.0
    return seeded
