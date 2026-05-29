"""Headless batch initial conditions helpers and storage."""

from __future__ import annotations

import math
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .validation import validate_name

__all__ = [
    "BatchInitialConditionsStore",
    "InitialConditionImportEvent",
    "ReactionInitialConditionImportExtraction",
    "batch_initial_conditions_store_is_true_placeholder",
    "dataset_base_label",
    "extract_reaction_dsl_initial_condition_imports",
    "migrate_reaction_dsl_initial_concentration_sets",
    "reaction_dsl_with_initial_condition_import_provenance",
    "reaction_dsl_with_parseable_initial_concentrations",
    "strip_named_reaction_dsl_initial_concentration_sets",
    "strip_reaction_dsl_initial_concentrations",
    "resolve_run_scope",
    "seed_batch_set_from_dataset_first_row",
]


_INITIAL_CONC_PROVENANCE_COMMENT = (
    "# Initial Conditions moved to Batch Initial Conditions table ({set_name}): "
    "{assignments}. Table values are authoritative."
)


@dataclass(frozen=True)
class InitialConditionImportEvent:
    import_id: str
    source_kind: str
    source_name: str
    values: Dict[str, float]
    insert_index: int

    @property
    def value_bearing(self) -> bool:
        return bool(dict(self.values or {}))


@dataclass(frozen=True)
class ReactionInitialConditionImportExtraction:
    clean_reactions_text: str
    imports: Tuple[InitialConditionImportEvent, ...] = ()


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


def _format_initial_condition_provenance_comment(set_name: str, seed: Dict[str, float]) -> str | None:
    values = dict(seed or {})
    if not values:
        return None
    assignments = ", ".join(
        f"{species}={float(value):.6g}"
        for species, value in values.items()
        if str(species)
    )
    if not assignments:
        return None
    return _INITIAL_CONC_PROVENANCE_COMMENT.format(
        set_name=validate_name(str(set_name)),
        assignments=assignments,
    )


def _render_import_provenance(
    clean_text: str,
    imports: Sequence[InitialConditionImportEvent],
    *,
    destination_names: Mapping[str, str],
) -> str:
    lines = str(clean_text or "").splitlines()
    stubs_by_index: Dict[int, List[str]] = {}
    for event in imports:
        if not event.value_bearing:
            continue
        destination = str(destination_names.get(event.import_id, event.source_name) or event.source_name)
        stub = _format_initial_condition_provenance_comment(destination, dict(event.values))
        if stub is None:
            continue
        insert_at = max(0, min(int(event.insert_index), len(lines)))
        stubs_by_index.setdefault(insert_at, []).append(stub)

    if not stubs_by_index:
        return str(clean_text or "")

    out: List[str] = []
    for idx in range(len(lines) + 1):
        out.extend(stubs_by_index.get(idx, []))
        if idx < len(lines):
            out.append(lines[idx])
    return "\n".join(out)


_NAMED_INITIAL_SET_HEADER_RE = re.compile(
    r"^(?P<name>[^#=][^=]*?)\s*=\s*\{\s*(?:#.*)?$",
    flags=re.IGNORECASE,
)

_NAMED_INITIAL_SET_ONE_LINE_RE = re.compile(
    r"^(?P<name>[^#=][^=]*?)\s*=\s*\{\s*(?P<body>.*?)\s*\}\s*(?:#.*)?$",
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


def _parse_initials_from_named_block_body(set_name: str, body: str) -> Dict[str, float]:
    stripped = str(body or "").strip()
    if not stripped:
        return {}
    lowered = stripped.lower()
    if lowered.startswith("init:") or lowered.startswith("initial:"):
        return _extract_initials_from_init_line(stripped)
    if stripped.startswith("[") and "=" in stripped:
        return _extract_initials_from_bracket_line(stripped)
    raise ValueError(
        "Unsupported inline initial-concentration line in "
        f"{set_name!r}: {stripped!r}"
    )


def _strip_inline_initial_condition_comment(text: str) -> str:
    return str(text or "").split("#", 1)[0].strip()


def _looks_like_one_line_initial_condition_body(body: str) -> bool:
    stripped = _strip_inline_initial_condition_comment(body)
    if not stripped:
        return True
    lowered = stripped.lower()
    return bool(
        lowered.startswith("init:")
        or lowered.startswith("initial:")
        or (stripped.startswith("[") and "=" in stripped)
    )


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
    one_line_match = _NAMED_INITIAL_SET_ONE_LINE_RE.match(str(lines[start_idx] or "").strip())
    if one_line_match is not None:
        set_name = validate_name(str(one_line_match.group("name") or ""))
        if _is_unsupported_named_initial_set_name(set_name):
            return None
        body = str(one_line_match.group("body") or "")
        if not _looks_like_one_line_initial_condition_body(body):
            return None
        block_seed = _parse_initials_from_named_block_body(
            set_name,
            _strip_inline_initial_condition_comment(body),
        )
        return set_name, block_seed, int(start_idx)

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


def _extract_named_initial_concentration_sets(
    text: str,
) -> tuple[str, Tuple[InitialConditionImportEvent, ...]]:
    lines = str(text or "").splitlines()
    if not lines:
        return str(text or ""), ()

    removed: set[int] = set()
    events: list[tuple[str, Dict[str, float], int]] = []
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

        removed.update(range(int(idx), int(cursor) + 1))
        if any(existing_name == set_name for existing_name, _seed, _start in events):
            raise ValueError(f"Duplicate inline initial-concentration set name: {set_name!r}")
        events.append((set_name, block_seed, int(idx)))
        idx = cursor + 1

    if not events:
        return str(text or ""), ()

    clean_text = "\n".join(raw for idx, raw in enumerate(lines) if idx not in removed)
    import_events = tuple(
        InitialConditionImportEvent(
            import_id=f"named:{index}:{set_name}",
            source_kind="named_block",
            source_name=str(set_name),
            values=dict(seed),
            insert_index=sum(1 for line_idx in range(start_idx) if line_idx not in removed),
        )
        for index, (set_name, seed, start_idx) in enumerate(events)
    )
    return clean_text, import_events


def _extract_legacy_initial_concentrations(
    reaction_text: str,
    *,
    set_name: str,
) -> tuple[Dict[str, float], str, int | None, set[int]]:
    text = str(reaction_text or "")
    if not text.strip():
        return {}, text, None, set()

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
        return {}, text, None, set()

    insert_index = sum(1 for line_idx in range(min(removed)) if line_idx not in removed)
    rewritten = "\n".join(raw for idx, raw in enumerate(lines) if idx not in removed)
    return seed, rewritten, insert_index, set(removed)


def _shift_insert_index_after_removed_lines(index: int, removed: set[int]) -> int:
    return int(index) - sum(1 for line_idx in removed if line_idx < int(index))


def extract_reaction_dsl_initial_condition_imports(
    reaction_text: str,
    *,
    default_set_name: str = "set1",
) -> ReactionInitialConditionImportExtraction:
    named_clean_text, named_events = _extract_named_initial_concentration_sets(str(reaction_text or ""))
    legacy_seed, clean_text, legacy_insert_index, legacy_removed = _extract_legacy_initial_concentrations(
        named_clean_text,
        set_name=str(default_set_name),
    )

    events: list[InitialConditionImportEvent] = [
        InitialConditionImportEvent(
            import_id=event.import_id,
            source_kind=event.source_kind,
            source_name=event.source_name,
            values=dict(event.values),
            insert_index=_shift_insert_index_after_removed_lines(event.insert_index, legacy_removed),
        )
        for event in named_events
    ]

    if legacy_seed:
        normalized_default = validate_name(str(default_set_name))
        matching_index = next(
            (
                index
                for index, event in enumerate(events)
                if event.source_name == normalized_default
            ),
            None,
        )
        if matching_index is None:
            events.append(
                InitialConditionImportEvent(
                    import_id=f"anonymous:{len(events)}:{normalized_default}",
                    source_kind="anonymous_inline",
                    source_name=normalized_default,
                    values=dict(legacy_seed),
                    insert_index=int(legacy_insert_index or 0),
                )
            )
        else:
            existing = events[matching_index]
            if dict(existing.values):
                raise ValueError(
                    "Duplicate inline initial-concentration set name while importing anonymous inline initials: "
                    f"{normalized_default!r}"
                )
            events[matching_index] = InitialConditionImportEvent(
                import_id=existing.import_id,
                source_kind=existing.source_kind,
                source_name=existing.source_name,
                values=dict(legacy_seed),
                insert_index=existing.insert_index,
            )

    return ReactionInitialConditionImportExtraction(
        clean_reactions_text=str(clean_text),
        imports=tuple(events),
    )


def reaction_dsl_with_initial_condition_import_provenance(
    extraction: ReactionInitialConditionImportExtraction,
    *,
    destination_names: Mapping[str, str] | None = None,
) -> str:
    destinations = dict(destination_names or {})
    return _render_import_provenance(
        str(extraction.clean_reactions_text or ""),
        tuple(extraction.imports or ()),
        destination_names=destinations,
    )


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

    extraction = extract_reaction_dsl_initial_condition_imports(
        text,
        default_set_name=str(default_set_name),
    )
    seeds_by_name = {
        str(event.source_name): dict(event.values)
        for event in extraction.imports
        if event.value_bearing
    }
    rewritten = reaction_dsl_with_initial_condition_import_provenance(
        extraction,
        destination_names={event.import_id: event.source_name for event in extraction.imports},
    )
    return seeds_by_name, rewritten


def reaction_dsl_with_parseable_initial_concentrations(
    reaction_text: str,
    *,
    preferred_set_name: str | None = None,
    default_set_name: str = "set1",
) -> str:
    """
    Convert authoring-time IC import syntax to parser-safe reaction DSL.

    The returned text contains no named import blocks. If authoring IC seeds are
    present, one seed set is rendered as ordinary `initial:` DSL for direct
    parser/solver consumers that do not have a Batch Initial Conditions table.
    """
    extraction = extract_reaction_dsl_initial_condition_imports(
        str(reaction_text or ""),
        default_set_name=str(default_set_name),
    )
    seeds_by_name = {
        str(event.source_name): dict(event.values)
        for event in extraction.imports
        if event.value_bearing
    }
    rewritten = str(extraction.clean_reactions_text)
    if not seeds_by_name:
        return str(rewritten)

    selected_name = str(preferred_set_name or "").strip()
    if selected_name not in seeds_by_name:
        selected_name = next(iter(seeds_by_name.keys()))
    seed = dict(seeds_by_name.get(selected_name) or {})
    if not seed:
        return str(rewritten)

    assignments = [
        f"{species}={float(value):.6g}"
        for species, value in seed.items()
    ]
    initial_line = "initial: " + ", ".join(assignments)
    if str(rewritten).strip():
        return str(rewritten).rstrip() + "\n" + initial_line
    return initial_line


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
        self._explicit_serialized_initial_conditions = False

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
        visible_species = [str(s) for s in visible if str(s)] if isinstance(visible, list) else []
        if not visible_species and isinstance(sets_in, list):
            seen_species: set[str] = set()
            for entry in sets_in:
                if not isinstance(entry, dict):
                    continue
                values = entry.get("values")
                if not isinstance(values, dict):
                    continue
                for key in values.keys():
                    species = str(key)
                    if species and species not in seen_species:
                        visible_species.append(species)
                        seen_species.add(species)
        if visible_species:
            store.set_species(visible_species)
        explicit_visible_species = isinstance(visible, list) and any(str(s) for s in visible)
        explicit_values = False
        if isinstance(sets_in, list):
            for entry in sets_in:
                if not isinstance(entry, dict):
                    continue
                values = entry.get("values")
                if isinstance(values, dict) and any(str(k) for k in values.keys()):
                    explicit_values = True
                    break
        store._explicit_serialized_initial_conditions = bool(explicit_visible_species or explicit_values)
        return store

    def _normalize_set_ids(self) -> None:
        used: set[str] = set()
        for batch in self._sets:
            sid = str(getattr(batch, "set_id", "") or "").strip()
            if not sid or sid in used:
                sid = _new_batch_set_id()
            batch.set_id = sid
            used.add(sid)


def batch_initial_conditions_store_is_true_placeholder(
    store: BatchInitialConditionsStore,
) -> bool:
    """Return True when a store carries only the default empty table placeholder."""
    if bool(getattr(store, "_explicit_serialized_initial_conditions", False)):
        return False
    if int(store.row_count()) != 1:
        return False
    if list(store.set_names() or []) != ["set1"]:
        return False
    values = dict(store.values_for_set("set1") or {})
    for raw in values.values():
        text = str(raw).strip()
        if not text:
            continue
        try:
            parsed = float(text)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(parsed) or abs(float(parsed)) > 1e-12:
            return False
    return True


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
