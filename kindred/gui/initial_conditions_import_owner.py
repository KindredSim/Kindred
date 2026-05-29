from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from PySide6 import QtCore

from kindred.core.batch_initial_conditions import (
    BatchInitialConditionsStore,
    InitialConditionImportEvent,
    batch_initial_conditions_store_is_true_placeholder,
    extract_reaction_dsl_initial_condition_imports,
    reaction_dsl_with_initial_condition_import_provenance,
)
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.units import UnitsModel
from kindred.core.validation import try_parse_finite_float

logger = logging.getLogger(__name__)


class InitialConditionsImportStatus(str, Enum):
    NO_IMPORT = "NO_IMPORT"
    APPLIED = "APPLIED"
    CANCELLED_OVERWRITE = "CANCELLED_OVERWRITE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class InitialConditionsReconciliationResult:
    status: InitialConditionsImportStatus
    source: MechanismAuthoringSource
    affected_rows: tuple[int, ...] = ()
    affected_set_names: tuple[str, ...] = ()
    species_changed: bool = False
    error_message: str = ""


@dataclass(frozen=True)
class InitialConditionsReconciliationPlan:
    status: InitialConditionsImportStatus
    source: MechanismAuthoringSource
    seed_sets: Mapping[str, Mapping[str, object]]
    species_names: tuple[str, ...]
    affected_set_names: tuple[str, ...] = ()
    error_message: str = ""


class InitialConditionsImportOwner:
    """Owns GUI Initial Conditions import and table reconciliation policy."""

    def __init__(
        self,
        *,
        batch_store_getter: Callable[[], BatchInitialConditionsStore],
        batch_model_getter: Callable[[], Any],
        confirm_overwrite: Callable[[Sequence[str]], bool],
        notify_rows_changed: Callable[[Sequence[int]], None],
        sync_species_columns: Callable[[Sequence[str]], None],
        temperature_getter: Callable[[], float],
    ) -> None:
        self._batch_store_getter = batch_store_getter
        self._batch_model_getter = batch_model_getter
        self._confirm_overwrite = confirm_overwrite
        self._notify_rows_changed = notify_rows_changed
        self._sync_species_columns = sync_species_columns
        self._temperature_getter = temperature_getter

    def prepare_reconciliation(
        self,
        source: MechanismAuthoringSource,
        *,
        prompt_overwrite: bool,
    ) -> InitialConditionsReconciliationPlan:
        original_source = source
        try:
            extraction = extract_reaction_dsl_initial_condition_imports(
                str(source.reactions_text or ""),
                default_set_name="set1",
            )
        except Exception as exc:
            logger.debug("Failed to parse mechanism-text Initial Conditions", exc_info=True)
            return InitialConditionsReconciliationPlan(
                InitialConditionsImportStatus.ERROR,
                original_source,
                seed_sets={},
                species_names=(),
                error_message=str(exc),
            )

        value_imports = tuple(event for event in extraction.imports if event.value_bearing)
        destination_names = self._destination_names_for_imports(value_imports)
        value_seed_sets = {
            str(destination_names.get(event.import_id, event.source_name)): dict(event.values)
            for event in value_imports
        }
        rewritten_reactions = reaction_dsl_with_initial_condition_import_provenance(
            extraction,
            destination_names=destination_names,
        )
        rewritten_source = source.with_reactions_text(str(rewritten_reactions))
        if value_seed_sets:
            if prompt_overwrite:
                affected = self._useful_initial_condition_overwrite_set_names(value_seed_sets)
                if affected and not self._confirm_overwrite(tuple(affected)):
                    return InitialConditionsReconciliationPlan(
                        InitialConditionsImportStatus.CANCELLED_OVERWRITE,
                        original_source,
                        seed_sets={},
                        species_names=(),
                        affected_set_names=tuple(affected),
                    )

        species_names = self._species_names_for_source_or_import_seeds(
            rewritten_source,
            seed_sets=value_seed_sets,
        )

        species_changed = list(species_names) != list(self._batch_store_getter().visible_species() or ())
        source_changed = rewritten_source != original_source
        status = (
            InitialConditionsImportStatus.APPLIED
            if value_seed_sets or species_changed or source_changed
            else InitialConditionsImportStatus.NO_IMPORT
        )
        return InitialConditionsReconciliationPlan(
            status,
            rewritten_source,
            seed_sets=value_seed_sets,
            species_names=tuple(species_names),
            affected_set_names=tuple(str(name) for name in value_seed_sets.keys() if str(name)),
        )

    def apply_reconciliation_plan(
        self,
        plan: InitialConditionsReconciliationPlan,
    ) -> InitialConditionsReconciliationResult:
        if plan.status == InitialConditionsImportStatus.NO_IMPORT:
            return InitialConditionsReconciliationResult(
                InitialConditionsImportStatus.NO_IMPORT,
                plan.source,
                affected_set_names=tuple(plan.affected_set_names),
            )
        if plan.status in {
            InitialConditionsImportStatus.CANCELLED_OVERWRITE,
            InitialConditionsImportStatus.ERROR,
        }:
            return InitialConditionsReconciliationResult(
                plan.status,
                plan.source,
                affected_set_names=tuple(plan.affected_set_names),
                error_message=str(plan.error_message or ""),
            )
        species_changed = self._apply_visible_species(tuple(plan.species_names))
        migrated_rows = self._materialize_imported_sets(seed_sets=plan.seed_sets)
        if migrated_rows:
            self._notify_rows_changed(migrated_rows)
        return InitialConditionsReconciliationResult(
            InitialConditionsImportStatus.APPLIED,
            plan.source,
            affected_rows=tuple(int(row) for row in migrated_rows),
            affected_set_names=tuple(plan.affected_set_names),
            species_changed=bool(species_changed),
        )

    def pending_initials_for_source_set(
        self,
        source: MechanismAuthoringSource,
        *,
        set_name: str,
    ) -> dict[str, float]:
        target_name = str(set_name or "").strip() or "set1"
        try:
            extraction = extract_reaction_dsl_initial_condition_imports(
                str(source.reactions_text or ""),
                default_set_name=target_name,
            )
        except Exception:
            logger.debug("Failed to parse pending draft Initial Conditions", exc_info=True)
            return {}
        for event in extraction.imports:
            if not bool(event.value_bearing):
                continue
            if str(event.source_name or "") != target_name:
                continue
            initials: dict[str, float] = {}
            for species, value in dict(event.values or {}).items():
                parsed, ok = try_parse_finite_float(value)
                if ok:
                    initials[str(species)] = float(parsed)
            return initials
        return {}

    def _apply_visible_species(self, species_names: Sequence[str]) -> bool:
        names = [str(name) for name in species_names if str(name)]
        store = self._batch_store_getter()
        before = list(store.visible_species() or ())
        if names == before:
            return False
        self._sync_species_columns(names)
        return True

    def _species_names_for_source(self, source: MechanismAuthoringSource) -> tuple[str, ...]:
        try:
            temperature_k = float(self._temperature_getter())
        except Exception:
            temperature_k = 298.15
        parse_source = source.without_reaction_initial_concentrations()
        mechanism = parse_dsl_to_mechanism(
            str(parse_source.full_dsl or ""),
            initials={},
            units=UnitsModel(temperature_K=float(temperature_k), energy_unit="kJ/mol"),
        )
        return tuple(str(name) for name in mechanism.species_names())

    def _species_names_for_source_or_import_seeds(
        self,
        source: MechanismAuthoringSource,
        *,
        seed_sets: Mapping[str, Mapping[str, object]],
    ) -> tuple[str, ...]:
        try:
            species_names = tuple(str(name) for name in self._species_names_for_source(source) if str(name))
            if species_names or not seed_sets:
                return species_names
        except Exception:
            logger.debug("Deferred mechanism validation while reconciling Initial Conditions species", exc_info=True)
            if not seed_sets:
                return tuple(self._batch_store_getter().visible_species() or ())
        species_names: list[str] = []
        seen: set[str] = set()
        for seed in dict(seed_sets or {}).values():
            for species in dict(seed or {}).keys():
                name = str(species)
                if name and name not in seen:
                    species_names.append(name)
                    seen.add(name)
        return tuple(species_names)

    def _batch_store_is_pristine_default_placeholder(self) -> bool:
        return batch_initial_conditions_store_is_true_placeholder(self._batch_store_getter())

    def _destination_names_for_imports(
        self,
        imports: Sequence[InitialConditionImportEvent],
    ) -> dict[str, str]:
        destinations: dict[str, str] = {}
        if not imports:
            return destinations
        existing = set(self._batch_store_getter().set_names() or [])
        reserved = set(existing)
        for event in imports:
            import_id = str(event.import_id or "")
            source_name = str(event.source_name or "").strip() or "set1"
            source_kind = str(event.source_kind or "")
            destination = source_name
            if (
                source_kind == "anonymous_inline"
                and source_name == "set1"
                and not self._batch_store_is_pristine_default_placeholder()
            ):
                index = 1
                while True:
                    candidate = f"set{index}"
                    if candidate not in reserved:
                        destination = candidate
                        break
                    index += 1
            destinations[import_id] = destination
            reserved.add(destination)
        return destinations

    def _useful_initial_condition_overwrite_set_names(
        self,
        seed_sets: Mapping[str, Mapping[str, object]],
    ) -> list[str]:
        store = self._batch_store_getter()
        if self._batch_store_is_pristine_default_placeholder():
            return []
        affected: list[str] = []
        for set_name, seed in dict(seed_sets or {}).items():
            row = store.row_for_set(str(set_name))
            if row is None:
                continue
            if dict(seed or {}):
                affected.append(str(set_name))
        return affected

    def _materialize_imported_sets(
        self,
        *,
        seed_sets: Mapping[str, Mapping[str, object]],
    ) -> list[int]:
        store = self._batch_store_getter()
        rows: list[int] = []
        ordered_names = [
            str(name)
            for name, seed in dict(seed_sets or {}).items()
            if str(name).strip() and dict(seed or {})
        ]
        if not ordered_names:
            return rows

        reuse_default_row = bool(
            self._batch_store_is_pristine_default_placeholder()
            and str(ordered_names[0]) != "set1"
        )
        model = self._batch_model_getter()
        model_attached = bool(
            model is not None
            and hasattr(model, "store")
            and callable(getattr(model, "store"))
            and model.store() is store
        )
        row_by_name: dict[str, int] = {}
        if reuse_default_row:
            store.set_set_name(0, str(ordered_names[0]))
            row_by_name[str(ordered_names[0])] = 0

        seen_rows: set[int] = set()
        visible_species = tuple(store.visible_species() or ())
        for set_name in ordered_names:
            seed = dict(seed_sets.get(str(set_name)) or {})
            row_idx = row_by_name.get(str(set_name))
            if row_idx is None:
                existing_row = store.row_for_set(str(set_name))
                if existing_row is None:
                    insert_at = int(store.row_count())
                    if model_attached:
                        model.beginInsertRows(QtCore.QModelIndex(), insert_at, insert_at)
                    try:
                        row_idx = int(store.ensure_set(str(set_name)))
                    finally:
                        if model_attached:
                            model.endInsertRows()
                else:
                    row_idx = int(existing_row)
                row_by_name[str(set_name)] = int(row_idx)
            for species in visible_species:
                store.set_value(int(row_idx), str(species), "0.0")
            for species, value in seed.items():
                parsed, ok = try_parse_finite_float(value)
                if not ok:
                    continue
                store.set_value(int(row_idx), str(species), f"{float(parsed):.6g}")
            if int(row_idx) not in seen_rows:
                rows.append(int(row_idx))
                seen_rows.add(int(row_idx))
        return rows
