"""Committed per-dataset fitting settings owner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from kindred.gui.controllers.dataset_errors import DatasetOwnerError
from kindred.gui.controllers.dataset_registry import DatasetRecord, DatasetRegistry


@dataclass
class DatasetFitSettings:
    """Per-dataset fitting settings."""

    weight: float = 1.0
    initial_conditions: Dict[str, float] = field(default_factory=dict)
    fit_flags: Dict[str, bool] = field(default_factory=dict)
    log10_flags: Dict[str, bool] = field(default_factory=dict)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    batch_set: Optional[str] = None
    batch_set_id: Optional[str] = None

    def ensure_species(self, species_names: Sequence[str], defaults: Optional[Dict[str, float]] = None) -> None:
        defaults = defaults or {}
        for name in species_names:
            if name not in self.initial_conditions:
                self.initial_conditions[name] = float(defaults.get(name, 0.0))
            self.fit_flags.setdefault(name, False)
            self.log10_flags.setdefault(name, False)
            self.bounds.setdefault(name, (0.0, max(10.0, self.initial_conditions[name] * 10 or 10.0)))


class DatasetFitSettingsStore:
    """Own settings keyed by committed registry dataset ids."""

    def __init__(self, registry: DatasetRegistry) -> None:
        self._registry = registry
        self._settings_by_id: Dict[str, DatasetFitSettings] = {}

    def ensure_for_records(self, records: Sequence[DatasetRecord]) -> None:
        for record in records:
            self._settings_by_id.setdefault(str(record.dataset_id), DatasetFitSettings())

    def get_fit_settings(self, dataset_id: str) -> DatasetFitSettings:
        record = self._require_record(dataset_id)
        try:
            return self._settings_by_id[str(record.dataset_id)]
        except KeyError as exc:
            raise DatasetOwnerError(f"Dataset '{dataset_id}' has no committed fit settings.") from exc

    def update_fit_settings(self, dataset_id: str, settings: DatasetFitSettings) -> None:
        record = self._require_record(dataset_id)
        if not isinstance(settings, DatasetFitSettings):
            raise DatasetOwnerError("Dataset fit settings update received an invalid settings object.")
        self._settings_by_id[str(record.dataset_id)] = settings

    def iter_fit_settings(self) -> List[Tuple[str, DatasetFitSettings]]:
        pairs: List[Tuple[str, DatasetFitSettings]] = []
        for record in self._registry.records():
            settings = self._settings_by_id.get(str(record.dataset_id))
            if settings is not None:
                pairs.append((str(record.display_name), settings))
        return pairs

    def remove_dataset_by_id(self, dataset_id: str) -> None:
        self._settings_by_id.pop(str(dataset_id), None)

    def clear(self) -> None:
        self._settings_by_id.clear()

    def datasets_mapped_to_batch_sets(
        self, *, set_ids: Sequence[str], set_names: Sequence[str]
    ) -> List[str]:
        id_targets = {str(v) for v in (set_ids or []) if str(v)}
        name_targets = {str(v) for v in (set_names or []) if str(v)}
        affected: List[str] = []
        for dataset_name, settings in self.iter_fit_settings():
            mapped_id = str(getattr(settings, "batch_set_id", "") or "").strip()
            mapped_name = str(getattr(settings, "batch_set", "") or "").strip()
            if mapped_id and mapped_id in id_targets:
                affected.append(str(dataset_name))
                continue
            if mapped_name and mapped_name in name_targets:
                affected.append(str(dataset_name))
        return affected

    def unmap_batch_sets(self, *, set_ids: Sequence[str], set_names: Sequence[str]) -> List[str]:
        id_targets = {str(v) for v in (set_ids or []) if str(v)}
        name_targets = {str(v) for v in (set_names or []) if str(v)}
        affected: List[str] = []
        for dataset_name, settings in self.iter_fit_settings():
            mapped_id = str(getattr(settings, "batch_set_id", "") or "").strip()
            mapped_name = str(getattr(settings, "batch_set", "") or "").strip()
            if (mapped_id and mapped_id in id_targets) or (mapped_name and mapped_name in name_targets):
                settings.batch_set = None
                settings.batch_set_id = None
                affected.append(str(dataset_name))
        return affected

    def _require_record(self, dataset_id: str) -> DatasetRecord:
        return self._registry.require_record_by_id(dataset_id)
