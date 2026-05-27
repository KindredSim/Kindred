"""Committed dataset registry owner."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from kindred.gui.dataset_payload_snapshot import copy_dataset_payload
from kindred.gui.controllers.dataset_errors import DatasetOwnerError


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    display_name: str
    payload: Dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", copy_dataset_payload(self.payload))


class DatasetRegistry:
    """Own committed dataset records and payload snapshots."""

    _RESERVED_DISPLAY_NAME_RE = re.compile(r"^dataset-\d+$")

    def __init__(self) -> None:
        self._records_by_id: Dict[str, DatasetRecord] = {}
        self._ids_by_display_name: Dict[str, str] = {}
        self._next_id = 1

    def commit_datasets(
        self,
        datasets: Iterable[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[List[DatasetRecord], List[Tuple[str, DatasetOwnerError]]]:
        records: List[DatasetRecord] = []
        errors: List[Tuple[str, DatasetOwnerError]] = []
        for raw_name, payload in datasets:
            name = str(raw_name)
            try:
                records.append(self.commit_dataset(name, payload))
            except DatasetOwnerError as exc:
                errors.append((name, exc))
        return records, errors

    def commit_dataset(self, display_name: str, payload: Dict[str, Any]) -> DatasetRecord:
        clean_payload = self._copy_payload(payload)
        species = clean_payload.get("species")
        if not isinstance(species, dict) or not species:
            raise DatasetOwnerError("Dataset contains no numeric species columns.")
        unique_name = self._unique_display_name(str(display_name))
        dataset_id = self._allocate_id()
        record = DatasetRecord(
            dataset_id=dataset_id,
            display_name=unique_name,
            payload=clean_payload,
        )
        self._records_by_id[dataset_id] = record
        self._ids_by_display_name[unique_name] = dataset_id
        return self._clone_record(record)

    def records(self) -> Tuple[DatasetRecord, ...]:
        return tuple(self._clone_record(record) for record in self._records_by_id.values())

    def presentation_payloads_by_display_name(self) -> Dict[str, Dict[str, Any]]:
        return {
            str(record.display_name): self._copy_payload(record.payload)
            for record in self._records_by_id.values()
        }

    def payload_for_id(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        record = self.record_by_id(dataset_id)
        if record is None:
            return None
        return self._copy_payload(record.payload)

    def record_by_id(self, dataset_id: str) -> Optional[DatasetRecord]:
        record = self._records_by_id.get(str(dataset_id))
        if record is None:
            return None
        return self._clone_record(record)

    def require_record_by_id(self, dataset_id: str) -> DatasetRecord:
        record = self.record_by_id(dataset_id)
        if record is None:
            raise DatasetOwnerError(f"Unknown committed dataset id: {dataset_id}")
        return record

    def record_by_display_name(self, display_name: str) -> Optional[DatasetRecord]:
        dataset_id = self._ids_by_display_name.get(str(display_name))
        if dataset_id is None:
            return None
        record = self._records_by_id.get(str(dataset_id))
        if record is None:
            return None
        return self._clone_record(record)

    def remove_by_id(self, dataset_id: str) -> Optional[DatasetRecord]:
        record = self.record_by_id(dataset_id)
        if record is None:
            return None
        stored = self._records_by_id.pop(str(record.dataset_id), None)
        if stored is None:
            return None
        self._ids_by_display_name.pop(str(stored.display_name), None)
        return self._clone_record(stored)

    def clear(self) -> None:
        self._records_by_id.clear()
        self._ids_by_display_name.clear()
        self._next_id = 1

    def _allocate_id(self) -> str:
        dataset_id = f"dataset-{self._next_id}"
        self._next_id += 1
        return dataset_id

    def _unique_display_name(self, raw_name: str) -> str:
        name = str(raw_name).strip() or "dataset"
        if name not in self._ids_by_display_name and not self._is_reserved_display_name(name):
            return name
        base, ext = os.path.splitext(name)
        counter = 1
        while True:
            candidate = f"{base}_{counter}{ext}"
            if candidate not in self._ids_by_display_name and not self._is_reserved_display_name(candidate):
                return candidate
            counter += 1

    @classmethod
    def _is_reserved_display_name(cls, name: str) -> bool:
        return bool(cls._RESERVED_DISPLAY_NAME_RE.fullmatch(str(name).strip()))

    @classmethod
    def _clone_record(cls, record: DatasetRecord) -> DatasetRecord:
        return DatasetRecord(
            dataset_id=str(record.dataset_id),
            display_name=str(record.display_name),
            payload=cls._copy_payload(record.payload),
        )

    @staticmethod
    def _copy_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        return copy_dataset_payload(payload)
