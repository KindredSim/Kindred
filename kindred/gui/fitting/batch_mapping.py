"""Shared helpers for dataset-to-batch-set mapping flows."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence

from PySide6 import QtCore, QtWidgets

from kindred.core.batch_initial_conditions import dataset_base_label, seed_batch_set_from_dataset_first_row

logger = logging.getLogger(__name__)

T0_SEED_TOL_S = 1e-9

__all__ = [
    "BatchMappingResolution",
    "T0_SEED_TOL_S",
    "apply_batch_mapping_to_settings",
    "create_and_seed_batch_set",
    "default_batch_set_name_for_dataset",
    "emit_batch_row_data_changed",
    "ensure_batch_set_row",
    "pick_existing_batch_set",
    "prompt_dataset_batch_mapping_choice",
    "resolve_saved_batch_mapping",
    "select_batch_set",
    "unique_batch_set_name",
]


@dataclass(frozen=True)
class BatchMappingResolution:
    status: str
    batch_set: Optional[str] = None
    batch_set_id: Optional[str] = None
    created: bool = False
    seeded: bool = False


def _record_failure(
    record_failure: Optional[Callable[..., None]],
    key: str,
    *,
    message: str,
    exc: Optional[Exception] = None,
) -> None:
    if not callable(record_failure):
        return
    try:
        record_failure(key, message=message, exc=exc)
    except Exception:
        logger.debug("Failed to record batch-mapping helper failure %s", key, exc_info=True)


def unique_batch_set_name(existing_names: Sequence[str], preferred: str) -> str:
    base_name = str(preferred or "").strip() or "set"
    existing = {str(name) for name in (existing_names or []) if str(name).strip()}
    if base_name not in existing:
        return base_name
    counter = 1
    while True:
        candidate = f"{base_name}_{counter}"
        if candidate not in existing:
            return candidate
        counter += 1


def default_batch_set_name_for_dataset(dataset_name: str) -> str:
    label = str(dataset_name or "").strip()
    if not label:
        return ""
    if "::" in label:
        file_part, sheet_part = label.split("::", 1)
        file_stem = os.path.splitext(os.path.basename(str(file_part).strip()))[0].strip()
        sheet_label = str(sheet_part).strip()
        if file_stem and sheet_label:
            return f"{file_stem}_{sheet_label}"
        if file_stem:
            return file_stem
        if sheet_label:
            return sheet_label
    return dataset_base_label(label) or label


def ensure_batch_set_row(batch_store: Any, batch_model: Any, set_name: str) -> tuple[int, bool]:
    if batch_store is None or batch_model is None:
        raise RuntimeError("Batch initial conditions table is unavailable.")
    existing = batch_store.row_for_set(set_name)
    if existing is not None:
        return int(existing), False
    insert_at = int(batch_store.row_count())
    batch_model.beginInsertRows(QtCore.QModelIndex(), insert_at, insert_at)
    try:
        idx = int(batch_store.ensure_set(set_name))
    finally:
        batch_model.endInsertRows()
    return idx, True


def select_batch_set(
    batch_store: Any,
    batch_model: Any,
    batch_table: Any,
    set_name: str,
    *,
    record_failure: Optional[Callable[..., None]] = None,
    failure_key_prefix: str = "batch_mapping",
) -> None:
    if batch_store is None or batch_model is None or batch_table is None:
        return
    row = batch_store.row_for_set(set_name)
    if row is None:
        return
    idx = batch_model.index(int(row), 0)
    batch_table.setCurrentIndex(idx)
    selection_model = batch_table.selectionModel()
    if selection_model is not None:
        signals_blocked = False
        try:
            selection_model.blockSignals(True)
            signals_blocked = True
        except Exception as exc:
            _record_failure(
                record_failure,
                f"{failure_key_prefix}.selection_model.block_signals.true",
                message="Failed to block batch selection signals while selecting a mapped batch set",
                exc=exc,
            )
        try:
            selection_model.clearSelection()
            selection_model.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
        finally:
            if signals_blocked:
                try:
                    selection_model.blockSignals(False)
                except Exception as exc:
                    _record_failure(
                        record_failure,
                        f"{failure_key_prefix}.selection_model.block_signals.false",
                        message="Failed to unblock batch selection signals while selecting a mapped batch set",
                        exc=exc,
                    )
    try:
        batch_table.scrollTo(idx)
    except Exception as exc:
        _record_failure(
            record_failure,
            f"{failure_key_prefix}.batch_table.scroll_to",
            message="Failed to scroll the batch table to a mapped batch set",
            exc=exc,
        )


def emit_batch_row_data_changed(
    batch_model: Any,
    row_idx: int,
    *,
    record_failure: Optional[Callable[..., None]] = None,
    failure_key: str = "batch_mapping.batch_model.data_changed",
) -> None:
    if batch_model is None or int(batch_model.columnCount()) <= 1:
        return
    try:
        top_left = batch_model.index(int(row_idx), 1)
        bottom_right = batch_model.index(int(row_idx), max(1, int(batch_model.columnCount()) - 1))
        batch_model.dataChanged.emit(
            top_left,
            bottom_right,
            [QtCore.Qt.DisplayRole, QtCore.Qt.BackgroundRole],
        )
    except Exception as exc:
        _record_failure(
            record_failure,
            failure_key,
            message="Failed to emit dataChanged after updating a batch set from dataset mapping",
            exc=exc,
        )


def resolve_saved_batch_mapping(settings: Any, batch_store: Any) -> BatchMappingResolution:
    mapped_name = str(getattr(settings, "batch_set", "") or "").strip() or None
    mapped_id = str(getattr(settings, "batch_set_id", "") or "").strip() or None
    if batch_store is not None and mapped_id:
        mapped_row = batch_store.row_for_set_id(mapped_id)
        if mapped_row is not None:
            resolved_name = str(batch_store.set_name_for_row(int(mapped_row)))
            settings.batch_set = resolved_name
            settings.batch_set_id = mapped_id
            return BatchMappingResolution(
                status="mapped",
                batch_set=resolved_name,
                batch_set_id=mapped_id,
            )
    if batch_store is not None and mapped_name:
        mapped_row = batch_store.row_for_set(mapped_name)
        if mapped_row is not None:
            resolved_id = str(batch_store.set_id_for_row(int(mapped_row)))
            settings.batch_set = mapped_name
            settings.batch_set_id = resolved_id
            return BatchMappingResolution(
                status="mapped",
                batch_set=mapped_name,
                batch_set_id=resolved_id,
            )
    settings.batch_set = None
    settings.batch_set_id = None
    return BatchMappingResolution(status="unmapped")


def prompt_dataset_batch_mapping_choice(
    parent: QtWidgets.QWidget,
    dataset_name: str,
    create_set_name: str,
    *,
    title: str,
    skip_label: str,
    skip_description: str,
    running_under_pytest: bool = False,
    pytest_default_action: str = "create",
) -> str:
    if running_under_pytest:
        return str(pytest_default_action)

    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    box.setWindowTitle(str(title))
    box.setText(f"Dataset '{dataset_name}' has no saved batch mapping.")
    box.setInformativeText(
        "\n".join(
            [
                f"Create new: Create batch set '{create_set_name}'",
                "Map existing: Choose an existing batch set",
                f"{skip_label}: {skip_description}",
            ]
        )
    )
    create_button = box.addButton("Create new", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    map_button = box.addButton("Map existing", QtWidgets.QMessageBox.ButtonRole.ActionRole)
    skip_button = box.addButton(str(skip_label), QtWidgets.QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(create_button)
    box.exec()
    clicked = box.clickedButton()
    if clicked is map_button:
        return "map"
    if clicked is create_button:
        return "create"
    if clicked is skip_button:
        return "skip"
    return "skip"


def pick_existing_batch_set(
    parent: QtWidgets.QWidget,
    dataset_name: str,
    batch_set_names: Sequence[str],
    *,
    title: str,
    empty_message_title: str,
    empty_message_text: str,
) -> Optional[str]:
    if not list(batch_set_names):
        QtWidgets.QMessageBox.warning(parent, str(empty_message_title), str(empty_message_text))
        return None
    choice, ok = QtWidgets.QInputDialog.getItem(
        parent,
        str(title),
        f"Select a batch set for dataset '{dataset_name}':",
        [str(name) for name in batch_set_names],
        0,
        False,
    )
    if not ok or not str(choice).strip():
        return None
    return str(choice).strip()


def apply_batch_mapping_to_settings(settings: Any, batch_store: Any, target_set: str) -> BatchMappingResolution:
    settings.batch_set = str(target_set)
    settings.batch_set_id = None
    mapped_id = None
    if batch_store is not None:
        mapped_row = batch_store.row_for_set(str(target_set))
        if mapped_row is not None:
            mapped_id = str(batch_store.set_id_for_row(int(mapped_row)))
            settings.batch_set_id = mapped_id
    return BatchMappingResolution(
        status="mapped",
        batch_set=str(target_set),
        batch_set_id=mapped_id,
    )


def create_and_seed_batch_set(
    *,
    dataset_name: str,
    dataset_payload: Dict[str, object],
    mechanism_species: Sequence[str],
    batch_store: Any,
    batch_model: Any,
    set_name: Optional[str] = None,
    record_failure: Optional[Callable[..., None]] = None,
    failure_key_prefix: str = "batch_mapping",
    tol: float = T0_SEED_TOL_S,
) -> tuple[int, bool, bool]:
    preferred_name = str(set_name or "").strip() or default_batch_set_name_for_dataset(dataset_name) or str(dataset_name)
    row_idx, created = ensure_batch_set_row(batch_store, batch_model, preferred_name)
    seeded = False
    if created and batch_store is not None:
        seeded_values = seed_batch_set_from_dataset_first_row(
            dataset_payload,
            mechanism_species,
            tol=float(tol),
        )
        if seeded_values:
            seeded = True
            for species, value in seeded_values.items():
                try:
                    batch_store.set_value(int(row_idx), str(species), f"{float(value):.6g}")
                except Exception as exc:
                    _record_failure(
                        record_failure,
                        f"{failure_key_prefix}.seed.set_value",
                        message=f"Failed to seed batch set value for species '{species}' from dataset mapping",
                        exc=exc,
                    )
            emit_batch_row_data_changed(
                batch_model,
                int(row_idx),
                record_failure=record_failure,
                failure_key=f"{failure_key_prefix}.seed.data_changed",
            )
    return int(row_idx), bool(created), bool(seeded)
