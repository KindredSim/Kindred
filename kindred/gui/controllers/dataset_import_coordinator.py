"""Commit policy for completed dataset import sessions."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from PySide6 import QtWidgets

from kindred.gui.controllers.dataset_fit_settings_store import DatasetFitSettingsStore
from kindred.gui.controllers.dataset_registry import DatasetRecord, DatasetRegistry
from kindred.gui.controllers.dataset_view_publisher import DatasetViewPublisher
from kindred.gui.widgets.dataset_import_session import DatasetImportCompletion

logger = logging.getLogger(__name__)


class DatasetImportCoordinator:
    """Commit valid import completions into registry/settings/views."""

    def __init__(
        self,
        *,
        registry: DatasetRegistry,
        fit_settings_store: DatasetFitSettingsStore,
        view_publisher: DatasetViewPublisher,
        import_panel: Any,
        status_setter: Callable[[str], None],
        overlay_sync: Callable[[], None],
        color_sync: Callable[[], None],
        batch_store_getter: Callable[[], Any],
        batch_model_getter: Callable[[], Any],
        batch_table_getter: Callable[[], Any],
        mechanism_text_getter: Callable[[], str],
        extract_mechanism_initials: Callable[[str], Dict[str, float]],
        sync_batch_species_columns: Callable[[Sequence[str]], None],
        record_failure: Callable[..., None],
        runtime_inputs_changed: Callable[[], None],
        datasets_removed: Callable[[Sequence[str]], None],
        message_parent: Any,
    ) -> None:
        self._registry = registry
        self._fit_settings_store = fit_settings_store
        self._view_publisher = view_publisher
        self._import_panel = import_panel
        self._status_setter = status_setter
        self._overlay_sync = overlay_sync
        self._color_sync = color_sync
        self._batch_store_getter = batch_store_getter
        self._batch_model_getter = batch_model_getter
        self._batch_table_getter = batch_table_getter
        self._mechanism_text_getter = mechanism_text_getter
        self._extract_mechanism_initials = extract_mechanism_initials
        self._sync_batch_species_columns = sync_batch_species_columns
        self._record_failure = record_failure
        self._runtime_inputs_changed = runtime_inputs_changed
        self._datasets_removed = datasets_removed
        self._message_parent = message_parent

    def handle_completion(self, completion: DatasetImportCompletion) -> None:
        """Commit one completed import session, unless it was canceled/superseded."""
        if str(getattr(completion, "outcome", "completed") or "completed") != "completed":
            return
        if completion.canceled or completion.superseded:
            return
        ready_items = [(str(unit.display_name), unit.payload) for unit in completion.units]
        if not ready_items:
            return

        if len(ready_items) == 1:
            logger.info("Dataset loaded: %s", ready_items[0][0])
        else:
            logger.info("Datasets loaded: %s", ", ".join(name for name, _data in ready_items))

        self._color_sync()
        records, errors = self._registry.commit_datasets(ready_items)
        for name, exc in errors:
            logger.warning("Dataset '%s' missing usable species: %s", name, exc)
            QtWidgets.QMessageBox.warning(
                self._message_parent,
                "Dataset Skipped",
                f"Dataset '{name}' cannot be visualized:\n\n{exc}",
            )
        if not records:
            self._render_registry()
            return

        self._fit_settings_store.ensure_for_records(records)
        self._view_publisher.publish_records(records)
        self._render_registry()

        if len(records) == 1:
            record = records[0]
            t = np.asarray(record.payload.get("t", [])).reshape(-1)
            self._status_setter(f"Dataset '{record.display_name}' loaded ({len(t)} points)")
        else:
            self._status_setter(f"{len(records)} datasets loaded")
        self._overlay_sync()
        self._prompt_for_import_batch_mapping(records)

    def remove_dataset_by_id(self, dataset_id: str) -> None:
        """Remove one committed dataset from every owner."""
        record = self._registry.record_by_id(dataset_id)
        if record is None:
            return
        self._fit_settings_store.remove_dataset_by_id(str(record.dataset_id))
        removed_record = self._registry.remove_by_id(str(record.dataset_id))
        if removed_record is not None:
            self._view_publisher.remove_record(removed_record)
            self._status_setter(f"Dataset '{removed_record.display_name}' removed")
            self._notify_datasets_removed((str(removed_record.dataset_id),))
        self._render_registry()
        self._overlay_sync()

    def clear_all_datasets(self) -> None:
        """Clear import work and committed dataset owners."""
        removed_ids = tuple(str(record.dataset_id) for record in self._registry.records())
        clear_panel = getattr(self._import_panel, "clear_datasets", None)
        if callable(clear_panel):
            clear_panel()
        self._registry.clear()
        self._fit_settings_store.clear()
        self._view_publisher.clear_all()
        self._render_registry()
        self._overlay_sync()
        if removed_ids:
            self._notify_datasets_removed(removed_ids)
        else:
            self._notify_runtime_inputs_changed()

    def _notify_datasets_removed(self, dataset_ids: Sequence[str]) -> None:
        try:
            self._datasets_removed(tuple(str(dataset_id) for dataset_id in dataset_ids if str(dataset_id)))
        except Exception as exc:
            self._record_failure(
                "dataset_import.datasets_removed",
                message="Failed to notify active fitting windows about removed datasets",
                exc=exc,
            )

    def _notify_runtime_inputs_changed(self) -> None:
        try:
            self._runtime_inputs_changed()
        except Exception as exc:
            self._record_failure(
                "dataset_import.runtime_inputs_changed",
                message="Failed to notify active fitting windows about dataset runtime input changes",
                exc=exc,
            )

    def _render_registry(self) -> None:
        render = getattr(self._import_panel, "render_registry_snapshot", None)
        if callable(render):
            render(self._registry.records())

    def _prompt_for_import_batch_mapping(self, records: Sequence[DatasetRecord]) -> None:
        if not records:
            return
        batch_store = self._batch_store_getter()
        batch_model = self._batch_model_getter()
        batch_store_rows = int(batch_store.row_count()) if batch_store is not None else 0
        batch_model_rows = int(batch_model.rowCount()) if batch_model is not None else 0
        if not (batch_store_rows > 0 or batch_model_rows > 0):
            return

        mechanism_species: List[str] = []
        try:
            mechanism_species = list((self._extract_mechanism_initials(self._mechanism_text_getter()) or {}).keys())
        except Exception as exc:
            self._record_failure(
                "dataset_import.batch_mapping.extract_mechanism_initials",
                message="Failed to extract mechanism initials while preparing import-time batch mapping",
                exc=exc,
            )
        try:
            self._sync_batch_species_columns(mechanism_species)
        except Exception as exc:
            self._record_failure(
                "dataset_import.batch_mapping.sync_batch_species_columns",
                message="Failed to sync batch species columns while preparing import-time batch mapping",
                exc=exc,
            )

        for record in records:
            self._maybe_prompt_for_import_batch_mapping(record, mechanism_species)

    def _maybe_prompt_for_import_batch_mapping(
        self,
        record: DatasetRecord,
        mechanism_species: Sequence[str],
    ) -> None:
        from kindred.gui.fitting.batch_mapping import (
            T0_SEED_TOL_S,
            apply_batch_mapping_to_settings,
            create_and_seed_batch_set,
            default_batch_set_name_for_dataset,
            pick_existing_batch_set,
            prompt_dataset_batch_mapping_choice,
            resolve_saved_batch_mapping,
            select_batch_set,
            unique_batch_set_name,
        )

        batch_store = self._batch_store_getter()
        batch_model = self._batch_model_getter()
        if batch_store is None or batch_model is None:
            return
        try:
            self._sync_batch_species_columns(list(mechanism_species))
        except Exception as exc:
            self._record_failure(
                "dataset_import.batch_mapping.sync_batch_species_columns",
                message="Failed to sync batch species columns while preparing import-time batch mapping",
                exc=exc,
            )

        dataset_name = str(record.display_name)
        settings = self._fit_settings_store.get_fit_settings(str(record.dataset_id))
        resolved = resolve_saved_batch_mapping(settings, batch_store)
        if resolved.status == "mapped":
            self._fit_settings_store.update_fit_settings(str(record.dataset_id), settings)
            return

        batch_set_names = list(batch_store.set_names() or [])
        if not batch_set_names:
            return

        create_set_name = unique_batch_set_name(
            batch_set_names,
            default_batch_set_name_for_dataset(dataset_name) or dataset_name,
        )
        running_under_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
        action = prompt_dataset_batch_mapping_choice(
            self._message_parent,
            dataset_name,
            create_set_name,
            title="Import Set Mapping",
            skip_label="Skip",
            skip_description="Leave this dataset unmapped",
            running_under_pytest=running_under_pytest,
            pytest_default_action="skip",
        )
        if action == "skip":
            self._fit_settings_store.update_fit_settings(str(record.dataset_id), settings)
            return
        if action == "map":
            target_set = pick_existing_batch_set(
                self._message_parent,
                dataset_name,
                batch_set_names,
                title="Map Dataset to Set",
                empty_message_title="Import Set Mapping",
                empty_message_text="No sets exist to map to. Create a set first.",
            )
            if not target_set:
                self._fit_settings_store.update_fit_settings(str(record.dataset_id), settings)
                return
            apply_batch_mapping_to_settings(settings, batch_store, target_set)
            self._fit_settings_store.update_fit_settings(str(record.dataset_id), settings)
            return

        _row_idx, created, seeded = create_and_seed_batch_set(
            dataset_name=dataset_name,
            dataset_payload=record.payload,
            mechanism_species=mechanism_species,
            batch_store=batch_store,
            batch_model=batch_model,
            set_name=create_set_name,
            record_failure=self._record_failure,
            failure_key_prefix="dataset_import.batch_mapping",
        )
        if created and not seeded and not mechanism_species:
            self._fit_settings_store.update_fit_settings(str(record.dataset_id), settings)
            return
        if created and batch_store is not None and not seeded:
            try:
                t_arr = np.asarray((record.payload or {}).get("t", []), dtype=float).reshape(-1)
                t0 = float(t_arr[0]) if t_arr.size else float("nan")
            except Exception:
                t0 = float("nan")
            if not (abs(t0) <= T0_SEED_TOL_S):
                if running_under_pytest:
                    response = QtWidgets.QMessageBox.StandardButton.Cancel
                else:
                    response = QtWidgets.QMessageBox.warning(
                        self._message_parent,
                        "Import Set Mapping",
                        (
                            f"Dataset '{dataset_name}' does not start at t~=0 "
                            f"(t0={t0:.6g} s; tol={T0_SEED_TOL_S:.1e} s).\n\n"
                            "OK: Map this dataset to the new zeroed set\n"
                            "Cancel: Leave this dataset unmapped and edit the new set manually"
                        ),
                        QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
                        QtWidgets.QMessageBox.StandardButton.Cancel,
                    )
                if response == QtWidgets.QMessageBox.StandardButton.Cancel:
                    select_batch_set(
                        batch_store,
                        batch_model,
                        self._batch_table_getter(),
                        create_set_name,
                        record_failure=self._record_failure,
                        failure_key_prefix="dataset_import.batch_mapping",
                    )
                    if not running_under_pytest:
                        QtWidgets.QMessageBox.information(
                            self._message_parent,
                            "Import Set Mapping",
                            (
                                f"Set '{create_set_name}' was created.\n\n"
                                "Edit its initial concentrations in the Initial Conditions table, "
                                "then map the dataset when it is ready."
                            ),
                        )
                    self._fit_settings_store.update_fit_settings(str(record.dataset_id), settings)
                    return
        apply_batch_mapping_to_settings(settings, batch_store, create_set_name)
        self._fit_settings_store.update_fit_settings(str(record.dataset_id), settings)
