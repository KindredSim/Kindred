# kindred/gui/widgets/data_manager.py
"""Data manager panel for loading and managing experimental datasets."""

from __future__ import annotations

import csv
from contextlib import closing, suppress
import itertools
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from kindred.core.datasets.csv_import import (
    CsvImportInterrupted,
    parse_csv_rows,
)
from kindred.core.datasets.excel_import import (
    list_sheets,
    read_excel_sheet_rows,
)
from kindred.core.datasets.observation_payload import (
    dense_view_from_observations,
    observations_from_payload,
    scale_payload_in_place,
)
from kindred.gui.widgets.import_config import (
    ImportConfig,
    ResolvedSheetPlan,
    SheetImportIntent,
    UnitDetection,
    UserImportIntent,
    detect_units_from_row_mapping,
    rebuild_intent_for_target,
    resolve_import_plans,
)
from kindred.gui.widgets.import_config_dialog import ImportConfigDialog, ImportDialogResult
from kindred.gui.controllers.dataset_registry import DatasetRecord
from kindred.gui.widgets.dataset_import_session import (
    DatasetImportCompletion,
    DatasetImportSession,
    DatasetImportUnit,
)

logger = logging.getLogger(__name__)

__all__ = ["DataManagerPanel"]

class CSVLoaderWorker(QtCore.QThread):
    """Background worker for loading CSV files without blocking UI."""

    loaded = QtCore.Signal(str, dict)  # name, data
    cancelled = QtCore.Signal(str)  # dataset name
    error = QtCore.Signal(str)  # error message
    progress = QtCore.Signal(int)  # progress percentage

    def __init__(self, plan: ResolvedSheetPlan):
        super().__init__()
        self.filepath = plan.filepath
        self._time_column = plan.time_column or None
        self._species_columns = list(plan.species_columns)
        self._skip_unit_row = plan.skip_unit_row

    def _load_csv_payload(self) -> dict:
        with open(self.filepath, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise ValueError("CSV file is empty")
            normalized_header = [str(column).strip() for column in header]
            row_iter = iter(reader)
            first_row = next(row_iter, None)
            if first_row is None:
                raise ValueError("CSV file is empty")
            if self._skip_unit_row:
                rows = row_iter
            else:
                rows = itertools.chain((first_row,), row_iter)
            normalized_rows = (
                {
                    normalized_header[index]: (row[index] if index < len(row) else "")
                    for index in range(len(normalized_header))
                }
                for row in rows
            )
            _time_source, data = parse_csv_rows(
                normalized_rows,
                time_column=self._time_column,
                species_columns=self._species_columns,
                interruption_checker=self.isInterruptionRequested,
                source_label=os.path.basename(self.filepath),
            )
        return data

    def run(self):
        """Load CSV file in background thread."""
        filename = self.filepath
        dataset_name = os.path.basename(filename)

        def _check_cancel(stage: str) -> bool:
            if self.isInterruptionRequested():
                logger.info(f"CSV import interrupted {stage}: {dataset_name}")
                self.cancelled.emit(dataset_name)
                return True
            return False

        # Log start
        logger.info(f"Starting CSV import: {dataset_name} from {filename}")

        try:
            self.progress.emit(10)

            # Check for interruption before file I/O
            if _check_cancel("before reading"):
                return

            self.progress.emit(30)

            # Stream CSV rows through the shared core importer so interruption can
            # be observed during file iteration rather than only after full read.
            data = self._load_csv_payload()
            logger.debug(
                "Parsed CSV dataset: time column '%s', species columns: %s",
                data.get("metadata", {}).get("time_column"),
                list((data.get("observations") or {}).keys()),
            )

            self.progress.emit(80)

            if _check_cancel("after parsing"):
                return

            self.progress.emit(100)

            # Log success before emitting signal
            observations = data.get('observations', {})
            total_points = sum(len(spec.get("t", [])) for spec in observations.values()) if isinstance(observations, dict) else 0
            logger.info(
                "CSV import completed: %s (%d rows, %d species: %s)",
                dataset_name,
                total_points,
                len(observations),
                list(observations.keys()),
            )

            self.loaded.emit(dataset_name, data)

        except CsvImportInterrupted:
            self.cancelled.emit(dataset_name)
            return
        except Exception as e:
            logger.error(f"CSV import failed: {dataset_name} - {type(e).__name__}: {e}", exc_info=True)
            self.error.emit(f"{type(e).__name__}: {str(e)}")


class ExcelLoaderWorker(QtCore.QThread):
    """Background worker for loading Excel sheets without blocking UI."""

    loaded = QtCore.Signal(str, dict)  # name, data
    cancelled = QtCore.Signal(str)  # dataset name
    error = QtCore.Signal(str)  # error message
    progress = QtCore.Signal(int)  # progress percentage

    def __init__(self, filepath: str, plans: Sequence[ResolvedSheetPlan]):
        super().__init__()
        self.filepath = str(filepath)
        self._plans = list(plans)

    def _load_sheet_payload(self, plan: ResolvedSheetPlan) -> Tuple[str, dict]:
        with closing(read_excel_sheet_rows(self.filepath, plan.sheet_name)) as rows:
            row_iter = iter(rows)
            first_row = next(row_iter, None)
            if first_row is None:
                raise ValueError(f"Sheet '{plan.sheet_name}' is empty.")
            first_row_mapping = dict(first_row)
            if plan.skip_unit_row:
                rows_to_parse = row_iter
            else:
                rows_to_parse = itertools.chain((first_row_mapping,), row_iter)
            _time_source, data = parse_csv_rows(
                rows_to_parse,
                time_column=plan.time_column,
                species_columns=list(plan.species_columns),
                interruption_checker=self.isInterruptionRequested,
                source_label=os.path.basename(self.filepath),
                sheet_name=plan.sheet_name,
            )
        return f"{os.path.basename(self.filepath)}::{plan.sheet_name}", data

    def run(self):
        dataset_name = os.path.basename(self.filepath)
        total = len(self._plans)

        if total <= 0:
            self.error.emit("No Excel sheets were selected for import.")
            return

        try:
            self.progress.emit(0)
            for index, plan in enumerate(self._plans, start=1):
                if self.isInterruptionRequested():
                    logger.info("Excel import interrupted before sheet %s: %s", plan.sheet_name, dataset_name)
                    self.cancelled.emit(dataset_name)
                    return
                try:
                    loaded_name, data = self._load_sheet_payload(plan)
                except CsvImportInterrupted:
                    logger.info("Excel import interrupted while parsing sheet %s: %s", plan.sheet_name, dataset_name)
                    self.cancelled.emit(dataset_name)
                    return
                except Exception as exc:
                    logger.error(
                        "Excel import failed: %s sheet %s - %s: %s",
                        dataset_name, plan.sheet_name, type(exc).__name__, exc, exc_info=True,
                    )
                    self.error.emit(f"Sheet '{plan.sheet_name}': {type(exc).__name__}: {exc}")
                else:
                    self.loaded.emit(loaded_name, data)
                self.progress.emit(int((index / total) * 100))
                if self.isInterruptionRequested():
                    logger.info("Excel import interrupted after sheet %s: %s", plan.sheet_name, dataset_name)
                    self.cancelled.emit(dataset_name)
                    return
        except Exception as exc:
            logger.error("Excel import failed: %s - %s: %s", dataset_name, type(exc).__name__, exc, exc_info=True)
            self.error.emit(f"{type(exc).__name__}: {exc}")

class DataManagerPanel(QtWidgets.QWidget):
    """
    Import panel for loading and previewing committed experimental datasets.

    Features:
    - Load CSV and Excel files with experimental data
    - Auto-detect time column (tries: time, time_s, t, Time, T, x)
    - Extract all numeric columns as species
    - Preview loaded datasets
    - Import configuration dialog with unit selection
    - Multiple dataset support

    Signals:
        importCompleted(object): Emitted with one DatasetImportCompletion when an import session ends
        datasetRemovalRequested(str): Emitted when the user requests removal of a committed dataset
    """

    importCompleted = QtCore.Signal(object)
    datasetRemovalRequested = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """
        Initialize data manager panel.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        # Header
        layout.addWidget(QtWidgets.QLabel("<b>Loaded Datasets</b>"))

        # Dataset list
        self._dataset_list = QtWidgets.QListWidget()
        self._dataset_list.currentItemChanged.connect(self._on_dataset_selected)
        layout.addWidget(self._dataset_list)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self._load_btn = QtWidgets.QPushButton("Load Data")
        self._load_btn.setObjectName("loadDataButton")
        self._load_btn.clicked.connect(self._load_dataset)
        self._remove_btn = QtWidgets.QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_dataset)
        btn_layout.addWidget(self._load_btn)
        btn_layout.addWidget(self._remove_btn)
        layout.addLayout(btn_layout)

        # Preview area
        layout.addWidget(QtWidgets.QLabel("Preview:"))
        self._preview_label = QtWidgets.QLabel("No dataset selected")
        self._preview_label.setContentsMargins(10, 10, 10, 10)
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setWordWrap(True)
        layout.addWidget(self._preview_label)
        self._preview_label.hide()

        # Rendered committed registry records for list/preview only.
        self._visible_records_by_id: Dict[str, DatasetRecord] = {}

        # Active worker handles live in DatasetImportSession until completion.
        self._progress_dialog: Optional[QtWidgets.QProgressDialog] = None
        self._cancel_requested = False
        self._import_completion_emitted = False
        self._load_generation = 0
        self._active_import_session: Optional[DatasetImportSession] = None
        self._retired_import_sessions_by_worker: Dict[int, DatasetImportSession] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def trigger_load_dialog(self) -> None:
        """Programmatically invoke the load dialog (mirrors Load button)."""
        self._load_dataset()

    def clear_datasets(self) -> None:
        """Cancel active import work and reset rendered registry state."""
        had_active_load = self._deauthorize_active_import(
            emit_canceled_finished=True,
            outcome="cleared_or_reset",
        )
        if not had_active_load:
            self._import_completion_emitted = False
        self.render_registry_snapshot(())

    def render_registry_snapshot(self, records: Sequence[DatasetRecord]) -> None:
        """Render committed registry records without becoming their owner."""
        self._visible_records_by_id = {str(record.dataset_id): record for record in records}
        self._dataset_list.clear()
        for record in records:
            item = QtWidgets.QListWidgetItem(str(record.display_name))
            item.setData(Qt.UserRole, str(record.dataset_id))
            self._dataset_list.addItem(item)
        self._preview_label.setText("No dataset selected")
        self._preview_label.hide()

    def _deauthorize_active_import(
        self,
        *,
        emit_canceled_finished: bool,
        outcome: str = "superseded",
    ) -> bool:
        """Cancel the active import generation without blocking on worker teardown."""
        session = self._active_import_session
        had_active_load = session is not None and not self._import_completion_emitted
        if session is None:
            return False

        self._load_generation += 1
        workers = session.deauthorize(str(outcome or "superseded"))
        self._retain_deauthorized_import_session(session, workers)
        self._cancel_requested = False
        if not had_active_load:
            self._import_completion_emitted = False
        self._finalize_progress_dialog()
        if had_active_load and emit_canceled_finished:
            self._emit_import_completed(
                canceled=True,
                superseded=str(outcome) == "superseded",
                discard_units=True,
                outcome=str(outcome or "superseded"),
            )
        else:
            self._active_import_session = None
        return had_active_load

    def _retain_deauthorized_import_session(
        self,
        session: DatasetImportSession,
        workers: Sequence[QtCore.QThread],
    ) -> None:
        for worker in workers:
            self._retired_import_sessions_by_worker[id(worker)] = session

    def _import_session_for_worker(
        self,
        worker: Optional[QtCore.QThread],
    ) -> Optional[DatasetImportSession]:
        if worker is None:
            return None
        active = self._active_import_session
        if active is not None and active.owns_worker(worker):
            return active
        return self._retired_import_sessions_by_worker.get(id(worker))

    def _release_retired_import_worker(
        self,
        worker: Optional[QtCore.QThread],
        session: Optional[DatasetImportSession],
    ) -> None:
        if worker is None or session is None:
            return
        self._retired_import_sessions_by_worker.pop(id(worker), None)
        if not getattr(session, "_workers", None):
            stale_worker_ids = [
                worker_id
                for worker_id, retired_session in self._retired_import_sessions_by_worker.items()
                if retired_session is session
            ]
            for worker_id in stale_worker_ids:
                self._retired_import_sessions_by_worker.pop(worker_id, None)

    def _load_dataset(self):
        """Load dataset(s) using per-file import configuration and background workers."""
        self._deauthorize_active_import(emit_canceled_finished=True, outcome="superseded")

        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Load Dataset(s)",
            "",
            "Data Files (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx);;Text Files (*.txt);;All Files (*)"
        )

        if not filenames:
            return

        configs: List[ImportConfig] = []
        index = 0
        while index < len(filenames):
            filepath = str(filenames[index])
            remaining = len(filenames) - index - 1
            result = self._collect_import_config(filepath, remaining)
            if result.action == "cancel":
                return
            if result.action == "skip":
                index += 1
                continue
            if result.config is None:
                index += 1
                continue
            configs.append(result.config)
            if result.config.file_intent.apply_to_remaining:
                source_intent = result.config.remaining_file_template
                if source_intent is None:
                    raise RuntimeError(
                        "apply_to_remaining is True but remaining_file_template is None"
                    )
                source_sheet_names = result.config.file_intent.sheet_names
                for remaining_idx in range(index + 1, len(filenames)):
                    remaining_path = str(filenames[remaining_idx])
                    try:
                        remaining_config = self._build_remaining_file_config(
                            remaining_path, source_intent,
                            source_sheet_names=source_sheet_names,
                        )
                    except (ValueError, OSError, UnicodeDecodeError) as exc:
                        QtWidgets.QMessageBox.critical(
                            self,
                            "Import Error",
                            f"Cannot apply settings to "
                            f"'{os.path.basename(remaining_path)}':\n\n{exc}",
                        )
                        continue
                    configs.append(remaining_config)
                break
            index += 1

        if not configs:
            return

        expected_count = sum(self._expected_dataset_count_for_config(config) for config in configs)
        if expected_count <= 0:
            return

        self._load_generation += 1
        current_generation = self._load_generation
        self._cancel_requested = False
        self._import_completion_emitted = False
        self._active_import_session = DatasetImportSession(
            session_id=current_generation,
            expected_units=expected_count,
        )

        dataset_label = f"{expected_count} dataset{'s' if expected_count != 1 else ''}"
        logger.info("User initiated dataset import: %s", dataset_label)

        self._progress_dialog = QtWidgets.QProgressDialog(
            f"Loading {dataset_label}...", "Cancel", 0, 100, self
        )
        self._progress_dialog.setWindowTitle("Loading Datasets")
        self._progress_dialog.setWindowModality(Qt.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.canceled.connect(self._on_load_canceled)
        self._progress_dialog.show()

        for config in configs:
            result_units = self._expected_dataset_count_for_config(config)
            if config.file_type == "excel":
                worker: QtCore.QThread = ExcelLoaderWorker(config.filepath, list(config.plans))
                worker.loaded.connect(self._on_excel_loaded)
            else:
                worker = CSVLoaderWorker(config.plans[0])
                worker.loaded.connect(self._on_csv_loaded)
            setattr(worker, "_expected_result_count", result_units)
            setattr(worker, "_accounted_result_count", 0)
            setattr(worker, "_load_generation", current_generation)
            worker.progress.connect(self._on_load_progress)
            worker.cancelled.connect(self._on_csv_cancelled)
            worker.error.connect(self._on_csv_error)
            worker.finished.connect(self._on_worker_done)
            session = self._active_import_session
            if session is not None:
                session.register_worker(worker, expected_units=result_units, config=config)
            worker.start()

    def _expected_dataset_count_for_config(self, config: ImportConfig) -> int:
        return len(config.plans)

    def _collect_import_config(self, filepath: str, remaining_count: int) -> ImportDialogResult:
        try:
            dialog = ImportConfigDialog(filepath, remaining_count=remaining_count, parent=self)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to open import configuration for '{os.path.basename(filepath)}':\n\n{type(exc).__name__}: {exc}",
            )
            return ImportDialogResult(config=None, action="skip")
        dialog.exec()
        return dialog.get_result()

    def _build_remaining_file_config(
        self,
        filepath: str,
        source_intent: SheetImportIntent,
        *,
        source_sheet_names: Tuple[str, ...] = (),
    ) -> ImportConfig:
        """Build an ImportConfig for a remaining file using the source intent.

        Raises ValueError if the source intent is incompatible with the file.
        """
        lower = filepath.lower()
        file_type = "excel" if lower.endswith(".xlsx") else "csv"

        per_sheet_intents: Dict[Optional[str], SheetImportIntent] = {}
        per_sheet_detections: Dict[Optional[str], UnitDetection] = {}
        per_sheet_columns: Dict[Optional[str], List[str]] = {}
        target_sheet_names: Tuple[str, ...] = ()

        # Scope detection to source-selected columns only
        selected_keys = set(source_intent.species_columns)
        if source_intent.time_column:
            selected_keys.add(source_intent.time_column)

        if file_type == "csv":
            with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    raise ValueError("CSV file is empty")
                columns = [h.strip() for h in header]
                first_row_raw = next(reader, None)
            if first_row_raw is not None:
                row_values = [c.strip() if c else "" for c in first_row_raw]
                row_mapping = dict(zip(columns, row_values))
                det = detect_units_from_row_mapping(
                    row_mapping, list(selected_keys)
                )
            else:
                det = UnitDetection.empty()
            per_sheet_intents[None] = rebuild_intent_for_target(
                source_intent, det,
            )
            per_sheet_detections[None] = det
            per_sheet_columns[None] = columns
        else:
            sheets = list_sheets(filepath)
            if not sheets:
                raise ValueError("Excel file has no sheets")
            # Filter sheets by source's checked set
            if source_sheet_names:
                source_set = set(source_sheet_names)
                sheets = [s for s in sheets if s in source_set]
                if not sheets:
                    raise ValueError("No matching sheets found")
            for sheet_name in sheets:
                with closing(read_excel_sheet_rows(filepath, sheet_name)) as rows:
                    first_row = next(iter(rows), None)
                if first_row is None:
                    per_sheet_columns[sheet_name] = []
                    per_sheet_detections[sheet_name] = UnitDetection.empty()
                else:
                    per_sheet_columns[sheet_name] = list(first_row.keys())
                    per_sheet_detections[sheet_name] = detect_units_from_row_mapping(
                        dict(first_row), list(selected_keys)
                    )
                per_sheet_intents[sheet_name] = rebuild_intent_for_target(
                    source_intent, per_sheet_detections[sheet_name],
                )
            target_sheet_names = tuple(sheets)

        plans = resolve_import_plans(
            filepath, file_type,
            per_sheet_intents, per_sheet_detections, per_sheet_columns,
        )

        file_intent = UserImportIntent(
            sheet_names=target_sheet_names,
            apply_to_remaining=False,
        )
        return ImportConfig(
            filepath=filepath,
            file_type=file_type,
            file_intent=file_intent,
            per_sheet_intents=tuple(per_sheet_intents.items()),
            plans=tuple(plans),
        )

    def _on_load_progress(self, percent: int):
        """Handle progress updates from dataset import worker(s)."""
        if self._progress_dialog:
            session = self._active_import_session
            expected = int(getattr(session, "expected_units", 0) or 0)
            completed = int(getattr(session, "_completed_units", 0) or 0)
            if expected > 0:
                completed_fraction = completed / expected
                current_file_fraction = (1.0 / expected) * (percent / 100.0)
                overall_percent = int((completed_fraction + current_file_fraction) * 100)
                self._progress_dialog.setValue(min(overall_percent, 99))

    def _on_load_canceled(self):
        """Handle cancel request during dataset loading."""
        logger.info("Dataset import cancellation requested by user")
        self._cancel_requested = True
        self._deauthorize_active_import(
            emit_canceled_finished=True,
            outcome="user_canceled",
        )
        self._cancel_requested = False

    def _on_csv_cancelled(self, name: str):
        """Handle dataset import cancellation without surfacing an error dialog."""
        sender = self.sender()
        if sender is not None and int(getattr(sender, "_load_generation", -1)) != int(self._load_generation):
            return
        logger.info("Dataset import canceled: %s", name)
        session = self._active_import_session
        if session is not None and sender is not None:
            session.note_worker_units_processed(sender, session.remaining_worker_result_count(sender))
        self._maybe_finalize_load_cycle()

    def _emit_import_completed(
        self,
        *,
        canceled: bool,
        superseded: bool = False,
        discard_units: bool = False,
        outcome: str = "completed",
    ) -> None:
        """Emit one completion object for the active import session."""
        if self._import_completion_emitted:
            return
        self._import_completion_emitted = True
        session = self._active_import_session
        self._active_import_session = None
        if session is None:
            completion = DatasetImportCompletion(
                session_id=int(self._load_generation),
                units=(),
                outcome=str(outcome or "completed"),
                errors=(),
                canceled=bool(canceled),
                superseded=bool(superseded),
            )
        else:
            completion = session.completion(
                canceled=bool(canceled),
                superseded=bool(superseded),
                discard_units=bool(discard_units),
                outcome=str(outcome or "completed"),
            )
        self.importCompleted.emit(completion)

    def _finalize_progress_dialog(self) -> None:
        """Close and clear the progress dialog if it exists."""
        if not self._progress_dialog:
            return
        try:
            self._progress_dialog.canceled.disconnect(self._on_load_canceled)
        except (TypeError, RuntimeError):
            pass
        with suppress(RuntimeError, TypeError):
            self._progress_dialog.setValue(100)
        self._progress_dialog.close()
        self._progress_dialog = None

    def _maybe_finalize_load_cycle(self):
        """Close progress UI and emit completion when all workers are gone."""
        session = self._active_import_session
        if session is not None and session.complete_ready():
            self._finalize_progress_dialog()
            outcome = "user_canceled" if self._cancel_requested else "completed"
            self._emit_import_completed(
                canceled=self._cancel_requested,
                discard_units=bool(self._cancel_requested),
                outcome=outcome,
            )
            self._cancel_requested = False

    def _finalize_worker(self, worker: QtCore.QThread) -> bool:
        """Disconnect and delete a stopped import worker."""
        if not worker:
            return False

        try:
            if worker.isRunning():
                worker.requestInterruption()
                return False
        except RuntimeError:
            return False

        for signal in (
            getattr(worker, "progress", None),
            getattr(worker, "loaded", None),
            getattr(worker, "error", None),
            getattr(worker, "cancelled", None),
            getattr(worker, "finished", None),
        ):
            if signal is None:
                continue
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                # Already disconnected or deleted
                pass

        worker.deleteLater()

        return True

    def _on_worker_done(self) -> None:
        """Marshal worker cleanup back onto the GUI thread."""
        worker = self.sender()
        session = self._import_session_for_worker(worker)
        finalized = self._finalize_worker(worker)
        if not finalized:
            return
        if session is not None and worker is not None:
            session.mark_worker_terminal(worker)
            self._release_retired_import_worker(worker, session)
        self._maybe_finalize_load_cycle()

    def _apply_unit_conversion(self, data: dict, plan: ResolvedSheetPlan) -> None:
        scale_payload_in_place(
            data,
            time_factor=plan.time_factor,
            conc_factors=plan.conc_factors,
        )
        metadata = data.setdefault("metadata", {})
        metadata["original_time_unit"] = plan.original_time_unit
        metadata["original_concentration_units"] = dict(plan.original_conc_units)

    def _finalize_loaded_dataset(self, worker: Optional[QtCore.QThread], name: str, data: dict) -> None:
        if worker is not None and int(getattr(worker, "_load_generation", -1)) != int(self._load_generation):
            return
        session = self._active_import_session
        if session is None or (worker is not None and not session.owns_worker(worker)):
            return
        config = session.worker_config(worker) if worker is not None else None
        plan: Optional[ResolvedSheetPlan] = None
        if config is not None:
            if "::" in name:
                sheet_name = name.split("::", 1)[1]
                plan = next((p for p in config.plans if p.sheet_name == sheet_name), None)
            elif config.plans:
                plan = config.plans[0]
        try:
            if plan is not None:
                self._apply_unit_conversion(data, plan)
        except Exception as exc:
            if worker is not None:
                session.note_worker_units_processed(worker, 1)
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to apply import settings:\n\n{type(exc).__name__}: {exc}",
            )
            self._maybe_finalize_load_cycle()
            return

        session.add_unit(
            DatasetImportUnit(
                display_name=str(name),
                payload=dict(data),
                source_path=str(getattr(worker, "filepath", "") or ""),
            )
        )
        if worker is not None:
            session.note_worker_units_processed(worker, 1)
        self._maybe_finalize_load_cycle()

    def _on_csv_loaded(self, name: str, data: dict):
        """Handle successful CSV load with import settings applied."""
        self._finalize_loaded_dataset(self.sender(), name, data)

    def _on_excel_loaded(self, name: str, data: dict):
        """Handle successful Excel sheet load with import settings applied."""
        self._finalize_loaded_dataset(self.sender(), name, data)

    def _on_csv_error(self, error_msg: str):
        """Handle dataset load error."""
        sender = self.sender()
        if sender is not None and int(getattr(sender, "_load_generation", -1)) != int(self._load_generation):
            return
        session = self._active_import_session
        if session is None or (sender is not None and not session.owns_worker(sender)):
            return
        if sender is not None:
            session.note_worker_units_processed(sender, 1)
        session.add_error(str(error_msg))

        QtWidgets.QMessageBox.critical(
            self,
            "Load Error",
            f"Failed to load dataset:\n\n{error_msg}"
        )

        self._maybe_finalize_load_cycle()

        # Note: Worker already logged the error, no need to log again here

    def _remove_dataset(self):
        """Remove selected dataset."""
        current = self._dataset_list.currentItem()
        if not current:
            return

        dataset_id = str(current.data(Qt.UserRole) or "")
        if dataset_id:
            self.datasetRemovalRequested.emit(dataset_id)

    def _on_dataset_selected(self, current, previous):
        """Update preview when dataset is selected."""
        if not current:
            self._preview_label.setText("No dataset selected")
            self._preview_label.hide()
            return

        dataset_id = str(current.data(Qt.UserRole) or "")
        record = self._visible_records_by_id.get(dataset_id)
        if record is None:
            self._preview_label.hide()
            return

        name = str(record.display_name)
        data = record.payload
        self._preview_label.show()
        observations = observations_from_payload(data)
        t, species = dense_view_from_observations(observations)

        metadata = data.get('metadata', {})
        mapping_source = metadata.get('mapping_source', 'auto')
        preview = f"<b>{name}</b><br>"
        preview += f"{len(t)} time points<br>"
        preview += f"Time column: {metadata.get('time_column', 'unknown')} ({mapping_source})<br>"
        preview += f"{len(species)} species: {', '.join(list(species.keys())[:5])}"
        if len(species) > 5:
            preview += f", ... ({len(species)-5} more)"

        self._preview_label.setText(preview)
