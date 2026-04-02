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
    read_excel_sheet_rows,
)
from kindred.gui.widgets.import_config import (
    ImportConfig,
    ResolvedSheetPlan,
)
from kindred.gui.widgets.import_config_dialog import ImportConfigDialog, ImportDialogResult

logger = logging.getLogger(__name__)

__all__ = ["DataManagerPanel"]

class CSVLoaderWorker(QtCore.QThread):
    """Background worker for loading CSV files without blocking UI."""

    finished = QtCore.Signal(str, dict)  # name, data
    cancelled = QtCore.Signal(str)  # dataset name
    error = QtCore.Signal(str)  # error message
    progress = QtCore.Signal(int)  # progress percentage
    done = QtCore.Signal()

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
                list(data['species'].keys()),
            )

            self.progress.emit(80)

            if _check_cancel("after parsing"):
                return

            self.progress.emit(100)

            # Log success before emitting signal
            species = data.get('species', {})
            logger.info(
                "CSV import completed: %s (%d rows, %d species: %s)",
                dataset_name,
                len(data.get('t', [])),
                len(species),
                list(species.keys()),
            )

            self.finished.emit(dataset_name, data)

        except CsvImportInterrupted:
            self.cancelled.emit(dataset_name)
            return
        except Exception as e:
            logger.error(f"CSV import failed: {dataset_name} - {type(e).__name__}: {e}", exc_info=True)
            self.error.emit(f"{type(e).__name__}: {str(e)}")
        finally:
            self.done.emit()


class ExcelLoaderWorker(QtCore.QThread):
    """Background worker for loading Excel sheets without blocking UI."""

    finished = QtCore.Signal(str, dict)  # name, data
    cancelled = QtCore.Signal(str)  # dataset name
    error = QtCore.Signal(str)  # error message
    progress = QtCore.Signal(int)  # progress percentage
    done = QtCore.Signal()

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
            )
        return f"{os.path.basename(self.filepath)}::{plan.sheet_name}", data

    def run(self):
        dataset_name = os.path.basename(self.filepath)
        total = len(self._plans)

        if total <= 0:
            self.error.emit("No Excel sheets were selected for import.")
            self.done.emit()
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
                    self.finished.emit(loaded_name, data)
                self.progress.emit(int((index / total) * 100))
                if self.isInterruptionRequested():
                    logger.info("Excel import interrupted after sheet %s: %s", plan.sheet_name, dataset_name)
                    self.cancelled.emit(dataset_name)
                    return
        except Exception as exc:
            logger.error("Excel import failed: %s - %s: %s", dataset_name, type(exc).__name__, exc, exc_info=True)
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.done.emit()


class DataManagerPanel(QtWidgets.QWidget):
    """
    Data manager for loading/managing experimental datasets.

    Features:
    - Load CSV and Excel files with experimental data
    - Auto-detect time column (tries: time, time_s, t, Time, T, x)
    - Extract all numeric columns as species
    - Preview loaded datasets
    - Import configuration dialog with unit selection
    - Multiple dataset support

    Signals:
        datasetLoaded(str, dict): Emitted when dataset is loaded
            - name: Dataset name (filename)
            - data: {'t': array, 'species': {name: array}}
        datasetRemoved(str): Emitted when a dataset is removed from the panel
        loadFinished(bool): Emitted once when a load cycle completes or is canceled
            - canceled: True if the operation was canceled
    """

    datasetLoaded = QtCore.Signal(str, dict)  # name, data dict
    datasetRemoved = QtCore.Signal(str)  # name
    loadFinished = QtCore.Signal(bool)  # canceled flag

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

        # Store loaded datasets {name: {t: array, species: {name: array}}}
        self._datasets: Dict[str, Dict] = {}

        # Track active import workers across CSV and Excel loads.
        self._csv_workers: List[QtCore.QThread] = []
        self._pending_files_count = 0
        self._completed_files_count = 0
        self._progress_dialog: Optional[QtWidgets.QProgressDialog] = None
        self._cancel_requested = False
        self._load_finished_emitted = False
        self._pending_import_configs: Dict[str, ImportConfig] = {}
        self._pending_import_units_remaining: Dict[str, int] = {}
        self._load_generation = 0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def trigger_load_dialog(self) -> None:
        """Programmatically invoke the load dialog (mirrors Load button)."""
        self._load_dataset()

    def clear_datasets(self) -> None:
        """Clear loaded datasets and reset dataset-panel UI state."""
        self._load_generation += 1
        workers = list(self._csv_workers)
        for worker in workers:
            with suppress(RuntimeError):
                worker.requestInterruption()
        for worker in workers:
            self._cleanup_worker(worker)
        self._pending_files_count = 0
        self._completed_files_count = 0
        self._pending_import_configs.clear()
        self._pending_import_units_remaining.clear()
        self._cancel_requested = False
        self._load_finished_emitted = False
        self._finalize_progress_dialog()
        self._datasets.clear()
        self._dataset_list.clear()
        self._preview_label.setText("No dataset selected")
        self._preview_label.hide()

    def _load_dataset(self):
        """Load dataset(s) using per-file import configuration and background workers."""
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
                break
            index += 1

        if not configs:
            return

        expected_count = sum(self._expected_dataset_count_for_config(config) for config in configs)
        if expected_count <= 0:
            return

        self._load_generation += 1
        current_generation = self._load_generation
        self._pending_files_count = expected_count
        self._completed_files_count = 0
        self._cancel_requested = False
        self._load_finished_emitted = False
        self._pending_import_configs.clear()
        self._pending_import_units_remaining.clear()

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
            self._pending_import_configs[config.filepath] = config
            self._pending_import_units_remaining[config.filepath] = result_units
            if config.file_type == "excel":
                worker: QtCore.QThread = ExcelLoaderWorker(config.filepath, list(config.plans))
                worker.finished.connect(self._on_excel_loaded)
            else:
                worker = CSVLoaderWorker(config.plans[0])
                worker.finished.connect(self._on_csv_loaded)
            setattr(worker, "_expected_result_count", result_units)
            setattr(worker, "_accounted_result_count", 0)
            setattr(worker, "_load_generation", current_generation)
            worker.progress.connect(self._on_load_progress)
            worker.cancelled.connect(self._on_csv_cancelled)
            worker.error.connect(self._on_csv_error)
            worker.done.connect(self._on_worker_done)
            self._csv_workers.append(worker)
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

    def _on_load_progress(self, percent: int):
        """Handle progress updates from dataset import worker(s)."""
        if self._progress_dialog:
            if self._pending_files_count > 0:
                completed_fraction = self._completed_files_count / self._pending_files_count
                current_file_fraction = (1.0 / self._pending_files_count) * (percent / 100.0)
                overall_percent = int((completed_fraction + current_file_fraction) * 100)
                self._progress_dialog.setValue(min(overall_percent, 99))

    def _on_load_canceled(self):
        """Handle cancel request during dataset loading."""
        for worker in self._csv_workers:
            if worker:
                worker.requestInterruption()

        logger.info("Dataset import cancellation requested by user")
        self._cancel_requested = True

        # Close progress dialog immediately
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None

        self._maybe_finalize_load_cycle()

    def _on_csv_cancelled(self, name: str):
        """Handle dataset import cancellation without surfacing an error dialog."""
        sender = self.sender()
        if sender is not None and int(getattr(sender, "_load_generation", -1)) != int(self._load_generation):
            return
        logger.info("Dataset import canceled: %s", name)
        self._note_worker_units_processed(sender, self._remaining_worker_result_count(sender))
        self._maybe_finalize_load_cycle()

    def _emit_load_finished(self, canceled: bool) -> None:
        """Ensure overall completion/cancellation signal fires only once."""
        if self._load_finished_emitted:
            return
        self._load_finished_emitted = True
        self.loadFinished.emit(canceled)

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
        if (
            self._pending_files_count > 0
            and self._completed_files_count >= self._pending_files_count
            and not self._csv_workers
        ):
            self._finalize_progress_dialog()
            self._pending_files_count = 0
            self._completed_files_count = 0
            self._pending_import_configs.clear()
            self._pending_import_units_remaining.clear()
            self._emit_load_finished(self._cancel_requested)
            self._cancel_requested = False

    def _remaining_worker_result_count(self, worker: Optional[QtCore.QThread]) -> int:
        if worker is None:
            return 0
        expected = int(getattr(worker, "_expected_result_count", 0) or 0)
        accounted = int(getattr(worker, "_accounted_result_count", 0) or 0)
        return max(0, expected - accounted)

    def _note_worker_units_processed(self, worker: Optional[QtCore.QThread], count: int) -> None:
        if worker is None:
            return
        if int(getattr(worker, "_load_generation", -1)) != int(self._load_generation):
            return
        expected = int(getattr(worker, "_expected_result_count", 0) or 0)
        accounted = int(getattr(worker, "_accounted_result_count", 0) or 0)
        delta = max(0, min(int(count), max(0, expected - accounted)))
        if delta <= 0:
            return
        setattr(worker, "_accounted_result_count", accounted + delta)
        self._completed_files_count += delta
        filepath = str(getattr(worker, "filepath", "") or "")
        if not filepath:
            return
        remaining = max(0, int(self._pending_import_units_remaining.get(filepath, 0) or 0) - delta)
        if remaining <= 0:
            self._pending_import_units_remaining.pop(filepath, None)
            self._pending_import_configs.pop(filepath, None)
            return
        self._pending_import_units_remaining[filepath] = remaining

    def _cleanup_worker(self, worker: QtCore.QThread):
        """Clean up worker thread after completion or error."""
        if not worker:
            return

        self._note_worker_units_processed(worker, self._remaining_worker_result_count(worker))

        try:
            worker.requestInterruption()
        except RuntimeError:
            pass

        if worker.isRunning():
            worker.quit()
            worker.wait(2000)  # Wait up to 2 seconds
        else:
            worker.quit()
            worker.wait(2000)

        # Disconnect signals to prevent memory leaks
        for signal in (
            getattr(worker, "progress", None),
            getattr(worker, "finished", None),
            getattr(worker, "error", None),
            getattr(worker, "cancelled", None),
            getattr(worker, "done", None),
        ):
            if signal is None:
                continue
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                # Already disconnected or deleted
                pass

        # Remove from workers list
        if worker in self._csv_workers:
            self._csv_workers.remove(worker)

        # Delete worker
        worker.deleteLater()

        self._maybe_finalize_load_cycle()

    def _on_worker_done(self) -> None:
        """Marshal worker cleanup back onto the GUI thread."""
        self._cleanup_worker(self.sender())

    def _apply_unit_conversion(self, data: dict, plan: ResolvedSheetPlan) -> None:
        metadata = data.setdefault("metadata", {})
        if plan.time_factor != 1.0:
            data["t"] = data["t"] * plan.time_factor
        if plan.conc_factor != 1.0:
            for species_name in list((data.get("species") or {}).keys()):
                data["species"][species_name] = data["species"][species_name] * plan.conc_factor
        metadata["original_time_unit"] = plan.original_time_unit
        metadata["original_concentration_unit"] = plan.original_conc_unit

    def _finalize_loaded_dataset(self, worker: Optional[QtCore.QThread], name: str, data: dict) -> None:
        if worker is not None and int(getattr(worker, "_load_generation", -1)) != int(self._load_generation):
            return
        config = self._pending_import_configs.get(str(getattr(worker, "filepath", "") or ""))
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
            self._note_worker_units_processed(worker, 1)
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to apply import settings:\n\n{type(exc).__name__}: {exc}",
            )
            self._maybe_finalize_load_cycle()
            return

        unique_name = self._make_unique_dataset_name(name)
        self._datasets[unique_name] = data
        self._dataset_list.addItem(unique_name)
        self.datasetLoaded.emit(unique_name, data)
        self._note_worker_units_processed(worker, 1)
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
        self._note_worker_units_processed(sender, 1)

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

        name = current.text()
        self._dataset_list.takeItem(self._dataset_list.row(current))
        if name in self._datasets:
            del self._datasets[name]
            self.datasetRemoved.emit(name)

    def _on_dataset_selected(self, current, previous):
        """Update preview when dataset is selected."""
        if not current:
            self._preview_label.setText("No dataset selected")
            self._preview_label.hide()
            return

        name = current.text()
        if name not in self._datasets:
            self._preview_label.hide()
            return

        data = self._datasets[name]
        self._preview_label.show()
        t = data['t']
        species = data['species']

        metadata = data.get('metadata', {})
        mapping_source = metadata.get('mapping_source', 'auto')
        preview = f"<b>{name}</b><br>"
        preview += f"{len(t)} time points<br>"
        preview += f"Time column: {metadata.get('time_column', 'unknown')} ({mapping_source})<br>"
        preview += f"{len(species)} species: {', '.join(list(species.keys())[:5])}"
        if len(species) > 5:
            preview += f", ... ({len(species)-5} more)"

        self._preview_label.setText(preview)

    def _make_unique_dataset_name(self, name: str) -> str:
        """
        Generate a unique dataset name by appending _1, _2, etc. if needed.

        Parameters
        ----------
        name : str
            Desired dataset name (usually base filename)

        Returns
        -------
        str
            Unique name not already in self._datasets
        """
        if name not in self._datasets:
            return name

        # Extract base name and extension
        base, ext = os.path.splitext(name)

        # Try name_1, name_2, etc.
        counter = 1
        while True:
            candidate = f"{base}_{counter}{ext}"
            if candidate not in self._datasets:
                return candidate
            counter += 1

    def get_datasets(self) -> Dict[str, Dict]:
        """
        Get all loaded datasets.

        Returns
        -------
        dict
            Dictionary of {name: {'t': array, 'species': {name: array}}}
        """
        return self._datasets

    def get_selected_dataset(self) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Get currently selected dataset.

        Returns
        -------
        tuple
            (name, data) or (None, None) if no selection
        """
        current = self._dataset_list.currentItem()
        if not current:
            return None, None

        name = current.text()
        return name, self._datasets.get(name)

    def get_dataset(self, name: str) -> Optional[Dict]:
        """
        Retrieve a dataset by name.

        Parameters
        ----------
        name : str
            Dataset identifier (usually filename)

        Returns
        -------
        dict or None
            Dataset payload or None if not loaded.
        """
        return self._datasets.get(name)
