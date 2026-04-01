# kindred/gui/widgets/data_manager.py
"""Data manager panel for loading and managing experimental datasets."""

from __future__ import annotations

import csv
from contextlib import closing, suppress
from dataclasses import replace
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
from kindred.core.datasets.units import looks_like_unit_row, parse_concentration_unit, parse_time_unit, parse_unit
from kindred.gui.widgets.import_config_dialog import ImportConfig, ImportConfigDialog, ImportDialogResult

logger = logging.getLogger(__name__)

__all__ = ["DataManagerPanel"]

class CSVLoaderWorker(QtCore.QThread):
    """Background worker for loading CSV files without blocking UI."""

    finished = QtCore.Signal(str, dict)  # name, data
    cancelled = QtCore.Signal(str)  # dataset name
    error = QtCore.Signal(str)  # error message
    progress = QtCore.Signal(int)  # progress percentage
    done = QtCore.Signal()

    def __init__(
        self,
        filepath: str,
        time_column: Optional[str] = None,
        species_columns: Optional[Sequence[str]] = None,
        unit_row_detected: bool = False,
    ):
        """
        Initialize CSV loader worker.

        Parameters
        ----------
        filepath : str
            Path to CSV file to load
        """
        super().__init__()
        self.filepath = filepath
        self._time_column = (time_column or "").strip() or None
        self._species_columns = [col.strip() for col in (species_columns or []) if col and col.strip()]
        self._unit_row_detected = bool(unit_row_detected)

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
            if self._unit_row_detected:
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

    def __init__(self, config: ImportConfig):
        super().__init__()
        self.filepath = str(config.filepath)
        self._config = config

    def _load_sheet_payload(self, sheet_name: str) -> Tuple[str, dict]:
        with closing(read_excel_sheet_rows(self.filepath, sheet_name)) as rows:
            row_iter = iter(rows)
            first_row = next(row_iter, None)
            if first_row is None:
                raise ValueError(f"Sheet '{sheet_name}' is empty.")
            first_row_mapping = dict(first_row)
            if self._config.unit_row_detected:
                rows_to_parse = row_iter
            else:
                rows_to_parse = itertools.chain((first_row_mapping,), row_iter)
            _time_source, data = parse_csv_rows(
                rows_to_parse,
                time_column=self._config.time_column,
                species_columns=self._config.species_columns,
                interruption_checker=self.isInterruptionRequested,
            )
        return f"{os.path.basename(self.filepath)}::{sheet_name}", data

    def run(self):
        """Load selected Excel sheet(s) in a background thread."""
        dataset_name = os.path.basename(self.filepath)
        sheet_names = list(self._config.sheet_names or [])
        total_sheets = len(sheet_names)

        if total_sheets <= 0:
            self.error.emit("No Excel sheets were selected for import.")
            self.done.emit()
            return

        try:
            self.progress.emit(0)
            for index, sheet_name in enumerate(sheet_names, start=1):
                if self.isInterruptionRequested():
                    logger.info("Excel import interrupted before sheet %s: %s", sheet_name, dataset_name)
                    self.cancelled.emit(dataset_name)
                    return
                try:
                    loaded_name, data = self._load_sheet_payload(sheet_name)
                except CsvImportInterrupted:
                    logger.info("Excel import interrupted while parsing sheet %s: %s", sheet_name, dataset_name)
                    self.cancelled.emit(dataset_name)
                    return
                except Exception as exc:
                    logger.error(
                        "Excel import failed: %s sheet %s - %s: %s",
                        dataset_name,
                        sheet_name,
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                    self.error.emit(f"Sheet '{sheet_name}': {type(exc).__name__}: {exc}")
                else:
                    self.finished.emit(loaded_name, data)
                self.progress.emit(int((index / total_sheets) * 100))
                if self.isInterruptionRequested():
                    logger.info("Excel import interrupted after sheet %s: %s", sheet_name, dataset_name)
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
            if not self._config_has_compatible_unit_detection(result.config):
                index += 1
                continue
            configs.append(result.config)
            if result.config.apply_to_remaining:
                remaining_files = filenames[index + 1 :]
                for remaining_index, remaining_filepath in enumerate(remaining_files):
                    cloned_config = self._clone_config_for_file(result.config, str(remaining_filepath))
                    if cloned_config is not None:
                        configs.append(cloned_config)
                        continue
                    remaining_after = len(remaining_files) - remaining_index - 1
                    fallback_result = self._collect_import_config(
                        str(remaining_filepath),
                        max(0, remaining_after),
                    )
                    if fallback_result.action == "cancel":
                        return
                    if fallback_result.action == "import" and fallback_result.config is not None:
                        if not self._config_has_compatible_unit_detection(fallback_result.config):
                            continue
                        configs.append(fallback_result.config)
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
                worker: QtCore.QThread = ExcelLoaderWorker(config)
                worker.finished.connect(self._on_excel_loaded)
            else:
                worker = CSVLoaderWorker(
                    config.filepath,
                    time_column=config.time_column or None,
                    species_columns=config.species_columns or None,
                    unit_row_detected=bool(config.unit_row_detected),
                )
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

    @staticmethod
    def _file_type_for_path(filepath: str) -> Optional[str]:
        lower_path = str(filepath).lower()
        if lower_path.endswith(".xls") and not lower_path.endswith(".xlsx"):
            return None
        if lower_path.endswith(".xlsx"):
            return "excel"
        return "csv"

    def _expected_dataset_count_for_config(self, config: ImportConfig) -> int:
        if str(config.file_type) == "excel":
            return len(config.sheet_names or [])
        return 1

    def _clone_config_for_file(self, source_config: ImportConfig, target_filepath: str) -> Optional[ImportConfig]:
        target_file_type = self._file_type_for_path(target_filepath)
        if target_file_type != str(source_config.file_type):
            return None
        required_columns = [str(source_config.time_column)] + [str(name) for name in (source_config.species_columns or [])]
        if target_file_type == "excel":
            selected_sheets = list(source_config.sheet_names or [])
            try:
                available_sheets = set(list_sheets(target_filepath))
            except Exception:
                return None
            for sheet_name in selected_sheets:
                if sheet_name not in available_sheets:
                    return None
                try:
                    columns = self._excel_columns_for_sheet(target_filepath, sheet_name)
                except Exception:
                    return None
                if not self._columns_cover_required_fields(columns, required_columns):
                    return None
                if (
                    self._detected_excel_sheet_unit_signature(source_config.filepath, sheet_name, required_columns)
                    != self._detected_excel_sheet_unit_signature(target_filepath, sheet_name, required_columns)
                ):
                    return None
            cloned_config = replace(
                source_config,
                filepath=target_filepath,
                sheet_names=list(selected_sheets),
                apply_to_remaining=False,
            )
            if not self._config_has_compatible_unit_detection(cloned_config, show_message=False):
                return None
            return cloned_config
        try:
            columns = self._csv_columns_for_file(target_filepath)
        except Exception:
            return None
        if not self._columns_cover_required_fields(columns, required_columns):
            return None
        if self._detected_csv_unit_signature(source_config.filepath, required_columns) != self._detected_csv_unit_signature(
            target_filepath,
            required_columns,
        ):
            return None
        return replace(source_config, filepath=target_filepath, apply_to_remaining=False)

    def _config_has_compatible_unit_detection(
        self,
        config: ImportConfig,
        *,
        show_message: bool = True,
    ) -> bool:
        relevant_columns = self._unit_detection_columns(config)
        if str(config.file_type) == "csv":
            detected = self._detected_csv_unit_signature(config.filepath, relevant_columns)
            if len(tuple(detected["concentration"])) > 1:
                if show_message:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Load Error",
                        "Detected multiple concentration units in the selected data. Import each unit group separately.",
                    )
                return False
            return True
        if str(config.file_type) != "excel":
            return True
        sheet_names = list(config.sheet_names or [])
        signatures: list[dict[str, object]] = []
        for sheet_name in sheet_names:
            detected = self._detected_excel_sheet_unit_signature(config.filepath, sheet_name, relevant_columns)
            signatures.append(detected)
            if len(tuple(detected["concentration"])) > 1:
                if show_message:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Load Error",
                        "Detected multiple concentration units in the selected data. Import each unit group separately.",
                    )
                return False
        if len(sheet_names) <= 1:
            return True
        signature_set = {
            (
                bool(detected["has_unit_row"]),
                detected["time"],
                tuple(detected["concentration"]),
            )
            for detected in signatures
        }
        if len(signature_set) > 1:
            if show_message:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Load Error",
                    "Selected Excel sheets have different detected unit rows. Import each sheet separately.",
                )
            return False
        return True

    @staticmethod
    def _unit_detection_columns(config: ImportConfig) -> List[str]:
        columns: List[str] = []
        if str(config.time_column or "").strip():
            columns.append(str(config.time_column).strip())
        columns.extend(
            str(name).strip()
            for name in (config.species_columns or [])
            if str(name).strip()
        )
        return columns

    @staticmethod
    def _build_unit_signature(
        row_mapping: Dict[str, object],
        relevant_columns: Sequence[str],
    ) -> Dict[str, object]:
        columns = [str(column).strip() for column in relevant_columns if str(column).strip()]
        if not columns:
            return {"has_unit_row": False, "time": None, "concentration": tuple()}
        normalized_values = [str(row_mapping.get(column, "")).strip() for column in columns]
        if not looks_like_unit_row(normalized_values):
            return {"has_unit_row": False, "time": None, "concentration": tuple()}
        time_factor: Optional[float] = None
        concentration_factors: set[float] = set()
        for value in normalized_values:
            if not value:
                continue
            try:
                category, factor = parse_unit(value)
            except ValueError:
                continue
            if category == "time":
                time_factor = float(factor)
            elif category == "concentration":
                concentration_factors.add(float(factor))
        return {
            "has_unit_row": True,
            "time": time_factor,
            "concentration": tuple(sorted(concentration_factors)),
        }

    @classmethod
    def _detected_csv_unit_signature(cls, filepath: str, relevant_columns: Sequence[str]) -> Dict[str, object]:
        with open(filepath, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            first_row = next(reader, None)
        if header is None or first_row is None:
            return {"has_unit_row": False, "time": None, "concentration": tuple()}
        normalized_header = [str(column).strip() for column in header]
        row_mapping = {
            normalized_header[index]: (first_row[index] if index < len(first_row) else "")
            for index in range(len(normalized_header))
        }
        return cls._build_unit_signature(row_mapping, relevant_columns)

    @classmethod
    def _detected_excel_sheet_unit_signature(
        cls,
        filepath: str,
        sheet_name: str,
        relevant_columns: Sequence[str],
    ) -> Dict[str, object]:
        with closing(read_excel_sheet_rows(filepath, sheet_name)) as rows:
            first_row = next(iter(rows), None)
        if first_row is None:
            return {"has_unit_row": False, "time": None, "concentration": tuple()}
        return cls._build_unit_signature(dict(first_row), relevant_columns)

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

    @staticmethod
    def _columns_cover_required_fields(columns: Sequence[str], required_columns: Sequence[str]) -> bool:
        available = {str(column).strip() for column in columns if str(column).strip()}
        required = {str(column).strip() for column in required_columns if str(column).strip()}
        return bool(required) and required.issubset(available)

    @staticmethod
    def _csv_columns_for_file(filepath: str) -> List[str]:
        with open(filepath, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        if header is None:
            return []
        return [str(value).strip() for value in header]

    @staticmethod
    def _excel_columns_for_sheet(filepath: str, sheet_name: str) -> List[str]:
        with closing(read_excel_sheet_rows(filepath, sheet_name)) as rows:
            first_row = next(iter(rows), None)
        if first_row is None:
            return []
        return [str(value) for value in first_row.keys()]

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

    def _apply_unit_conversion(self, data: dict, config: ImportConfig) -> None:
        metadata = data.setdefault("metadata", {})
        time_unit = config.time_unit or "s"
        concentration_unit = config.concentration_unit or "M"
        if config.time_unit is not None:
            time_factor = parse_time_unit(config.time_unit)
            if time_factor != 1.0:
                data["t"] = data["t"] * time_factor
        if config.concentration_unit is not None:
            concentration_factor = parse_concentration_unit(config.concentration_unit)
            if concentration_factor != 1.0:
                for species_name in list((data.get("species") or {}).keys()):
                    data["species"][species_name] = data["species"][species_name] * concentration_factor
        metadata["original_time_unit"] = time_unit
        metadata["original_concentration_unit"] = concentration_unit

    def _finalize_loaded_dataset(self, worker: Optional[QtCore.QThread], name: str, data: dict) -> None:
        if worker is not None and int(getattr(worker, "_load_generation", -1)) != int(self._load_generation):
            return
        config = self._pending_import_configs.get(str(getattr(worker, "filepath", "") or ""))
        try:
            if config is not None:
                self._apply_unit_conversion(data, config)
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
