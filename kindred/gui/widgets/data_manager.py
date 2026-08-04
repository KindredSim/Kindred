# kindred/gui/widgets/data_manager.py
"""Data manager panel for loading and managing experimental datasets."""

from __future__ import annotations

import csv
from contextlib import closing, suppress
import io
import itertools
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from kindred.core.datasets.csv_import import (
    CsvImportInterrupted,
    PhysicalColumn,
    identify_physical_columns,
    parse_physical_rows,
    validate_tabular_source_bytes,
)
from kindred.core.datasets.excel_import import (
    list_sheets_from_bytes,
    read_excel_sheet_rows_from_bytes,
)
from kindred.core.datasets.observation_payload import (
    DatasetObservationPayload,
)
from kindred.gui.widgets.import_config import (
    ImportConfig,
    ImportSourceSnapshot,
    ResolvedSheetPlan,
    SheetImportIntent,
    UnitDetection,
    UserImportIntent,
    capture_import_source,
    detect_units_from_physical_row,
    rebuild_intent_for_target,
    resolve_import_plans,
)
from kindred.gui.widgets.import_config_dialog import ImportConfigDialog, ImportDialogResult
from kindred.gui.controllers.dataset_errors import (
    DatasetImportAdmissionError,
)
from kindred.gui.dataset_import_staging import (
    DatasetImportCancellation,
    DatasetImportFailure,
    DatasetImportProgress,
    DatasetImportRequest,
    DatasetImportStaging,
    DatasetImportWorker,
    DatasetImportWorkerRequest,
)
from kindred.gui.dataset_payload_snapshot import (
    ActiveDatasetLinkSnapshot,
    DataManagerSnapshot,
)
from kindred.core.project_document import RawDatasetRecord, thaw_json_value
from kindred.gui.display_name_policy import DATASET_LIST_LABEL_MAX_CHARS, compact_dataset_label

logger = logging.getLogger(__name__)

_MAX_SELECTED_SOURCES = 16
_MAX_SELECTED_SOURCE_BYTES = 128 * 1024 * 1024

__all__ = [
    "CSVLoaderWorker",
    "DataManagerPanel",
    "DatasetDeleteIntent",
    "DatasetExportOriginalIntent",
    "DatasetFitInclusionIntent",
    "DatasetMappingIntent",
    "DatasetObservationMappingIntent",
    "DatasetRenameIntent",
    "ExcelLoaderWorker",
]


@dataclass(frozen=True)
class DatasetRenameIntent:
    dataset_id: str
    display_name: str


@dataclass(frozen=True)
class DatasetDeleteIntent:
    dataset_id: str


@dataclass(frozen=True)
class DatasetExportOriginalIntent:
    dataset_id: str


@dataclass(frozen=True)
class DatasetMappingIntent:
    dataset_id: str
    action: str
    set_id: str | None = None
    occupied_choice: str | None = None


@dataclass(frozen=True)
class DatasetObservationMappingIntent:
    dataset_id: str


@dataclass(frozen=True)
class DatasetFitInclusionIntent:
    dataset_id: str
    include: bool


class CSVLoaderWorker(DatasetImportWorker):
    """Background worker for loading CSV files without blocking UI."""

    def __init__(
        self,
        source: ImportSourceSnapshot,
        plan: ResolvedSheetPlan,
    ):
        super().__init__(source)
        if source.file_type != "csv":
            raise ValueError("CSVLoaderWorker requires an admitted CSV snapshot.")
        if plan.sheet_name is not None:
            raise ValueError("CSVLoaderWorker cannot accept a worksheet plan.")
        self._plan = plan
        self._time_column = plan.time_column
        self._species_columns = tuple(plan.species_columns)
        self._skip_unit_row = plan.skip_unit_row

    def _load_csv_payload(self) -> DatasetObservationPayload:
        validate_tabular_source_bytes(
            self.source.exact_source_bytes,
            source_format="csv",
            source_label=self.source.original_filename,
        )
        with io.TextIOWrapper(
            io.BytesIO(self.source.exact_source_bytes),
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise ValueError("CSV file is empty")
            columns = identify_physical_columns(header)
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
                    column: (
                        row[column.source_index]
                        if column.source_index < len(row)
                        else ""
                    )
                    for column in columns
                }
                for row in rows
            )
            _time_source, data = parse_physical_rows(
                normalized_rows,
                columns=columns,
                time_column=self._time_column,
                species_columns=self._species_columns,
                interruption_checker=self.isInterruptionRequested,
                source_label=self.source.original_filename,
                source_row_start=3 if self._skip_unit_row else 2,
                time_factor=self._plan.time_factor,
                observation_factors=self._plan.conc_factors,
                metadata_extras={
                    "original_time_unit": self._plan.original_time_unit,
                    "original_concentration_units": dict(
                        self._plan.original_conc_units
                    ),
                },
            )
        return data

    def run(self):
        """Load CSV file in background thread."""
        filename = self.source.filepath
        dataset_name = self.source.original_filename

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
                "Parsed CSV dataset with observation columns: %s",
                [series.column_name for series in data.observations],
            )

            self.progress.emit(80)

            if _check_cancel("after parsing"):
                return

            self.progress.emit(100)

            # Log success before emitting signal
            observations = data.observations
            total_points = sum(len(series.t) for series in observations)
            logger.info(
                "CSV import completed: %s (%d rows, %d species: %s)",
                dataset_name,
                total_points,
                len(observations),
                [series.column_name for series in observations],
            )

            self.loaded.emit(dataset_name, data)

        except CsvImportInterrupted:
            self.cancelled.emit(dataset_name)
            return
        except Exception as e:
            logger.error(f"CSV import failed: {dataset_name} - {type(e).__name__}: {e}", exc_info=True)
            self.error.emit(f"{type(e).__name__}: {str(e)}")


class ExcelLoaderWorker(DatasetImportWorker):
    """Background worker for loading Excel sheets without blocking UI."""

    def __init__(
        self,
        source: ImportSourceSnapshot,
        plans: Sequence[ResolvedSheetPlan],
    ):
        super().__init__(source)
        if source.file_type != "excel":
            raise ValueError(
                "ExcelLoaderWorker requires an admitted XLSX snapshot."
            )
        owned_plans = tuple(plans)
        if not owned_plans or any(
            plan.sheet_name is None
            for plan in owned_plans
        ):
            raise ValueError(
                "ExcelLoaderWorker requires nonempty plans for the admitted "
                "workbook."
            )
        self._plans = owned_plans

    def _load_sheet_payload(
        self,
        plan: ResolvedSheetPlan,
    ) -> Tuple[str, DatasetObservationPayload]:
        with closing(
            read_excel_sheet_rows_from_bytes(
                self.source.exact_source_bytes,
                plan.sheet_name,
                source_label=self.source.original_filename,
            )
        ) as rows:
            row_iter = iter(rows)
            first_row = next(row_iter, None)
            if first_row is None:
                raise ValueError(f"Sheet '{plan.sheet_name}' is empty.")
            columns = tuple(first_row)
            if plan.skip_unit_row:
                rows_to_parse = row_iter
            else:
                rows_to_parse = itertools.chain((first_row,), row_iter)
            _time_source, data = parse_physical_rows(
                rows_to_parse,
                columns=columns,
                time_column=plan.time_column,
                species_columns=plan.species_columns,
                interruption_checker=self.isInterruptionRequested,
                source_label=self.source.original_filename,
                sheet_name=plan.sheet_name,
                source_row_start=3 if plan.skip_unit_row else 2,
                time_factor=plan.time_factor,
                observation_factors=plan.conc_factors,
                metadata_extras={
                    "original_time_unit": plan.original_time_unit,
                    "original_concentration_units": dict(
                        plan.original_conc_units
                    ),
                },
            )
        return f"{self.source.original_filename}::{plan.sheet_name}", data

    def run(self):
        dataset_name = self.source.original_filename
        total = len(self._plans)

        if total <= 0:
            self.error.emit("No Excel sheets were selected for import.")
            return

        try:
            validate_tabular_source_bytes(
                self.source.exact_source_bytes,
                source_format="xlsx",
                source_label=self.source.original_filename,
            )
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

    The panel retains only rendered snapshots and import UI configuration.
    DatasetImportStaging owns workers and uncommitted output; ProjectSession
    callbacks own every canonical mutation.
    """

    datasetRenameRequested = QtCore.Signal(object)
    datasetDeleteRequested = QtCore.Signal(object)
    datasetExportOriginalRequested = QtCore.Signal(object)
    datasetMappingRequested = QtCore.Signal(object)
    datasetObservationMappingRequested = QtCore.Signal(object)
    datasetFitInclusionRequested = QtCore.Signal(object)
    def __init__(
        self,
        staging: DatasetImportStaging,
        *,
        origin_tab_id: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """
        Initialize data manager panel.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        if not isinstance(staging, DatasetImportStaging):
            raise TypeError("DataManagerPanel requires the application DatasetImportStaging owner.")
        if (
            type(origin_tab_id) is not str
            or not origin_tab_id
            or origin_tab_id != origin_tab_id.strip()
        ):
            raise ValueError(
                "DataManagerPanel requires one stable edge-normalized origin "
                "tab ID."
            )
        self._staging = staging
        self._origin_tab_id = origin_tab_id

        layout = QtWidgets.QVBoxLayout(self)

        # Header
        layout.addWidget(QtWidgets.QLabel("<b>Loaded Datasets</b>"))

        # Dataset list
        self._dataset_list = QtWidgets.QListWidget()
        self._dataset_list.setUniformItemSizes(True)
        self._dataset_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._dataset_list.currentItemChanged.connect(self._on_dataset_selected)
        layout.addWidget(self._dataset_list)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self._load_btn = QtWidgets.QPushButton("Load Data")
        self._load_btn.setObjectName("loadDataButton")
        self._load_btn.clicked.connect(self._load_dataset)
        self._rename_btn = QtWidgets.QPushButton("Rename")
        self._rename_btn.clicked.connect(self._rename_dataset)
        self._remove_btn = QtWidgets.QPushButton("Delete")
        self._remove_btn.clicked.connect(self._remove_dataset)
        self._export_original_btn = QtWidgets.QPushButton("Export Original…")
        self._export_original_btn.clicked.connect(self._export_original)
        btn_layout.addWidget(self._load_btn)
        btn_layout.addWidget(self._rename_btn)
        btn_layout.addWidget(self._remove_btn)
        btn_layout.addWidget(self._export_original_btn)
        layout.addLayout(btn_layout)

        mapping_group = QtWidgets.QGroupBox("Active-tab mapping")
        mapping_layout = QtWidgets.QVBoxLayout(mapping_group)
        self._mapping_status = QtWidgets.QLabel("Select a dataset")
        self._mapping_status.setWordWrap(True)
        mapping_layout.addWidget(self._mapping_status)
        mapping_actions = QtWidgets.QHBoxLayout()
        self._create_set_btn = QtWidgets.QPushButton("Create Set From t=0")
        self._create_set_btn.clicked.connect(
            lambda: self._emit_mapping_action("create_set_from_t0")
        )
        self._map_existing_btn = QtWidgets.QPushButton("Map Existing")
        self._map_existing_btn.clicked.connect(self._map_existing)
        self._leave_unmapped_btn = QtWidgets.QPushButton("Leave Unmapped")
        self._leave_unmapped_btn.clicked.connect(
            lambda: self._emit_mapping_action("leave_unmapped")
        )
        mapping_actions.addWidget(self._create_set_btn)
        mapping_actions.addWidget(self._map_existing_btn)
        mapping_actions.addWidget(self._leave_unmapped_btn)
        mapping_layout.addLayout(mapping_actions)
        self._observation_mapping_btn = QtWidgets.QPushButton("Observation Mapping…")
        self._observation_mapping_btn.clicked.connect(self._request_observation_mapping)
        mapping_layout.addWidget(self._observation_mapping_btn)
        self._fit_include = QtWidgets.QCheckBox("Include mapped dataset in Fit")
        self._fit_include.toggled.connect(self._on_fit_include_toggled)
        mapping_layout.addWidget(self._fit_include)
        layout.addWidget(mapping_group)

        # Preview area
        layout.addWidget(QtWidgets.QLabel("Preview:"))
        self._preview_label = QtWidgets.QLabel("No dataset selected")
        self._preview_label.setContentsMargins(10, 10, 10, 10)
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setWordWrap(True)
        layout.addWidget(self._preview_label)
        self._preview_label.hide()

        # Rendered global-catalog records for list/preview only.
        self._snapshot = DataManagerSnapshot((), None, (), ())
        self._visible_records_by_id: Dict[str, RawDatasetRecord] = {}

        self._progress_dialog: Optional[QtWidgets.QProgressDialog] = None
        self._initiating_import_request: DatasetImportRequest | None = None
        self._staging.busyChanged.connect(self._on_staging_busy_changed)
        self._staging.completionAvailable.connect(self._on_completion_available)
        self._staging.workerProgressed.connect(self._on_import_progress)
        self._staging.workerFailed.connect(self._on_import_failure)
        self._staging.workerCanceled.connect(self._on_import_cancellation)
        self._set_mapping_controls_enabled(False)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def clear_projection(self) -> None:
        """Clear only rendered Data state and revoke all Data actions."""
        self._snapshot = DataManagerSnapshot((), None, (), ())
        self._visible_records_by_id.clear()
        self._dataset_list.clear()
        self._preview_label.setText("No dataset selected")
        self._preview_label.hide()
        self._mapping_status.setText("Select a dataset")
        self._set_mapping_controls_enabled(False)
        self._rename_btn.setEnabled(False)
        self._remove_btn.setEnabled(False)
        self._export_original_btn.setEnabled(False)
        blocker = QtCore.QSignalBlocker(self._fit_include)
        try:
            self._fit_include.setChecked(False)
        finally:
            del blocker

    def trigger_load_dialog(self) -> None:
        """Programmatically invoke the load dialog (mirrors Load button)."""
        self._load_dataset()

    def render_snapshot(self, snapshot: DataManagerSnapshot) -> None:
        """Render one global-catalog/active-tab snapshot without retaining truth."""
        if not isinstance(snapshot, DataManagerSnapshot):
            raise TypeError("DataManagerPanel requires a DataManagerSnapshot.")
        selected_id = self._selected_dataset_id()
        self._snapshot = snapshot
        self._visible_records_by_id = {
            str(record.dataset_id): record for record in snapshot.datasets
        }
        self._dataset_list.clear()
        self._preview_label.setText("No dataset selected")
        self._preview_label.hide()
        selected_row = -1
        for row, record in enumerate(snapshot.datasets):
            compact = compact_dataset_label(record.display_name, max_chars=DATASET_LIST_LABEL_MAX_CHARS)
            item = QtWidgets.QListWidgetItem(compact.display)
            item.setToolTip(compact.full)
            item.setData(Qt.UserRole, str(record.dataset_id))
            item.setData(Qt.UserRole + 1, compact.full)
            self._dataset_list.addItem(item)
            if str(record.dataset_id) == selected_id:
                selected_row = row
        if selected_row >= 0:
            self._dataset_list.setCurrentRow(selected_row)
        self._load_btn.setEnabled(not self._staging.busy)
        self._update_mapping_area()

    def _load_dataset(self):
        """Load dataset(s) using per-file import configuration and background workers."""
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Load Dataset(s)",
            "",
            "Data Files (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx)",
        )

        if not filenames:
            return
        if len(filenames) > _MAX_SELECTED_SOURCES:
            QtWidgets.QMessageBox.critical(
                self,
                "Unsupported Dataset Selection",
                "Select no more than 16 dataset sources at once.",
            )
            return
        try:
            selected_size = sum(os.stat(filepath).st_size for filepath in filenames)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Unsupported Dataset Selection",
                f"Cannot inspect the selected dataset sources:\n\n{exc}",
            )
            return
        if selected_size > _MAX_SELECTED_SOURCE_BYTES:
            QtWidgets.QMessageBox.critical(
                self,
                "Unsupported Dataset Selection",
                "The selected dataset sources exceed the 128 MiB total limit.",
            )
            return
        try:
            sources = tuple(
                capture_import_source(filepath)
                for filepath in filenames
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Unsupported Dataset Source",
                str(exc),
            )
            return
        if sum(len(source.exact_source_bytes) for source in sources) > _MAX_SELECTED_SOURCE_BYTES:
            QtWidgets.QMessageBox.critical(
                self,
                "Unsupported Dataset Selection",
                "The selected dataset sources exceed the 128 MiB total limit.",
            )
            return

        configs: List[ImportConfig] = []
        index = 0
        while index < len(sources):
            source = sources[index]
            remaining = len(sources) - index - 1
            result = self._collect_import_config(source, remaining)
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
                for remaining_idx in range(index + 1, len(sources)):
                    remaining_source = sources[remaining_idx]
                    try:
                        remaining_config = self._build_remaining_file_config(
                            remaining_source,
                            source_intent,
                            source_sheet_names=source_sheet_names,
                        )
                    except (ValueError, OSError, UnicodeDecodeError) as exc:
                        QtWidgets.QMessageBox.critical(
                            self,
                            "Import Error",
                            f"Cannot apply settings to "
                            f"'{remaining_source.original_filename}':\n\n{exc}",
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

        try:
            import_request: DatasetImportRequest = self._staging.begin(
                expected_units=expected_count,
                origin_tab_id=self._origin_tab_id,
            )
        except DatasetImportAdmissionError as exc:
            QtWidgets.QMessageBox.warning(self, "Dataset Import", str(exc))
            return
        self._initiating_import_request = import_request
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
            worker: CSVLoaderWorker | ExcelLoaderWorker | None = None
            request: DatasetImportWorkerRequest | None = None
            try:
                if config.source.file_type == "excel":
                    worker = ExcelLoaderWorker(
                        config.source,
                        list(config.plans),
                    )
                else:
                    worker = CSVLoaderWorker(
                        config.source,
                        config.plans[0],
                    )
                request = self._staging.register_worker(
                    import_request,
                    worker,
                    source=config.source,
                    admitted_plans=tuple(config.plans),
                )
                worker.start()
            except Exception as exc:
                self._staging.cancel_active(outcome="worker_start_failed")
                if request is not None and worker is not None:
                    try:
                        running = bool(worker.isRunning())
                    except RuntimeError:
                        running = False
                    if not running:
                        self._staging.release_unstarted_worker(
                            request,
                            worker=worker,
                        )
                elif worker is not None:
                    with suppress(RuntimeError):
                        worker.deleteLater()
                QtWidgets.QMessageBox.critical(
                    self,
                    "Dataset Import",
                    f"Failed to start dataset import:\n\n{type(exc).__name__}: {exc}",
                )
                return

    def _expected_dataset_count_for_config(self, config: ImportConfig) -> int:
        return len(config.plans)

    def _collect_import_config(
        self,
        source: ImportSourceSnapshot,
        remaining_count: int,
    ) -> ImportDialogResult:
        try:
            dialog = ImportConfigDialog(
                source,
                remaining_count=remaining_count,
                parent=self,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to open import configuration for "
                f"'{source.original_filename}':\n\n{type(exc).__name__}: {exc}",
            )
            return ImportDialogResult(config=None, action="skip")
        dialog.exec()
        return dialog.get_result()

    def _build_remaining_file_config(
        self,
        source: ImportSourceSnapshot,
        source_intent: SheetImportIntent,
        *,
        source_sheet_names: Tuple[str, ...] = (),
    ) -> ImportConfig:
        """Build an ImportConfig for a remaining file using the source intent.

        Raises ValueError if the source intent is incompatible with the file.
        """
        file_type = source.file_type

        per_sheet_intents: Dict[Optional[str], SheetImportIntent] = {}
        per_sheet_detections: Dict[Optional[str], UnitDetection] = {}
        per_sheet_columns: Dict[Optional[str], List[PhysicalColumn]] = {}
        target_sheet_names: Tuple[str, ...] = ()

        # Scope detection to source-selected columns only
        selected_keys = set(source_intent.species_columns)
        if source_intent.time_column:
            selected_keys.add(source_intent.time_column)

        if file_type == "csv":
            with io.TextIOWrapper(
                io.BytesIO(source.exact_source_bytes),
                encoding="utf-8-sig",
                newline="",
            ) as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    raise ValueError("CSV file is empty")
                columns = list(identify_physical_columns(header))
                first_row_raw = next(reader, None)
            if first_row_raw is not None:
                row_values = [c.strip() if c else "" for c in first_row_raw]
                row_mapping = {
                    column: (
                        row_values[column.source_index]
                        if column.source_index < len(row_values)
                        else ""
                    )
                    for column in columns
                }
                det = detect_units_from_physical_row(
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
            sheets = list_sheets_from_bytes(
                source.exact_source_bytes,
                source_label=source.original_filename,
            )
            if not sheets:
                raise ValueError("Excel file has no sheets")
            # Filter sheets by source's checked set
            if source_sheet_names:
                source_set = set(source_sheet_names)
                sheets = [s for s in sheets if s in source_set]
                if not sheets:
                    raise ValueError("No matching sheets found")
            for sheet_name in sheets:
                with closing(
                    read_excel_sheet_rows_from_bytes(
                        source.exact_source_bytes,
                        sheet_name,
                        source_label=source.original_filename,
                    )
                ) as rows:
                    first_row = next(iter(rows), None)
                if first_row is None:
                    per_sheet_columns[sheet_name] = []
                    per_sheet_detections[sheet_name] = UnitDetection.empty()
                else:
                    per_sheet_columns[sheet_name] = list(first_row.keys())
                    per_sheet_detections[sheet_name] = detect_units_from_physical_row(
                        dict(first_row), list(selected_keys)
                    )
                per_sheet_intents[sheet_name] = rebuild_intent_for_target(
                    source_intent, per_sheet_detections[sheet_name],
                )
            target_sheet_names = tuple(sheets)

        plans = resolve_import_plans(
            source,
            per_sheet_intents, per_sheet_detections, per_sheet_columns,
        )

        file_intent = UserImportIntent(
            sheet_names=target_sheet_names,
            apply_to_remaining=False,
        )
        return ImportConfig(
            source=source,
            file_intent=file_intent,
            plans=tuple(plans),
        )

    def _on_import_progress(self, event: DatasetImportProgress) -> None:
        """Render one staging-owned typed worker progress event."""
        if not isinstance(event, DatasetImportProgress):
            raise TypeError("Dataset import progress event has the wrong type.")
        if not self._owns_worker_request(event.request):
            return
        if self._progress_dialog:
            self._progress_dialog.setValue(min(event.percent, 99))

    def _on_import_failure(self, event: DatasetImportFailure) -> None:
        """Present one staging-owned typed worker failure."""
        if not isinstance(event, DatasetImportFailure):
            raise TypeError("Dataset import failure event has the wrong type.")
        if not self._owns_worker_request(event.request):
            return
        QtWidgets.QMessageBox.critical(
            self,
            "Load Error",
            f"Failed to load dataset:\n\n{event.message}",
        )

    def _on_import_cancellation(
        self,
        event: DatasetImportCancellation,
    ) -> None:
        """Observe one staging-owned typed worker cancellation."""
        if not isinstance(event, DatasetImportCancellation):
            raise TypeError("Dataset import cancellation event has the wrong type.")
        if not self._owns_worker_request(event.request):
            return
        logger.info("Dataset import canceled: %s", event.source_label)

    def _on_load_canceled(self):
        """Handle cancel request during dataset loading."""
        logger.info("Dataset import cancellation requested by user")
        self._staging.cancel_active(outcome="user_canceled")

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

    def _remove_dataset(self):
        """Emit one global-delete intent; confirmation belongs to integration."""
        dataset_id = self._selected_dataset_id()
        if not dataset_id:
            return
        self.datasetDeleteRequested.emit(DatasetDeleteIntent(dataset_id))

    def _export_original(self) -> None:
        """Request the integration-owned destination dialog for exact bytes."""
        dataset_id = self._selected_dataset_id()
        if not dataset_id:
            return
        self.datasetExportOriginalRequested.emit(
            DatasetExportOriginalIntent(dataset_id)
        )

    def _rename_dataset(self) -> None:
        dataset_id = self._selected_dataset_id()
        record = self._visible_records_by_id.get(dataset_id)
        if record is None:
            return
        value, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Rename Dataset",
            "Dataset name:",
            text=str(record.display_name),
        )
        if accepted:
            self.datasetRenameRequested.emit(
                DatasetRenameIntent(dataset_id=dataset_id, display_name=str(value))
            )

    def _on_dataset_selected(self, current, previous):
        """Update preview when dataset is selected."""
        if not current:
            self._preview_label.setText("No dataset selected")
            self._preview_label.hide()
            self._update_mapping_area()
            return

        dataset_id = str(current.data(Qt.UserRole) or "")
        record = self._visible_records_by_id.get(dataset_id)
        if record is None:
            self._preview_label.hide()
            self._update_mapping_area()
            return

        name = str(record.display_name)
        self._preview_label.show()
        unique_times = {
            time_value
            for series in record.observations
            for time_value in series.t
        }
        observation_labels = tuple(
            series.column_name for series in record.observations
        )

        thawed_metadata = thaw_json_value(record.source_metadata)
        metadata = dict(thawed_metadata) if isinstance(thawed_metadata, dict) else {}
        mapping_source = metadata.get("mapping_source", "normalized")
        preview = f"<b>{name}</b><br>"
        preview += (
            f"Original source: {record.original_filename} "
            f"({record.original_format})<br>"
        )
        preview += f"{len(unique_times)} time points<br>"
        preview += f"Time column: {metadata.get('time_column', 'unknown')} ({mapping_source})<br>"
        preview += (
            f"{len(observation_labels)} species: "
            f"{', '.join(observation_labels[:5])}"
        )
        if len(observation_labels) > 5:
            preview += f", ... ({len(observation_labels)-5} more)"

        self._preview_label.setText(preview)
        self._update_mapping_area()

    def _selected_dataset_id(self) -> str:
        current = self._dataset_list.currentItem()
        return "" if current is None else str(current.data(Qt.UserRole) or "")

    def _active_link(self, dataset_id: str) -> ActiveDatasetLinkSnapshot | None:
        return self._snapshot.link_by_dataset_id().get(str(dataset_id))

    def _set_mapping_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._create_set_btn,
            self._map_existing_btn,
            self._leave_unmapped_btn,
            self._observation_mapping_btn,
            self._fit_include,
        ):
            widget.setEnabled(bool(enabled))

    def _update_mapping_area(self) -> None:
        dataset_id = self._selected_dataset_id()
        record = self._visible_records_by_id.get(dataset_id)
        enabled = record is not None and self._snapshot.active_tab_id is not None
        self._rename_btn.setEnabled(record is not None)
        self._remove_btn.setEnabled(record is not None)
        self._export_original_btn.setEnabled(record is not None)
        self._set_mapping_controls_enabled(enabled)
        link = self._active_link(dataset_id)
        self._leave_unmapped_btn.setEnabled(
            bool(enabled and (link is None or link.set_id is None))
        )
        self._fit_include.blockSignals(True)
        try:
            self._fit_include.setChecked(bool(link and link.include_in_fit))
            self._fit_include.setEnabled(bool(enabled and link and link.set_id))
        finally:
            self._fit_include.blockSignals(False)
        if record is None:
            self._mapping_status.setText("Select a dataset")
        elif self._snapshot.active_tab_id is None:
            self._mapping_status.setText("No active mechanism tab; raw dataset remains available.")
        elif link is None or link.set_id is None:
            self._mapping_status.setText("Unmapped in the active mechanism tab")
        else:
            mapped_name = link.set_name or link.set_id
            mapped_columns = len(link.observation_mapping)
            self._mapping_status.setText(
                f"Mapped to {mapped_name}; {mapped_columns} confirmed observation mapping(s)"
            )

    def _emit_mapping_action(
        self,
        action: str,
        *,
        set_id: str | None = None,
        occupied_choice: str | None = None,
    ) -> None:
        dataset_id = self._selected_dataset_id()
        if not dataset_id:
            return
        self.datasetMappingRequested.emit(
            DatasetMappingIntent(
                dataset_id=dataset_id,
                action=str(action),
                set_id=set_id,
                occupied_choice=occupied_choice,
            )
        )

    def _map_existing(self) -> None:
        if not self._snapshot.available_sets:
            QtWidgets.QMessageBox.information(
                self,
                "Map Existing",
                "The active mechanism tab has no available initial-condition set.",
            )
            return
        labels = [name for _set_id, name in self._snapshot.available_sets]
        label, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Map Existing",
            "Initial-condition set:",
            labels,
            editable=False,
        )
        if not accepted:
            return
        set_id = next(
            set_id
            for set_id, name in self._snapshot.available_sets
            if name == str(label)
        )
        selected_dataset_id = self._selected_dataset_id()
        occupant = next(
            (
                link.dataset_id
                for link in self._snapshot.active_links
                if link.set_id == set_id and link.dataset_id != selected_dataset_id
            ),
            None,
        )
        occupied_choice = None
        if occupant is not None:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Icon.Question)
            box.setWindowTitle("Set Already Mapped")
            box.setText(f"Set '{label}' is already mapped in the active tab.")
            override_button = box.addButton("Override", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            copy_button = box.addButton("Create Copy", QtWidgets.QMessageBox.ButtonRole.ActionRole)
            cancel_button = box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(cancel_button)
            box.exec()
            clicked = box.clickedButton()
            if clicked is override_button:
                occupied_choice = "override"
            elif clicked is copy_button:
                occupied_choice = "create_copy"
            else:
                return
        self._emit_mapping_action(
            "map_existing",
            set_id=set_id,
            occupied_choice=occupied_choice,
        )

    def _request_observation_mapping(self) -> None:
        dataset_id = self._selected_dataset_id()
        if not dataset_id:
            return
        self.datasetObservationMappingRequested.emit(
            DatasetObservationMappingIntent(dataset_id=dataset_id)
        )

    def _on_fit_include_toggled(self, checked: bool) -> None:
        dataset_id = self._selected_dataset_id()
        link = self._active_link(dataset_id)
        if (
            not dataset_id
            or link is None
            or link.set_id is None
        ):
            return
        self.datasetFitInclusionRequested.emit(
            DatasetFitInclusionIntent(dataset_id=dataset_id, include=bool(checked))
        )

    def _on_completion_available(self, import_id: int) -> None:
        request = self._initiating_import_request
        if request is None or request.import_id != int(import_id):
            return
        self._finalize_progress_dialog()
        self._initiating_import_request = None

    def _on_staging_busy_changed(self, busy: bool) -> None:
        self._load_btn.setEnabled(not bool(busy))
        if not busy:
            self._finalize_progress_dialog()
            self._initiating_import_request = None

    def _owns_worker_request(
        self,
        request: DatasetImportWorkerRequest,
    ) -> bool:
        initiating = self._initiating_import_request
        return bool(
            initiating is not None
            and request.import_request is initiating
        )
