"""Application-global sequential owner for dataset-import batches."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import itertools
import logging

from PySide6 import QtCore

from kindred.core.datasets.csv_import import (
    CsvImportInterrupted,
    parse_physical_rows,
    read_csv_rows_from_bytes,
)
from kindred.core.datasets.excel_import import ExcelWorkbookPair
from kindred.core.datasets.observation_payload import DatasetObservationPayload
from kindred.core.datasets.foundation import (
    ImportedSource,
    RawDatasetRecord,
    validate_raw_dataset_record,
)
from kindred.gui.controllers.dataset_errors import (
    DatasetImportAdmissionError,
)
from kindred.core.project_document import new_dataset_id
from kindred.gui.import_workflow import (
    ImportConfig,
    ResolvedSheetPlan,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetImportCompletion",
    "DatasetImportStaging",
    "DatasetImportUnit",
    "RawDatasetBatchProposal",
    "build_raw_record_batch_proposal",
]


@dataclass(frozen=True)
class DatasetImportUnit:
    source: ImportedSource
    display_name: str
    payload: DatasetObservationPayload

    def __post_init__(self) -> None:
        if not isinstance(self.source, ImportedSource):
            raise TypeError(
                "Imported datasets require their immutable source snapshot."
            )
        if not isinstance(self.display_name, str):
            raise TypeError("Imported dataset display name must be a string.")
        name = self.display_name.strip()
        if not name:
            raise ValueError("Imported datasets require a nonblank display name.")
        if not isinstance(self.payload, DatasetObservationPayload):
            raise TypeError(
                "Imported dataset payload must be a DatasetObservationPayload."
            )
        object.__setattr__(self, "display_name", name)


@dataclass(frozen=True, slots=True)
class DatasetImportCompletion:
    """One immutable, fully settled dataset-import result."""

    origin_tab_id: str
    units: tuple[DatasetImportUnit, ...]


@dataclass(frozen=True, slots=True)
class RawDatasetBatchProposal:
    """One complete immutable all-or-none project dataset proposal."""

    origin_tab_id: str
    sources: tuple[ImportedSource, ...]
    records: tuple[RawDatasetRecord, ...]

    def __post_init__(self) -> None:
        source_ids = tuple(source.source_id for source in self.sources)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Raw dataset batch source IDs must be unique.")
        referenced = {record.source_id for record in self.records}
        if referenced != set(source_ids):
            raise ValueError(
                "Raw dataset batch sources must exactly cover dataset source references."
            )


class _DatasetImportCancelled(Exception):
    """Internal cooperative cancellation marker."""


class _DatasetImportBatchWorker(QtCore.QThread):
    """Read one admitted batch serially in source and sheet order."""

    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()
    progressed = QtCore.Signal(int)

    def __init__(
        self,
        *,
        origin_tab_id: str,
        configs: tuple[ImportConfig, ...],
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        self._origin_tab_id = origin_tab_id
        self._configs = configs
        self._total_units = sum(len(config.plans) for config in configs)
        self._completed_units = 0

    def cancel(self) -> None:
        self.requestInterruption()

    def _check_interrupted(self) -> None:
        if self.isInterruptionRequested():
            raise _DatasetImportCancelled()

    @staticmethod
    def _parse_rows(
        rows,
        *,
        columns,
        plan: ResolvedSheetPlan,
        source: ImportedSource,
        sheet_name: str | None = None,
        first_row=None,
        interruption_checker,
    ) -> DatasetObservationPayload:
        row_iter = iter(rows)
        if first_row is None:
            first_row = next(row_iter, None)
        if first_row is None:
            label = (
                f"Sheet {sheet_name!r} in workbook {source.original_filename!r}"
                if sheet_name is not None
                else f"CSV file {source.original_filename!r}"
            )
            raise ValueError(f"{label} is empty.")
        rows_to_parse = (
            row_iter
            if plan.skip_unit_row
            else itertools.chain((first_row,), row_iter)
        )
        return parse_physical_rows(
            rows_to_parse,
            columns=columns,
            time_column=plan.time_column,
            species_columns=plan.species_columns,
            interruption_checker=interruption_checker,
            source_label=source.original_filename,
            sheet_name=sheet_name,
            source_row_start=3 if plan.skip_unit_row else 2,
            time_unit=plan.time_unit,
            observation_units=plan.observation_units,
            metadata_extras={
                "skip_unit_row": plan.skip_unit_row,
            },
        )

    def _load_csv(self, config: ImportConfig) -> DatasetImportUnit:
        source = config.source
        plan = config.plans[0]
        try:
            columns, rows = read_csv_rows_from_bytes(
                source.exact_source_bytes,
                source_label=source.original_filename,
            )
            with closing(rows):
                payload = self._parse_rows(
                    rows,
                    columns=columns,
                    plan=plan,
                    source=source,
                    interruption_checker=lambda: self.isInterruptionRequested(),
                )
        except (_DatasetImportCancelled, CsvImportInterrupted):
            raise
        except Exception as exc:
            raise ValueError(
                f"CSV file {source.original_filename!r}: {exc}"
            ) from exc
        return DatasetImportUnit(
            source=source,
            display_name=source.original_filename,
            payload=payload,
        )

    def _load_excel(self, config: ImportConfig) -> tuple[DatasetImportUnit, ...]:
        source = config.source
        units: list[DatasetImportUnit] = []
        # Keep one workbook pair open for all selected sheets in this source.
        try:
            with ExcelWorkbookPair(
                source.exact_source_bytes,
                source_label=source.original_filename,
            ) as workbooks:
                for plan in config.plans:
                    self._check_interrupted()
                    sheet_name = plan.sheet_name
                    if sheet_name is None:
                        raise ValueError("XLSX import plans require a sheet name.")
                    try:
                        rows = workbooks.read_sheet_rows(sheet_name)
                        with closing(rows):
                            row_iter = iter(rows)
                            first_row = next(row_iter, None)
                            if first_row is None:
                                raise ValueError("sheet is empty")
                            payload = self._parse_rows(
                                row_iter,
                                columns=tuple(first_row),
                                plan=plan,
                                source=source,
                                sheet_name=sheet_name,
                                first_row=first_row,
                                interruption_checker=lambda: self.isInterruptionRequested(),
                            )
                    except (_DatasetImportCancelled, CsvImportInterrupted):
                        raise
                    except Exception as exc:
                        raise ValueError(
                            f"Workbook {source.original_filename!r}, sheet "
                            f"{sheet_name!r}: {exc}"
                        ) from exc
                    units.append(
                        DatasetImportUnit(
                            source=source,
                            display_name=f"{source.original_filename}::{sheet_name}",
                            payload=payload,
                        )
                    )
                    self._completed_units += 1
                    self._publish_plan_progress()
                    self._check_interrupted()
        except (_DatasetImportCancelled, CsvImportInterrupted):
            raise
        except Exception as exc:
            if str(source.original_filename) in str(exc):
                raise
            raise ValueError(
                f"Workbook {source.original_filename!r}: {exc}"
            ) from exc
        return tuple(units)

    def _publish_plan_progress(self) -> None:
        """Publish completed-plan progress while reserving 100 for success."""
        if self._completed_units < self._total_units:
            self.progressed.emit(
                (self._completed_units * 100) // self._total_units
            )

    def run(self) -> None:
        units: list[DatasetImportUnit] = []
        try:
            self._check_interrupted()
            for config in self._configs:
                self._check_interrupted()
                source = config.source
                if source.original_format == "xlsx":
                    units.extend(self._load_excel(config))
                else:
                    units.append(self._load_csv(config))
                    self._completed_units += 1
                    self._publish_plan_progress()
                self._check_interrupted()
            self._check_interrupted()
            self.progressed.emit(100)
            completion = DatasetImportCompletion(
                origin_tab_id=self._origin_tab_id,
                units=tuple(units),
            )
            self.completed.emit(completion)
        except (_DatasetImportCancelled, CsvImportInterrupted):
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("Dataset import failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def build_raw_record_batch_proposal(
    completion: DatasetImportCompletion,
    *,
    existing_records: tuple[RawDatasetRecord, ...],
) -> RawDatasetBatchProposal:
    """Build the complete immutable batch for one ProjectDocument admission."""
    used_names = {record.display_name.casefold() for record in existing_records}
    candidates: list[RawDatasetRecord] = []
    sources_by_id: dict[str, ImportedSource] = {}
    for unit in completion.units:
        raw_name = unit.display_name.strip()
        stem, dot, suffix = raw_name.rpartition(".")
        base = stem if dot and stem else raw_name
        extension = f".{suffix}" if dot and stem else ""
        display_name = raw_name
        counter = 1
        while display_name.casefold() in used_names:
            display_name = f"{base}_{counter}{extension}"
            counter += 1
        used_names.add(display_name.casefold())
        source = unit.source
        previous_source = sources_by_id.get(source.source_id)
        if previous_source is not None and previous_source != source:
            raise ValueError(
                f"Import captured multiple distinct sources with source ID {source.source_id!r}."
            )
        sources_by_id[source.source_id] = source
        candidate = RawDatasetRecord(
            dataset_id=new_dataset_id(),
            display_name=display_name,
            source_id=source.source_id,
            source_metadata=unit.payload.source_metadata,
            observations=unit.payload.observations,
        )
        validate_raw_dataset_record(candidate)
        candidates.append(candidate)
    return RawDatasetBatchProposal(
        origin_tab_id=completion.origin_tab_id,
        sources=tuple(sources_by_id.values()),
        records=tuple(candidates),
    )


class DatasetImportStaging(QtCore.QObject):
    """Own one global batch worker until terminal settlement."""

    completionAvailable = QtCore.Signal(object)
    busyChanged = QtCore.Signal(bool)
    application_close_settlement_changed = QtCore.Signal()
    progressed = QtCore.Signal(int)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, *, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._active_worker: _DatasetImportBatchWorker | None = None
        self._application_close_fenced = False

    @property
    def busy(self) -> bool:
        return self._active_worker is not None

    @property
    def application_close_settled(self) -> bool:
        """Close settlement derives directly from worker custody."""
        return self._active_worker is None

    def start_import(
        self,
        *,
        origin_tab_id: str,
        configs: tuple[ImportConfig, ...],
    ) -> None:
        if self._application_close_fenced:
            raise DatasetImportAdmissionError("Dataset import admission is fenced.")
        if self._active_worker is not None:
            raise DatasetImportAdmissionError("A dataset import is already active.")
        if not configs:
            raise DatasetImportAdmissionError(
                "Dataset import requires at least one configuration."
            )
        worker = _DatasetImportBatchWorker(
            origin_tab_id=origin_tab_id,
            configs=configs,
            parent=self,
        )
        self._active_worker = worker
        worker.completed.connect(self.completionAvailable)
        worker.failed.connect(self.failed)
        worker.cancelled.connect(self.cancelled)
        worker.progressed.connect(self.progressed)
        worker.finished.connect(self._worker_finished)
        self.busyChanged.emit(True)
        try:
            worker.start()
        except Exception as exc:
            try:
                unstarted = not worker.isRunning() and not worker.isFinished()
            except RuntimeError:
                unstarted = False
            if unstarted:
                self._active_worker = None
                self.busyChanged.emit(False)
                try:
                    worker.deleteLater()
                except RuntimeError:
                    pass
                raise DatasetImportAdmissionError(
                    f"Failed to start dataset import: {type(exc).__name__}: {exc}"
                ) from exc
            raise

    def cancel_active(self) -> None:
        worker = self._active_worker
        if worker is not None:
            worker.cancel()

    def fence_application_close_admission(self) -> None:
        self._application_close_fenced = True

    def cancel_for_application_close(self) -> None:
        self._application_close_fenced = True
        self.cancel_active()

    @QtCore.Slot()
    def _worker_finished(self) -> None:
        worker = self.sender()
        if worker is not self._active_worker:
            return
        self._active_worker = None
        worker.deleteLater()
        self.busyChanged.emit(False)
        self.application_close_settlement_changed.emit()
