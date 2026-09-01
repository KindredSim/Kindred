"""Direct owner for dataset-import workers and uncommitted normalized output."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
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
    validate_imported_source,
    validate_raw_dataset_record,
)
from kindred.gui.controllers.dataset_errors import (
    DatasetImportAdmissionError,
    DatasetImportStateError,
)
from kindred.core.project_document import (
    new_dataset_id,
)
from kindred.gui.import_workflow import (
    ImportSourceSnapshot,
    ResolvedSheetPlan,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetImportCompletion",
    "DatasetImportRequest",
    "DatasetImportStaging",
    "DatasetImportWorkerAdmission",
    "RawDatasetBatchProposal",
    "build_raw_record_batch_proposal",
]


@dataclass(frozen=True)
class DatasetImportUnit:
    source: ImportSourceSnapshot
    display_name: str
    payload: DatasetObservationPayload

    def __post_init__(self) -> None:
        if not isinstance(self.source, ImportSourceSnapshot):
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


@dataclass(frozen=True, slots=True)
class DatasetImportRequest:
    origin_tab_id: str
    expected_units: int


@dataclass(frozen=True, slots=True)
class DatasetImportWorkerAdmission:
    """One immutable source/plan member of an admitted ordered import roster."""

    source: ImportSourceSnapshot
    admitted_plans: tuple[ResolvedSheetPlan, ...]


@dataclass(frozen=True, slots=True)
class _DatasetImportSuccess:
    unit: DatasetImportUnit


_UNSUCCESSFUL_OUTCOME = object()
_CANCELED_OUTCOME = _UNSUCCESSFUL_OUTCOME


class DatasetImportWorker(QtCore.QThread):
    """Exact signal contract for workers retained by DatasetImportStaging."""

    loaded = QtCore.Signal(str, object)
    cancelled = QtCore.Signal(str)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int)

    def __init__(self, source: ImportSourceSnapshot) -> None:
        super().__init__()
        self._source = source

    @property
    def source(self) -> ImportSourceSnapshot:
        return self._source


class _CSVLoaderWorker(DatasetImportWorker):
    """Background worker for one admitted CSV source."""

    def __init__(
        self,
        source: ImportSourceSnapshot,
        plan: ResolvedSheetPlan,
    ) -> None:
        super().__init__(source)
        self._plan = plan

    def _load_payload(self) -> DatasetObservationPayload:
        columns, rows = read_csv_rows_from_bytes(
            self.source.exact_source_bytes,
            source_label=self.source.original_filename,
        )
        with closing(rows):
            row_iter = iter(rows)
            first_row = next(row_iter, None)
            if first_row is None:
                raise ValueError("CSV file is empty")
            rows_to_parse = row_iter if self._plan.skip_unit_row else itertools.chain((first_row,), row_iter)
            payload = parse_physical_rows(
                rows_to_parse,
                columns=columns,
                time_column=self._plan.time_column,
                species_columns=self._plan.species_columns,
                interruption_checker=self.isInterruptionRequested,
                source_label=self.source.original_filename,
                source_row_start=3 if self._plan.skip_unit_row else 2,
                time_factor=self._plan.time_factor,
                observation_factors=self._plan.conc_factors,
                metadata_extras={
                    "original_time_unit": self._plan.original_time_unit,
                    "original_concentration_units": dict(
                        self._plan.original_conc_units
                    ),
                    "skip_unit_row": self._plan.skip_unit_row,
                },
            )
        return payload

    def run(self) -> None:
        dataset_name = self.source.original_filename
        try:
            self.progress.emit(10)
            if self.isInterruptionRequested():
                self.cancelled.emit(dataset_name)
                return
            self.progress.emit(30)
            payload = self._load_payload()
            self.progress.emit(80)
            if self.isInterruptionRequested():
                self.cancelled.emit(dataset_name)
                return
            self.progress.emit(100)
            self.loaded.emit(dataset_name, payload)
        except CsvImportInterrupted:
            self.cancelled.emit(dataset_name)
        except Exception as exc:
            logger.exception("CSV import failed: %s", dataset_name)
            self.error.emit(f"{type(exc).__name__}: {exc}")


class _ExcelLoaderWorker(DatasetImportWorker):
    """Background worker for one admitted XLSX source."""

    def __init__(
        self,
        source: ImportSourceSnapshot,
        plans: tuple[ResolvedSheetPlan, ...],
    ) -> None:
        super().__init__(source)
        self._plans = plans

    def _load_sheet_payload(
        self,
        plan: ResolvedSheetPlan,
        workbooks: ExcelWorkbookPair,
    ) -> tuple[str, DatasetObservationPayload]:
        sheet_name = plan.sheet_name
        rows = workbooks.read_sheet_rows(sheet_name)
        with closing(rows):
            row_iter = iter(rows)
            first_row = next(row_iter, None)
            if first_row is None:
                raise ValueError(f"Sheet {sheet_name!r} is empty.")
            columns = tuple(first_row)
            rows_to_parse = (
                row_iter
                if plan.skip_unit_row
                else itertools.chain((first_row,), row_iter)
            )
            payload = parse_physical_rows(
                rows_to_parse,
                columns=columns,
                time_column=plan.time_column,
                species_columns=plan.species_columns,
                interruption_checker=self.isInterruptionRequested,
                source_label=self.source.original_filename,
                sheet_name=sheet_name,
                source_row_start=3 if plan.skip_unit_row else 2,
                time_factor=plan.time_factor,
                observation_factors=plan.conc_factors,
                metadata_extras={
                    "original_time_unit": plan.original_time_unit,
                    "original_concentration_units": dict(plan.original_conc_units),
                    "skip_unit_row": plan.skip_unit_row,
                },
            )
        return f"{self.source.original_filename}::{sheet_name}", payload

    def run(self) -> None:
        dataset_name = self.source.original_filename
        try:
            self.progress.emit(0)
            with ExcelWorkbookPair(
                self.source.exact_source_bytes,
                source_label=self.source.original_filename,
            ) as workbooks:
                for index, plan in enumerate(self._plans, start=1):
                    if self.isInterruptionRequested():
                        self.cancelled.emit(dataset_name)
                        return
                    try:
                        loaded_name, payload = self._load_sheet_payload(
                            plan,
                            workbooks,
                        )
                    except CsvImportInterrupted:
                        self.cancelled.emit(dataset_name)
                        return
                    except Exception as exc:
                        logger.exception(
                            "Excel import failed: %s sheet %s",
                            dataset_name,
                            plan.sheet_name,
                        )
                        self.error.emit(
                            f"Sheet {plan.sheet_name!r}: {type(exc).__name__}: {exc}"
                        )
                    else:
                        self.loaded.emit(loaded_name, payload)
                    self.progress.emit(int((index / len(self._plans)) * 100))
                    if self.isInterruptionRequested():
                        self.cancelled.emit(dataset_name)
                        return
        except Exception as exc:
            logger.exception("Excel import failed: %s", dataset_name)
            self.error.emit(f"{type(exc).__name__}: {exc}")

def build_raw_record_batch_proposal(
    completion: DatasetImportCompletion,
    *,
    existing_records: tuple[RawDatasetRecord, ...],
) -> RawDatasetBatchProposal:
    """Build the complete immutable batch for one ProjectDocument admission.

    This is identity/name allocation only. ``ProjectDocument.add_datasets``
    validates the whole returned tuple in one all-or-none project proposal.
    Optional active-tab mapping remains a later transaction over committed raw
    records.
    """
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
        source = ImportedSource(
            source_id=unit.source.source_id,
            original_filename=unit.source.original_filename,
            original_format=unit.source.archive_format,
            exact_source_bytes=unit.source.exact_source_bytes,
        )
        validate_imported_source(source)
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


@dataclass
class _WorkerRecord:
    """Private admitted worker state owned by one active import."""

    source: ImportSourceSnapshot
    admitted_plans: tuple[ResolvedSheetPlan, ...]
    admitted_ordinals: tuple[int, ...]
    worker: DatasetImportWorker | None = None
    progress: int = 0
    accounted_units: int = 0


@dataclass
class _ActiveImport:
    request: DatasetImportRequest
    workers: list[_WorkerRecord] = field(default_factory=list)
    outcomes: list[_DatasetImportSuccess | object | None] = field(default_factory=list)
    completed_units: int = 0
    terminal_outcome: str | None = None
    launch_started: bool = False
    aggregate_progress: int = 0


class DatasetImportStaging(QtCore.QObject):
    """Own live QThreads until finish and exactly one normalized completion.

    The object admits one complete raw-record batch before a GUI-thread
    integration owner commits it through ProjectSession and only then offers
    optional active-tab mapping.
    """

    completionAvailable = QtCore.Signal(object)
    busyChanged = QtCore.Signal(bool)
    application_close_settlement_changed = QtCore.Signal()
    workerProgressed = QtCore.Signal(int)
    workerFailed = QtCore.Signal(str)
    workerCanceled = QtCore.Signal(str)

    def __init__(
        self,
        *,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._active: _ActiveImport | None = None
        self._application_close_settled = True
        self._application_close_fenced = False

    @property
    def busy(self) -> bool:
        return self._active is not None

    def begin(
        self,
        *,
        origin_tab_id: str,
        roster: tuple[DatasetImportWorkerAdmission, ...],
    ) -> DatasetImportRequest:
        """Admit one complete immutable source/plan roster in user order."""
        if self._application_close_fenced:
            raise DatasetImportAdmissionError("Dataset import admission is fenced.")
        if self.busy:
            raise DatasetImportAdmissionError("A dataset import is already active or awaiting commit.")
        expected = sum(len(item.admitted_plans) for item in roster)
        if expected <= 0:
            raise DatasetImportAdmissionError("Dataset import requires at least one result unit.")
        request = DatasetImportRequest(
            origin_tab_id=origin_tab_id,
            expected_units=expected,
        )
        worker_records: list[_WorkerRecord] = []
        next_ordinal = 0
        for admission in roster:
            unit_count = len(admission.admitted_plans)
            ordinals = tuple(range(next_ordinal, next_ordinal + unit_count))
            next_ordinal += unit_count
            worker_records.append(
                _WorkerRecord(
                    source=admission.source,
                    admitted_plans=admission.admitted_plans,
                    admitted_ordinals=ordinals,
                )
            )
        self._active = _ActiveImport(
            request=request,
            workers=worker_records,
            outcomes=[None] * expected,
        )
        self._application_close_settled = False
        self.busyChanged.emit(True)
        return request

    def start_import(
        self,
        import_request: DatasetImportRequest,
    ) -> None:
        """Acquire, connect, install, and start every admitted worker."""

        active = self._require_active(import_request)
        if self._application_close_fenced or active.terminal_outcome is not None:
            raise DatasetImportAdmissionError(
                "Dataset import start is fenced."
            )
        if active.launch_started:
            raise DatasetImportStateError("Dataset import roster already started.")
        active.launch_started = True
        for record in tuple(active.workers):
            if active.terminal_outcome is not None:
                break
            worker: DatasetImportWorker | None = None
            try:
                worker = self._acquire_worker(record)
                self._install_worker(active, record, worker)
                worker.start()
            except Exception as exc:
                self.cancel_active(outcome="worker_start_failed")
                if worker is not None:
                    try:
                        worker.requestInterruption()
                    except RuntimeError:
                        pass
                    if self._worker_is_verified_unstarted(worker):
                        owned = self._lookup_worker_callback(
                            import_request,
                            worker,
                        )
                        if owned is not None:
                            self._release_worker(*owned)
                        else:
                            try:
                                worker.deleteLater()
                            except RuntimeError:
                                pass
                raise DatasetImportAdmissionError(
                    "Failed to start dataset import: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    @staticmethod
    def _acquire_worker(
        record: _WorkerRecord,
    ) -> DatasetImportWorker:
        if record.source.file_type == "excel":
            return _ExcelLoaderWorker(
                record.source,
                record.admitted_plans,
            )
        return _CSVLoaderWorker(
            record.source,
            record.admitted_plans[0],
        )

    def _install_worker(
        self,
        active: _ActiveImport,
        record: _WorkerRecord,
        worker: DatasetImportWorker,
    ) -> _WorkerRecord:
        if any(
            registered.worker is worker
            for registered in active.workers
        ):
            raise DatasetImportStateError(
                "Dataset import worker is already registered."
            )
        if not any(admitted is record for admitted in active.workers):
            raise DatasetImportStateError(
                "Dataset import worker record is not admitted."
            )
        if record.worker is not None:
            raise DatasetImportStateError(
                "Dataset import worker record is already installed."
            )
        record.worker = worker
        worker.loaded.connect(
            lambda display_name, payload, token=active.request, worker=worker: (
                self._worker_loaded(
                    token,
                    worker,
                    display_name,
                    payload,
                )
            )
        )
        worker.progress.connect(
            lambda percent, token=active.request, worker=worker: (
                self._worker_progressed(
                    token,
                    worker,
                    percent,
                )
            )
        )
        worker.cancelled.connect(
            lambda source_label, token=active.request, worker=worker: (
                self._worker_canceled(
                    token,
                    worker,
                    source_label,
                )
            )
        )
        worker.error.connect(
            lambda message, token=active.request, worker=worker: (
                self._worker_failed(
                    token,
                    worker,
                    message,
                )
            )
        )
        worker.finished.connect(
            lambda token=active.request, worker=worker: (
                self._worker_finished(
                    token,
                    worker,
                )
            )
        )
        return record

    @staticmethod
    def _worker_is_verified_unstarted(worker: DatasetImportWorker) -> bool:
        try:
            return bool(
                not worker.isRunning()
                and not worker.isFinished()
            )
        except RuntimeError:
            return False

    @property
    def application_close_settled(self) -> bool:
        """Return whether this owner has completed all admitted import work."""

        return self._application_close_settled

    def _publish_application_close_settlement(self) -> None:
        if self._active is not None:
            return
        if self._application_close_settled:
            return
        self._application_close_settled = True
        self.application_close_settlement_changed.emit()

    def _worker_loaded(
        self,
        import_request: DatasetImportRequest,
        worker: DatasetImportWorker,
        display_name: str,
        payload: DatasetObservationPayload,
    ) -> bool:
        owned = self._lookup_worker_callback(import_request, worker)
        if owned is None:
            return False
        active, record = owned
        if active.terminal_outcome is not None:
            return False
        try:
            plan = record.admitted_plans[record.accounted_units]
            expected_display_name = record.source.original_filename
            if plan.sheet_name is not None:
                expected_display_name = (
                    f"{expected_display_name}::{plan.sheet_name}"
                )
            if display_name != expected_display_name:
                raise ValueError(
                    "Dataset worker result label does not match its next "
                    "admitted typed plan."
                )
            unit = DatasetImportUnit(
                source=record.source,
                display_name=display_name,
                payload=payload,
            )
        except (IndexError, TypeError, ValueError) as exc:
            self._record_failure(
                active,
                record,
                f"{type(exc).__name__}: {exc}",
            )
            return False
        ordinal = record.admitted_ordinals[record.accounted_units]
        self._store_outcomes(
            active,
            record,
            (ordinal,),
            _DatasetImportSuccess(
                unit=unit,
            ),
        )
        return True

    def _worker_progressed(
        self,
        import_request: DatasetImportRequest,
        worker: DatasetImportWorker,
        percent: int,
    ) -> bool:
        owned = self._lookup_worker_callback(import_request, worker)
        if owned is None:
            return False
        active, record = owned
        if active.terminal_outcome is not None:
            return False
        progress = percent
        previous = record.progress
        if progress <= previous:
            return True
        record.progress = progress
        self._publish_aggregate_progress(active)
        return True

    def _publish_aggregate_progress(
        self,
        active: _ActiveImport,
        *,
        force_complete: bool = False,
    ) -> None:
        """Publish monotonic progress weighted by admitted dataset units."""
        if active.terminal_outcome is not None:
            return
        if not active.workers:
            return
        if force_complete:
            if active.completed_units != active.request.expected_units:
                return
            aggregate = 100
        else:
            weighted_progress = sum(
                record.progress * len(record.admitted_plans)
                for record in active.workers
            )
            total_units = active.request.expected_units
            aggregate = (weighted_progress + total_units // 2) // total_units
            # Worker progress may reach 100 before its result signal has been
            # validated and settled.  Reserve terminal 100 for the successful
            # completion path below.
            aggregate = min(aggregate, 99)
        if aggregate <= active.aggregate_progress:
            return
        active.aggregate_progress = aggregate
        self.workerProgressed.emit(aggregate)

    def _worker_failed(
        self,
        import_request: DatasetImportRequest,
        worker: DatasetImportWorker,
        message: str,
    ) -> bool:
        owned = self._lookup_worker_callback(import_request, worker)
        if owned is None:
            return False
        active, record = owned
        if active.terminal_outcome is not None:
            return False
        self._record_failure(active, record, message)
        return True

    def _worker_canceled(
        self,
        import_request: DatasetImportRequest,
        worker: DatasetImportWorker,
        source_label: str,
    ) -> bool:
        owned = self._lookup_worker_callback(import_request, worker)
        if owned is None:
            return False
        active, record = owned
        if active.terminal_outcome is None:
            ordinals = self._unsettled_ordinals(active, record)
            self._store_outcomes(
                active,
                record,
                ordinals,
                _CANCELED_OUTCOME,
            )
            self._latch_cancellation(active, outcome="worker_canceled")
        self.workerCanceled.emit(source_label)
        return True

    def _worker_finished(
        self,
        import_request: DatasetImportRequest,
        worker: DatasetImportWorker,
    ) -> None:
        """Release custody after the finished signal or a verified stopped wait."""

        owned = self._lookup_worker_callback(import_request, worker)
        if owned is None:
            return
        active, record = owned
        self._release_worker(active, record)

    def _release_worker(
        self,
        active: _ActiveImport,
        record: _WorkerRecord,
    ) -> None:
        worker = record.worker
        if worker is None:
            return
        missing = self._remaining_worker_units(record)
        failure: str | None = None
        if missing and active.terminal_outcome is None:
            ordinals = self._unsettled_ordinals(active, record)
            message = (
                "Dataset importer ended without returning "
                f"{missing} expected result{'s' if missing != 1 else ''}."
            )
            self._store_outcomes(
                active,
                record,
                ordinals,
                _UNSUCCESSFUL_OUTCOME,
            )
            failure = message
        if failure is not None:
            self._latch_failure(active)
        record.worker = None
        try:
            worker.deleteLater()
        except RuntimeError:
            pass
        if failure is not None:
            self.workerFailed.emit(failure)
        self._finish_if_ready()

    def cancel_active(self, *, outcome: str = "user_canceled") -> bool:
        active = self._active
        if active is None:
            return False
        if active.terminal_outcome is None:
            self._latch_cancellation(
                active,
                outcome=str(outcome or "user_canceled"),
            )
        self._finish_if_ready()
        return True

    def fence_application_close_admission(self) -> None:
        """Reject dataset-import admission for the active Close attempt."""

        self._application_close_fenced = True

    def cancel_for_application_close(self) -> None:
        """Suppress publication and request cancellation without waiting."""

        self.fence_application_close_admission()
        self.cancel_active(outcome="application_close")

    def _finish_if_ready(self) -> None:
        active = self._active
        if active is None or any(record.worker is not None for record in active.workers):
            return
        if active.terminal_outcome is not None:
            self._active = None
            self.busyChanged.emit(False)
            self._publish_application_close_settlement()
            return
        if active.completed_units < active.request.expected_units:
            return
        successes = tuple(
            outcome
            for outcome in active.outcomes
            if isinstance(outcome, _DatasetImportSuccess)
        )
        if len(successes) != active.request.expected_units:
            raise DatasetImportStateError(
                "A successful import must settle every ordinal with one unit."
            )
        completion = DatasetImportCompletion(
            origin_tab_id=active.request.origin_tab_id,
            units=tuple(outcome.unit for outcome in successes),
        )
        self._publish_aggregate_progress(active, force_complete=True)
        self._active = None
        self.busyChanged.emit(False)
        self.completionAvailable.emit(completion)
        self._publish_application_close_settlement()

    def _require_active(self, import_request: DatasetImportRequest) -> _ActiveImport:
        active = self._active
        if (
            active is None
            or active.request is not import_request
        ):
            raise DatasetImportStateError("Dataset import callback is stale.")
        return active

    def _lookup_worker_callback(
        self,
        import_request: DatasetImportRequest,
        worker: DatasetImportWorker,
    ) -> tuple[_ActiveImport, _WorkerRecord] | None:
        active = self._active
        if active is None or active.request is not import_request:
            return None
        for record in active.workers:
            if record.worker is worker:
                return active, record
        return None

    def _record_failure(
        self,
        active: _ActiveImport,
        record: _WorkerRecord,
        message: str,
    ) -> None:
        ordinals = self._unsettled_ordinals(active, record)
        self._store_outcomes(
            active,
            record,
            ordinals,
            _UNSUCCESSFUL_OUTCOME,
        )
        self._latch_failure(active)
        self.workerFailed.emit(message)

    def _latch_failure(self, active: _ActiveImport) -> None:
        if active.terminal_outcome is not None:
            return
        active.terminal_outcome = "worker_failed"
        self._settle_canceled_ordinals(active)
        for record in tuple(active.workers):
            worker = record.worker
            if worker is None:
                continue
            try:
                worker.requestInterruption()
            except RuntimeError:
                continue

    def _latch_cancellation(
        self,
        active: _ActiveImport,
        *,
        outcome: str,
    ) -> None:
        if active.terminal_outcome is not None:
            return
        active.terminal_outcome = str(outcome)
        self._settle_canceled_ordinals(active)
        for record in tuple(active.workers):
            worker = record.worker
            if worker is None:
                continue
            try:
                worker.requestInterruption()
            except RuntimeError:
                continue

    def _settle_canceled_ordinals(self, active: _ActiveImport) -> None:
        for record in active.workers:
            ordinals = self._unsettled_ordinals(active, record)
            if not ordinals:
                continue
            self._store_outcomes(
                active,
                record,
                ordinals,
                _CANCELED_OUTCOME,
            )

    @staticmethod
    def _unsettled_ordinals(
        active: _ActiveImport,
        record: _WorkerRecord,
    ) -> tuple[int, ...]:
        return tuple(
            ordinal
            for ordinal in record.admitted_ordinals
            if active.outcomes[ordinal] is None
        )

    @staticmethod
    def _store_outcomes(
        active: _ActiveImport,
        record: _WorkerRecord,
        ordinals: tuple[int, ...],
        outcome: _DatasetImportSuccess | object,
    ) -> None:
        if any(
            ordinal not in record.admitted_ordinals
            or active.outcomes[ordinal] is not None
            for ordinal in ordinals
        ) or len(ordinals) != len(set(ordinals)):
            raise DatasetImportStateError(
                "Dataset import ordinal was settled twice or by the wrong worker."
            )
        for ordinal in ordinals:
            active.outcomes[ordinal] = outcome
        record.accounted_units += len(ordinals)
        if record.accounted_units == len(record.admitted_plans):
            if isinstance(outcome, _DatasetImportSuccess):
                record.progress = 100
        active.completed_units += len(ordinals)

    @staticmethod
    def _remaining_worker_units(
        record: _WorkerRecord,
    ) -> int:
        return max(
            0,
            len(record.admitted_plans) - record.accounted_units,
        )
