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
    "DatasetImportCancellation",
    "DatasetImportCompletion",
    "DatasetImportFailure",
    "DatasetImportProgress",
    "DatasetImportRequest",
    "DatasetImportStaging",
    "DatasetImportWorkerAdmission",
    "DatasetImportWorkerRequest",
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
    import_id: int
    origin_tab_id: str
    expected_units: int


@dataclass(frozen=True, slots=True)
class DatasetImportWorkerAdmission:
    """One immutable source/plan member of an admitted ordered import roster."""

    source: ImportSourceSnapshot
    admitted_plans: tuple[ResolvedSheetPlan, ...]


@dataclass(frozen=True, slots=True)
class DatasetImportWorkerRequest:
    import_request: DatasetImportRequest
    worker_id: int
    source: ImportSourceSnapshot
    expected_units: int
    admitted_plans: tuple[ResolvedSheetPlan, ...]
    admitted_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        plans = self.admitted_plans
        if len(plans) != self.expected_units:
            raise ValueError(
                "A dataset import worker must retain one typed plan per "
                "expected result."
            )
        ordinals = self.admitted_ordinals
        if (
            len(ordinals) != self.expected_units
            or tuple(range(ordinals[0], ordinals[0] + len(ordinals))) != ordinals
            or ordinals[-1] >= self.import_request.expected_units
        ):
            raise ValueError(
                "A dataset import worker requires one contiguous admitted "
                "ordinal per expected result."
            )

    @property
    def import_id(self) -> int:
        return self.import_request.import_id


@dataclass(frozen=True, slots=True)
class DatasetImportProgress:
    request: DatasetImportRequest
    percent: int


@dataclass(frozen=True, slots=True)
class DatasetImportFailure:
    request: DatasetImportWorkerRequest
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.message, str):
            raise TypeError("Dataset import failure message must be a string.")
        message = self.message.strip()
        if not message:
            raise ValueError("Dataset import failure message must be nonblank.")
        object.__setattr__(self, "message", message)


@dataclass(frozen=True, slots=True)
class DatasetImportCancellation:
    request: DatasetImportWorkerRequest
    source_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_label, str):
            raise TypeError(
                "Dataset import cancellation source label must be a string."
            )
        source_label = self.source_label.strip()
        if not source_label:
            raise ValueError(
                "Dataset import cancellation source label must be nonblank."
            )
        object.__setattr__(self, "source_label", source_label)


@dataclass(frozen=True, slots=True)
class _DatasetImportSuccess:
    unit: DatasetImportUnit


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
class _RegisteredWorker:
    request: DatasetImportWorkerRequest
    worker: DatasetImportWorker
    accounted_units: int = 0


@dataclass
class _ActiveImport:
    request: DatasetImportRequest
    worker_requests: tuple[DatasetImportWorkerRequest, ...]
    workers: dict[int, _RegisteredWorker] = field(default_factory=dict)
    outcomes: list[
        _DatasetImportSuccess
        | DatasetImportFailure
        | DatasetImportCancellation
        | None
    ] = field(default_factory=list)
    completed_units: int = 0
    terminal_outcome: str | None = None
    launch_started: bool = False
    latest_progress_by_worker: dict[int, int] = field(default_factory=dict)
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
    workerProgressed = QtCore.Signal(object)
    workerFailed = QtCore.Signal(object)
    workerCanceled = QtCore.Signal(object)

    def __init__(
        self,
        *,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._next_import_id = 1
        self._next_worker_id = 1
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
        import_id = self._next_import_id
        self._next_import_id += 1
        request = DatasetImportRequest(
            import_id=import_id,
            origin_tab_id=origin_tab_id,
            expected_units=expected,
        )
        worker_requests: list[DatasetImportWorkerRequest] = []
        next_ordinal = 0
        for admission in roster:
            worker_id = self._next_worker_id
            self._next_worker_id += 1
            unit_count = len(admission.admitted_plans)
            ordinals = tuple(range(next_ordinal, next_ordinal + unit_count))
            next_ordinal += unit_count
            worker_requests.append(
                DatasetImportWorkerRequest(
                    import_request=request,
                    worker_id=worker_id,
                    source=admission.source,
                    expected_units=unit_count,
                    admitted_plans=admission.admitted_plans,
                    admitted_ordinals=ordinals,
                )
            )
        worker_requests_tuple = tuple(worker_requests)
        self._active = _ActiveImport(
            request=request,
            worker_requests=worker_requests_tuple,
            outcomes=[None] * expected,
            latest_progress_by_worker={
                worker_request.worker_id: 0
                for worker_request in worker_requests_tuple
            },
        )
        self._application_close_settled = False
        self.busyChanged.emit(True)
        return request

    def start_import(
        self,
        import_request: DatasetImportRequest,
    ) -> None:
        """Acquire, connect, install, and start every admitted worker."""

        active = self._require_active(import_request.import_id)
        if active.request is not import_request:
            raise DatasetImportStateError("Dataset import request is stale.")
        if self._application_close_fenced or active.terminal_outcome is not None:
            raise DatasetImportAdmissionError(
                "Dataset import start is fenced."
            )
        if active.launch_started:
            raise DatasetImportStateError("Dataset import roster already started.")
        active.launch_started = True
        for request in active.worker_requests:
            if active.terminal_outcome is not None:
                break
            worker: DatasetImportWorker | None = None
            registration: _RegisteredWorker | None = None
            try:
                worker = self._acquire_worker(request)
                registration = self._install_worker(active, request, worker)
                worker.start()
            except Exception as exc:
                self.cancel_active(outcome="worker_start_failed")
                if registration is not None and worker is not None:
                    try:
                        worker.requestInterruption()
                    except RuntimeError:
                        pass
                    if self._worker_is_verified_unstarted(worker):
                        owned = self._lookup_worker_callback(request, worker)
                        if owned is not None:
                            self._release_worker(*owned)
                elif worker is not None:
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
        request: DatasetImportWorkerRequest,
    ) -> DatasetImportWorker:
        if request.source.file_type == "excel":
            return _ExcelLoaderWorker(
                request.source,
                request.admitted_plans,
            )
        return _CSVLoaderWorker(
            request.source,
            request.admitted_plans[0],
        )

    def _install_worker(
        self,
        active: _ActiveImport,
        request: DatasetImportWorkerRequest,
        worker: DatasetImportWorker,
    ) -> _RegisteredWorker:
        if any(
            registration.worker is worker
            for registration in active.workers.values()
        ):
            raise DatasetImportStateError(
                "Dataset import worker is already registered."
            )
        if request.worker_id in active.workers:
            raise DatasetImportStateError(
                "Dataset import worker request is already installed."
            )
        registration = _RegisteredWorker(
            request=request,
            worker=worker,
        )
        worker.loaded.connect(
            lambda display_name, payload, request=request, worker=worker: (
                self._worker_loaded(
                    request,
                    worker,
                    display_name,
                    payload,
                )
            )
        )
        worker.progress.connect(
            lambda percent, request=request, worker=worker: (
                self._worker_progressed(
                    request,
                    worker,
                    percent,
                )
            )
        )
        worker.cancelled.connect(
            lambda source_label, request=request, worker=worker: (
                self._worker_canceled(
                    request,
                    worker,
                    source_label,
                )
            )
        )
        worker.error.connect(
            lambda message, request=request, worker=worker: (
                self._worker_failed(
                    request,
                    worker,
                    message,
                )
            )
        )
        worker.finished.connect(
            lambda request=request, worker=worker: (
                self._worker_finished(
                    request,
                    worker,
                )
            )
        )
        active.workers[request.worker_id] = registration
        return registration

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
        request: DatasetImportWorkerRequest,
        worker: DatasetImportWorker,
        display_name: str,
        payload: DatasetObservationPayload,
    ) -> bool:
        owned = self._lookup_worker_callback(request, worker)
        if owned is None:
            return False
        active, registration = owned
        if active.terminal_outcome is not None:
            return False
        try:
            plan = request.admitted_plans[registration.accounted_units]
            expected_display_name = request.source.original_filename
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
                source=request.source,
                display_name=display_name,
                payload=payload,
            )
        except (IndexError, TypeError, ValueError) as exc:
            self._record_failure(
                active,
                request,
                f"{type(exc).__name__}: {exc}",
            )
            return False
        ordinal = request.admitted_ordinals[registration.accounted_units]
        self._store_outcomes(
            active,
            request,
            (ordinal,),
            _DatasetImportSuccess(
                unit=unit,
            ),
        )
        return True

    def _worker_progressed(
        self,
        request: DatasetImportWorkerRequest,
        worker: DatasetImportWorker,
        percent: int,
    ) -> bool:
        owned = self._lookup_worker_callback(request, worker)
        if owned is None:
            return False
        active = owned[0]
        if active.terminal_outcome is not None:
            return False
        progress = percent
        previous = active.latest_progress_by_worker[request.worker_id]
        if progress <= previous:
            return True
        active.latest_progress_by_worker[request.worker_id] = progress
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
        if not active.worker_requests:
            return
        if force_complete:
            if active.completed_units != active.request.expected_units:
                return
            aggregate = 100
        else:
            weighted_progress = sum(
                active.latest_progress_by_worker[worker_request.worker_id]
                * worker_request.expected_units
                for worker_request in active.worker_requests
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
        self.workerProgressed.emit(
            DatasetImportProgress(request=active.request, percent=aggregate)
        )

    def _worker_failed(
        self,
        request: DatasetImportWorkerRequest,
        worker: DatasetImportWorker,
        message: str,
    ) -> bool:
        owned = self._lookup_worker_callback(request, worker)
        if owned is None:
            return False
        active = owned[0]
        if active.terminal_outcome is not None:
            return False
        self._record_failure(active, request, message)
        return True

    def _worker_canceled(
        self,
        request: DatasetImportWorkerRequest,
        worker: DatasetImportWorker,
        source_label: str,
    ) -> bool:
        owned = self._lookup_worker_callback(request, worker)
        if owned is None:
            return False
        active = owned[0]
        if active.terminal_outcome is None:
            ordinals = self._unsettled_ordinals(active, request)
            event = DatasetImportCancellation(
                request=request,
                source_label=source_label,
            )
            self._store_outcomes(
                active,
                request,
                ordinals,
                event,
            )
            self._latch_cancellation(active, outcome="worker_canceled")
        else:
            event = DatasetImportCancellation(
                request=request,
                source_label=source_label,
            )
        self.workerCanceled.emit(event)
        return True

    def _worker_finished(
        self,
        request: DatasetImportWorkerRequest,
        worker: DatasetImportWorker,
    ) -> None:
        """Release custody after the finished signal or a verified stopped wait."""

        owned = self._lookup_worker_callback(request, worker)
        if owned is None:
            return
        active, registration = owned
        self._release_worker(active, registration)

    def _release_worker(
        self,
        active: _ActiveImport,
        registration: _RegisteredWorker,
    ) -> None:
        request = registration.request
        worker = registration.worker
        missing = self._remaining_worker_units(registration)
        failure: DatasetImportFailure | None = None
        if missing and active.terminal_outcome is None:
            ordinals = self._unsettled_ordinals(active, request)
            message = (
                "Dataset importer ended without returning "
                f"{missing} expected result{'s' if missing != 1 else ''}."
            )
            failure = DatasetImportFailure(
                request=request,
                message=message,
            )
            self._store_outcomes(
                active,
                request,
                ordinals,
                failure,
            )
        if failure is not None:
            self._latch_failure(active)
        active.workers.pop(request.worker_id)
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
        if active is None or active.workers:
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

    def _require_active(self, import_id: int) -> _ActiveImport:
        active = self._active
        if (
            active is None
            or active.request.import_id != import_id
        ):
            raise DatasetImportStateError("Dataset import callback is stale.")
        return active

    def _lookup_worker_callback(
        self,
        request: DatasetImportWorkerRequest,
        worker: DatasetImportWorker,
    ) -> tuple[_ActiveImport, _RegisteredWorker] | None:
        active = self._active
        if (
            active is None
            or active.request.import_id != request.import_id
        ):
            return None
        registration = active.workers.get(request.worker_id)
        if registration is None:
            return None
        if (
            registration.request is not request
            or registration.worker is not worker
        ):
            raise DatasetImportStateError(
                "Dataset import callback identity does not match its worker."
            )
        return active, registration

    def _record_failure(
        self,
        active: _ActiveImport,
        request: DatasetImportWorkerRequest,
        message: str,
    ) -> None:
        ordinals = self._unsettled_ordinals(active, request)
        event = DatasetImportFailure(
            request=request,
            message=message,
        )
        self._store_outcomes(
            active,
            request,
            ordinals,
            event,
        )
        self._latch_failure(active)
        self.workerFailed.emit(event)

    def _latch_failure(self, active: _ActiveImport) -> None:
        if active.terminal_outcome is not None:
            return
        active.terminal_outcome = "worker_failed"
        self._settle_canceled_ordinals(active)
        for registration in tuple(active.workers.values()):
            try:
                registration.worker.requestInterruption()
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
        for registration in tuple(active.workers.values()):
            try:
                registration.worker.requestInterruption()
            except RuntimeError:
                continue

    def _settle_canceled_ordinals(self, active: _ActiveImport) -> None:
        for request in active.worker_requests:
            ordinals = self._unsettled_ordinals(active, request)
            if not ordinals:
                continue
            event = DatasetImportCancellation(
                request=request,
                source_label=request.source.original_filename,
            )
            self._store_outcomes(
                active,
                request,
                ordinals,
                event,
            )

    @staticmethod
    def _unsettled_ordinals(
        active: _ActiveImport,
        request: DatasetImportWorkerRequest,
    ) -> tuple[int, ...]:
        return tuple(
            ordinal
            for ordinal in request.admitted_ordinals
            if active.outcomes[ordinal] is None
        )

    @staticmethod
    def _store_outcomes(
        active: _ActiveImport,
        request: DatasetImportWorkerRequest,
        ordinals: tuple[int, ...],
        outcome: (
            _DatasetImportSuccess
            | DatasetImportFailure
            | DatasetImportCancellation
        ),
    ) -> None:
        if any(
            ordinal not in request.admitted_ordinals
            or active.outcomes[ordinal] is not None
            for ordinal in ordinals
        ) or len(ordinals) != len(set(ordinals)):
            raise DatasetImportStateError(
                "Dataset import ordinal was settled twice or by the wrong worker."
            )
        for ordinal in ordinals:
            active.outcomes[ordinal] = outcome
        registration = active.workers.get(request.worker_id)
        if registration is not None:
            if registration.request is not request:
                raise DatasetImportStateError(
                    "Dataset import registration request identity changed."
                )
            registration.accounted_units += len(ordinals)
            if registration.accounted_units == request.expected_units:
                if isinstance(outcome, _DatasetImportSuccess):
                    active.latest_progress_by_worker[request.worker_id] = 100
        active.completed_units += len(ordinals)

    @staticmethod
    def _remaining_worker_units(
        registration: _RegisteredWorker,
    ) -> int:
        return max(
            0,
            registration.request.expected_units
            - registration.accounted_units,
        )
