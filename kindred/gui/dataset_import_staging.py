"""Direct owner for dataset-import workers and uncommitted normalized output."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from PySide6 import QtCore

from kindred.core.datasets.observation_payload import DatasetObservationPayload
from kindred.gui.controllers.dataset_errors import (
    DatasetImportAdmissionError,
    DatasetImportStateError,
)
from kindred.core.dataset_mapping import validate_raw_dataset_record
from kindred.core.project_document import (
    RawDatasetRecord,
    new_dataset_id,
)
from kindred.gui.widgets.import_config import (
    ImportSourceSnapshot,
    ResolvedSheetPlan,
)

_MAX_IMPORT_UNITS = 64
_MAX_IMPORT_RETAINED_OBSERVATION_CELLS = 1_000_000


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


@dataclass(frozen=True)
class _DatasetImportCompletion:
    import_id: int
    origin_tab_id: str
    units: tuple[DatasetImportUnit, ...]


@dataclass(frozen=True, slots=True)
class RawDatasetBatchProposal:
    """One complete immutable all-or-none project dataset proposal."""

    import_id: int
    origin_tab_id: str
    records: tuple[RawDatasetRecord, ...]

    def __post_init__(self) -> None:
        if type(self.import_id) is not int or self.import_id <= 0:
            raise ValueError("Raw dataset batch import ID must be positive.")
        if (
            type(self.origin_tab_id) is not str
            or not self.origin_tab_id
            or self.origin_tab_id != self.origin_tab_id.strip()
        ):
            raise ValueError(
                "Raw dataset batch origin tab ID must be nonblank and "
                "edge-normalized."
            )
        if type(self.records) is not tuple or any(
            not isinstance(record, RawDatasetRecord)
            for record in self.records
        ):
            raise TypeError(
                "Raw dataset batch records must be an exact "
                "RawDatasetRecord tuple."
            )


@dataclass(frozen=True, slots=True)
class DatasetImportRequest:
    import_id: int
    origin_tab_id: str
    expected_units: int

    def __post_init__(self) -> None:
        if type(self.import_id) is not int or self.import_id <= 0:
            raise ValueError("Dataset import ID must be a positive integer.")
        if (
            type(self.origin_tab_id) is not str
            or not self.origin_tab_id
            or self.origin_tab_id != self.origin_tab_id.strip()
        ):
            raise ValueError(
                "Dataset import origin tab ID must be nonblank and "
                "edge-normalized."
            )
        if type(self.expected_units) is not int or self.expected_units <= 0:
            raise ValueError(
                "Dataset import expected_units must be a positive integer."
            )


@dataclass(frozen=True, slots=True)
class DatasetImportWorkerRequest:
    import_request: DatasetImportRequest
    worker_id: int
    source: ImportSourceSnapshot
    expected_units: int
    admitted_plans: tuple[ResolvedSheetPlan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.import_request, DatasetImportRequest):
            raise TypeError(
                "Dataset import worker requires DatasetImportRequest."
            )
        if type(self.worker_id) is not int or self.worker_id <= 0:
            raise ValueError("Dataset import worker ID must be a positive integer.")
        if not isinstance(self.source, ImportSourceSnapshot):
            raise TypeError(
                "Dataset import worker source must be ImportSourceSnapshot."
            )
        if type(self.expected_units) is not int or self.expected_units <= 0:
            raise ValueError(
                "Dataset import worker expected_units must be a positive integer."
            )
        plans = self.admitted_plans
        if type(plans) is not tuple or any(
            not isinstance(plan, ResolvedSheetPlan)
            for plan in plans
        ):
            raise TypeError(
                "Dataset import worker admitted plans must be an exact "
                "ResolvedSheetPlan tuple."
            )
        if len(plans) != self.expected_units:
            raise ValueError(
                "A dataset import worker must retain one typed plan per "
                "expected result."
            )
        if self.source.file_type == "csv":
            if len(plans) != 1 or plans[0].sheet_name is not None:
                raise ValueError(
                    "A CSV import worker must admit exactly one non-sheet plan."
                )
            return
        if any(
            plan.sheet_name is None
            for plan in plans
        ):
            raise ValueError(
                "Every admitted XLSX plan must identify one worksheet in the "
                "worker source."
            )

    @property
    def import_id(self) -> int:
        return self.import_request.import_id


@dataclass(frozen=True, slots=True)
class DatasetImportProgress:
    request: DatasetImportWorkerRequest
    percent: int

    def __post_init__(self) -> None:
        if not isinstance(self.request, DatasetImportWorkerRequest):
            raise TypeError(
                "Dataset import progress requires DatasetImportWorkerRequest."
            )
        if type(self.percent) is not int or not 0 <= self.percent <= 100:
            raise ValueError(
                "Dataset import progress must be an integer from 0 to 100."
            )


@dataclass(frozen=True, slots=True)
class DatasetImportFailure:
    request: DatasetImportWorkerRequest
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, DatasetImportWorkerRequest):
            raise TypeError(
                "Dataset import failure requires DatasetImportWorkerRequest."
            )
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
        if not isinstance(self.request, DatasetImportWorkerRequest):
            raise TypeError(
                "Dataset import cancellation requires DatasetImportWorkerRequest."
            )
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


class DatasetImportWorker(QtCore.QThread):
    """Exact signal contract for workers retained by DatasetImportStaging."""

    loaded = QtCore.Signal(str, object)
    cancelled = QtCore.Signal(str)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int)

    def __init__(self, source: ImportSourceSnapshot) -> None:
        super().__init__()
        if not isinstance(source, ImportSourceSnapshot):
            raise TypeError(
                "Dataset import workers require one immutable source snapshot."
            )
        self._source = source

    @property
    def source(self) -> ImportSourceSnapshot:
        return self._source

def _raw_record_batch_proposal(
    completion: _DatasetImportCompletion,
    *,
    existing_records: tuple[RawDatasetRecord, ...],
) -> RawDatasetBatchProposal:
    """Build the complete immutable batch for one ProjectDocument admission.

    This is identity/name allocation only. ``ProjectDocument.add_datasets``
    validates the whole returned tuple in one all-or-none project proposal.
    Optional active-tab mapping remains a later transaction over committed raw
    records.
    """
    if not isinstance(completion, _DatasetImportCompletion):
        raise TypeError(
            "Raw dataset batch proposal requires a dataset import completion."
        )
    if type(existing_records) is not tuple or any(
        not isinstance(record, RawDatasetRecord)
        for record in existing_records
    ):
        raise TypeError(
            "Existing raw datasets must be an exact RawDatasetRecord tuple."
        )
    used_names = {record.display_name.casefold() for record in existing_records}
    candidates: list[RawDatasetRecord] = []
    for unit in completion.units:
        raw_name = unit.display_name.strip() or "dataset"
        stem, dot, suffix = raw_name.rpartition(".")
        base = stem if dot and stem else raw_name
        extension = f".{suffix}" if dot and stem else ""
        display_name = raw_name
        counter = 1
        while display_name.casefold() in used_names:
            display_name = f"{base}_{counter}{extension}"
            counter += 1
        used_names.add(display_name.casefold())
        candidate = RawDatasetRecord(
            dataset_id=new_dataset_id(),
            display_name=display_name,
            original_filename=unit.source.original_filename,
            original_format=unit.source.archive_format,
            exact_source_bytes=unit.source.exact_source_bytes,
            source_metadata=unit.payload.source_metadata,
            observations=unit.payload.observations,
        )
        validate_raw_dataset_record(candidate)
        candidates.append(candidate)
    return RawDatasetBatchProposal(
        import_id=completion.import_id,
        origin_tab_id=completion.origin_tab_id,
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
    workers: dict[int, _RegisteredWorker] = field(default_factory=dict)
    units: list[DatasetImportUnit] = field(default_factory=list)
    completed_units: int = 0
    retained_cells: int = 0
    terminal_outcome: str | None = None


class DatasetImportStaging(QtCore.QObject):
    """Own live QThreads until finish and exactly one normalized completion.

    The object admits one complete raw-record batch before a GUI-thread
    integration owner commits it through ProjectSession and only then offers
    optional active-tab mapping.
    """

    completionAvailable = QtCore.Signal(int)
    busyChanged = QtCore.Signal(bool)
    owned_qthreads_changed = QtCore.Signal()
    discarded = QtCore.Signal(str)
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
        self._completion: _DatasetImportCompletion | None = None
        self._project_replacement_admission_fenced = False
        self._application_close_fenced = False

    @property
    def busy(self) -> bool:
        return self._active is not None or self._completion is not None

    def begin(
        self,
        *,
        expected_units: int,
        origin_tab_id: str,
    ) -> DatasetImportRequest:
        """Admit a new import on the GUI thread."""
        if (
            self._application_close_fenced
            or self._project_replacement_admission_fenced
        ):
            raise DatasetImportAdmissionError("Dataset import admission is fenced.")
        if self.busy:
            raise DatasetImportAdmissionError("A dataset import is already active or awaiting commit.")
        expected = int(expected_units)
        if expected <= 0:
            raise DatasetImportAdmissionError("Dataset import requires at least one result unit.")
        if expected > _MAX_IMPORT_UNITS:
            raise DatasetImportAdmissionError(
                "A dataset import cannot contain more than 64 resolved units."
            )
        import_id = self._next_import_id
        self._next_import_id += 1
        request = DatasetImportRequest(
            import_id=import_id,
            origin_tab_id=origin_tab_id,
            expected_units=expected,
        )
        self._active = _ActiveImport(request=request)
        self.busyChanged.emit(True)
        return request

    def register_worker(
        self,
        import_request: DatasetImportRequest,
        worker: DatasetImportWorker,
        *,
        source: ImportSourceSnapshot,
        admitted_plans: tuple[ResolvedSheetPlan, ...],
    ) -> DatasetImportWorkerRequest:
        if not isinstance(import_request, DatasetImportRequest):
            raise TypeError(
                "Dataset import worker registration requires DatasetImportRequest."
            )
        active = self._require_active(import_request.import_id)
        if active.request is not import_request:
            raise DatasetImportStateError(
                "Dataset import request is stale."
            )
        if (
            self._application_close_fenced
            or self._project_replacement_admission_fenced
            or active.terminal_outcome is not None
        ):
            raise DatasetImportAdmissionError(
                "Dataset import worker admission is fenced."
            )
        if not isinstance(worker, DatasetImportWorker):
            raise TypeError(
                "Dataset import workers must implement DatasetImportWorker."
            )
        if not isinstance(source, ImportSourceSnapshot):
            raise TypeError(
                "Dataset import worker registration requires one immutable "
                "source snapshot."
            )
        if worker.source != source:
            raise DatasetImportStateError(
                "Dataset import worker source does not match the admitted "
                "immutable snapshot."
            )
        if any(
            registration.worker is worker
            for registration in active.workers.values()
        ):
            raise DatasetImportStateError(
                "Dataset import worker is already registered."
            )
        plans = tuple(admitted_plans)
        expected = len(plans)
        if expected <= 0:
            raise DatasetImportStateError(
                "Dataset import workers must account for at least one result unit."
            )
        registered_total = sum(
            registration.request.expected_units
            for registration in active.workers.values()
        )
        if registered_total + expected > active.request.expected_units:
            raise DatasetImportStateError(
                "Dataset import workers exceed the admitted result-unit count."
            )
        if registered_total + expected > _MAX_IMPORT_UNITS:
            raise DatasetImportStateError(
                "A dataset import cannot contain more than 64 resolved units."
            )
        worker_id = self._next_worker_id
        self._next_worker_id += 1
        request = DatasetImportWorkerRequest(
            import_request=active.request,
            worker_id=worker_id,
            source=source,
            expected_units=expected,
            admitted_plans=plans,
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
        active.workers[worker_id] = registration
        self.owned_qthreads_changed.emit()
        return request

    def owned_qthreads(self) -> tuple[QtCore.QThread, ...]:
        """Return the exact native threads still retained by this owner."""

        active = self._active
        if active is None:
            return ()
        return tuple(
            registration.worker
            for registration in active.workers.values()
        )

    def release_unstarted_worker(
        self,
        request: DatasetImportWorkerRequest,
        *,
        worker: DatasetImportWorker,
    ) -> None:
        """Release a registered worker after start failed before it ran."""

        active, registration = self._require_registered(request, worker)
        try:
            running = bool(worker.isRunning())
        except RuntimeError:
            running = False
        if running:
            raise DatasetImportStateError(
                "A running dataset import worker cannot be released as unstarted."
            )
        self._release_worker(active, registration)

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
                registration,
                request,
                f"{type(exc).__name__}: {exc}",
            )
            return False
        retained_cells = sum(len(series.t) for series in payload.observations)
        if (
            active.retained_cells + retained_cells
            > _MAX_IMPORT_RETAINED_OBSERVATION_CELLS
        ):
            self._record_failure(
                active,
                registration,
                request,
                "The import exceeds the 1000000 retained-observation-cell limit.",
            )
            return False
        self._account_worker_units(active, registration, 1)
        active.units.append(unit)
        active.retained_cells += retained_cells
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
        active, _registration = owned
        if active.terminal_outcome is not None:
            return False
        progress = max(0, min(100, int(percent)))
        self.workerProgressed.emit(
            DatasetImportProgress(request=request, percent=progress)
        )
        return True

    def _worker_failed(
        self,
        request: DatasetImportWorkerRequest,
        worker: DatasetImportWorker,
        message: str,
    ) -> bool:
        owned = self._lookup_worker_callback(request, worker)
        if owned is None:
            return False
        active, registration = owned
        if active.terminal_outcome is not None:
            return False
        self._record_failure(active, registration, request, message)
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
        active, registration = owned
        event = DatasetImportCancellation(
            request=request,
            source_label=source_label,
        )
        if active.terminal_outcome is None:
            self.cancel_active(outcome="worker_canceled")
        missing = self._remaining_worker_units(registration)
        self._account_worker_units(active, registration, missing)
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
            message = (
                "Dataset importer ended without returning "
                f"{missing} expected result{'s' if missing != 1 else ''}."
            )
            failure = DatasetImportFailure(request=request, message=message)
        self._account_worker_units(active, registration, missing)
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
        self.owned_qthreads_changed.emit()

    def cancel_active(self, *, outcome: str = "user_canceled") -> bool:
        active = self._active
        if active is None:
            return False
        if active.terminal_outcome is None:
            active.terminal_outcome = str(outcome or "user_canceled")
            active.units.clear()
            active.retained_cells = 0
            for registration in tuple(active.workers.values()):
                try:
                    registration.worker.requestInterruption()
                except RuntimeError:
                    continue
        self._finish_if_ready()
        return True

    def take_batch_for_commit(
        self,
        import_id: int,
        *,
        existing_records: tuple[RawDatasetRecord, ...],
    ) -> RawDatasetBatchProposal | None:
        """Take one complete staging-admitted raw-record batch."""
        completion = self._completion
        if completion is None or completion.import_id != int(import_id):
            return None
        try:
            return _raw_record_batch_proposal(
                completion,
                existing_records=existing_records,
            )
        finally:
            self._completion = None
            self.busyChanged.emit(False)

    def fence_application_close_admission(self) -> None:
        """Permanently reject every later dataset-import admission."""

        self._application_close_fenced = True

    def cancel_for_application_close(self) -> None:
        """Suppress publication and request cancellation without waiting."""

        self.fence_application_close_admission()
        self.cancel_active(outcome="application_close")

    def fence_for_project_replacement(self) -> None:
        """Temporarily reject import admission during project replacement."""

        self._project_replacement_admission_fenced = True

    def resume_after_project_replacement(self) -> None:
        """Release only the reversible project-replacement fence."""

        self._project_replacement_admission_fenced = False

    def cancel_and_settle_for_project_replacement(self, deadline: float) -> bool:
        """Blocking cleanup solely for an accepted project replacement."""

        self.fence_for_project_replacement()
        self.cancel_active(outcome="project_replacement")
        active = self._active
        if active is None:
            return True
        for registration in tuple(active.workers.values()):
            worker = registration.worker
            remaining = max(0.0, float(deadline) - time.monotonic())
            if remaining <= 0.0:
                return False
            try:
                worker.wait(max(1, int(remaining * 1000.0)))
            except RuntimeError:
                pass
            try:
                running = bool(worker.isRunning())
            except RuntimeError:
                running = False
            if running:
                return False
            self._worker_finished(registration.request, worker)
        return self._active is None

    def _finish_if_ready(self) -> None:
        active = self._active
        if active is None or active.workers:
            return
        if active.terminal_outcome is not None:
            outcome = active.terminal_outcome
            self._active = None
            self.discarded.emit(f"Dataset import ended: {outcome}.")
            self.busyChanged.emit(False)
            return
        if active.completed_units < active.request.expected_units:
            return
        completion = _DatasetImportCompletion(
            import_id=active.request.import_id,
            origin_tab_id=active.request.origin_tab_id,
            units=tuple(active.units),
        )
        self._active = None
        self._completion = completion
        try:
            self.completionAvailable.emit(completion.import_id)
        finally:
            if self._completion is completion:
                self._completion = None
                self.discarded.emit(
                    "Dataset import completion was not consumed synchronously."
                )
                self.busyChanged.emit(False)

    def _require_active(self, import_id: int) -> _ActiveImport:
        active = self._active
        if (
            active is None
            or active.request.import_id != int(import_id)
        ):
            raise DatasetImportStateError("Dataset import callback is stale.")
        return active

    def _require_registered(
        self,
        request: DatasetImportWorkerRequest,
        worker: DatasetImportWorker,
    ) -> tuple[_ActiveImport, _RegisteredWorker]:
        active, registration = self._require_request(request)
        if not isinstance(worker, DatasetImportWorker):
            raise TypeError(
                "Dataset import callback requires DatasetImportWorker."
            )
        if registration.worker is not worker:
            raise DatasetImportStateError(
                "Dataset import callback identity does not match its worker."
            )
        return active, registration

    def _require_request(
        self,
        request: DatasetImportWorkerRequest,
    ) -> tuple[_ActiveImport, _RegisteredWorker]:
        if not isinstance(request, DatasetImportWorkerRequest):
            raise TypeError(
                "Dataset import callback requires DatasetImportWorkerRequest."
            )
        active = self._require_active(request.import_id)
        registration = active.workers.get(request.worker_id)
        if (
            registration is None
            or registration.request is not request
        ):
            raise DatasetImportStateError(
                "Dataset import callback identity is not registered."
            )
        return active, registration

    def _lookup_worker_callback(
        self,
        request: DatasetImportWorkerRequest,
        worker: DatasetImportWorker,
    ) -> tuple[_ActiveImport, _RegisteredWorker] | None:
        if not isinstance(request, DatasetImportWorkerRequest):
            raise TypeError(
                "Dataset import callback requires DatasetImportWorkerRequest."
            )
        if not isinstance(worker, DatasetImportWorker):
            raise TypeError(
                "Dataset import callback requires DatasetImportWorker."
            )
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
        registration: _RegisteredWorker,
        request: DatasetImportWorkerRequest,
        message: str,
    ) -> None:
        event = DatasetImportFailure(request=request, message=message)
        self._latch_failure(active)
        self.workerFailed.emit(event)

    @staticmethod
    def _latch_failure(active: _ActiveImport) -> None:
        if active.terminal_outcome is not None:
            return
        active.terminal_outcome = "worker_failed"
        active.units.clear()
        active.retained_cells = 0
        for registration in tuple(active.workers.values()):
            try:
                registration.worker.requestInterruption()
            except RuntimeError:
                continue

    @staticmethod
    def _remaining_worker_units(
        registration: _RegisteredWorker,
    ) -> int:
        return max(
            0,
            registration.request.expected_units
            - registration.accounted_units,
        )

    @staticmethod
    def _account_worker_units(
        active: _ActiveImport,
        registration: _RegisteredWorker,
        count: int,
    ) -> None:
        requested = int(count)
        if requested < 0:
            raise DatasetImportStateError(
                "Dataset import result accounting cannot be negative."
            )
        remaining = DatasetImportStaging._remaining_worker_units(registration)
        if requested > remaining:
            raise DatasetImportStateError(
                "Dataset import worker produced more results than admitted."
            )
        registration.accounted_units += requested
        active.completed_units += requested
