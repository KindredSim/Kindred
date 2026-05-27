"""Active dataset import session records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from kindred.gui.dataset_payload_snapshot import copy_dataset_payload


@dataclass(frozen=True)
class DatasetImportUnit:
    display_name: str
    payload: Dict[str, Any]
    source_path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", copy_dataset_payload(self.payload))


@dataclass(frozen=True)
class DatasetImportCompletion:
    session_id: int
    units: Tuple[DatasetImportUnit, ...]
    outcome: str = "completed"
    errors: Tuple[str, ...] = ()
    canceled: bool = False
    superseded: bool = False


@dataclass
class DatasetImportSession:
    session_id: int
    expected_units: int
    _units: list[DatasetImportUnit] = field(default_factory=list)
    _errors: list[str] = field(default_factory=list)
    _workers: dict[int, Any] = field(default_factory=dict)
    _expected_by_worker: dict[int, int] = field(default_factory=dict)
    _accounted_by_worker: dict[int, int] = field(default_factory=dict)
    _config_by_worker: dict[int, Any] = field(default_factory=dict)
    _done_workers: set[int] = field(default_factory=set)
    _completed_units: int = 0
    _terminal_outcome: str | None = None

    @property
    def terminal(self) -> bool:
        return self._terminal_outcome is not None

    @property
    def terminal_outcome(self) -> str | None:
        return self._terminal_outcome

    def register_worker(self, worker: Any, *, expected_units: int, config: Any) -> None:
        worker_id = id(worker)
        self._workers[worker_id] = worker
        self._expected_by_worker[worker_id] = max(0, int(expected_units or 0))
        self._accounted_by_worker[worker_id] = 0
        self._config_by_worker[worker_id] = config

    def deauthorize(self, outcome: str) -> tuple[Any, ...]:
        self._terminal_outcome = str(outcome or "superseded")
        self._units.clear()
        for worker in tuple(self._workers.values()):
            try:
                worker.requestInterruption()
            except RuntimeError:
                pass
        return tuple(self._workers.values())

    def owns_worker(self, worker: Any) -> bool:
        return id(worker) in self._workers

    def worker_config(self, worker: Any) -> Any:
        return self._config_by_worker.get(id(worker))

    def remaining_worker_result_count(self, worker: Any) -> int:
        worker_id = id(worker)
        expected = int(self._expected_by_worker.get(worker_id, 0) or 0)
        accounted = int(self._accounted_by_worker.get(worker_id, 0) or 0)
        return max(0, expected - accounted)

    def note_worker_units_processed(self, worker: Any, count: int) -> None:
        worker_id = id(worker)
        if worker_id not in self._workers:
            return
        expected = int(self._expected_by_worker.get(worker_id, 0) or 0)
        accounted = int(self._accounted_by_worker.get(worker_id, 0) or 0)
        delta = max(0, min(int(count), max(0, expected - accounted)))
        if delta <= 0:
            return
        self._accounted_by_worker[worker_id] = accounted + delta
        self._completed_units += delta

    def account_remaining_worker_units(self, worker: Any) -> int:
        remaining = self.remaining_worker_result_count(worker)
        self.note_worker_units_processed(worker, remaining)
        return remaining

    def mark_worker_done(self, worker: Any) -> None:
        worker_id = id(worker)
        if worker_id not in self._workers:
            return
        self._done_workers.add(worker_id)
        self._workers.pop(worker_id, None)
        self._expected_by_worker.pop(worker_id, None)
        self._accounted_by_worker.pop(worker_id, None)
        self._config_by_worker.pop(worker_id, None)

    def mark_worker_terminal(self, worker: Any) -> None:
        self.account_remaining_worker_units(worker)
        self.mark_worker_done(worker)

    def complete_ready(self) -> bool:
        return self._completed_units >= max(0, int(self.expected_units or 0)) and not self._workers

    def add_unit(self, unit: DatasetImportUnit) -> None:
        if self.terminal:
            return
        self._units.append(unit)

    def add_error(self, message: str) -> None:
        if self.terminal:
            return
        text = str(message).strip()
        if text:
            self._errors.append(text)

    def completion(
        self,
        *,
        canceled: bool,
        superseded: bool,
        discard_units: bool = False,
        outcome: str = "completed",
    ) -> DatasetImportCompletion:
        terminal_outcome = str(outcome or "completed")
        discard = bool(discard_units or terminal_outcome != "completed")
        units: Tuple[DatasetImportUnit, ...] = () if discard else tuple(self._units)
        return DatasetImportCompletion(
            session_id=int(self.session_id),
            units=units,
            outcome=terminal_outcome,
            errors=tuple(self._errors),
            canceled=bool(canceled),
            superseded=bool(superseded),
        )
