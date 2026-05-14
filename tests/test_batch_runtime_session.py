from __future__ import annotations

from typing import Any

import pytest

from kindred.core.batch_containment import BatchCompletionRecord, BatchLaneOutcome, BatchRequestMetadata

pytestmark = pytest.mark.unit


class _FakeLaneOwner:
    def __init__(self) -> None:
        self.submitted: list[dict[str, Any]] = []
        self.polled: list[Any] = []
        self.shutdown_calls: list[bool] = []
        self.reset_calls = 0
        self.soft_supersede_calls = 0
        self.pool_stale = False
        self.has_pool = True
        self.current_generation = 3
        self.current_max_workers = 2
        self.active_count = 0
        self.worker_count = 0
        self.lane_pool_factory = lambda _max_lanes, _limit_blas: object()
        self.max_parallel_workers = 2
        self.limit_blas_threads_per_worker = True
        self.record_nonfatal_exception = lambda _msg, _exc: None
        self.metadata_by_set_id: dict[str, dict[str, Any]] = {}
        self.discarded: list[str] = []
        self.enqueued: list[str] = []
        self.drained = 0
        self.cleared = 0
        self.join_calls: list[float] = []
        self.lane_pool_token_value: int | None = 123

    @property
    def is_pool_stale(self) -> bool:
        return bool(self.pool_stale)

    @property
    def warm_failure(self) -> str | None:
        return None

    def ensure_lane_pool(self, *, max_lanes: int):
        return {"max_lanes": int(max_lanes)}

    def ensure_warm_lane_pool(self, *, max_lanes: int, wait: bool = True):
        return {"max_lanes": int(max_lanes), "wait": bool(wait)}

    def has_lane_pool(self) -> bool:
        return bool(self.has_pool)

    def has_ready_lane_pool(self, *, max_lanes: int) -> bool:
        _ = max_lanes
        return bool(self.has_pool and not self.pool_stale)

    def active_request_count(self) -> int:
        return int(self.active_count)

    def has_active_requests(self) -> bool:
        return bool(self.active_count)

    def request_worker_count(self) -> int:
        return int(self.worker_count)

    def join_active_requests(self, *, timeout_s: float = 2.0) -> None:
        self.join_calls.append(float(timeout_s))

    def lane_pool_token(self) -> int | None:
        return self.lane_pool_token_value

    def active_request_metadata(self, set_id: str) -> dict[str, Any]:
        return dict(self.metadata_by_set_id.get(str(set_id), {}))

    def discard_request(self, set_id: str) -> None:
        self.discarded.append(str(set_id))

    def clear_stale_requests(self) -> None:
        self.cleared += 1

    def drain_completion_queue(self) -> None:
        self.drained += 1

    def enqueue_completion(self, set_id: str) -> None:
        self.enqueued.append(str(set_id))

    def submit_task(self, task, **kwargs):
        call = {"task": dict(task or {}), **dict(kwargs)}
        self.submitted.append(call)
        self.active_count += 1
        return object()

    def poll_completed_records(self):
        records = list(self.polled)
        self.polled = []
        self.active_count = 0
        return records

    def soft_supersede(self) -> tuple[int, int]:
        self.soft_supersede_calls += 1
        running = int(self.active_count)
        self.active_count = 0
        return (0, running)

    def reset_active_run_state(self) -> None:
        self.reset_calls += 1
        self.active_count = 0

    def reset_run_state(self) -> None:
        self.reset_calls += 1
        self.active_count = 0
        self.polled = []
        self.drained += 1

    def mark_pool_stale(self) -> None:
        self.pool_stale = True

    def shutdown(self, *, force_terminate: bool, record_nonfatal_exception) -> None:
        _ = record_nonfatal_exception
        self.shutdown_calls.append(bool(force_terminate))
        self.active_count = 0
        self.has_pool = False
        self.pool_stale = False


class _PolledCompletion:
    def __init__(self, record: BatchCompletionRecord, *, source: str = "scan") -> None:
        self.set_id = record.set_id
        self.record = record
        self.source = source
        self.completed_ts = record.completed_ts


def _completion_record(
    set_id: str,
    *,
    run_id: int = 10,
    request_id: int = 20,
    generation: int = 3,
) -> BatchCompletionRecord:
    metadata = BatchRequestMetadata(
        set_id=set_id,
        set_name=f"Set {set_id}",
        run_id=run_id,
        request_id=request_id,
        generation=generation,
        preview_owner_epoch=9,
        expected_owner_epoch=9,
    )
    outcome = BatchLaneOutcome(
        lane_id="lane-a",
        run_id=run_id,
        request_id=request_id,
        set_id=set_id,
        owner_epoch=9,
        success=True,
        payload={"set_id": set_id},
    )
    return BatchCompletionRecord(metadata=metadata, outcome=outcome, completed_ts=123.0)


def test_session_begin_owns_explicit_lifecycle_state() -> None:
    from kindred.core.batch_runtime_session import BatchRuntimeSession, BatchRuntimeSessionRequest, BatchRuntimeSessionState

    owner = _FakeLaneOwner()
    session = BatchRuntimeSession(owner)

    session.begin(
        BatchRuntimeSessionRequest(
            run_id=10,
            request_id=20,
            fast_mode=True,
            queue_ids=("a", "b"),
            queue_names=("Set A", "Set B"),
            keep_lane_pool_alive=True,
            preview_owner_epoch=9,
            active_timeout_s=2.5,
            cache_key="cache-a",
        )
    )

    snapshot = session.snapshot()
    assert snapshot.state is BatchRuntimeSessionState.RUNNING
    assert snapshot.active is True
    assert snapshot.run_id == 10
    assert snapshot.request_id == 20
    assert snapshot.queue_ids == ("a", "b")
    assert snapshot.completed_set_ids == ()
    assert snapshot.keep_lane_pool_alive is True
    assert snapshot.current_generation == 3
    assert snapshot.has_lane_pool is True


def test_session_submit_uses_run_identity_without_controller_repeating_it() -> None:
    from kindred.core.batch_runtime_session import BatchRuntimeSession, BatchRuntimeSessionRequest

    owner = _FakeLaneOwner()
    session = BatchRuntimeSession(owner)
    session.begin(
        BatchRuntimeSessionRequest(
            run_id=10,
            request_id=20,
            fast_mode=False,
            queue_ids=("a",),
            queue_names=("Set A",),
            keep_lane_pool_alive=False,
            preview_owner_epoch=9,
            active_timeout_s=2.5,
        )
    )

    identity = object()

    session.submit_task(
        {"payload": 1},
        set_id="a",
        set_name="Set A",
        expected_owner_epoch=9,
        callback_identity=identity,
    )

    assert owner.submitted == [
        {
            "task": {"payload": 1},
            "run_id": 10,
            "request_id": 20,
            "set_id": "a",
            "set_name": "Set A",
            "preview_owner_epoch": 9,
            "active_timeout_s": 2.5,
            "expected_owner_epoch": 9,
            "request_metadata": {"callback_identity": identity},
        }
    ]


def test_session_submit_rejects_missing_callback_identity() -> None:
    from kindred.core.batch_runtime_session import BatchRuntimeSession, BatchRuntimeSessionRequest

    owner = _FakeLaneOwner()
    session = BatchRuntimeSession(owner)
    session.begin(
        BatchRuntimeSessionRequest(
            run_id=10,
            request_id=20,
            fast_mode=False,
            queue_ids=("a",),
            queue_names=("Set A",),
            keep_lane_pool_alive=False,
        )
    )

    with pytest.raises(ValueError, match="callback_identity"):
        session.submit_task({"payload": 1}, set_id="a", set_name="Set A", callback_identity=None)


def test_session_submit_stores_callback_identity_in_request_metadata() -> None:
    from kindred.core.batch_runtime_session import BatchRuntimeSession, BatchRuntimeSessionRequest

    owner = _FakeLaneOwner()
    session = BatchRuntimeSession(owner)
    session.begin(
        BatchRuntimeSessionRequest(
            run_id=10,
            request_id=20,
            fast_mode=False,
            queue_ids=("a",),
            queue_names=("Set A",),
            keep_lane_pool_alive=False,
            preview_owner_epoch=9,
            active_timeout_s=2.5,
        )
    )
    identity = object()

    session.submit_task({"payload": 1}, set_id="a", set_name="Set A", callback_identity=identity)

    assert owner.submitted[-1]["request_metadata"] == {"callback_identity": identity}


def test_session_poll_consumes_records_and_completes_current_run() -> None:
    from kindred.core.batch_runtime_session import BatchRuntimeSession, BatchRuntimeSessionRequest, BatchRuntimeSessionState

    owner = _FakeLaneOwner()
    owner.polled = [_PolledCompletion(_completion_record("a")), _PolledCompletion(_completion_record("b"))]
    session = BatchRuntimeSession(owner)
    session.begin(
        BatchRuntimeSessionRequest(
            run_id=10,
            request_id=20,
            fast_mode=False,
            queue_ids=("a", "b"),
            queue_names=("Set A", "Set B"),
            keep_lane_pool_alive=False,
            preview_owner_epoch=9,
            active_timeout_s=2.5,
        )
    )

    polled = session.poll_completed_records()

    assert [item.set_id for item in polled] == ["a", "b"]
    snapshot = session.snapshot()
    assert snapshot.completed_set_ids == ("a", "b")
    assert snapshot.active is False
    assert snapshot.state is BatchRuntimeSessionState.COMPLETED


def test_session_returns_stale_polled_records_without_marking_complete() -> None:
    from kindred.core.batch_runtime_session import BatchRuntimeSession, BatchRuntimeSessionRequest, BatchRuntimeSessionState

    owner = _FakeLaneOwner()
    owner.polled = [_PolledCompletion(_completion_record("a", run_id=99))]
    session = BatchRuntimeSession(owner)
    session.begin(
        BatchRuntimeSessionRequest(
            run_id=10,
            request_id=20,
            fast_mode=False,
            queue_ids=("a",),
            queue_names=("Set A",),
            keep_lane_pool_alive=False,
            preview_owner_epoch=9,
            active_timeout_s=2.5,
        )
    )

    returned = session.poll_completed_records()
    assert [item.set_id for item in returned] == ["a"]
    stale = returned[0].record.request_metadata["runtime_session_stale"]
    assert stale["expected_run_id"] == 10
    assert stale["actual_run_id"] == 99
    snapshot = session.snapshot()
    assert snapshot.completed_set_ids == ()
    assert snapshot.active is True
    assert snapshot.state is BatchRuntimeSessionState.RUNNING


def test_session_soft_supersede_is_a_lifecycle_transition_not_a_clear_call() -> None:
    from kindred.core.batch_runtime_session import BatchRuntimeSession, BatchRuntimeSessionRequest, BatchRuntimeSessionState

    owner = _FakeLaneOwner()
    owner.active_count = 2
    session = BatchRuntimeSession(owner)
    session.begin(
        BatchRuntimeSessionRequest(
            run_id=10,
            request_id=20,
            fast_mode=True,
            queue_ids=("a", "b"),
            queue_names=("Set A", "Set B"),
            keep_lane_pool_alive=True,
            preview_owner_epoch=9,
            active_timeout_s=2.5,
        )
    )
    owner.active_count = 2

    cancelled, running = session.soft_supersede_active_run()

    assert (cancelled, running) == (0, 2)
    assert owner.soft_supersede_calls == 1
    snapshot = session.snapshot()
    assert snapshot.active is False
    assert snapshot.state is BatchRuntimeSessionState.SUPERSEDED


def test_session_finish_keeps_or_shuts_down_lane_pool_by_policy() -> None:
    from kindred.core.batch_runtime_session import BatchRuntimeSession, BatchRuntimeSessionRequest

    owner = _FakeLaneOwner()
    session = BatchRuntimeSession(owner)
    request = BatchRuntimeSessionRequest(
        run_id=10,
        request_id=20,
        fast_mode=True,
        queue_ids=("a",),
        queue_names=("Set A",),
        keep_lane_pool_alive=True,
        preview_owner_epoch=9,
        active_timeout_s=2.5,
    )
    session.begin(request)
    resets_after_begin = int(owner.reset_calls)

    session.finish_after_run(keep_lane_pool_alive=True, record_nonfatal_exception=lambda _msg, _exc: None)

    assert owner.reset_calls == resets_after_begin + 1
    assert owner.shutdown_calls == []

    session.begin(request)
    session.finish_after_run(keep_lane_pool_alive=False, record_nonfatal_exception=lambda _msg, _exc: None)

    assert owner.shutdown_calls == [False]
