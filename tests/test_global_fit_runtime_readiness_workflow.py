from __future__ import annotations

import threading
import time

import multiprocessing
import numpy as np
import pytest

from kindred.core.fitting_containment import WarmFittingEvaluatorLane
from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
from kindred.core.fitting_runtime_session import FittingRuntimeLedger, FittingRuntimeSession
from kindred.core.simulation_preparation import PreparedSimulationMetadata
from kindred.gui.fitting.run_stamp import hash_global_fit_run_stamp
from kindred.gui.fitting.runtime_readiness import (
    FittingRuntimeLaunchDecisionState,
    FittingRuntimeIdentity,
    FittingRuntimeReadinessController,
    FittingRuntimeReadinessState,
)
from tests.workflow_helpers import latest_fit_window, seed_simple_mechanism, seed_two_datasets


class _BlockingRuntimeSession:
    def __init__(self) -> None:
        self.warm_entered = threading.Event()
        self.release_warm = threading.Event()
        self.close_calls: list[bool] = []
        self.cancel_calls = 0

    def warm(self, *, cancellation_check=None, lane_count=None) -> None:
        self.warm_entered.set()
        self.release_warm.wait(timeout=2.0)

    def cancel_run(self) -> None:
        self.cancel_calls += 1
        self.close(kill=True)
        self.release_warm.set()

    def close(self, *, kill: bool = False) -> None:
        self.close_calls.append(bool(kill))
        self.release_warm.set()

    def is_ready(self, *, lane_count=None) -> bool:
        return False


class _ReadyRuntimeSession(_BlockingRuntimeSession):
    def warm(self, *, cancellation_check=None, lane_count=None) -> None:
        self.warm_entered.set()
        self.release_warm.set()

    def is_ready(self, *, lane_count=None) -> bool:
        return not self.close_calls


class _DetachedCancelRuntimeSession(_BlockingRuntimeSession):
    def cancel_run(self) -> None:
        self.cancel_calls += 1
        self.release_warm.set()


class _FakeRuntimeLane:
    def __init__(self, _payload):
        self.close_calls: list[bool] = []
        self.warm_calls = 0

    def warm(self, *, cancellation_check=None) -> None:
        self.warm_calls += 1

    def close(self, *, kill: bool = False) -> None:
        self.close_calls.append(bool(kill))


class _FakeRuntimeScheduler:
    def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.shutdown_calls: list[tuple[bool, bool | None]] = []

    def shutdown(self, *, wait: bool = True, cancel_futures: bool | None = None) -> None:
        self.shutdown_calls.append((bool(wait), cancel_futures))


def _ready_then_waiting_fit_lane_child(process_payload, input_queue, output_queue, owner_epoch):
    output_queue.put(
        {
            "kind": "ready",
            "owner_epoch": int(owner_epoch),
        }
    )
    input_queue.get()


def _serial_evaluator() -> SerialFittingEvaluator:
    context = prepare_fitting_execution_context(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        param_names=["k1"],
        t_end=1.0,
        num_points=3,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    return SerialFittingEvaluator(context)


def _dataset_spec() -> FitDatasetSpec:
    return FitDatasetSpec(
        dataset_id="ds1",
        t_exp=np.asarray([0.0, 0.5, 1.0], dtype=float),
        species_list=["B"],
        y_matrix=np.asarray([[0.0, 0.1, 0.2]], dtype=float),
        point_count=3,
        x_name="t",
        x_obs=None,
        x_mode="auto",
    )


def _symbolic_prepared_metadata() -> PreparedSimulationMetadata:
    return PreparedSimulationMetadata(
        version=1,
        mechanism_text_sha256="0" * 64,
        mechanism_text_len=42,
        param_names=["k1"],
        t_end=1.0,
        num_points=3,
        temperature_K=298.15,
        solver_requested="BDF",
        solver_normalized="BDF",
        solver_warning=None,
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=True,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
        symbolic_jacobian_identity={"kind": "jacobian", "fingerprint": "prepared-symbolic"},
    )


def _identity(*, stamp_hash: str, evaluator: SerialFittingEvaluator, lane_count: int = 2) -> FittingRuntimeIdentity:
    return FittingRuntimeIdentity(
        datasets=(_dataset_spec(),),
        config={
            "parameters": {"k1": 0.2},
            "bounds": {"k1": (0.01, 1.0)},
            "method": "trf",
            "max_nfev": 5,
            "ftol": 1e-8,
            "xtol": 1e-8,
            "seed": None,
            "log10_params": {},
        },
        dataset_overrides=(),
        weights=None,
        requested_solver="BDF",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=evaluator,
        stamp={"kind": "runtime-readiness-workflow", "hash": stamp_hash},
        stamp_hash=stamp_hash,
        stamp_short=stamp_hash[:12],
        lane_count=lane_count,
        base_evaluator=evaluator,
    )


def _symbolic_deferred_identity() -> FittingRuntimeIdentity:
    stamp = {
        "kind": "runtime-readiness-workflow",
        "runtime_request": {"solver": "BDF", "param_names": ["k1"]},
    }
    stamp_hash = hash_global_fit_run_stamp(stamp)
    evaluator = _serial_evaluator()
    object.__setattr__(evaluator._context, "prepared_metadata", _symbolic_prepared_metadata())
    return FittingRuntimeIdentity(
        datasets=(_dataset_spec(),),
        config={
            "parameters": {"k1": 0.2},
            "bounds": {"k1": (0.01, 1.0)},
            "method": "trf",
            "max_nfev": 5,
            "ftol": 1e-8,
            "xtol": 1e-8,
            "seed": None,
            "log10_params": {},
        },
        dataset_overrides=(),
        weights=None,
        requested_solver="BDF",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=None,
        stamp=stamp,
        stamp_hash=stamp_hash,
        stamp_short=stamp_hash[:12],
        lane_count=1,
        fit_evaluator_factory=lambda: evaluator,
        base_evaluator=None,
    )


def _drain_preparation(controller: FittingRuntimeReadinessController, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if controller.handle_worker_finished():
            return
        time.sleep(0.005)
    raise AssertionError("fitting runtime preparation did not finish")


def test_close_during_passive_preparation_releases_active_runtime_once():
    session = _BlockingRuntimeSession()
    controller = FittingRuntimeReadinessController(
        session_factory=lambda _evaluator, _lane_count: session,
        finished_callback=lambda: None,
    )
    identity = _identity(stamp_hash="close-active-preparation", evaluator=_serial_evaluator())

    controller.set_desired_identity(identity)
    assert session.warm_entered.wait(timeout=2.0)

    controller.close(kill=True)
    _drain_preparation(controller)

    snapshot = controller.snapshot()
    assert snapshot.state is FittingRuntimeReadinessState.CLOSED
    assert snapshot.session is None
    assert session.cancel_calls == 0
    assert session.close_calls == [True]
    assert snapshot.ledger.session_closes == 1


def test_close_releases_cancel_detached_preparation_session():
    session = _DetachedCancelRuntimeSession()
    controller = FittingRuntimeReadinessController(
        session_factory=lambda _evaluator, _lane_count: session,
        finished_callback=lambda: None,
    )
    identity = _identity(stamp_hash="detached-cancel-preparation", evaluator=_serial_evaluator())

    controller.set_desired_identity(identity)
    assert session.warm_entered.wait(timeout=2.0)

    assert controller.cancel(kill=True) is False
    _drain_preparation(controller)
    assert session.cancel_calls == 0
    assert session.close_calls == [True]

    assert controller.close(kill=True) is True

    assert controller.snapshot().state is FittingRuntimeReadinessState.CLOSED
    assert session.close_calls == [True]


def test_supersede_releases_active_preparation_runtime_before_new_ready_session():
    first = _BlockingRuntimeSession()
    second = _ReadyRuntimeSession()
    sessions = [first, second]
    controller = FittingRuntimeReadinessController(
        session_factory=lambda _evaluator, _lane_count: sessions.pop(0),
        finished_callback=lambda: None,
    )

    controller.set_desired_identity(
        _identity(stamp_hash="superseded-active-preparation", evaluator=_serial_evaluator())
    )
    assert first.warm_entered.wait(timeout=2.0)

    controller.set_desired_identity(
        _identity(stamp_hash="replacement-preparation", evaluator=_serial_evaluator())
    )
    _drain_preparation(controller)
    _drain_preparation(controller)

    snapshot = controller.snapshot()
    assert snapshot.state is FittingRuntimeReadinessState.READY
    assert snapshot.session is second
    assert first.close_calls == [True]
    assert second.close_calls == []


def test_cancel_releases_ready_runtime_session_without_window_compensation():
    session = _ReadyRuntimeSession()
    controller = FittingRuntimeReadinessController(
        session_factory=lambda _evaluator, _lane_count: session,
        finished_callback=lambda: None,
    )
    identity = _identity(stamp_hash="cancel-ready-session", evaluator=_serial_evaluator())

    controller.set_desired_identity(identity)
    _drain_preparation(controller)
    assert controller.snapshot().state is FittingRuntimeReadinessState.READY

    assert controller.cancel(kill=True) is True

    snapshot = controller.snapshot()
    assert snapshot.state is FittingRuntimeReadinessState.EMPTY
    assert snapshot.session is None
    assert session.close_calls == [True]


def test_symbolic_prepared_runtime_reuses_original_launch_request_hash():
    session = _ReadyRuntimeSession()
    controller = FittingRuntimeReadinessController(
        session_factory=lambda _evaluator, _lane_count: session,
        finished_callback=lambda: None,
    )
    identity = _symbolic_deferred_identity()

    controller.set_desired_identity(identity)
    _drain_preparation(controller)
    ready_snapshot = controller.snapshot()
    assert ready_snapshot.state is FittingRuntimeReadinessState.READY
    assert ready_snapshot.identity is not None
    assert ready_snapshot.identity.stamp_hash != identity.stamp_hash

    decision = controller.prepare_or_accept_launch(identity)

    assert decision.state is FittingRuntimeLaunchDecisionState.ACCEPTED
    assert decision.accepted_launch is not None
    assert decision.accepted_launch.identity.launch_request_hash == identity.stamp_hash


def test_real_runtime_session_cancel_kills_bounded_lanes_and_scheduler():
    ledger = FittingRuntimeLedger()
    session = FittingRuntimeSession(
        process_payload={"kind": "public-runtime-close-contract"},
        max_lanes=2,
        lane_factory=_FakeRuntimeLane,
        executor_factory=_FakeRuntimeScheduler,
        ledger=ledger,
    )

    session.warm(lane_count=2)
    session.cancel_run()
    session.close(kill=True)

    assert ledger.lane_creations == 2
    assert ledger.lane_closes == 2
    assert ledger.lane_kills == 2
    assert ledger.scheduler_creations == 1
    assert ledger.scheduler_shutdowns == 1
    assert session.is_ready(lane_count=2) is False


def test_warm_fitting_evaluator_lane_kill_close_terminates_spawned_process():
    before_pids = {child.pid for child in multiprocessing.active_children()}
    lane = WarmFittingEvaluatorLane(
        {},
        ready_timeout_s=3.0,
        mp_context=multiprocessing.get_context("spawn"),
        child_target=_ready_then_waiting_fit_lane_child,
    )

    lane.warm()
    new_children = [
        child
        for child in multiprocessing.active_children()
        if child.pid not in before_pids
    ]
    assert len(new_children) == 1
    child = new_children[0]
    assert child.is_alive()

    lane.close(kill=True)

    deadline = time.monotonic() + 2.0
    while child.is_alive() and time.monotonic() < deadline:
        child.join(timeout=0.01)
    assert child.is_alive() is False


@pytest.mark.gui
def test_fitting_window_close_event_drains_active_runtime_preparation(
    main_window,
    monkeypatch,
    qt_app,
):
    seed_two_datasets(main_window)
    seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )
    main_window._run_global_fit()
    window = latest_fit_window(main_window)

    session = _BlockingRuntimeSession()
    controller = FittingRuntimeReadinessController(
        session_factory=lambda _evaluator, _lane_count: session,
        finished_callback=lambda: None,
    )
    window._fit_runtime_readiness = controller
    window.fit_runtime_preparation_owner.refresh_pending = False
    identity = _identity(stamp_hash="window-close-active-preparation", evaluator=_serial_evaluator())

    try:
        controller.set_desired_identity(identity)
        assert session.warm_entered.wait(timeout=2.0)

        assert window.close() is False
        _drain_preparation(controller)
        window.fit_runtime_preparation_owner.poll_preparation()
        qt_app.processEvents()

        snapshot = controller.snapshot()
        assert snapshot.state is FittingRuntimeReadinessState.CLOSED
        assert snapshot.session is None
        assert session.close_calls == [True]
        assert snapshot.ledger.close_requests >= 2
    finally:
        if window.isVisible():
            window.close()
