from __future__ import annotations

import multiprocessing
import os
import queue
import time
from typing import Any, Mapping

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _process_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context("spawn")


def _require_spawn_queue_support() -> multiprocessing.context.BaseContext:
    mp_context = _process_context()
    try:
        probe_queue: multiprocessing.Queue = mp_context.Queue(maxsize=1)
    except (OSError, PermissionError) as exc:
        pytest.skip(f"multiprocessing spawn Queue unavailable in this environment: {exc}")
    else:
        probe_queue.close()
        probe_queue.join_thread()
    return mp_context


def _basic_context():
    from kindred.core.fitting_evaluation import prepare_fitting_execution_context

    return prepare_fitting_execution_context(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        param_names=["k"],
        t_end=1.0,
        num_points=5,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )


def _basic_evaluator():
    from kindred.core.fitting_evaluation import SerialFittingEvaluator

    return SerialFittingEvaluator(_basic_context())


def _spawn_roundtrip_child(payload: Mapping[str, Any], requests: list[dict[str, float]], output_queue) -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator

    evaluator = SerialFittingEvaluator.from_process_payload(payload)
    evaluator._ensure_prepared()
    prepared_id = id(evaluator._prepared_run)
    outputs = []
    for params in requests:
        result = evaluator.evaluate_series(params)
        outputs.append(
            {
                "t_size": int(np.asarray(result.t, dtype=float).size),
                "prepared_id": int(id(evaluator._prepared_run)),
                "b_last": float(np.asarray(result.species["B"], dtype=float)[-1]),
            }
        )
    output_queue.put(
        {
            "pid": int(os.getpid()),
            "prepared_id": int(prepared_id),
            "outputs": outputs,
        }
    )


def _accepted_then_hang_lane_child(_payload, input_queue, output_queue, owner_epoch: int) -> None:
    output_queue.put(
        {
            "kind": "ready",
            "owner_epoch": int(owner_epoch),
            "pid": int(os.getpid()),
        }
    )
    request = input_queue.get()
    output_queue.put(
        {
            "kind": "accepted",
            "owner_epoch": int(owner_epoch),
            "request_id": int(request["request_id"]),
        }
    )
    while True:
        time.sleep(0.05)


def _hang_or_result_lane_child(_payload, input_queue, output_queue, owner_epoch: int) -> None:
    output_queue.put(
        {
            "kind": "ready",
            "owner_epoch": int(owner_epoch),
            "pid": int(os.getpid()),
        }
    )
    while True:
        request = input_queue.get()
        if str(request.get("kind") or "") == "close":
            return
        request_id = int(request["request_id"])
        output_queue.put(
            {
                "kind": "accepted",
                "owner_epoch": int(owner_epoch),
                "request_id": request_id,
            }
        )
        params = dict(request.get("params") or {})
        if str(params.get("mode") or "") == "hang":
            while True:
                time.sleep(0.05)
        t = np.linspace(0.0, 1.0, 3)
        output_queue.put(
            {
                "kind": "result",
                "owner_epoch": int(owner_epoch),
                "request_id": request_id,
                "payload": {"t": t, "species": {"A": np.ones_like(t)}},
            }
        )


def test_spawn_child_reconstructs_serial_evaluator_once_and_reuses_prepared_state() -> None:
    mp_context = _require_spawn_queue_support()
    evaluator = _basic_evaluator()
    payload = evaluator.to_process_payload()
    output_queue = mp_context.Queue(maxsize=1)
    proc = mp_context.Process(
        target=_spawn_roundtrip_child,
        args=(
            payload,
            [
                {"k": 0.2, "init:A": 1.0},
                {"k": 0.2, "init:A": 2.0},
            ],
            output_queue,
        ),
    )

    proc.start()
    try:
        message = output_queue.get(timeout=10.0)
    except queue.Empty:
        proc.terminate()
        proc.join(timeout=1.0)
        pytest.fail("spawn child did not return SerialFittingEvaluator roundtrip result")
    finally:
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
        output_queue.close()
        output_queue.join_thread()

    assert proc.exitcode == 0
    outputs = message["outputs"]
    assert len(outputs) == 2
    assert outputs[0]["t_size"] == 5
    assert outputs[0]["prepared_id"] == message["prepared_id"]
    assert outputs[1]["prepared_id"] == message["prepared_id"]
    assert outputs[0]["b_last"] != pytest.approx(outputs[1]["b_last"])


def test_reply_gate_rejects_stale_epoch_and_request_id() -> None:
    from kindred.core.fitting_containment import _FittingLaneReplyGate

    gate = _FittingLaneReplyGate(owner_epoch=7)

    assert gate.is_current({"owner_epoch": 7, "request_id": 3}, request_id=3)
    assert not gate.is_current({"owner_epoch": 6, "request_id": 3}, request_id=3)
    assert not gate.is_current({"owner_epoch": 7, "request_id": 2}, request_id=3)
    assert not gate.is_current({"owner_epoch": 8, "request_id": 3}, request_id=3)


def test_warm_lane_times_out_after_accepted_request_and_kills_child() -> None:
    from kindred.core.fitting_containment import FittingLaneTimeout, WarmFittingEvaluatorLane

    mp_context = _require_spawn_queue_support()
    lane = WarmFittingEvaluatorLane(
        {},
        request_timeout_s=0.2,
        ready_timeout_s=10.0,
        accept_timeout_s=2.0,
        mp_context=mp_context,
        child_target=_accepted_then_hang_lane_child,
    )

    with pytest.raises(FittingLaneTimeout) as exc_info:
        lane.evaluate_series_with_parameter_origins(
            {"k": 0.2},
            {"k": "optimizer_shared"},
            failed_params={"k": 0.2},
        )

    assert exc_info.value.details["fatal"] is False
    assert exc_info.value.details["failure"]["kind"] == "timeout"
    assert lane._process is None
    lane.close()


def test_warm_lane_restarts_after_timeout_for_later_request() -> None:
    from kindred.core.fitting_containment import FittingLaneTimeout, WarmFittingEvaluatorLane

    mp_context = _require_spawn_queue_support()
    lane = WarmFittingEvaluatorLane(
        {},
        request_timeout_s=0.2,
        ready_timeout_s=10.0,
        accept_timeout_s=2.0,
        mp_context=mp_context,
        child_target=_hang_or_result_lane_child,
    )

    with pytest.raises(FittingLaneTimeout):
        lane.evaluate_series_with_parameter_origins({"mode": "hang"}, {}, failed_params={})

    result = lane.evaluate_series_with_parameter_origins({"mode": "ok"}, {}, failed_params={})

    assert np.asarray(result.species["A"], dtype=float).tolist() == [1.0, 1.0, 1.0]
    lane.close()


def test_warm_lane_reconstruction_failure_is_fatal_with_child_diagnostics() -> None:
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.fitting_containment import WarmFittingEvaluatorLane

    mp_context = _require_spawn_queue_support()
    lane = WarmFittingEvaluatorLane(
        {"not": "a valid fitting payload"},
        request_timeout_s=0.2,
        ready_timeout_s=10.0,
        accept_timeout_s=2.0,
        mp_context=mp_context,
    )

    with pytest.raises(FitSimulationError) as exc_info:
        lane.evaluate_series_with_parameter_origins({"k": 0.2}, {}, failed_params={"k": 0.2})

    err = exc_info.value
    assert err.details["fatal"] is True
    assert err.details["failure"]["kind"] == "fitting_containment_prewarm"
    assert "missing required keys" in err.details["failure"]["message"].lower()
    lane.close()


def test_warm_lane_cancellation_kills_in_progress_request_without_penalty_mapping() -> None:
    from kindred.core.exceptions import FittingCancelled
    from kindred.core.fitting_containment import WarmFittingEvaluatorLane

    mp_context = _require_spawn_queue_support()
    lane = WarmFittingEvaluatorLane(
        {},
        request_timeout_s=5.0,
        ready_timeout_s=10.0,
        accept_timeout_s=2.0,
        mp_context=mp_context,
        child_target=_accepted_then_hang_lane_child,
    )
    started = time.monotonic()

    def _cancel_after_request_is_active() -> bool:
        return (time.monotonic() - started) >= 0.1

    with pytest.raises(FittingCancelled):
        lane.evaluate_series_with_parameter_origins(
            {"k": 0.2},
            {"k": "optimizer_shared"},
            failed_params={"k": 0.2},
            cancellation_check=_cancel_after_request_is_active,
        )

    assert lane._process is None
    lane.close()


def test_contained_evaluator_timeout_maps_to_nonfatal_fit_simulation_error() -> None:
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.fitting_containment import ContainedSerialFittingEvaluator, FittingLaneTimeout

    class _TimeoutLane:
        close_calls = 0

        def evaluate_series_with_parameter_origins(self, *_args, **_kwargs):
            raise FittingLaneTimeout(0.25)

        def close(self) -> None:
            self.close_calls += 1

    lane = _TimeoutLane()
    evaluator = ContainedSerialFittingEvaluator(
        _basic_evaluator(),
        lane_factory=lambda _payload, **_kwargs: lane,
        request_timeout_s=0.25,
    )

    with pytest.raises(FitSimulationError) as exc_info:
        evaluator.evaluate_series_with_parameter_origins(
            {"k": 0.2, "init:A": 1.0},
            {"k": "optimizer_shared", "init:A": "optimizer_dataset"},
            failed_params={"k": 0.2, "ds::init:A": 1.0},
        )

    err = exc_info.value
    assert err.details["fatal"] is False
    assert err.details["failure"]["kind"] == "timeout"
    assert err.details["failure"]["details"]["active_solve_timeout_s"] == pytest.approx(0.25)
    assert err.failed_params == {"k": 0.2, "ds::init:A": 1.0}

    evaluator.close()
    assert lane.close_calls == 1


def test_contained_evaluator_lane_protocol_error_is_fatal() -> None:
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.fitting_containment import ContainedSerialFittingEvaluator, FittingLaneProtocolError

    class _ProtocolFailureLane:
        def evaluate_series_with_parameter_origins(self, *_args, **_kwargs):
            raise FittingLaneProtocolError("bad reply")

        def close(self) -> None:
            return None

    evaluator = ContainedSerialFittingEvaluator(
        _basic_evaluator(),
        lane_factory=lambda _payload, **_kwargs: _ProtocolFailureLane(),
    )

    with pytest.raises(FitSimulationError) as exc_info:
        evaluator.evaluate_series({"k": 0.2, "init:A": 1.0})

    assert exc_info.value.details["fatal"] is True
    assert exc_info.value.details["failure"]["kind"] == "fitting_containment_protocol"


def test_fit_global_wraps_exact_serial_evaluator_with_runtime_session_by_default(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_runtime_session import FittingRuntimeSession

    wrapped = {"count": 0}
    class _FakeRuntimeSession:
        def __init__(self, evaluator):
            self._evaluator = evaluator

        def begin_run(self) -> None:
            return None

        def evaluator(self, *, cancellation_check=None):
            return self._evaluator

        def close(self, *, kill: bool = False) -> None:
            return None

    def _spy_from_serial(cls, evaluator, *args, **kwargs):
        wrapped["count"] += 1
        return _FakeRuntimeSession(evaluator)

    monkeypatch.setattr(FittingRuntimeSession, "from_serial_evaluator", classmethod(_spy_from_serial))
    monkeypatch.setattr(
        global_fitting,
        "fit_parameters",
        lambda objective_func, initial_params, **_kwargs: _fit_result_from_objective(objective_func, initial_params),
    )

    t = np.linspace(0.0, 1.0, 5)
    result = global_fitting.fit_global(
        _basic_evaluator(),
        datasets=[{"id": "ds1", "t": t, "y": np.zeros_like(t), "species": "B"}],
        shared_params={"k": 0.2},
        method="trf",
        max_nfev=1,
    )

    assert wrapped["count"] == 1
    assert result.completion.status in {"ok", "warn"}


def _fit_result_from_objective(objective_func, initial_params):
    from kindred.core.fitting_optimization import FitResult

    keys = list(initial_params)
    x0 = np.asarray([float(initial_params[key]) for key in keys], dtype=float)
    residuals = np.asarray(objective_func(x0), dtype=float).reshape(-1)
    return FitResult(
        success=True,
        parameters={key: float(value) for key, value in zip(keys, x0)},
        uncertainties=None,
        chi_squared=float(np.mean(residuals**2)) if residuals.size else 0.0,
        r_squared=1.0,
        residuals=residuals,
        nfev=1,
        message="fake optimizer success",
        covariance=None,
    )


def test_fit_global_candidate_timeout_uses_penalty_and_final_replay_keeps_other_dataset(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_containment import FittingLaneTimeout
    from kindred.core.fitting_runtime_session import FittingRuntimeSession

    class _TimeoutDatasetReplayLane:
        def __init__(self):
            self.calls = 0

        def evaluate_series_with_parameter_origins(self, params, *_args, **_kwargs):
            self.calls += 1
            if self.calls in {1, 3}:
                raise FittingLaneTimeout(0.2)
            t = np.linspace(0.0, 1.0, 4)
            value = float(dict(params).get("k", 0.0))
            return {"t": t, "species": {"A": np.full_like(t, value)}}

        def close(self) -> None:
            return None

    lane = _TimeoutDatasetReplayLane()

    class _RuntimeEvaluator:
        def evaluate_fitting_runtime_batch(self, requests, *, cancellation_check=None):
            out = []
            for request in requests:
                try:
                    out.append(
                        lane.evaluate_series_with_parameter_origins(
                            request.params,
                            request.origins,
                            failed_params=request.failed_params,
                        )
                    )
                except BaseException as exc:  # noqa: BLE001 - objective owns penalty/final policy
                    out.append(exc)
            return out

        def evaluate_series(self, params):
            return lane.evaluate_series_with_parameter_origins(params)

    class _RuntimeSession:
        def begin_run(self):
            return 1

        def evaluator(self, *, cancellation_check=None):
            return _RuntimeEvaluator()

        def close(self, *, kill: bool = False):
            return None

    monkeypatch.setattr(
        FittingRuntimeSession,
        "from_serial_evaluator",
        classmethod(lambda cls, *_args, **_kwargs: _RuntimeSession()),
    )
    monkeypatch.setattr(
        global_fitting,
        "fit_parameters",
        lambda objective_func, initial_params, **_kwargs: _fit_result_from_objective(objective_func, initial_params),
    )

    t = np.linspace(0.0, 1.0, 4)
    result = global_fitting.fit_global(
        _basic_evaluator(),
        datasets=[
            {"id": "ds-timeout", "t": t.copy(), "y": np.zeros_like(t), "species": "A"},
            {"id": "ds-ok", "t": t.copy(), "y": np.zeros_like(t), "species": "A"},
        ],
        shared_params={"k": 0.2},
        method="trf",
        max_nfev=1,
    )

    assert result.completion.status == "fail"
    assert set(result.completion.dataset_failures) == {"ds-timeout"}
    assert result.completion.dataset_failures["ds-timeout"].failure["kind"] == "timeout"
    assert "ds-ok" in result.model_series
    assert "ds-timeout" not in result.model_series
