from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_global_fitting_runtime_session_reuses_warm_lanes_and_records_ledger() -> None:
    from kindred.core.fitting_runtime_session import (
        FittingRuntimeLedger,
        FittingRuntimeRequest,
        FittingRuntimeSession,
    )

    events: list[str] = []

    class _FakeLane:
        def __init__(self, process_payload, **_kwargs):
            self.process_payload = dict(process_payload)
            self.closed = False
            events.append(f"lane:create:{self.process_payload['identity']}")

        def warm(self, *, cancellation_check=None) -> None:
            assert cancellation_check is None or cancellation_check() is False
            events.append("lane:warm")

        def evaluate_series_with_parameter_origins(self, params, origins=None, *, failed_params=None, cancellation_check=None):
            assert self.closed is False
            assert cancellation_check is None or cancellation_check() is False
            events.append(f"lane:evaluate:{params['dataset']}")
            t = np.asarray([0.0, 1.0], dtype=float)
            return {"t": t, "species": {"A": np.asarray([float(params["value"]), 2.0], dtype=float)}}

        def close(self, *, kill: bool = False) -> None:
            self.closed = True
            events.append(f"lane:close:{kill}")

    ledger = FittingRuntimeLedger()
    session = FittingRuntimeSession(
        process_payload={"identity": "same-runtime"},
        max_lanes=2,
        lane_factory=_FakeLane,
        ledger=ledger,
    )

    session.warm(cancellation_check=lambda: False)
    first = session.evaluate_batch(
        [
            FittingRuntimeRequest(params={"dataset": "ds1", "value": 1.0}),
            FittingRuntimeRequest(params={"dataset": "ds2", "value": 3.0}),
        ],
        cancellation_check=lambda: False,
    )
    second = session.evaluate_batch(
        [FittingRuntimeRequest(params={"dataset": "ds3", "value": 5.0})],
        cancellation_check=lambda: False,
    )

    assert [float(result.species["A"][0]) for result in first] == [1.0, 3.0]
    assert [float(result.species["A"][0]) for result in second] == [5.0]
    assert ledger.snapshot()["payload_preparations"] == 1
    assert ledger.snapshot()["lane_creations"] == 2
    assert ledger.snapshot()["lane_warms"] == 2
    assert ledger.snapshot()["candidate_evaluations"] == 3
    assert events.count("lane:create:same-runtime") == 2
    assert events.count("lane:warm") == 2

    session.close()
    assert events[-2:] == ["lane:close:False", "lane:close:False"]


def test_global_fitting_runtime_session_cancel_closes_warm_lanes_with_kill() -> None:
    from kindred.core.fitting_runtime_session import FittingRuntimeLedger, FittingRuntimeSession

    closed: list[bool] = []

    class _FakeLane:
        def __init__(self, *_args, **_kwargs):
            return None

        def warm(self, *, cancellation_check=None) -> None:
            return None

        def close(self, *, kill: bool = False) -> None:
            closed.append(bool(kill))

    ledger = FittingRuntimeLedger()
    session = FittingRuntimeSession(
        process_payload={"identity": "cancel-runtime"},
        max_lanes=2,
        lane_factory=_FakeLane,
        ledger=ledger,
    )
    session.warm()

    session.cancel_run()

    assert closed == [True, True]
    assert ledger.snapshot()["run_cancellations"] == 1
    assert ledger.snapshot()["lane_kills"] == 2


def test_runtime_evaluator_raises_runtime_failures_in_single_evaluation() -> None:
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.fitting_containment import FittingLaneProtocolError
    from kindred.core.fitting_runtime_session import FittingRuntimeSession

    class _BrokenLane:
        def __init__(self, *_args, **_kwargs):
            return None

        def warm(self, *, cancellation_check=None) -> None:
            return None

        def evaluate_series_with_parameter_origins(self, *_args, **_kwargs):
            raise FittingLaneProtocolError("bad runtime reply")

        def close(self, *, kill: bool = False) -> None:
            return None

    session = FittingRuntimeSession(
        process_payload={"identity": "single-eval-error"},
        max_lanes=1,
        lane_factory=_BrokenLane,
    )

    with pytest.raises(FitSimulationError) as exc_info:
        session.evaluator(cancellation_check=lambda: False).evaluate_series({"k": 1.0})

    assert exc_info.value.details["failure"]["kind"] == "fitting_containment_protocol"


def test_runtime_session_invalidates_failed_warm_lane_before_next_evaluation() -> None:
    from kindred.core.fitting_containment import FittingLaneTimeout
    from kindred.core.fitting_runtime_session import (
        FittingRuntimeLedger,
        FittingRuntimeRequest,
        FittingRuntimeSession,
    )

    events: list[str] = []

    class _TimeoutLane:
        def __init__(self, *_args, **_kwargs):
            self.index = events.count("create") + 1
            events.append("create")

        def warm(self, *, cancellation_check=None) -> None:
            events.append(f"warm:{self.index}")

        def evaluate_series_with_parameter_origins(self, *_args, **_kwargs):
            events.append(f"evaluate:{self.index}")
            raise FittingLaneTimeout(1.0, failed_params={"k": float(self.index)})

        def close(self, *, kill: bool = False) -> None:
            events.append(f"close:{self.index}:{kill}")

    ledger = FittingRuntimeLedger()
    session = FittingRuntimeSession(
        process_payload={"identity": "timeout-runtime"},
        max_lanes=1,
        lane_factory=_TimeoutLane,
        ledger=ledger,
    )

    first = session.evaluate_batch([FittingRuntimeRequest(params={"k": 1.0})])[0]
    second = session.evaluate_batch([FittingRuntimeRequest(params={"k": 2.0})])[0]

    assert isinstance(first, FittingLaneTimeout)
    assert isinstance(second, FittingLaneTimeout)
    assert events == [
        "create",
        "warm:1",
        "evaluate:1",
        "close:1:True",
        "create",
        "warm:2",
        "evaluate:2",
        "close:2:True",
    ]
    assert ledger.snapshot()["lane_creations"] == 2
    assert ledger.snapshot()["lane_warms"] == 2


def test_runtime_session_replaces_invalidated_lane_within_same_batch_group() -> None:
    from kindred.core.fitting_containment import FittingLaneTimeout
    from kindred.core.fitting_runtime_session import (
        FittingRuntimeRequest,
        FittingRuntimeSession,
    )

    events: list[str] = []

    class _FlakyLane:
        def __init__(self, *_args, **_kwargs):
            self.index = events.count("create") + 1
            self.calls = 0
            events.append("create")

        def warm(self, *, cancellation_check=None) -> None:
            events.append(f"warm:{self.index}")

        def evaluate_series_with_parameter_origins(self, params, *_args, **_kwargs):
            self.calls += 1
            events.append(f"evaluate:{self.index}:{params['dataset']}")
            if self.index == 1:
                raise FittingLaneTimeout(1.0, failed_params={"dataset": float(params["dataset"])})
            t = np.asarray([0.0], dtype=float)
            return {"t": t, "species": {"A": np.asarray([float(params["dataset"])], dtype=float)}}

        def close(self, *, kill: bool = False) -> None:
            events.append(f"close:{self.index}:{kill}")

    session = FittingRuntimeSession(
        process_payload={"identity": "same-batch-runtime"},
        max_lanes=1,
        lane_factory=_FlakyLane,
    )

    results = session.evaluate_batch(
        [
            FittingRuntimeRequest(params={"dataset": 1.0}),
            FittingRuntimeRequest(params={"dataset": 2.0}),
        ]
    )

    assert isinstance(results[0], FittingLaneTimeout)
    assert float(results[1].species["A"][0]) == pytest.approx(2.0)
    assert events == [
        "create",
        "warm:1",
        "evaluate:1:1.0",
        "close:1:True",
        "create",
        "warm:2",
        "evaluate:2:2.0",
    ]


def test_runtime_session_candidate_ledger_is_deterministic_for_concurrent_lanes() -> None:
    from kindred.core.fitting_runtime_session import (
        FittingRuntimeLedger,
        FittingRuntimeRequest,
        FittingRuntimeSession,
    )

    class _SuccessLane:
        def __init__(self, *_args, **_kwargs):
            return None

        def warm(self, *, cancellation_check=None) -> None:
            return None

        def evaluate_series_with_parameter_origins(self, params, *_args, **_kwargs):
            t = np.asarray([0.0], dtype=float)
            return {"t": t, "species": {"A": np.asarray([float(params["dataset"])], dtype=float)}}

        def close(self, *, kill: bool = False) -> None:
            return None

    ledger = FittingRuntimeLedger()
    session = FittingRuntimeSession(
        process_payload={"identity": "ledger-runtime"},
        max_lanes=4,
        lane_factory=_SuccessLane,
        ledger=ledger,
    )

    requests = [FittingRuntimeRequest(params={"dataset": float(index)}) for index in range(40)]
    results = session.evaluate_batch(requests)

    assert len(results) == 40
    assert ledger.snapshot()["candidate_evaluations"] == 40


def test_runtime_session_payload_build_failure_preserves_containment_diagnostic() -> None:
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.fitting_runtime_session import FittingRuntimeSession

    class _UnserializableEvaluator:
        def to_process_payload(self):
            raise ValueError("cannot pickle closure")

    with pytest.raises(FitSimulationError) as exc_info:
        FittingRuntimeSession.from_serial_evaluator(
            _UnserializableEvaluator(),  # type: ignore[arg-type]
            max_lanes=1,
        )

    assert exc_info.value.details["failure"]["kind"] == "fitting_containment_payload"
    assert "Failed to build fitting containment payload" in str(exc_info.value)


def test_runtime_session_close_shields_lane_close_failures() -> None:
    from kindred.core.fitting_runtime_session import FittingRuntimeLedger, FittingRuntimeSession

    events: list[str] = []

    class _CloseLane:
        def __init__(self, *_args, **_kwargs):
            self.index = events.count("create") + 1
            events.append("create")

        def warm(self, *, cancellation_check=None) -> None:
            return None

        def close(self, *, kill: bool = False) -> None:
            events.append(f"close:{self.index}:{kill}")
            if self.index == 1:
                raise RuntimeError("close failed")

    ledger = FittingRuntimeLedger()
    session = FittingRuntimeSession(
        process_payload={"identity": "close-runtime"},
        max_lanes=2,
        lane_factory=_CloseLane,
        ledger=ledger,
    )
    session.warm()

    session.close(kill=True)

    assert events == ["create", "create", "close:1:True", "close:2:True"]
    assert ledger.snapshot()["lane_closes"] == 2
    assert ledger.snapshot()["lane_kills"] == 2


def test_global_fit_dataset_evaluation_uses_runtime_batch_boundary() -> None:
    from kindred.core.analysis.fit_dataset_payload import coerce_fit_dataset_specs
    from kindred.core.analysis.global_fitting import (
        _ObjectiveDatasetInput,
        _evaluate_dataset_simulations,
    )

    payloads = coerce_fit_dataset_specs(
        [
            {"id": "ds1", "t": np.asarray([0.0, 1.0]), "y": np.asarray([1.0, 2.0]), "species": "A"},
            {"id": "ds2", "t": np.asarray([0.0, 1.0]), "y": np.asarray([3.0, 4.0]), "species": "A"},
        ]
    )
    items = [
        _ObjectiveDatasetInput(
            index=index,
            payload=payload,
            full_params={"dataset": payload.dataset_id, "value": float(index + 1)},
            parameter_origins={},
            failed_param_snapshot={},
        )
        for index, payload in enumerate(payloads)
    ]
    calls: list[list[str]] = []

    class _BatchEvaluator:
        def evaluate_fitting_runtime_batch(self, requests, *, cancellation_check=None):
            assert cancellation_check is None or cancellation_check() is False
            calls.append([str(request.params["dataset"]) for request in requests])
            out = []
            for request in requests:
                t = np.asarray([0.0, 1.0], dtype=float)
                out.append({"t": t, "species": {"A": np.full_like(t, float(request.params["value"]))}})
            return out

    results = _evaluate_dataset_simulations(
        _BatchEvaluator(),
        items,
        cancellation_check=lambda: False,
    )

    assert calls == [["ds1", "ds2"]]
    assert [result.index for result in results] == [0, 1]
    assert [float(result.sim_species["A"][0]) for result in results] == [1.0, 2.0]


def test_global_fit_dataset_evaluation_preserves_runtime_protocol_failures_as_fatal() -> None:
    from kindred.core.analysis.fit_dataset_payload import coerce_fit_dataset_specs
    from kindred.core.analysis.global_fitting import (
        _ObjectiveDatasetInput,
        _evaluate_dataset_simulations,
        _dataset_evaluation_is_fatal,
    )
    from kindred.core.fitting_containment import FittingLaneProtocolError

    payload = coerce_fit_dataset_specs(
        [{"id": "ds1", "t": np.asarray([0.0, 1.0]), "y": np.asarray([1.0, 2.0]), "species": "A"}]
    )[0]
    item = _ObjectiveDatasetInput(
        index=0,
        payload=payload,
        full_params={"k": 1.0},
        parameter_origins={},
        failed_param_snapshot={"k": 1.0},
    )

    class _BrokenRuntimeEvaluator:
        def evaluate_fitting_runtime_batch(self, requests, *, cancellation_check=None):
            return [FittingLaneProtocolError("bad runtime reply")]

    result = _evaluate_dataset_simulations(
        _BrokenRuntimeEvaluator(),
        [item],
        cancellation_check=lambda: False,
    )[0]

    assert _dataset_evaluation_is_fatal(result) is True
    assert result.error.details["failure"]["kind"] == "fitting_containment_protocol"


def test_global_fit_dataset_evaluation_preserves_raised_runtime_warm_protocol_failure_as_fatal() -> None:
    from kindred.core.analysis.fit_dataset_payload import coerce_fit_dataset_specs
    from kindred.core.analysis.global_fitting import (
        _ObjectiveDatasetInput,
        _dataset_evaluation_is_fatal,
        _evaluate_dataset_simulations,
    )
    from kindred.core.fitting_containment import FittingLaneProtocolError
    from kindred.core.fitting_runtime_session import FittingRuntimeSession

    payloads = coerce_fit_dataset_specs(
        [
            {"id": "ds1", "t": np.asarray([0.0, 1.0]), "y": np.asarray([1.0, 2.0]), "species": "A"},
            {"id": "ds2", "t": np.asarray([0.0, 1.0]), "y": np.asarray([3.0, 4.0]), "species": "A"},
        ]
    )
    items = [
        _ObjectiveDatasetInput(
            index=index,
            payload=payload,
            full_params={"k": 1.0},
            parameter_origins={},
            failed_param_snapshot={"k": 1.0},
        )
        for index, payload in enumerate(payloads)
    ]

    class _BrokenWarmLane:
        def __init__(self, *_args, **_kwargs):
            return None

        def warm(self, *, cancellation_check=None) -> None:
            raise FittingLaneProtocolError("bad startup reply")

    session = FittingRuntimeSession(
        process_payload={"identity": "broken-startup"},
        max_lanes=2,
        lane_factory=_BrokenWarmLane,
    )

    results = _evaluate_dataset_simulations(
        session.evaluator(cancellation_check=lambda: False),
        items,
        cancellation_check=lambda: False,
    )

    assert [result.index for result in results] == [0, 1]
    assert all(_dataset_evaluation_is_fatal(result) for result in results)
    assert {result.error.details["failure"]["kind"] for result in results} == {"fitting_containment_protocol"}
