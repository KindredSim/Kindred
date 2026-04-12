from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest


class _FinalPhaseFatalEvaluator:
    def __init__(self, *, t_axis, state):
        self._t_axis = np.asarray(t_axis, dtype=float).reshape(-1)
        self._state = state

    def _kindred_clone_fitting_evaluator_lane(self):
        with self._state["lock"]:
            self._state["clone_count"] += 1
            phase_counts = self._state.setdefault("phase_clone_counts", {})
            phase = str(self._state["phase"])
            phase_counts[phase] = int(phase_counts.get(phase, 0)) + 1
        return type(self)(t_axis=self._t_axis, state=self._state)

    def evaluate_series(self, params):
        from kindred.core.exceptions import FitSimulationError

        value = float(dict(params).get("init:A", 0.0))
        if self._state["phase"] == "final":
            with self._state["lock"]:
                self._state.setdefault("final_values", []).append(value)
        if self._state["phase"] == "final" and value == 10.0:
            raise FitSimulationError("final fatal lane failure", details={"fatal": True})
        return {
            "t": self._t_axis.copy(),
            "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
        }


class _LaneTrackingEvaluator:
    def __init__(self, *, t_axis, state, is_lane=False):
        self._t_axis = np.asarray(t_axis, dtype=float).reshape(-1)
        self._state = state
        self._is_lane = bool(is_lane)

    def _kindred_clone_fitting_evaluator_lane(self):
        with self._state["lock"]:
            self._state["clone_count"] += 1
        return type(self)(t_axis=self._t_axis, state=self._state, is_lane=True)

    def evaluate_series(self, params):
        return self.evaluate_series_with_parameter_origins(params, {})

    def evaluate_series_with_parameter_origins(self, params, origins=None, *, failed_params=None):
        if not self._is_lane:
            with self._state["lock"]:
                self._state["base_calls"] += 1
            value = float(dict(params).get("init:A", 0.0))
            return {
                "t": self._t_axis.copy(),
                "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
            }

        with self._state["lock"]:
            self._state["active"] += 1
            self._state["max_active"] = max(self._state["max_active"], self._state["active"])
        try:
            self._state["barrier"].wait(timeout=2.0)
            value = float(dict(params).get("init:A", 0.0))
            return {
                "t": self._t_axis.copy(),
                "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
            }
        finally:
            with self._state["lock"]:
                self._state["active"] -= 1


class _FatalObjectiveEvaluator:
    def __init__(self, *, t_axis, state):
        self._t_axis = np.asarray(t_axis, dtype=float).reshape(-1)
        self._state = state

    def _kindred_clone_fitting_evaluator_lane(self):
        with self._state["lock"]:
            self._state["clone_count"] += 1
        return type(self)(t_axis=self._t_axis, state=self._state)

    def evaluate_series(self, params):
        from kindred.core.exceptions import FitSimulationError

        value = float(dict(params).get("init:A", 0.0))
        with self._state["lock"]:
            self._state["values"].append(value)
        if value == 10.0:
            raise FitSimulationError("objective fatal lane failure", details={"fatal": True})
        return {
            "t": self._t_axis.copy(),
            "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
        }


class _CancellableLaneTrackingEvaluator:
    def __init__(self, *, t_axis, state, is_lane=False):
        self._t_axis = np.asarray(t_axis, dtype=float).reshape(-1)
        self._state = state
        self._is_lane = bool(is_lane)
        self._cancellation_check = None

    def _kindred_clone_fitting_evaluator_lane(self):
        with self._state["lock"]:
            self._state["clone_count"] += 1
        return type(self)(t_axis=self._t_axis, state=self._state, is_lane=True)

    def _kindred_set_fitting_cancellation_check(self, cancellation_check):
        self._cancellation_check = cancellation_check
        return self

    def evaluate_series(self, params):
        if not self._is_lane:
            with self._state["lock"]:
                self._state["base_calls"] += 1
            value = float(dict(params).get("init:A", 0.0))
            return {
                "t": self._t_axis.copy(),
                "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
            }

        with self._state["lock"]:
            self._state["active"] += 1
            self._state["max_active"] = max(self._state["max_active"], self._state["active"])
            if self._state["active"] == self._state["expected_active"]:
                self._state["all_active"].set()
        try:
            self._state["release"].wait(timeout=2.0)
            value = float(dict(params).get("init:A", 0.0))
            return {
                "t": self._t_axis.copy(),
                "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
            }
        finally:
            with self._state["lock"]:
                self._state["active"] -= 1


class _EvaluateOnlyNoClone:
    def __init__(self, *, t_axis, state):
        self._t_axis = np.asarray(t_axis, dtype=float).reshape(-1)
        self._state = state

    def evaluate_series(self, params):
        with self._state["lock"]:
            self._state["base_calls"] += 1
            self._state["active"] += 1
            self._state["max_active"] = max(self._state["max_active"], self._state["active"])
        try:
            value = float(dict(params).get("init:A", 0.0))
            return {
                "t": self._t_axis.copy(),
                "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
            }
        finally:
            with self._state["lock"]:
                self._state["active"] -= 1


class _SelfCloningEvaluator:
    def __init__(self, *, t_axis, state):
        self._t_axis = np.asarray(t_axis, dtype=float).reshape(-1)
        self._state = state

    def _kindred_clone_fitting_evaluator_lane(self):
        with self._state["lock"]:
            self._state["clone_count"] += 1
        return self

    def evaluate_series(self, params):
        with self._state["lock"]:
            self._state["base_calls"] += 1
            self._state["active"] += 1
            self._state["max_active"] = max(self._state["max_active"], self._state["active"])
        try:
            value = float(dict(params).get("init:A", 0.0))
            return {
                "t": self._t_axis.copy(),
                "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
            }
        finally:
            with self._state["lock"]:
                self._state["active"] -= 1


def _lane_state(parties: int = 2):
    return {
        "lock": threading.Lock(),
        "barrier": threading.Barrier(parties),
        "clone_count": 0,
        "base_calls": 0,
        "active": 0,
        "max_active": 0,
    }


def _payload(dataset_id: str, y_values) -> object:
    from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec

    y = np.asarray(y_values, dtype=float).reshape(1, -1)
    return FitDatasetSpec(
        dataset_id=str(dataset_id),
        t_exp=np.linspace(0.0, 1.0, y.shape[1]),
        species_list=["A"],
        y_matrix=y,
        point_count=int(y.size),
        x_name="t",
        x_obs=None,
        x_mode="auto",
        target_weights={},
    )


def test_global_fit_objective_parallel_lanes_preserve_residual_order_and_isolation() -> None:
    from kindred.core.analysis.global_fitting import _GlobalFitObjective, _build_parameter_layout
    from kindred.core.objective import ObjectiveContext

    payloads = [_payload("ds1", [0.0, 0.0]), _payload("ds2", [0.0, 0.0])]
    layout = _build_parameter_layout(
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_variable_params={},
        bounds=None,
        log10_params=None,
    )
    state = _lane_state()
    objective = _GlobalFitObjective(
        fit_evaluator=_LaneTrackingEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        weights={"ds1": 1.0, "ds2": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    residuals = objective(layout.x0.copy())

    np.testing.assert_allclose(residuals, np.asarray([1.0, 1.0, 10.0, 10.0], dtype=float))
    assert state["clone_count"] == 2
    assert state["base_calls"] == 0
    assert state["max_active"] == 2


def test_global_fit_objective_parallel_lanes_with_serial_fitting_evaluator(monkeypatch) -> None:
    from kindred.core.analysis.global_fitting import _GlobalFitObjective, _build_parameter_layout
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
    import kindred.core.fitting_evaluation as fitting_evaluation
    from kindred.core.objective import ObjectiveContext
    from kindred.core.simulator.solvers import SimulationOutput

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=2,
        solver="LSODA",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    state = _lane_state()

    def fake_solve_request(request):
        with state["lock"]:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            state["barrier"].wait(timeout=2.0)
            t = np.linspace(float(request.t_span[0]), float(request.t_span[1]), 2)
            y0 = np.asarray(request.y0, dtype=float).reshape(-1)
            return SimulationOutput(
                t=t,
                Y=np.vstack([np.full_like(t, y0[0]), np.full_like(t, y0[1])]),
                provenance={},
            )
        finally:
            with state["lock"]:
                state["active"] -= 1

    monkeypatch.setattr(fitting_evaluation, "_solve_request", fake_solve_request)

    payloads = [_payload("ds1", [0.0, 0.0]), _payload("ds2", [0.0, 0.0])]
    layout = _build_parameter_layout(
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_variable_params={},
        bounds=None,
        log10_params=None,
    )
    objective = _GlobalFitObjective(
        fit_evaluator=SerialFittingEvaluator(context),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        weights={"ds1": 1.0, "ds2": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    residuals = objective(layout.x0.copy())

    np.testing.assert_allclose(residuals, np.asarray([1.0, 1.0, 10.0, 10.0], dtype=float))
    assert state["max_active"] == 2


def test_global_fit_objective_parallel_fatal_lane_raises_original_error() -> None:
    from kindred.core.analysis.global_fitting import _GlobalFitObjective, _build_parameter_layout
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.objective import ObjectiveContext

    payloads = [
        _payload("ds1", [0.0, 0.0]),
        _payload("ds2", [0.0, 0.0]),
        _payload("ds3", [0.0, 0.0]),
        _payload("ds4", [0.0, 0.0]),
        _payload("ds5", [0.0, 0.0]),
    ]
    layout = _build_parameter_layout(
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_variable_params={},
        bounds=None,
        log10_params=None,
    )
    state = {"lock": threading.Lock(), "clone_count": 0, "values": []}
    objective = _GlobalFitObjective(
        fit_evaluator=_FatalObjectiveEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={
            "ds1": {"init:A": 1.0},
            "ds2": {"init:A": 10.0},
            "ds3": {"init:A": 3.0},
            "ds4": {"init:A": 4.0},
            "ds5": {"init:A": 5.0},
        },
        weights={payload.dataset_id: 1.0 for payload in payloads},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    with pytest.raises(FitSimulationError, match="objective fatal lane failure"):
        objective(layout.x0.copy())

    assert state["clone_count"] >= 2
    assert 10.0 in state["values"]


def test_global_fit_objective_cancel_signals_bounded_active_lane_set() -> None:
    from kindred.core.analysis.global_fitting import _GlobalFitObjective, _build_parameter_layout
    from kindred.core.exceptions import FittingCancelled
    from kindred.core.objective import ObjectiveContext

    payloads = [
        _payload("ds1", [0.0, 0.0]),
        _payload("ds2", [0.0, 0.0]),
        _payload("ds3", [0.0, 0.0]),
        _payload("ds4", [0.0, 0.0]),
        _payload("ds5", [0.0, 0.0]),
        _payload("ds6", [0.0, 0.0]),
    ]
    layout = _build_parameter_layout(
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_variable_params={},
        bounds=None,
        log10_params=None,
    )
    state = {
        "lock": threading.Lock(),
        "clone_count": 0,
        "base_calls": 0,
        "active": 0,
        "max_active": 0,
        "expected_active": 4,
        "all_active": threading.Event(),
        "release": threading.Event(),
    }
    cancelled = {"value": False}
    progress_calls = []
    result_box = {}

    objective = _GlobalFitObjective(
        fit_evaluator=_CancellableLaneTrackingEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={
            "ds1": {"init:A": 1.0},
            "ds2": {"init:A": 2.0},
            "ds3": {"init:A": 3.0},
            "ds4": {"init:A": 4.0},
            "ds5": {"init:A": 5.0},
            "ds6": {"init:A": 6.0},
        },
        weights={payload.dataset_id: 1.0 for payload in payloads},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=lambda *args: progress_calls.append(args),
        cancellation_check=lambda: bool(cancelled["value"]),
    )

    def _run_objective():
        try:
            result_box["residuals"] = objective(layout.x0.copy())
        except BaseException as exc:
            result_box["error"] = exc

    thread = threading.Thread(target=_run_objective)
    thread.start()
    assert state["all_active"].wait(timeout=2.0)

    cancelled["value"] = True
    state["release"].set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert isinstance(result_box.get("error"), FittingCancelled)
    assert "residuals" not in result_box
    assert progress_calls == []
    assert state["max_active"] == 4
    assert state["clone_count"] == 4


def test_global_fit_objective_uses_serial_path_for_custom_lanes_when_cancellable() -> None:
    from kindred.core.analysis.global_fitting import _GlobalFitObjective, _build_parameter_layout
    from kindred.core.objective import ObjectiveContext

    payloads = [_payload("ds1", [0.0, 0.0]), _payload("ds2", [0.0, 0.0])]
    layout = _build_parameter_layout(
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_variable_params={},
        bounds=None,
        log10_params=None,
    )
    state = _lane_state()
    objective = _GlobalFitObjective(
        fit_evaluator=_LaneTrackingEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        weights={"ds1": 1.0, "ds2": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=lambda: False,
    )

    residuals = objective(layout.x0.copy())

    np.testing.assert_allclose(residuals, np.asarray([1.0, 1.0, 10.0, 10.0], dtype=float))
    assert state["clone_count"] == 0
    assert state["base_calls"] == 2


def test_fit_global_least_squares_uses_parallel_dataset_lanes(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    state = _lane_state()

    def fake_least_squares(func, x0, **kwargs):
        residuals = np.asarray(func(np.asarray(x0, dtype=float)), dtype=float)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            success=True,
            message="ok",
            nfev=1,
            fun=residuals,
            jac=np.eye(len(residuals), len(np.asarray(x0, dtype=float))),
        )

    def fake_de(*_args, **_kwargs):
        raise AssertionError("DE should not be used for this test")

    monkeypatch.setattr(fitting_optimization, "load_scipy_optimize", lambda: (fake_least_squares, fake_de))

    result = global_fitting.fit_global(
        _LaneTrackingEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        [
            {"id": "ds1", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
            {"id": "ds2", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
        ],
        {"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        method="trf",
        max_nfev=1,
    )

    assert result.success is True
    assert state["clone_count"] >= 2
    assert state["base_calls"] == 0
    assert state["max_active"] == 2
    assert [info.dataset_id for info in result.dataset_info] == ["ds1", "ds2"]
    np.testing.assert_allclose(result.model_series["ds1"]["A"], np.asarray([1.0, 1.0]))
    np.testing.assert_allclose(result.model_series["ds2"]["A"], np.asarray([10.0, 10.0]))
    np.testing.assert_allclose(result.residual_series["ds1"]["A"], np.asarray([1.0, 1.0]))
    np.testing.assert_allclose(result.residual_series["ds2"]["A"], np.asarray([10.0, 10.0]))


def test_fit_global_de_uses_same_parallel_objective_without_de_workers(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    state = _lane_state()
    captured = {}

    def fake_least_squares(*_args, **_kwargs):
        raise AssertionError("least_squares should not be used for this test")

    def fake_de(func, bounds, **kwargs):
        captured["workers"] = kwargs.get("workers")
        x = np.asarray([(float(lo) + float(hi)) / 2.0 for lo, hi in bounds], dtype=float)
        _ = func(x)
        return SimpleNamespace(x=x, success=True, message="ok", nfev=1)

    monkeypatch.setattr(fitting_optimization, "load_scipy_optimize", lambda: (fake_least_squares, fake_de))

    result = global_fitting.fit_global(
        _LaneTrackingEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        [
            {"id": "ds1", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
            {"id": "ds2", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
        ],
        {"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        bounds={"k1": (0.1, 2.0)},
        method="de",
        max_nfev=1,
    )

    assert result.success is True
    assert captured["workers"] == 1
    assert state["clone_count"] >= 2
    assert state["base_calls"] == 0
    assert state["max_active"] == 2
    assert [info.dataset_id for info in result.dataset_info] == ["ds1", "ds2"]
    np.testing.assert_allclose(result.model_series["ds1"]["A"], np.asarray([1.0, 1.0]))
    np.testing.assert_allclose(result.model_series["ds2"]["A"], np.asarray([10.0, 10.0]))
    np.testing.assert_allclose(result.residual_series["ds1"]["A"], np.asarray([1.0, 1.0]))
    np.testing.assert_allclose(result.residual_series["ds2"]["A"], np.asarray([10.0, 10.0]))


def test_fit_global_evaluate_series_only_evaluator_stays_serial_for_public_surface(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    state = _lane_state()

    def fake_least_squares(func, x0, **kwargs):
        residuals = np.asarray(func(np.asarray(x0, dtype=float)), dtype=float)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            success=True,
            message="ok",
            nfev=1,
            fun=residuals,
            jac=np.eye(len(residuals), len(np.asarray(x0, dtype=float))),
        )

    def fake_de(*_args, **_kwargs):
        raise AssertionError("DE should not be used for this test")

    monkeypatch.setattr(fitting_optimization, "load_scipy_optimize", lambda: (fake_least_squares, fake_de))

    result = global_fitting.fit_global(
        _EvaluateOnlyNoClone(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        [
            {"id": "ds1", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
            {"id": "ds2", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
        ],
        {"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        method="trf",
        max_nfev=1,
    )

    assert result.success is True
    assert state["clone_count"] == 0
    assert state["base_calls"] >= 2
    assert state["max_active"] == 1
    np.testing.assert_allclose(result.model_series["ds1"]["A"], np.asarray([1.0, 1.0]))
    np.testing.assert_allclose(result.model_series["ds2"]["A"], np.asarray([10.0, 10.0]))


def test_fit_global_self_cloning_evaluator_falls_back_to_serial(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    state = _lane_state()

    def fake_least_squares(func, x0, **kwargs):
        residuals = np.asarray(func(np.asarray(x0, dtype=float)), dtype=float)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            success=True,
            message="ok",
            nfev=1,
            fun=residuals,
            jac=np.eye(len(residuals), len(np.asarray(x0, dtype=float))),
        )

    def fake_de(*_args, **_kwargs):
        raise AssertionError("DE should not be used for this test")

    monkeypatch.setattr(fitting_optimization, "load_scipy_optimize", lambda: (fake_least_squares, fake_de))

    result = global_fitting.fit_global(
        _SelfCloningEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        [
            {"id": "ds1", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
            {"id": "ds2", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
        ],
        {"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        method="trf",
        max_nfev=1,
    )

    assert result.success is True
    assert state["clone_count"] >= 2
    assert state["base_calls"] >= 2
    assert state["max_active"] == 1
    np.testing.assert_allclose(result.model_series["ds1"]["A"], np.asarray([1.0, 1.0]))
    np.testing.assert_allclose(result.model_series["ds2"]["A"], np.asarray([10.0, 10.0]))


def test_fit_global_final_assembly_records_parallel_fatal_lane_errors(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    state = {
        "lock": threading.Lock(),
        "clone_count": 0,
        "phase": "objective",
    }

    def fake_least_squares(func, x0, **kwargs):
        residuals = np.asarray(func(np.asarray(x0, dtype=float)), dtype=float)
        state["phase"] = "final"
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            success=True,
            message="ok",
            nfev=1,
            fun=residuals,
            jac=np.eye(len(residuals), len(np.asarray(x0, dtype=float))),
        )

    def fake_de(*_args, **_kwargs):
        raise AssertionError("DE should not be used for this test")

    monkeypatch.setattr(fitting_optimization, "load_scipy_optimize", lambda: (fake_least_squares, fake_de))

    result = global_fitting.fit_global(
        _FinalPhaseFatalEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        [
            {"id": "ds1", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
            {"id": "ds2", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
            {"id": "ds3", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
            {"id": "ds4", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
            {"id": "ds5", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)},
        ],
        {"k1": 1.0},
        dataset_params={
            "ds1": {"init:A": 1.0},
            "ds2": {"init:A": 10.0},
            "ds3": {"init:A": 3.0},
            "ds4": {"init:A": 4.0},
            "ds5": {"init:A": 5.0},
        },
        method="trf",
        max_nfev=1,
    )

    assert result.success is False
    assert "final fatal lane failure" in result.dataset_errors["ds2"]
    assert state["phase_clone_counts"]["final"] >= 5
    assert sorted(set(state["final_values"])) == [1.0, 3.0, 4.0, 5.0, 10.0]
    assert "ds5" in result.model_series
