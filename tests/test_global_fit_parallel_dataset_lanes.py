from __future__ import annotations

import threading
import time
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


class _SlotRetentionTrackingEvaluator:
    def __init__(self, *, t_axis, state, lane_id=None):
        self._t_axis = np.asarray(t_axis, dtype=float).reshape(-1)
        self._state = state
        self._lane_id = lane_id

    def _kindred_clone_fitting_evaluator_lane(self):
        with self._state["lock"]:
            lane_id = int(self._state["clone_count"])
            self._state["clone_count"] += 1
        return type(self)(t_axis=self._t_axis, state=self._state, lane_id=lane_id)

    def evaluate_series(self, params):
        value = float(dict(params).get("init:A", 0.0))
        with self._state["lock"]:
            if self._lane_id is None:
                self._state["base_calls"] += 1
            else:
                self._state.setdefault("lane_calls", []).append((int(self._lane_id), value))
        return {
            "t": self._t_axis.copy(),
            "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
        }


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


class _PauseAwareLaneTrackingEvaluator:
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
        value = float(dict(params).get("init:A", 0.0))
        if not self._is_lane:
            with self._state["lock"]:
                self._state["base_calls"] += 1
                self._state["starts"].append(value)
            return {
                "t": self._t_axis.copy(),
                "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
            }

        with self._state["lock"]:
            self._state["starts"].append(value)
            self._state["active"] += 1
            self._state["max_active"] = max(self._state["max_active"], self._state["active"])
            if self._state["active"] == self._state["expected_active"]:
                self._state["all_active"].set()
        try:
            if value <= 4.0:
                self._state["release"].wait(timeout=2.0)
            return {
                "t": self._t_axis.copy(),
                "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
            }
        finally:
            with self._state["lock"]:
                self._state["active"] -= 1


class _FatalCooperativeLaneEvaluator:
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
        from kindred.core.exceptions import FitSimulationError, FittingCancelled

        value = float(dict(params).get("init:A", 0.0))
        if not self._is_lane:
            return {
                "t": self._t_axis.copy(),
                "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
            }

        if value == 1.0:
            self._state["fatal_entered"].set()
            self._state["all_siblings_active"].wait(timeout=2.0)
            raise FitSimulationError("objective fatal lane failure", details={"fatal": True})

        with self._state["lock"]:
            self._state["siblings_active"] += 1
            if self._state["siblings_active"] == 3:
                self._state["all_siblings_active"].set()
        while not self._state["normal_release"].is_set():
            if self._cancellation_check is not None and self._cancellation_check():
                with self._state["lock"]:
                    self._state["cooperative_stops"] += 1
                raise FittingCancelled()
            self._state["poll"].wait(timeout=0.001)
        return {
            "t": self._t_axis.copy(),
            "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
        }


class _FatalWhilePauseCallbackBlocksEvaluator:
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
        from kindred.core.exceptions import FitSimulationError, FittingCancelled

        value = float(dict(params).get("init:A", 0.0))
        if not self._is_lane:
            return {
                "t": self._t_axis.copy(),
                "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
            }

        if value == 1.0:
            self._state["fatal_entered"].set()
            self._state["siblings_ready_to_poll"].wait(timeout=2.0)
            self._state["callback_should_block"].set()
            self._state["allow_sibling_poll"].set()
            time.sleep(0.05)
            raise FitSimulationError("objective fatal lane failure", details={"fatal": True})

        with self._state["lock"]:
            self._state["siblings_ready"] += 1
            if self._state["siblings_ready"] == 3:
                self._state["siblings_ready_to_poll"].set()
        self._state["allow_sibling_poll"].wait(timeout=2.0)
        while not self._state["normal_release"].is_set():
            if self._cancellation_check is not None and self._cancellation_check():
                with self._state["lock"]:
                    self._state["cooperative_stops"] += 1
                raise FittingCancelled()
            self._state["poll"].wait(timeout=0.001)
        return {
            "t": self._t_axis.copy(),
            "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
        }


class _PauseRefillThenFatalEvaluator:
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
        from kindred.core.exceptions import FitSimulationError, FittingCancelled

        value = float(dict(params).get("init:A", 0.0))
        if not self._is_lane:
            return {
                "t": self._t_axis.copy(),
                "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
            }

        with self._state["lock"]:
            self._state["starts"].append(value)
            if len(self._state["starts"]) == 4:
                self._state["initial_lanes_started"].set()

        if value == 1.0:
            self._state["release_nonfatal"].wait(timeout=2.0)
            return {
                "t": self._t_axis.copy(),
                "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
            }
        if value == 2.0:
            self._state["release_fatal"].wait(timeout=2.0)
            raise FitSimulationError("objective fatal lane failure", details={"fatal": True})

        while not self._state["normal_release"].is_set():
            if self._cancellation_check is not None and self._cancellation_check():
                with self._state["lock"]:
                    self._state["cooperative_stops"] += 1
                raise FittingCancelled()
            self._state["poll"].wait(timeout=0.001)
        return {
            "t": self._t_axis.copy(),
            "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
        }


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


def test_dataset_lane_pool_retains_only_bounded_reusable_slot_lanes() -> None:
    from kindred.core.analysis.global_fitting import (
        _MAX_PARALLEL_DATASET_LANES,
        _DatasetEvaluatorLanePool,
        _ObjectiveDatasetInput,
        _evaluate_dataset_simulations,
    )

    dataset_count = int(_MAX_PARALLEL_DATASET_LANES) + 6
    payloads = [_payload(f"ds{i}", [0.0, 0.0]) for i in range(dataset_count)]
    items = [
        _ObjectiveDatasetInput(
            index=idx,
            payload=payload,
            full_params={"init:A": float(idx + 1)},
            parameter_origins={},
            failed_param_snapshot={},
        )
        for idx, payload in enumerate(payloads)
    ]
    state = {"lock": threading.Lock(), "clone_count": 0, "base_calls": 0}
    evaluator = _SlotRetentionTrackingEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state)
    lane_pool = _DatasetEvaluatorLanePool(evaluator)

    results = _evaluate_dataset_simulations(evaluator, items, lane_pool=lane_pool)

    assert len(results) == dataset_count
    assert [result.index for result in results] == list(range(dataset_count))
    assert len(lane_pool._lanes) == int(_MAX_PARALLEL_DATASET_LANES)
    assert sorted(lane_pool._lanes) == list(range(int(_MAX_PARALLEL_DATASET_LANES)))
    assert state["clone_count"] == int(_MAX_PARALLEL_DATASET_LANES)
    assert state["base_calls"] == 0
    called_values = sorted(value for _lane_id, value in state["lane_calls"])
    assert called_values == [float(i + 1) for i in range(dataset_count)]


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
        solver="Radau",
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


def test_global_fit_objective_serializes_lsoda_lanes_without_penalty_residuals(monkeypatch) -> None:
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
    state = {
        "lock": threading.Lock(),
        "active": 0,
        "max_active": 0,
        "starts": 0,
        "first_entered": threading.Event(),
        "second_entered": threading.Event(),
        "release_first": threading.Event(),
    }

    def fake_solve_request(request):
        with state["lock"]:
            state["starts"] += 1
            call_index = int(state["starts"])
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            if call_index == 1:
                state["first_entered"].set()
            elif call_index == 2:
                state["second_entered"].set()
            if state["active"] > 1:
                raise RuntimeError("lsoda concurrency sentinel")
        try:
            if call_index == 1:
                state["release_first"].wait(timeout=2.0)
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
    ctx = ObjectiveContext()
    objective = _GlobalFitObjective(
        fit_evaluator=SerialFittingEvaluator(context),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        weights={"ds1": 1.0, "ds2": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ctx,
        progress_callback=None,
        cancellation_check=None,
    )
    result_box = {}

    def _run_objective():
        try:
            result_box["residuals"] = objective(layout.x0.copy())
        except BaseException as exc:
            result_box["error"] = exc

    thread = threading.Thread(target=_run_objective)
    thread.start()
    assert state["first_entered"].wait(timeout=2.0)
    assert not state["second_entered"].wait(timeout=0.1)
    state["release_first"].set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert "error" not in result_box
    assert state["max_active"] == 1
    assert ctx.last_error is None
    np.testing.assert_allclose(
        result_box["residuals"],
        np.asarray([1.0, 1.0, 10.0, 10.0], dtype=float),
    )


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


def test_global_fit_objective_pause_blocks_new_lane_submission_until_resume() -> None:
    from kindred.core.analysis.global_fitting import _GlobalFitObjective, _build_parameter_layout
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
        "starts": [],
        "clone_count": 0,
        "base_calls": 0,
        "active": 0,
        "max_active": 0,
        "expected_active": 4,
        "all_active": threading.Event(),
        "release": threading.Event(),
    }
    pause = {"value": False}
    resumed = threading.Event()
    resumed.set()

    def cancellation_check():
        return False

    cancellation_check._kindred_nonblocking_cancelled = lambda: False
    cancellation_check._kindred_nonblocking_paused = lambda: bool(pause["value"])
    cancellation_check._kindred_wait_for_resume = lambda timeout: resumed.wait(timeout=timeout)

    objective = _GlobalFitObjective(
        fit_evaluator=_PauseAwareLaneTrackingEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={f"ds{i}": {"init:A": float(i)} for i in range(1, 7)},
        weights={payload.dataset_id: 1.0 for payload in payloads},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=cancellation_check,
    )
    result_box = {}

    def _run_objective():
        try:
            result_box["residuals"] = objective(layout.x0.copy())
        except BaseException as exc:
            result_box["error"] = exc

    thread = threading.Thread(target=_run_objective)
    thread.start()
    assert state["all_active"].wait(timeout=2.0)

    pause["value"] = True
    resumed.clear()
    state["release"].set()
    assert not resumed.wait(timeout=0.1)

    with state["lock"]:
        starts_while_paused = list(state["starts"])
    assert starts_while_paused == [1.0, 2.0, 3.0, 4.0]
    assert thread.is_alive()

    pause["value"] = False
    resumed.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert "error" not in result_box
    with state["lock"]:
        assert sorted(state["starts"]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    np.testing.assert_allclose(
        result_box["residuals"],
        np.asarray([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0], dtype=float),
    )


def test_global_fit_objective_fatal_lane_cooperatively_stops_active_siblings() -> None:
    from kindred.core.analysis.global_fitting import _GlobalFitObjective, _build_parameter_layout
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.objective import ObjectiveContext

    payloads = [
        _payload("ds1", [0.0, 0.0]),
        _payload("ds2", [0.0, 0.0]),
        _payload("ds3", [0.0, 0.0]),
        _payload("ds4", [0.0, 0.0]),
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
        "siblings_active": 0,
        "cooperative_stops": 0,
        "fatal_entered": threading.Event(),
        "all_siblings_active": threading.Event(),
        "normal_release": threading.Event(),
        "poll": threading.Event(),
    }

    def cancellation_check():
        return False

    objective = _GlobalFitObjective(
        fit_evaluator=_FatalCooperativeLaneEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={f"ds{i}": {"init:A": float(i)} for i in range(1, 5)},
        weights={payload.dataset_id: 1.0 for payload in payloads},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=cancellation_check,
    )
    result_box = {}

    def _run_objective():
        try:
            result_box["residuals"] = objective(layout.x0.copy())
        except BaseException as exc:
            result_box["error"] = exc

    thread = threading.Thread(target=_run_objective)
    thread.start()
    assert state["fatal_entered"].wait(timeout=2.0)
    assert state["all_siblings_active"].wait(timeout=2.0)
    thread.join(timeout=0.5)

    if thread.is_alive():
        state["normal_release"].set()
        state["poll"].set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert isinstance(result_box.get("error"), FitSimulationError)
    assert "objective fatal lane failure" in str(result_box["error"])
    assert "residuals" not in result_box
    assert state["normal_release"].is_set() is False
    assert state["cooperative_stops"] == 3


def test_global_fit_objective_fatal_lane_stops_siblings_blocked_by_pause_callback() -> None:
    from kindred.core.analysis.global_fitting import _GlobalFitObjective, _build_parameter_layout
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.objective import ObjectiveContext

    payloads = [
        _payload("ds1", [0.0, 0.0]),
        _payload("ds2", [0.0, 0.0]),
        _payload("ds3", [0.0, 0.0]),
        _payload("ds4", [0.0, 0.0]),
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
        "siblings_ready": 0,
        "cooperative_stops": 0,
        "fatal_entered": threading.Event(),
        "siblings_ready_to_poll": threading.Event(),
        "allow_sibling_poll": threading.Event(),
        "callback_should_block": threading.Event(),
        "normal_release": threading.Event(),
        "poll": threading.Event(),
        "pause_gate": threading.Event(),
        "blocking_callback_calls": 0,
        "blocking_callback_waits": 0,
    }

    def blocking_pause_callback():
        with state["lock"]:
            state["blocking_callback_calls"] += 1
            should_block = state["callback_should_block"].is_set()
            if should_block:
                state["blocking_callback_waits"] += 1
        if should_block:
            state["pause_gate"].wait(timeout=2.0)
        return False

    blocking_pause_callback._kindred_nonblocking_cancelled = lambda: False
    blocking_pause_callback._kindred_nonblocking_paused = lambda: False

    objective = _GlobalFitObjective(
        fit_evaluator=_FatalWhilePauseCallbackBlocksEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={f"ds{i}": {"init:A": float(i)} for i in range(1, 5)},
        weights={payload.dataset_id: 1.0 for payload in payloads},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=blocking_pause_callback,
    )
    result_box = {}

    def _run_objective():
        try:
            result_box["residuals"] = objective(layout.x0.copy())
        except BaseException as exc:
            result_box["error"] = exc

    thread = threading.Thread(target=_run_objective)
    thread.start()
    assert state["fatal_entered"].wait(timeout=2.0)
    assert state["siblings_ready_to_poll"].wait(timeout=2.0)
    thread.join(timeout=0.5)

    finished_before_resume = not thread.is_alive()
    if thread.is_alive():
        state["pause_gate"].set()
        state["normal_release"].set()
        state["poll"].set()
        thread.join(timeout=2.0)

    assert finished_before_resume
    assert not thread.is_alive()
    assert isinstance(result_box.get("error"), FitSimulationError)
    assert "objective fatal lane failure" in str(result_box["error"])
    assert "residuals" not in result_box
    assert state["normal_release"].is_set() is False
    assert state["blocking_callback_waits"] == 0
    assert state["cooperative_stops"] == 3


def test_global_fit_objective_fatal_lane_wins_over_paused_refill_wait() -> None:
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
    state = {
        "lock": threading.Lock(),
        "clone_count": 0,
        "starts": [],
        "cooperative_stops": 0,
        "initial_lanes_started": threading.Event(),
        "release_nonfatal": threading.Event(),
        "release_fatal": threading.Event(),
        "normal_release": threading.Event(),
        "poll": threading.Event(),
    }
    pause = {"value": False}
    resumed = threading.Event()
    resumed.set()

    def cancellation_check():
        return False

    cancellation_check._kindred_nonblocking_cancelled = lambda: False
    cancellation_check._kindred_nonblocking_paused = lambda: bool(pause["value"])
    cancellation_check._kindred_wait_for_resume = lambda timeout: resumed.wait(timeout=timeout)

    objective = _GlobalFitObjective(
        fit_evaluator=_PauseRefillThenFatalEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={f"ds{i}": {"init:A": float(i)} for i in range(1, 6)},
        weights={payload.dataset_id: 1.0 for payload in payloads},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=cancellation_check,
    )
    result_box = {}

    def _run_objective():
        try:
            result_box["residuals"] = objective(layout.x0.copy())
        except BaseException as exc:
            result_box["error"] = exc

    thread = threading.Thread(target=_run_objective)
    thread.start()
    assert state["initial_lanes_started"].wait(timeout=2.0)

    pause["value"] = True
    resumed.clear()
    state["release_nonfatal"].set()
    time.sleep(0.05)
    with state["lock"]:
        starts_while_paused = list(state["starts"])
    assert starts_while_paused == [1.0, 2.0, 3.0, 4.0]

    state["release_fatal"].set()
    thread.join(timeout=0.5)
    finished_before_resume = not thread.is_alive()
    if thread.is_alive():
        pause["value"] = False
        resumed.set()
        state["normal_release"].set()
        state["poll"].set()
        thread.join(timeout=2.0)

    assert finished_before_resume
    assert not thread.is_alive()
    assert isinstance(result_box.get("error"), FitSimulationError)
    assert "objective fatal lane failure" in str(result_box["error"])
    assert "residuals" not in result_box
    with state["lock"]:
        assert 5.0 not in state["starts"]
    assert state["normal_release"].is_set() is False
    assert state["cooperative_stops"] == 2


def test_global_fit_objective_does_not_refill_before_inspecting_same_batch_fatal(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
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

    def ordered_wait(futures, return_when=None):
        future_list = list(futures)
        deadline = time.monotonic() + 2.0
        while not all(future.done() for future in future_list):
            if time.monotonic() > deadline:
                raise AssertionError("expected initial bounded lane batch to finish")
            time.sleep(0.001)
        ordered = sorted(
            future_list,
            key=lambda future: (future.result().index == 0, future.result().index),
        )
        return ordered, set()

    monkeypatch.setattr(global_fitting, "wait", ordered_wait)
    objective = _GlobalFitObjective(
        fit_evaluator=_FatalObjectiveEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={
            "ds1": {"init:A": 10.0},
            "ds2": {"init:A": 2.0},
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

    assert sorted(state["values"]) == [2.0, 3.0, 4.0, 10.0]
    assert 5.0 not in state["values"]


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


def test_fit_global_reuses_warmed_bounded_serial_lanes_across_fit_run(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_evaluation as fitting_evaluation
    import kindred.core.fitting_optimization as fitting_optimization
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
    from kindred.core.simulator.solvers import SimulationOutput

    state = {
        "phase": "setup",
        "objective_calls": 0,
        "prepare_counts": {},
        "lock": threading.Lock(),
    }
    original_assemble = global_fitting._assemble_global_fit_result
    original_prepare = fitting_evaluation.prepare_simulation_worker_run

    def counting_prepare(*args, **kwargs):
        with state["lock"]:
            phase = str(state["phase"])
            state["prepare_counts"][phase] = int(state["prepare_counts"].get(phase, 0)) + 1
        return original_prepare(*args, **kwargs)

    def fake_solve_request(request):
        t = np.linspace(
            float(request.t_span[0]),
            float(request.t_span[1]),
            int((request.grid or {}).get("N") or 2),
            dtype=float,
        )
        y0 = np.asarray(request.y0, dtype=float).reshape(-1)
        return SimulationOutput(
            t=t,
            Y=np.tile(y0.reshape(-1, 1), (1, t.size)),
            provenance={},
        )

    def fake_least_squares(fun, x0, **kwargs):
        state["phase"] = "objective_call_1"
        first_residuals = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)
        state["objective_calls"] += 1
        state["phase"] = "objective_call_2"
        second_residuals = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)
        state["objective_calls"] += 1
        state["phase"] = "optimal_residual_recheck"
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            success=True,
            message="ok",
            nfev=2,
            fun=second_residuals,
            jac=np.eye(len(first_residuals), len(np.asarray(x0, dtype=float))),
        )

    def fake_de(*_args, **_kwargs):
        raise AssertionError("DE should not be used for this test")

    def counting_assemble(*args, **kwargs):
        state["phase"] = "final_assembly"
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr(fitting_evaluation, "prepare_simulation_worker_run", counting_prepare)
    monkeypatch.setattr(fitting_evaluation, "_solve_request", fake_solve_request)
    monkeypatch.setattr(fitting_optimization, "load_scipy_optimize", lambda: (fake_least_squares, fake_de))
    monkeypatch.setattr(global_fitting, "_assemble_global_fit_result", counting_assemble)

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
        num_points=2,
        solver="Radau",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )

    dataset_count = int(global_fitting._MAX_PARALLEL_DATASET_LANES) + 2
    result = global_fitting.fit_global(
        SerialFittingEvaluator(context),
        [
            {"id": f"ds{i}", "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)}
            for i in range(1, dataset_count + 1)
        ],
        {"k1": 0.2},
        dataset_params={f"ds{i}": {"init:A": float(i)} for i in range(1, dataset_count + 1)},
        method="trf",
        max_nfev=2,
    )

    assert result.success is True
    assert state["objective_calls"] == 2
    assert state["prepare_counts"] == {
        "objective_call_1": int(global_fitting._MAX_PARALLEL_DATASET_LANES)
    }


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
    original_assemble = global_fitting._assemble_global_fit_result

    def fake_least_squares(func, x0, **kwargs):
        residuals = np.asarray(func(np.asarray(x0, dtype=float)), dtype=float)
        state["phase"] = "optimal_residual_recheck"
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

    def counting_assemble(*args, **kwargs):
        state["phase"] = "final"
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr(fitting_optimization, "load_scipy_optimize", lambda: (fake_least_squares, fake_de))
    monkeypatch.setattr(global_fitting, "_assemble_global_fit_result", counting_assemble)

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
    assert state["phase_clone_counts"] == {"objective": 4}
    assert sorted(set(state["final_values"])) == [1.0, 3.0, 4.0, 5.0, 10.0]
    assert "ds5" in result.model_series
