from __future__ import annotations

import threading
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest


def _sleep_for_process_termination_probe(seconds: float) -> int:
    import os
    import time

    time.sleep(float(seconds))
    return int(os.getpid())


class _FinalPhaseFatalEvaluator:
    def __init__(self, *, t_axis, state, lane_id=None):
        self._t_axis = np.asarray(t_axis, dtype=float).reshape(-1)
        self._state = state
        self._lane_id = lane_id

    def _kindred_clone_fitting_evaluator_lane(self):
        with self._state["lock"]:
            lane_id = int(self._state["clone_count"])
            self._state["clone_count"] += 1
            phase_counts = self._state.setdefault("phase_clone_counts", {})
            phase = str(self._state["phase"])
            phase_counts[phase] = int(phase_counts.get(phase, 0)) + 1
        return type(self)(t_axis=self._t_axis, state=self._state, lane_id=lane_id)

    def evaluate_series(self, params):
        from kindred.core.exceptions import FitSimulationError

        value = float(dict(params).get("init:A", 0.0))
        if self._state["phase"] == "final":
            with self._state["lock"]:
                self._state.setdefault("final_values", []).append(value)
                if self._lane_id is None:
                    self._state.setdefault("final_base_values", []).append(value)
                else:
                    self._state.setdefault("final_lane_values", []).append((int(self._lane_id), value))
        fatal_marker = float(dict(params).get("fatal_marker", 0.0))
        if self._state["phase"] == "final" and fatal_marker > 0.5:
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
        barrier_by_value = self._state.get("barrier_by_value")
        if barrier_by_value is not None:
            barrier = barrier_by_value.get(value)
            if barrier is not None:
                barrier.wait(timeout=5.0)
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

    lane_cap = int(_MAX_PARALLEL_DATASET_LANES)
    dataset_count = lane_cap * 2 + 1
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
    first_wave_barrier = threading.Barrier(lane_cap)
    second_wave_barrier = threading.Barrier(lane_cap)
    state = {"lock": threading.Lock(), "clone_count": 0, "base_calls": 0}
    state["barrier_by_value"] = {
        **{float(i + 1): first_wave_barrier for i in range(lane_cap)},
        **{float(i + 1): second_wave_barrier for i in range(lane_cap, lane_cap * 2)},
    }
    evaluator = _SlotRetentionTrackingEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state)
    lane_pool = _DatasetEvaluatorLanePool(evaluator)
    original_lane_for_slot = lane_pool.lane_for_slot
    retained_lane_object_ids_by_slot = {}
    retained_underlying_ids_by_slot = {}

    def tracking_lane_for_slot(slot):
        lane = original_lane_for_slot(slot)
        if lane is not None:
            slot_index = int(slot)
            lane_object_id = id(lane)
            underlying_id = id(getattr(lane, "_evaluator", lane))
            retained_lane_object_ids_by_slot.setdefault(slot_index, lane_object_id)
            retained_underlying_ids_by_slot.setdefault(slot_index, underlying_id)
            assert retained_lane_object_ids_by_slot[slot_index] == lane_object_id
            assert retained_underlying_ids_by_slot[slot_index] == underlying_id
        return lane

    lane_pool.lane_for_slot = tracking_lane_for_slot

    results = _evaluate_dataset_simulations(evaluator, items, lane_pool=lane_pool)

    assert len(results) == dataset_count
    assert [result.index for result in results] == list(range(dataset_count))
    assert len(lane_pool._lanes) == lane_cap
    assert sorted(lane_pool._lanes) == list(range(lane_cap))
    assert sorted(retained_lane_object_ids_by_slot) == list(range(lane_cap))
    assert sorted(retained_underlying_ids_by_slot) == list(range(lane_cap))
    retained_lane_ids = {
        getattr(lane, "_evaluator", lane)._lane_id for lane in lane_pool._lanes.values()
    }
    assert len(retained_lane_ids) == lane_cap
    assert state["clone_count"] == lane_cap
    assert state["base_calls"] == 0
    lane_ids_for_calls = [lane_id for lane_id, _value in state["lane_calls"]]
    lane_call_counts = Counter(lane_ids_for_calls)
    assert set(lane_call_counts) == retained_lane_ids
    assert sorted(lane_call_counts.values()) == ([2] * (lane_cap - 1)) + [3]
    called_values = sorted(value for _lane_id, value in state["lane_calls"])
    assert called_values == [float(i + 1) for i in range(dataset_count)]
    for result in results:
        expected_value = float(result.index + 1)
        np.testing.assert_allclose(result.sim_species["A"], np.full(2, expected_value, dtype=float))


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


def test_global_fit_objective_parallel_lanes_with_serial_fitting_evaluator() -> None:
    from kindred.core.analysis.global_fitting import _GlobalFitObjective, _build_parameter_layout
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
    from kindred.core.objective import ObjectiveContext

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
    class _InProcessSerialFittingEvaluator(SerialFittingEvaluator):
        pass

    reference = _GlobalFitObjective(
        fit_evaluator=_InProcessSerialFittingEvaluator(context),
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
    expected = reference(layout.x0.copy())

    try:
        residuals = objective(layout.x0.copy())
    finally:
        objective._lane_pool.close()

    assert residuals.shape == (4,)
    assert np.all(np.isfinite(residuals))
    np.testing.assert_allclose(residuals, expected)
    assert len(set(objective._lane_pool._kindred_process_worker_pids())) > 1


def test_global_fit_objective_process_lsoda_lanes_without_penalty_residuals() -> None:
    from kindred.core.analysis.global_fitting import _GlobalFitObjective, _build_parameter_layout
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
    from kindred.core.objective import ObjectiveContext

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
    class _InProcessSerialFittingEvaluator(SerialFittingEvaluator):
        pass

    reference = _GlobalFitObjective(
        fit_evaluator=_InProcessSerialFittingEvaluator(context),
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
    expected = reference(layout.x0.copy())

    try:
        residuals = objective(layout.x0.copy())
    finally:
        objective._lane_pool.close()

    assert ctx.last_error is None
    assert residuals.shape == (4,)
    assert np.all(np.isfinite(residuals))
    assert not np.allclose(residuals, np.full(4, 1e6, dtype=float))
    np.testing.assert_allclose(residuals, expected)
    assert len(set(objective._lane_pool._kindred_process_worker_pids())) > 1


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


def test_fit_global_de_restarts_process_lanes_after_fatal_penalty(monkeypatch) -> None:
    from concurrent.futures import Future

    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    created_pools = []
    captured = {}

    class _ScriptedProcessPool:
        def __init__(self, _payload, *, max_lanes):
            self.max_lanes = int(max_lanes)
            self.closed = False
            self.shutdown_calls = []
            self.recorded_payloads = []
            self.generation = len(created_pools) + 1
            created_pools.append(self)

        def submit(self, slot, task):
            future = Future()
            if self.generation == 1:
                future.set_result(
                    {
                        "ok": False,
                        "index": int(task["index"]),
                        "dataset_id": str(task["dataset_id"]),
                        "slot": int(slot),
                        "worker_pid": 20000 + int(slot),
                        "cold_start": True,
                        "prepare_count": 1,
                        "eval_count": 1,
                        "error": {
                            "kind": "fit_simulation",
                            "message": "fatal de probe failure",
                            "failed_params": {},
                            "details": {"fatal": True},
                            "context": None,
                            "error_provenance": {"dataset": str(task["dataset_id"])},
                            "final_error_message": "fatal de probe failure",
                        },
                    }
                )
                return future

            value = float(dict(task["full_params"]).get("init:A", 0.0))
            future.set_result(
                {
                    "ok": True,
                    "index": int(task["index"]),
                    "dataset_id": str(task["dataset_id"]),
                    "slot": int(slot),
                    "worker_pid": 20000 + self.generation * 10 + int(slot),
                    "sim_time": np.linspace(0.0, 0.1, 3),
                    "sim_species": {"A": np.full(3, value, dtype=float)},
                    "cold_start": self.generation == 2,
                    "prepare_count": 1,
                    "eval_count": 1,
                }
            )
            return future

        def record_result(self, payload):
            self.recorded_payloads.append(dict(payload))

        def worker_pids(self):
            return tuple(sorted({int(payload["worker_pid"]) for payload in self.recorded_payloads}))

        def slot_stats(self):
            return {
                int(payload["slot"]): {
                    "pid": int(payload["worker_pid"]),
                    "cold_starts": int(payload.get("prepare_count") or 0),
                    "eval_count": int(payload.get("eval_count") or 0),
                }
                for payload in self.recorded_payloads
            }

        def shutdown(self, *, wait=True, cancel_futures=True, terminate=False):
            self.closed = True
            self.shutdown_calls.append((wait, cancel_futures, terminate))

    def fake_least_squares(*_args, **_kwargs):
        raise AssertionError("least_squares should not be used for this test")

    def fake_de(func, bounds, **kwargs):
        captured["workers"] = kwargs.get("workers")
        x = np.asarray([(float(lo) + float(hi)) / 2.0 for lo, hi in bounds], dtype=float)
        first = float(func(x))
        second = float(func(x))
        captured["de_values"] = (first, second)
        return SimpleNamespace(x=x, success=True, message="ok", nfev=2)

    monkeypatch.setattr(global_fitting, "ProcessBackedFittingEvaluatorLanePool", _ScriptedProcessPool)
    monkeypatch.setattr(fitting_optimization, "load_scipy_optimize", lambda: (fake_least_squares, fake_de))

    result = global_fitting.fit_global(
        _build_process_lane_serial_evaluator(),
        [
            {"id": "ds1", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)},
            {"id": "ds2", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)},
        ],
        {"k1": 0.2},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        bounds={"k1": (0.1, 1.0)},
        method="de",
        max_nfev=2,
    )

    assert result.success is True
    assert captured["workers"] == 1
    assert captured["de_values"][0] > captured["de_values"][1]
    assert len(created_pools) == 2
    assert created_pools[0].shutdown_calls == [(False, True, True)]
    assert created_pools[1].shutdown_calls[-1] == (True, True, False)


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

    lane_cap = int(global_fitting._MAX_PARALLEL_DATASET_LANES)
    dataset_count = lane_cap + 3
    dataset_ids = [f"ds{i}" for i in range(1, dataset_count + 1)]
    early_fatal_dataset_id = "ds2"
    overflow_fatal_dataset_id = dataset_ids[lane_cap + 1]
    fatal_dataset_ids = {early_fatal_dataset_id, overflow_fatal_dataset_id}
    dataset_values = {
        dataset_id: (101.0 if dataset_id in {"ds1", early_fatal_dataset_id} else float(index + 100))
        for index, dataset_id in enumerate(dataset_ids, start=1)
    }
    fatal_markers = {
        dataset_id: (1.0 if dataset_id in fatal_dataset_ids else 0.0)
        for dataset_id in dataset_ids
    }
    result = global_fitting.fit_global(
        _FinalPhaseFatalEvaluator(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        [
            {"id": dataset_id, "t": np.linspace(0.0, 1.0, 2), "species": "A", "y": np.zeros(2)}
            for dataset_id in dataset_ids
        ],
        {"k1": 1.0},
        dataset_params={
            dataset_id: {"init:A": value, "fatal_marker": fatal_markers[dataset_id]}
            for dataset_id, value in dataset_values.items()
        },
        method="trf",
        max_nfev=1,
    )

    assert result.success is False
    assert set(result.dataset_errors) == fatal_dataset_ids
    for dataset_id in fatal_dataset_ids:
        assert "final fatal lane failure" in result.dataset_errors[dataset_id]
    assert state["phase_clone_counts"] == {
        "objective": min(dataset_count, lane_cap)
    }
    assert sorted(state["final_values"]) == sorted(dataset_values.values())
    assert state.get("final_base_values", []) == []
    assert sorted(value for _lane_id, value in state["final_lane_values"]) == sorted(dataset_values.values())
    assert sorted({lane_id for lane_id, _value in state["final_lane_values"]}) == list(range(lane_cap))
    successful_dataset_ids = [dataset_id for dataset_id in dataset_ids if dataset_id not in fatal_dataset_ids]
    assert list(result.model_series) == successful_dataset_ids
    assert [info.dataset_id for info in result.dataset_info] == dataset_ids
    assert sorted(result.residual_series) == sorted(dataset_ids)
    assert sorted(result.plot_model_x) == sorted(dataset_ids)
    assert sorted(result.plot_model_series) == sorted(successful_dataset_ids)
    dataset_info_by_id = {info.dataset_id: info for info in result.dataset_info}
    for dataset_id in successful_dataset_ids:
        np.testing.assert_allclose(
            result.model_series[dataset_id]["A"],
            np.full(2, dataset_values[dataset_id], dtype=float),
        )
        np.testing.assert_allclose(
            result.residual_series[dataset_id]["A"],
            np.full(2, dataset_values[dataset_id], dtype=float),
        )
        assert dataset_info_by_id[dataset_id].n_points == 2
    for dataset_id in fatal_dataset_ids:
        assert dataset_id not in result.model_series
        assert dataset_id not in result.plot_model_series
        assert "A" in result.residual_series[dataset_id]
        np.testing.assert_allclose(result.residual_series[dataset_id]["A"], np.full(2, 1e6, dtype=float))
        np.testing.assert_allclose(dataset_info_by_id[dataset_id].residuals, np.full(2, 1e6, dtype=float))
        assert dataset_info_by_id[dataset_id].n_points == 2


def _build_process_lane_serial_evaluator():
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    context = prepare_fitting_execution_context(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        param_names=["k1"],
        t_end=0.1,
        num_points=3,
        solver="Radau",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    return SerialFittingEvaluator(context)


def test_serial_fitting_process_dataset_evaluation_matches_serial_seam() -> None:
    from kindred.core.analysis import global_fitting

    evaluator = _build_process_lane_serial_evaluator()
    payloads = global_fitting._build_dataset_payloads(
        [
            {"id": "ds1", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)},
            {"id": "ds2", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)},
        ]
    )
    items = [
        global_fitting._ObjectiveDatasetInput(
            index=index,
            payload=payload,
            full_params={"k1": 0.2, "init:A": float(index + 1)},
            parameter_origins={},
            failed_param_snapshot={"k1": 0.2, f"{payload.dataset_id}::init:A": float(index + 1)},
        )
        for index, payload in enumerate(payloads)
    ]

    serial = global_fitting._evaluate_dataset_simulations_serial(
        evaluator,
        items,
        cancellation_check=None,
        stop_on_fatal=True,
    )
    lane_pool = global_fitting._DatasetEvaluatorLanePool(evaluator)
    try:
        process_backed = global_fitting._evaluate_dataset_simulations(
            evaluator,
            items,
            cancellation_check=None,
            lane_pool=lane_pool,
        )
    finally:
        lane_pool.close()

    assert len(process_backed) == len(serial)
    assert len(set(lane_pool._kindred_process_worker_pids())) > 1
    for actual, expected in zip(process_backed, serial):
        assert actual.index == expected.index
        assert actual.error is None
        assert expected.error is None
        np.testing.assert_allclose(actual.sim_time, expected.sim_time)
        assert set(actual.sim_species) == set(expected.sim_species)
        for name in expected.sim_species:
            np.testing.assert_allclose(actual.sim_species[name], expected.sim_species[name])


def test_fit_global_serial_evaluator_process_lanes_use_child_pids_and_reuse_warm_slots(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    captured = {}
    original_assemble = global_fitting._assemble_global_fit_result

    def fake_least_squares(fun, x0, **kwargs):
        first = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)
        second = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            success=True,
            message="ok",
            nfev=2,
            fun=second,
            jac=np.eye(len(first), len(np.asarray(x0, dtype=float))),
        )

    def fake_de(*_args, **_kwargs):
        raise AssertionError("DE should not be used for this test")

    def capture_assemble(*args, **kwargs):
        captured["lane_pool"] = kwargs["lane_pool"]
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr(fitting_optimization, "load_scipy_optimize", lambda: (fake_least_squares, fake_de))
    monkeypatch.setattr(global_fitting, "_assemble_global_fit_result", capture_assemble)

    lane_cap = int(global_fitting._MAX_PARALLEL_DATASET_LANES)
    dataset_count = lane_cap + 2
    result = global_fitting.fit_global(
        _build_process_lane_serial_evaluator(),
        [
            {"id": f"ds{i}", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)}
            for i in range(1, dataset_count + 1)
        ],
        {"k1": 0.2},
        dataset_params={f"ds{i}": {"init:A": float(i)} for i in range(1, dataset_count + 1)},
        method="trf",
        max_nfev=2,
    )

    lane_pool = captured["lane_pool"]
    process_pids = set(lane_pool._kindred_process_worker_pids())
    slot_stats = lane_pool._kindred_process_slot_stats()

    assert result.success is True
    assert len(process_pids) > 1
    assert len(slot_stats) == lane_cap
    assert {int(slot) for slot in slot_stats} == set(range(lane_cap))
    assert sum(int(stats["cold_starts"]) for stats in slot_stats.values()) == lane_cap
    assert all(int(stats["cold_starts"]) == 1 for stats in slot_stats.values())
    assert all(int(stats["eval_count"]) > int(stats["cold_starts"]) for stats in slot_stats.values())


def test_fit_global_closes_lane_pool_when_post_optimizer_reconstruction_overflows(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_optimization import FitResult

    lane_pools = []

    class _FakeLanePool:
        def __init__(self, _fit_evaluator) -> None:
            self.close_calls = []
            lane_pools.append(self)

        def close(self, *, wait=True, cancel_futures=True, terminate=False) -> None:
            self.close_calls.append((wait, cancel_futures, terminate))

    def fake_fit_parameters(*_args, **_kwargs):
        return FitResult(
            success=True,
            parameters={"k1": 10000.0},
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(1),
            nfev=1,
            message="extreme optimum",
            covariance=None,
        )

    def evaluator(_params):
        return {"t": np.array([0.0]), "species": {"A": np.array([1.0])}}

    monkeypatch.setattr(global_fitting, "_DatasetEvaluatorLanePool", _FakeLanePool)
    monkeypatch.setattr(global_fitting, "fit_parameters", fake_fit_parameters)

    with pytest.raises(OverflowError):
        global_fitting.fit_global(
            evaluator,
            [{"id": "ds1", "t": np.array([0.0]), "species": "A", "y": np.array([1.0])}],
            {"k1": 1.0},
            bounds={"k1": (1e-9, 1e9)},
            log10_params={"k1": True},
            method="trf",
            max_nfev=1,
        )

    assert len(lane_pools) == 1
    assert lane_pools[0].close_calls == [(True, True, False)]


def test_fit_global_process_submit_failure_aborts_batch_before_outer_close(monkeypatch) -> None:
    from concurrent.futures import Future

    from kindred.core.analysis import global_fitting

    lane_pools = []

    class _FakeProcessPool:
        max_lanes = 2

        def __init__(self) -> None:
            self.pending_future = Future()
            self.shutdown_calls = []
            self.submit_calls = 0

        def submit(self, slot, task):
            self.submit_calls += 1
            if self.submit_calls == 1:
                return self.pending_future
            raise RuntimeError("submit boom")

        def shutdown(self, *, wait, cancel_futures, terminate):
            self.shutdown_calls.append((wait, cancel_futures, terminate))

    class _FakeLanePool:
        def __init__(self, _fit_evaluator) -> None:
            self._process_pool = _FakeProcessPool()
            self.close_calls = []
            lane_pools.append(self)

        def process_pool(self):
            return self._process_pool

        def close(self, *, wait=True, cancel_futures=True, terminate=False) -> None:
            assert self._process_pool.shutdown_calls[:1] == [(False, True, True)]
            self.close_calls.append((wait, cancel_futures, terminate))
            self._process_pool.shutdown(wait=wait, cancel_futures=cancel_futures, terminate=terminate)

    def fake_fit_parameters(objective_func, initial_params, **_kwargs):
        objective_func(np.asarray([1.0], dtype=float))
        raise AssertionError("unreachable")

    monkeypatch.setattr(global_fitting, "_DatasetEvaluatorLanePool", _FakeLanePool)
    monkeypatch.setattr(global_fitting, "fit_parameters", fake_fit_parameters)

    result = global_fitting.fit_global(
        lambda _params: {"t": np.array([0.0]), "species": {"A": np.array([1.0])}},
        [
            {"id": "ds1", "t": np.array([0.0]), "species": "A", "y": np.array([1.0])},
            {"id": "ds2", "t": np.array([0.0]), "species": "A", "y": np.array([1.0])},
        ],
        {"k1": 1.0},
        method="trf",
        max_nfev=1,
    )

    assert result.success is False
    assert result.message == "Fitting failed: submit boom"
    assert len(lane_pools) == 1
    assert lane_pools[0]._process_pool.shutdown_calls[0] == (False, True, True)
    assert lane_pools[0].close_calls == [(True, True, False)]


def test_serial_fitting_evaluator_subclass_stays_on_in_process_path() -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_evaluation import SerialFittingEvaluator
    from kindred.core.fitting_process_lanes import fitting_process_lane_payload_from_evaluator

    class _CustomSerialFittingEvaluator(SerialFittingEvaluator):
        pass

    base = _build_process_lane_serial_evaluator()
    evaluator = _CustomSerialFittingEvaluator(base.context)
    payloads = global_fitting._build_dataset_payloads(
        [
            {"id": "ds1", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)},
            {"id": "ds2", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)},
        ]
    )
    items = [
        global_fitting._ObjectiveDatasetInput(
            index=index,
            payload=payload,
            full_params={"k1": 0.2, "init:A": float(index + 1)},
            parameter_origins={},
            failed_param_snapshot={"k1": 0.2, f"{payload.dataset_id}::init:A": float(index + 1)},
        )
        for index, payload in enumerate(payloads)
    ]
    lane_pool = global_fitting._DatasetEvaluatorLanePool(evaluator)

    results = global_fitting._evaluate_dataset_simulations(evaluator, items, lane_pool=lane_pool)

    assert fitting_process_lane_payload_from_evaluator(evaluator) is None
    assert lane_pool._kindred_process_worker_pids() == ()
    assert len(results) == 2
    assert all(result.error is None for result in results)


def test_serial_fitting_evaluator_process_export_failure_stays_on_in_process_path() -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_process_lanes import fitting_process_lane_payload_from_evaluator

    evaluator = _build_process_lane_serial_evaluator()

    def _boom():
        raise RuntimeError("export failed")

    evaluator._kindred_process_lane_payload = _boom
    payloads = global_fitting._build_dataset_payloads(
        [
            {"id": "ds1", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)},
            {"id": "ds2", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)},
        ]
    )
    items = [
        global_fitting._ObjectiveDatasetInput(
            index=index,
            payload=payload,
            full_params={"k1": 0.2, "init:A": float(index + 1)},
            parameter_origins={},
            failed_param_snapshot={"k1": 0.2, f"{payload.dataset_id}::init:A": float(index + 1)},
        )
        for index, payload in enumerate(payloads)
    ]
    lane_pool = global_fitting._DatasetEvaluatorLanePool(evaluator)

    results = global_fitting._evaluate_dataset_simulations(evaluator, items, lane_pool=lane_pool)

    assert fitting_process_lane_payload_from_evaluator(evaluator) is None
    assert lane_pool._kindred_process_worker_pids() == ()
    assert len(results) == 2
    assert all(result.error is None for result in results)


def test_process_lane_shutdown_terminates_private_workers_without_public_terminate() -> None:
    from kindred.core.fitting_process_lanes import ProcessBackedFittingEvaluatorLanePool, _ProcessLaneSlot

    class _FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.join_calls = []

        def is_alive(self):
            return not self.killed

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def join(self, timeout=None):
            self.join_calls.append(timeout)

    class _FakeExecutor:
        def __init__(self, process) -> None:
            self._processes = {1: process}
            self.shutdown_calls = []

        def shutdown(self, *, wait, cancel_futures):
            self.shutdown_calls.append((wait, cancel_futures))

    process = _FakeProcess()
    executor = _FakeExecutor(process)
    pool = ProcessBackedFittingEvaluatorLanePool({}, max_lanes=1)
    pool._slots[0] = _ProcessLaneSlot(slot=0, executor=executor)

    pool.shutdown(wait=False, cancel_futures=True, terminate=True)

    assert process.terminated is True
    assert process.killed is True
    assert process.join_calls == [0.2, 0.2]
    assert executor.shutdown_calls == [(False, True)]


def test_process_lane_shutdown_terminates_real_running_child_process() -> None:
    import multiprocessing as mp
    import time
    from concurrent.futures import ProcessPoolExecutor

    from kindred.core.fitting_process_lanes import ProcessBackedFittingEvaluatorLanePool

    executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
    processes = []
    try:
        executor.submit(_sleep_for_process_termination_probe, 30.0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            processes = list((getattr(executor, "_processes", None) or {}).values())
            if processes and all(process.is_alive() for process in processes):
                break
            time.sleep(0.01)
        assert processes

        handled = ProcessBackedFittingEvaluatorLanePool._terminate_executor_workers(
            executor,
            cancel_futures=True,
        )

        assert handled is True
        assert all(not process.is_alive() for process in processes)
    finally:
        if any(process.is_alive() for process in processes):
            ProcessBackedFittingEvaluatorLanePool._terminate_executor_workers(
                executor,
                cancel_futures=True,
            )
        executor.shutdown(wait=False, cancel_futures=True)


def test_process_dataset_fatal_result_terminates_active_sibling_lanes() -> None:
    from concurrent.futures import Future

    from kindred.core.analysis import global_fitting

    class _FakeProcessPool:
        max_lanes = 2

        def __init__(self) -> None:
            self.shutdown_calls = []
            self.recorded_payloads = []
            self.pending_future = None

        def submit(self, slot, task):
            future = Future()
            if int(slot) == 0:
                future.set_result(
                    {
                        "ok": False,
                        "index": int(task["index"]),
                        "dataset_id": str(task["dataset_id"]),
                        "slot": int(slot),
                        "worker_pid": 12345,
                        "cold_start": True,
                        "prepare_count": 1,
                        "eval_count": 1,
                        "error": {
                            "kind": "fit_simulation",
                            "message": "fatal child failure",
                            "failed_params": {},
                            "details": {"fatal": True},
                            "context": None,
                            "error_provenance": {"dataset": str(task["dataset_id"])},
                            "final_error_message": "fatal child failure",
                        },
                    }
                )
            else:
                self.pending_future = future
            return future

        def record_result(self, payload):
            self.recorded_payloads.append(dict(payload))

        def shutdown(self, *, wait, cancel_futures, terminate):
            self.shutdown_calls.append((wait, cancel_futures, terminate))
            if self.pending_future is not None:
                self.pending_future.cancel()

    class _FakeLanePool:
        def __init__(self, process_pool) -> None:
            self._process_pool = process_pool

        def process_pool(self):
            return self._process_pool

    payloads = global_fitting._build_dataset_payloads(
        [
            {"id": "ds1", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)},
            {"id": "ds2", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)},
        ]
    )
    items = [
        global_fitting._ObjectiveDatasetInput(
            index=index,
            payload=payload,
            full_params={"k1": 0.2, "init:A": float(index + 1)},
            parameter_origins={},
            failed_param_snapshot={},
        )
        for index, payload in enumerate(payloads)
    ]
    process_pool = _FakeProcessPool()

    results = global_fitting._evaluate_dataset_simulations_process(
        items,
        cancellation_check=None,
        stop_on_fatal=True,
        lane_pool=_FakeLanePool(process_pool),
    )

    assert len(results) == 1
    assert global_fitting._dataset_evaluation_is_fatal(results[0])
    assert process_pool.shutdown_calls == [(False, True, True)]


def test_process_submit_failure_terminates_active_batch() -> None:
    from concurrent.futures import Future

    from kindred.core.analysis import global_fitting

    class _FakeProcessPool:
        max_lanes = 2

        def __init__(self) -> None:
            self.shutdown_calls = []
            self.pending_future = Future()
            self.submit_calls = 0

        def submit(self, slot, task):
            self.submit_calls += 1
            if self.submit_calls == 1:
                return self.pending_future
            raise RuntimeError("submit boom")

        def shutdown(self, *, wait, cancel_futures, terminate):
            self.shutdown_calls.append((wait, cancel_futures, terminate))

        def record_result(self, payload):
            raise AssertionError("record_result should not be called")

    class _FakeLanePool:
        def __init__(self, process_pool) -> None:
            self._process_pool = process_pool

        def process_pool(self):
            return self._process_pool

    payloads = global_fitting._build_dataset_payloads(
        [
            {"id": "ds1", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)},
            {"id": "ds2", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)},
        ]
    )
    items = [
        global_fitting._ObjectiveDatasetInput(
            index=index,
            payload=payload,
            full_params={"k1": 0.2, "init:A": float(index + 1)},
            parameter_origins={},
            failed_param_snapshot={},
        )
        for index, payload in enumerate(payloads)
    ]
    process_pool = _FakeProcessPool()

    with pytest.raises(RuntimeError, match="submit boom"):
        global_fitting._evaluate_dataset_simulations_process(
            items,
            cancellation_check=None,
            stop_on_fatal=True,
            lane_pool=_FakeLanePool(process_pool),
        )

    assert process_pool.pending_future.cancelled() is True
    assert process_pool.shutdown_calls == [(False, True, True)]


def test_process_error_payload_preserves_error_context() -> None:
    from kindred.core.analysis import global_fitting

    payload = global_fitting._build_dataset_payloads(
        [{"id": "ds1", "t": np.linspace(0.0, 0.1, 3), "species": "A", "y": np.zeros(3)}]
    )[0]
    item = global_fitting._ObjectiveDatasetInput(
        index=0,
        payload=payload,
        full_params={"k1": 0.2},
        parameter_origins={},
        failed_param_snapshot={"k1": 0.2},
    )

    result = global_fitting._dataset_evaluation_from_process_payload(
        {
            "ok": False,
            "index": 0,
            "slot": 0,
            "worker_pid": 12345,
            "prepare_count": 1,
            "eval_count": 1,
            "error": {
                "kind": "fit_simulation",
                "message": "context failure",
                "failed_params": {"k1": 0.2},
                "details": {"fatal": True},
                "context": {"line": 7, "col": 3, "line_text": "bad", "file_path": None, "stack_trace": None},
                "error_provenance": {"dataset": "ds1"},
                "final_error_message": "context failure",
            },
        },
        item,
    )

    assert result.error is not None
    assert getattr(result.error, "context", None) is not None
    assert result.error.context.line == 7
    assert result.error.context.col == 3
    assert result.error.context.line_text == "bad"
