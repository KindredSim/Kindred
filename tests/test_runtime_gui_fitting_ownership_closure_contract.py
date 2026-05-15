from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def test_warm_simulation_owner_rejects_custom_child_lifecycle_branch() -> None:
    from kindred.core.simulation_containment import WarmSimulationOwner

    with pytest.raises(TypeError):
        WarmSimulationOwner({}, child_target=lambda *_args, **_kwargs: None)


def test_batch_runtime_session_declares_typed_lane_owner_contract() -> None:
    from inspect import signature
    from typing import get_type_hints

    import kindred.core.batch_runtime_session as session_module
    from kindred.core.batch_runtime_session import BatchRuntimeLaneOwnerProtocol, BatchRuntimeSession

    assert signature(BatchRuntimeSession).parameters["lane_owner"].annotation is not object
    assert get_type_hints(BatchRuntimeSession.__init__)["lane_owner"] is BatchRuntimeLaneOwnerProtocol
    source = session_module.BatchRuntimeSession.__init__.__code__.co_names
    assert "getattr" not in source
    assert "hasattr" not in source


def test_parallel_batch_ready_uses_action_required_lanes_instead_of_passive_capacity() -> None:
    from kindred.gui.controllers.parallel_batch_runtime_readiness_owner import (
        ParallelBatchRuntimeReadinessOwner,
    )

    class _BatchParallel:
        current_max_workers = 4
        is_pool_stale = False

        def __init__(self) -> None:
            self.ready_lane_checks: list[int] = []

        def has_ready_lane_pool(self, *, max_lanes: int) -> bool:
            self.ready_lane_checks.append(int(max_lanes))
            return int(max_lanes) == 2

    batch_parallel = _BatchParallel()
    owner = ParallelBatchRuntimeReadinessOwner(
        batch_parallel=batch_parallel,
        capacity_getter=lambda: 4,
    )

    assert owner.ready(required_lanes=2) is True
    assert batch_parallel.ready_lane_checks == [2]


def test_parallel_batch_waiting_retry_preserves_action_required_lanes() -> None:
    from kindred.gui.controllers.simulation_controller import SimulationController

    class _Readiness:
        def __init__(self) -> None:
            self.ensure_calls: list[tuple[bool, int | None]] = []

        def ensure(self, *, wait: bool = False, required_lanes: int | None = None) -> None:
            self.ensure_calls.append((bool(wait), required_lanes))

    class _ContextOwner:
        def mark_runtime_waiting(self, *, required_lanes: int | None = None):
            return {"runtime_waiting_required_lanes": required_lanes}

        def active_batch_state(self, _ctx):
            return SimpleNamespace(fast_mode=True, request_id=7)

    class _RunUi:
        def set_runtime_backed_run_controls_ready(self, _ready: bool) -> None:
            return None

        def set_run_button_enabled(self, _enabled: bool) -> None:
            return None

        def set_stop_button_enabled(self, _enabled: bool) -> None:
            return None

        def set_sim_progress_value(self, _value: int) -> None:
            return None

        def set_status_text(self, _text: str) -> None:
            return None

        def schedule_runtime_availability_refresh(self) -> None:
            return None

    controller = SimulationController.__new__(SimulationController)
    readiness = _Readiness()
    controller._parallel_batch_runtime_readiness_owner = readiness
    controller._batch_context_owner = _ContextOwner()
    controller._run_state = SimpleNamespace(simulation_running=True, slider_simulation_active=True)
    controller.ui = SimpleNamespace(run_ui=_RunUi())
    controller.queue_pending_slider_preview_replay = lambda **_kwargs: None
    controller.clear_pending_slider_preview_replay = lambda **_kwargs: None
    controller._run_simulation_from_slider = lambda: None
    controller._next_slider_preview_request_id = lambda: 9
    controller._parallel_batch_required_lanes_for_rows = lambda rows: len(rows)
    controller._ensure_parallel_batch_runtime_ready = readiness.ensure

    controller._handle_parallel_batch_runtime_waiting(
        rows=[0, 1],
        queue_ids=["set-1", "set-2"],
        runtime_snapshot=SimpleNamespace(should_poll=False),
    )

    assert readiness.ensure_calls == [(False, 2)]


def test_simulation_plan_rejects_cache_identity_schedule_mismatch() -> None:
    from kindred.core.intervention_schedule import InterventionSchedule
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
    from kindred.core.simulation_preparation import SimulationExecutionRequest

    selected_schedule = InterventionSchedule.from_payload(
        {
            "instant_events": [
                {"time": 1.0, "species": "A", "op": "add", "amount": 2.0}
            ]
        }
    )
    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0},
        t_span=(0.0, 10.0),
        solver_config={"solver": "BDF"},
        mechanism_text="A -> B; k=1",
        intervention_schedule=selected_schedule,
    )

    with pytest.raises(ValueError, match="intervention_schedule_fingerprint"):
        SimulationPlan.from_execution_request(
            request,
            execution_mode="ordinary",
            algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
            cache_identity_payload={
                "simulation_identity": {
                    "intervention_schedule_fingerprint": "raw-gui-dsl-schedule"
                }
            },
        )


def test_callback_identity_capture_requires_supplied_context_and_identity() -> None:
    from kindred.gui.controllers.simulation_controller import SimulationController

    class _ForbiddenBatchContextOwner:
        def callback_context_snapshot(self):
            raise AssertionError("callback identity must not rediscover current context")

        def simulation_plan_payload_for_set(self, _set_id):
            raise AssertionError("callback identity must not rediscover plan payload")

        def simulation_identity_for_set(self, _set_id):
            raise AssertionError("callback identity must not rediscover stored identity")

        def preview_batch_cache_token_for_set(self, _set_id):
            raise AssertionError("callback identity must not rediscover preview token")

    controller = SimulationController.__new__(SimulationController)
    controller._batch_context_owner = _ForbiddenBatchContextOwner()

    with pytest.raises(ValueError, match="callback identity"):
        controller._capture_simulation_callback_identity(
            run_id=1,
            fast_mode=False,
            request_id=2,
            owner_epoch=3,
            batch_set=None,
            batch_set_id=None,
            cache_key="cache-key",
            callback_context=None,
            simulation_identity=None,
            preview_batch_cache_token=None,
        )
    identity = controller._capture_simulation_callback_identity(
        run_id=1,
        fast_mode=False,
        request_id=2,
        owner_epoch=3,
        batch_set=None,
        batch_set_id=None,
        cache_key="cache-key",
        callback_context={},
        simulation_identity={},
        preview_batch_cache_token=None,
    )

    assert identity.batch_set_id is None


def test_completion_dispatch_rejects_missing_callback_identity(monkeypatch) -> None:
    from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
    from kindred.gui.controllers.simulation_controller import SimulationController

    controller = SimulationController.__new__(SimulationController)
    controller._completion_callback_owner = SimpleNamespace(
        handle_completion=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing callback identity must not reach completion owner")
        )
    )
    monkeypatch.setattr(
        SimulationCallbackIdentity,
        "capture",
        classmethod(
            lambda cls, **_kwargs: (_ for _ in ()).throw(
                AssertionError("completion dispatch must not rediscover callback identity")
            )
        ),
    )

    with pytest.raises(TypeError, match="callback_identity"):
        controller._on_simulation_complete(
            {"success": True},  # type: ignore[call-arg]
        )


def test_completion_callback_injects_captured_launch_provenance_before_publication() -> None:
    from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
    from kindred.gui.controllers.simulation_completion_callback import (
        SimulationCompletionCallbackDependencies,
        SimulationCompletionCallbackOwner,
    )

    published: list[dict] = []

    class _Publication:
        def publish_success(self, result, _state, **_kwargs):
            published.append(dict(result))

    class _Freshness:
        def assess_callback(self, callback_identity, **_kwargs):
            return SimpleNamespace(
                active_run_id=7,
                latest_request_id=11,
                shutdown_requested=False,
                current_global_epoch=0,
                callback_owner_epoch=callback_identity.owner_epoch,
                stale_run=False,
                runtime_input_stale=False,
                missing_owner_epoch=False,
                preview_owner_matches=True,
                superseded_fast_request=False,
            )

        def mark_stale_runtime_input_callback_consumed(self, **_kwargs) -> None:
            return None

    owner = SimulationCompletionCallbackOwner(
        ui=SimpleNamespace(slider=SimpleNamespace(slider_triggered_simulation=lambda: False)),
        batch_context_owner=SimpleNamespace(
            completion_policy_context=lambda _ctx=None: {"active": True},
            explicit_batch_coalescing_for_completion=lambda **_kwargs: False,
        ),
        completion_policy=SimpleNamespace(),
        lifecycle_effect_owner=SimpleNamespace(),
        publication_owner=_Publication(),
        dependencies=SimulationCompletionCallbackDependencies(
            freshness=_Freshness(),
            completion_policy_preview_ownership=lambda: None,
            completion_policy_pending_replay_state=lambda: None,
            apply_completion_policy_state_patch=lambda *_args, **_kwargs: None,
            apply_lifecycle_effects=lambda *_args, **_kwargs: None,
        ),
    )
    identity = SimulationCallbackIdentity.capture(
        run_id=7,
        fast_mode=False,
        request_id=11,
        owner_epoch=None,
        batch_set=None,
        batch_set_id=None,
        cache_key="cache",
        callback_context={"active": True, "run_id": 7, "request_id": 11},
        simulation_identity={"fingerprint": "submitted"},
        preview_batch_cache_token="",
        launch_provenance={
            "temperature_K": 310.0,
            "simulation_time": 12.5,
            "num_points_requested": 321,
            "mechanism_text": "reaction: A -> B; k=1",
        },
    )

    owner.handle_completion(
        {"success": True, "provenance": {"solver": "BDF"}},
        debug_batch_parallel=False,
        callback_identity=identity,
    )

    assert published
    assert published[0]["provenance"]["solver"] == "BDF"
    assert published[0]["provenance"]["launch_provenance"]["temperature_K"] == pytest.approx(310.0)
    assert published[0]["mechanism_text"] == "reaction: A -> B; k=1"


def test_completion_publication_does_not_recover_missing_cache_key_from_callback_context() -> None:
    from kindred.gui.controllers.simulation_completion_publication import (
        CompletionCallbackState,
        SimulationCompletionPublicationDependencies,
        SimulationCompletionPublicationOwner,
    )

    cache_truth_calls: list[dict] = []
    owner = SimulationCompletionPublicationOwner(
        ui=SimpleNamespace(),
        batch_context_owner=SimpleNamespace(),
        batch_cache=SimpleNamespace(),
        cache_admin=SimpleNamespace(
            publish_completion_cache_truth=lambda **kwargs: cache_truth_calls.append(dict(kwargs))
        ),
        completion_policy=SimpleNamespace(),
        lifecycle_effect_owner=SimpleNamespace(),
        result_materialization_owner=SimpleNamespace(),
        dependencies=SimulationCompletionPublicationDependencies(
            apply_lifecycle_effects=lambda *_args, **_kwargs: None,
            record_nonfatal_exception=lambda *_args, **_kwargs: None,
            queue_slider_plot_update=lambda *_args, **_kwargs: None,
            finalize_explicit_batch_dirty_reset=lambda *_args, **_kwargs: {},
            flush_slider_plot_updates=lambda *_args, **_kwargs: None,
            show_scoped_batch_failure_summary=lambda *_args, **_kwargs: None,
            has_deferred_preview_replay_intent=lambda: False,
            start_next_batch_simulation=lambda: None,
        ),
    )
    state = CompletionCallbackState(
        run_id=7,
        request_id=11,
        batch_set=None,
        batch_set_id=None,
        cache_key=None,
        policy_context=None,
        ctx={"cache_key": "context-cache"},
        shutdown_requested=False,
        is_preview=True,
        slider_triggered=False,
        explicit_batch_coalescing=False,
    )

    owner.publish_cache_truth(state)

    assert state.cache_key is None
    assert cache_truth_calls == []


def test_completion_provenance_uses_captured_launch_fields_over_current_ui_state() -> None:
    from kindred.gui.controllers.simulation_completion_publication import (
        CompletionResultState,
        SimulationCompletionPublicationDependencies,
        SimulationCompletionPublicationOwner,
    )

    class _Results:
        def __init__(self) -> None:
            self.annotation_payloads: list[dict] = []

        def publish_completion_intervention_annotations(self, payload):
            self.annotation_payloads.append(dict(payload or {}))

        def main_plot(self):
            return SimpleNamespace(overlay_snapshot=lambda: None)

    class _Solver:
        def temperature_spinbox_value(self) -> float:
            raise AssertionError("completion provenance must not read current UI temperature")

        def sim_time_spinbox_text(self) -> str:
            raise AssertionError("completion provenance must not read current UI simulation time text")

        def parse_sim_time_seconds(self) -> float:
            raise AssertionError("completion provenance must not parse current UI simulation time")

        def initial_solver_name(self) -> str:
            raise AssertionError("completion provenance must not read current UI solver name")

        def initial_rtol(self) -> float:
            raise AssertionError("completion provenance must not read current UI rtol")

        def initial_atol(self) -> float:
            raise AssertionError("completion provenance must not read current UI atol")

        def dsl_global_temperature_K(self, _mechanism_text: str):
            raise AssertionError("completion provenance must not rediscover DSL temperature")

        def num_points_spinbox_value(self) -> int:
            raise AssertionError("completion provenance must not read current UI point count")

    class _Provenance:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def publish_simulation_completion_provenance(self, **kwargs):
            self.calls.append(dict(kwargs))

    ui = SimpleNamespace(
        results=_Results(),
        solver=_Solver(),
        provenance=_Provenance(),
    )
    deps = SimulationCompletionPublicationDependencies(
        apply_lifecycle_effects=lambda *_args, **_kwargs: None,
        record_nonfatal_exception=lambda *_args, **_kwargs: None,
        queue_slider_plot_update=lambda *_args, **_kwargs: None,
        finalize_explicit_batch_dirty_reset=lambda *_args, **_kwargs: {},
        flush_slider_plot_updates=lambda *_args, **_kwargs: None,
        show_scoped_batch_failure_summary=lambda *_args, **_kwargs: None,
        has_deferred_preview_replay_intent=lambda: False,
        start_next_batch_simulation=lambda: None,
    )
    owner = SimulationCompletionPublicationOwner(
        ui=ui,
        batch_context_owner=SimpleNamespace(),
        batch_cache=SimpleNamespace(),
        cache_admin=SimpleNamespace(),
        completion_policy=SimpleNamespace(),
        lifecycle_effect_owner=SimpleNamespace(),
        result_materialization_owner=SimpleNamespace(),
        dependencies=deps,
    )
    completion = CompletionResultState(
        t=np.array([0.0, 1.0]),
        Y=np.array([[1.0, 0.5]]),
        species_names=["A"],
        algebra_scalars={},
        algebra_errors=[],
        solver_provenance={
            "launch_provenance": {
                "temperature_K": 310.0,
                "temperature_source": "captured",
                "simulation_time": 12.5,
                "num_points_requested": 321,
            }
        },
        mechanism=None,
        base_species_count=1,
        mechanism_text="A -> B; k=1",
        solver_config={"solver": "BDF", "solver_label": "BDF", "rtol": 1e-6, "atol": 1e-12},
        warnings=[],
        fallback_occurred=False,
        fallback_message=None,
        series={"A": np.array([1.0, 0.5])},
        is_primary=True,
        energy_mode=False,
        redraw_valid_set_ids=None,
        has_redraw_subset=False,
    )

    owner.publish_annotations_and_provenance(completion)

    published = ui.provenance.calls[-1]
    assert published["temperature_K"] == pytest.approx(310.0)
    assert published["temperature_source"] == "captured"
    assert published["simulation_time"] == pytest.approx(12.5)
    assert published["num_points_requested"] == 321


def test_fitting_readiness_uses_evaluator_state_owner_for_prepared_metadata(monkeypatch) -> None:
    from kindred.gui.fitting.evaluator_state import FittingEvaluatorStateOwner
    from kindred.gui.fitting.runtime_readiness import FittingRuntimeReadinessController

    sentinel = object()
    monkeypatch.setattr(
        FittingEvaluatorStateOwner,
        "prepared_simulation_meta_for",
        staticmethod(lambda evaluator: sentinel),
    )

    assert FittingRuntimeReadinessController._prepared_simulation_meta(object()) is sentinel


def test_fitting_worker_launch_is_owned_by_window_without_fake_sidecar_module() -> None:
    from pathlib import Path

    assert not Path("kindred/gui/fitting/worker_launch.py").exists()
    source = Path("kindred/gui/fitting/window.py").read_text(encoding="utf-8")

    assert "best_update_interval_s=" not in source
    assert "plot_update_interval_s=" not in source


def test_main_window_set_data_compatibility_surface_is_removed_without_test_shim() -> None:
    from pathlib import Path

    main_window_source = Path("kindred/gui/main_window.py").read_text(encoding="utf-8")
    solver_error_test = Path("tests/test_solver_error_gui.py").read_text(encoding="utf-8")

    assert "def set_data" not in main_window_source
    assert "Public API for setting data (compatibility)" not in main_window_source
    assert "main_window.set_data =" not in solver_error_test
