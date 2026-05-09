from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

import numpy as np
import pytest
from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.core.simulation_identity import SimulationIdentity
from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
from tests.batch_context_test_helpers import seed_batch_context


pytestmark = [pytest.mark.gui]


def _result_payload(*, marker: float = 1.0) -> Dict[str, Any]:
    return {
        "t": np.array([0.0, 1.0], dtype=float),
        "Y": np.array([[marker, marker]], dtype=float),
        "species_names": ["A"],
        "algebra_scalars": {},
        "mechanism": None,
        "mechanism_text": "reaction: A -> B ; k=0.1",
        "solver_config": {},
        "fallback_occurred": False,
        "fallback_message": None,
    }


def test_explicit_parallel_completions_do_not_redraw_per_completion(main_window, monkeypatch):
    main_window.simulation_controller.run_state.latest_sim_request_id = 501
    main_window.simulation_controller.run_state.active_run_id = 88
    main_window.simulation_controller.batch_cache.active_cache_key = "explicit-coalesce-key"
    seed_batch_context(main_window.simulation_controller.batch_context_owner, active=True, parallel=True, run_id=88, request_id=501, fast_mode=False, cache_key="explicit-coalesce-key", primary_set_id="set1", total=5, completed_set_ids=[], queue_ids=["set1", "set2", "set3", "set4", "set5"], queue_names=["set1", "set2", "set3", "set4", "set5"])

    batch_port = main_window.simulation_controller.ui.batch
    monkeypatch.setattr(batch_port, "batch_set_ids_for_scope", lambda _scope: ["set1", "set2", "set3"], raising=True)
    monkeypatch.setattr(batch_port, "batch_current_row", lambda: 0, raising=True)
    monkeypatch.setattr(batch_port, "batch_set_id_for_row", lambda _row: "set1", raising=True)

    display_calls: list[Dict[str, Any]] = []

    def _display(**kwargs):
        display_calls.append(dict(kwargs))
        return True

    monkeypatch.setattr(batch_port, "display_cached_batch_selection", _display, raising=True)

    main_window.simulation_controller.on_simulation_complete(
        _result_payload(marker=2.0),
        run_id=88,
        fast_mode=False,
        request_id=501,
        batch_set="set2",
        batch_set_id="set2",
        cache_key="explicit-coalesce-key",
    )
    main_window.simulation_controller.on_simulation_complete(
        _result_payload(marker=3.0),
        run_id=88,
        fast_mode=False,
        request_id=501,
        batch_set="set3",
        batch_set_id="set3",
        cache_key="explicit-coalesce-key",
    )

    assert display_calls == []

    timer = main_window.simulation_controller.plot_coalescer.timer
    assert timer.isActive() is True

    timer.timeout.emit()
    assert len(display_calls) == 1


def test_explicit_coalesced_flush_batches_multiple_set_ids(main_window, monkeypatch):
    main_window.simulation_controller.run_state.latest_sim_request_id = 777
    main_window.simulation_controller.run_state.active_run_id = 303
    main_window.simulation_controller.batch_cache.active_cache_key = "explicit-batch-key"
    seed_batch_context(main_window.simulation_controller.batch_context_owner, active=True, parallel=True, run_id=303, request_id=777, fast_mode=False, cache_key="explicit-batch-key", primary_set_id="set1", total=6, completed_set_ids=[], queue_ids=["set1", "set2", "set3", "set4", "set5", "set6"], queue_names=["set1", "set2", "set3", "set4", "set5", "set6"])

    batch_port = main_window.simulation_controller.ui.batch
    monkeypatch.setattr(batch_port, "batch_set_ids_for_scope", lambda _scope: ["set1", "set2", "set3", "set4"], raising=True)
    monkeypatch.setattr(batch_port, "shown_batch_set_ids", lambda: ["set1", "set2", "set3", "set4"], raising=True)
    monkeypatch.setattr(batch_port, "batch_current_row", lambda: 0, raising=True)
    monkeypatch.setattr(batch_port, "batch_set_id_for_row", lambda _row: "set1", raising=True)

    display_calls: list[Dict[str, Any]] = []

    def _display(**kwargs):
        display_calls.append(dict(kwargs))
        return True

    monkeypatch.setattr(batch_port, "display_cached_batch_selection", _display, raising=True)

    for sid in ("set2", "set3", "set4"):
        main_window.simulation_controller.on_simulation_complete(
            _result_payload(marker=4.0),
            run_id=303,
            fast_mode=False,
            request_id=777,
            batch_set=sid,
            batch_set_id=sid,
            cache_key="explicit-batch-key",
        )

    assert set(main_window.simulation_controller.plot_coalescer.pending.set_ids) == {"set2", "set3", "set4"}

    timer = main_window.simulation_controller.plot_coalescer.timer
    timer.timeout.emit()

    assert len(display_calls) == 1
    selected_sets = list(display_calls[0].get("selected_sets") or [])
    assert selected_sets == ["set1", "set2", "set3", "set4"]


def test_explicit_parallel_run_triggers_final_refresh_once(main_window, monkeypatch):
    main_window.simulation_controller.run_state.latest_sim_request_id = 909
    main_window.simulation_controller.run_state.active_run_id = 404
    main_window.simulation_controller.batch_cache.active_cache_key = "explicit-final-refresh-key"
    seed_batch_context(main_window.simulation_controller.batch_context_owner, active=True, parallel=True, run_id=404, request_id=909, fast_mode=False, cache_key="explicit-final-refresh-key", primary_set_id="set9", total=2, completed_set_ids=[], queue_ids=["set1", "set2"], queue_names=["set1", "set2"])

    publication_owner = main_window.simulation_controller._completion_publication_owner
    original_deps = publication_owner._deps
    # Force no pending coalesced keys so we can assert final forced refresh behavior.
    monkeypatch.setattr(
        main_window.simulation_controller.ui.batch,
        "display_cached_batch_selection",
        lambda **_kwargs: True,
        raising=True,
    )

    flush_calls: list[Dict[str, Any]] = []

    def _flush(*args, **kwargs):
        flush_calls.append(dict(kwargs))
        return None

    publication_owner._deps = replace(
        original_deps,
        queue_slider_plot_update=lambda **_kwargs: None,
        flush_slider_plot_updates=_flush,
    )

    main_window.simulation_controller.on_simulation_complete(
        _result_payload(marker=9.0),
        run_id=404,
        fast_mode=False,
        request_id=909,
        batch_set="set1",
        batch_set_id="set1",
        cache_key="explicit-final-refresh-key",
    )
    main_window.simulation_controller.on_simulation_complete(
        _result_payload(marker=10.0),
        run_id=404,
        fast_mode=False,
        request_id=909,
        batch_set="set2",
        batch_set_id="set2",
        cache_key="explicit-final-refresh-key",
    )

    assert len(flush_calls) == 1


def test_parallel_callback_completion_main_window_uses_shared_slim_context_and_per_set_identity(main_window):
    controller = main_window.simulation_controller
    run_id = 406
    request_id = 912
    cache_key = "preview-callback-main-window"
    preview_ownership = controller._set_preview_ownership(
        request_id=request_id,
        target_set_ids=("id1", "id2"),
    )
    controller.run_state.latest_sim_request_id = request_id
    controller.run_state.active_run_id = run_id
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=True,
        keep_lane_pool_alive=True,
        run_id=run_id,
        request_id=request_id,
        cache_key=cache_key,
        rows=[0, 1],
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        primary_set_id="id1",
        total=2,
        completed_set_ids=[],
        preview_scope_set_ids=("id1", "id2"),
        preview_owner_epoch=preview_ownership.epoch,
        simulation_identity_by_set_id={
            "id1": {"schema_id": "stale-context-id1"},
            "id2": {"schema_id": "stale-context-id2"},
        },
        preview_batch_cache_token_by_set_id={
            "id1": "stale-token-id1",
            "id2": "stale-token-id2",
        },
        pending_init_seed={},
        pending_init_applied=True,
    )
    callback_context = controller.batch_context_owner.callback_context_snapshot()
    identity_payload_by_set_id = {
        set_id: SimulationIdentity.build(
            schema_id="callback-schema",
            param_fingerprint=set_id,
            solver_config={"solver": "BDF"},
            t_end=1.0,
            preview_batch_cache_token=f"callback-token-{set_id}",
            execution_flags=("fast_mode",),
        ).to_payload()
        for set_id in ("id1", "id2")
    }
    identities = [
        SimulationCallbackIdentity.capture(
            run_id=run_id,
            fast_mode=True,
            request_id=request_id,
            owner_epoch=preview_ownership.epoch,
            batch_set=set_name,
            batch_set_id=set_id,
            cache_key=cache_key,
            callback_context=callback_context,
            simulation_identity=identity_payload_by_set_id[set_id],
            preview_batch_cache_token=f"callback-token-{set_id}",
        )
        for set_id, set_name in (("id1", "set1"), ("id2", "set2"))
    ]
    assert identities[0].callback_context is identities[1].callback_context

    for marker, identity in ((11.0, identities[0]), (12.0, identities[1])):
        controller.on_simulation_complete(
            _result_payload(marker=marker),
            callback_identity=identity,
        )

    for set_id in ("id1", "id2"):
        entry_result = controller.batch_cache.entry_for_set(
            cache_key=cache_key,
            set_id=set_id,
            is_preview=True,
        )
        assert entry_result.state == "valid"
        assert entry_result.entry is not None
        assert entry_result.entry["simulation_identity"] == identity_payload_by_set_id[set_id]
        assert entry_result.entry["preview_batch_cache_token"] == f"callback-token-{set_id}"

    assert controller.batch_cache.active_preview_cache_key == cache_key
    assert controller.batch_cache.preview_cache.get(BatchSimulationCache.entry_key(cache_key, "id1")) is not None
    assert controller.batch_cache.preview_cache.get(BatchSimulationCache.entry_key(cache_key, "id2")) is not None


def test_explicit_parallel_final_flush_preserves_valid_subset_after_earlier_timeout(main_window, monkeypatch):
    main_window.simulation_controller.run_state.latest_sim_request_id = 910
    main_window.simulation_controller.run_state.active_run_id = 405
    main_window.simulation_controller.batch_cache.active_cache_key = "explicit-final-subset-key"
    main_window.simulation_controller.batch_cache.active_cache_valid_set_ids = ("set2",)
    seed_batch_context(main_window.simulation_controller.batch_context_owner, active=True, parallel=True, run_id=405, request_id=910, fast_mode=False, cache_key="explicit-final-subset-key", primary_set_id="set1", total=2, completed_set_ids=[], queue_ids=["set1", "set2"], queue_names=["set1", "set2"], explicit_cache_valid_set_ids=("set2",))

    batch_port = main_window.simulation_controller.ui.batch
    monkeypatch.setattr(batch_port, "batch_set_ids_for_scope", lambda _scope: ["set1"], raising=True)
    monkeypatch.setattr(batch_port, "batch_current_row", lambda: 0, raising=True)
    monkeypatch.setattr(batch_port, "batch_set_id_for_row", lambda _row: "set1", raising=True)

    display_calls: list[Dict[str, Any]] = []

    def _display(**kwargs):
        display_calls.append(dict(kwargs))
        return False

    monkeypatch.setattr(batch_port, "display_cached_batch_selection", _display, raising=True)

    main_window.simulation_controller.on_simulation_complete(
        _result_payload(marker=2.0),
        run_id=405,
        fast_mode=False,
        request_id=910,
        batch_set="set2",
        batch_set_id="set2",
        cache_key="explicit-final-subset-key",
    )

    timer = main_window.simulation_controller.plot_coalescer.timer
    timer.timeout.emit()

    assert len(display_calls) == 1
    assert display_calls[0]["valid_set_ids"] == ("set2",)
    assert display_calls[0]["allow_fallback"] is False

    main_window.simulation_controller.on_simulation_complete(
        _result_payload(marker=3.0),
        run_id=405,
        fast_mode=False,
        request_id=910,
        batch_set="set1",
        batch_set_id="set1",
        cache_key="explicit-final-subset-key",
    )

    assert len(display_calls) == 2
    assert display_calls[1]["valid_set_ids"] == ("set2",)
    assert display_calls[1]["allow_fallback"] is False
