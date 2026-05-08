from __future__ import annotations

import numpy as np

from kindred.core.batch_cache_contracts import read_batch_cache_entry
from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.core.simulation_identity import SimulationIdentity


def test_core_batch_cache_contract_preserves_completion_payload_identity() -> None:
    cache = BatchSimulationCache()
    identity = SimulationIdentity.build(
        schema_id="schema-a",
        param_fingerprint="params-a",
        solver_config={"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
        t_end=1.0,
        intervention_schedule_fingerprint="schedule-a",
    )

    stored_key = cache.put_completion_entry(
        cache_key="ck",
        set_id="set-1",
        is_preview=True,
        t=np.asarray([0.0, 1.0]),
        series={"A": np.asarray([1.0, 0.25])},
        algebra_scalars={"Keq": "2.5", "bad": object()},
        mechanism="mechanism-object",
        mechanism_text="reaction: A -> B; k=1",
        simulation_identity=identity.to_payload(),
        solver_config={"solver": "BDF"},
        preview_batch_cache_token="preview-token",
        fallback_occurred=True,
        fallback_message="fallback used",
        solver_provenance={"normalized": "BDF"},
    )

    assert stored_key == "ck::set-1"
    result = cache.entry_for_set(cache_key="ck", set_id="set-1", is_preview=True)

    assert result.state == "valid"
    assert result.entry is not None
    assert result.entry["t"].tolist() == [0.0, 1.0]
    assert result.entry["series"]["A"].tolist() == [1.0, 0.25]
    assert result.entry["algebra_scalars"] == {"Keq": 2.5}
    assert result.entry["mechanism"] == "mechanism-object"
    assert result.entry["mechanism_text"] == "reaction: A -> B; k=1"
    assert result.entry["simulation_identity"]["schema_id"] == "schema-a"
    assert result.entry["simulation_identity"]["param_fingerprint"] == "params-a"
    assert result.entry["simulation_identity"]["intervention_schedule_fingerprint"] == "schedule-a"
    assert result.entry["solver_config"] == {"solver": "BDF"}
    assert result.entry["preview_batch_cache_token"] == "preview-token"
    assert result.entry["fallback_occurred"] is True
    assert result.entry["fallback_message"] == "fallback used"
    assert result.entry["solver_provenance"] == {"normalized": "BDF"}


def test_core_batch_cache_owner_distinguishes_invalid_entries_from_misses() -> None:
    cache = BatchSimulationCache()
    cache.result_cache.put("ck::bad", {"t": [0.0], "series": object()})

    invalid = cache.entry_for_set(cache_key="ck", set_id="bad", is_preview=False)
    missing = cache.entry_for_set(cache_key="ck", set_id="missing", is_preview=False)

    assert invalid.state == "invalid"
    assert missing.state == "missing"
    assert read_batch_cache_entry({"series": {"A": [1.0]}}).state == "invalid"


def test_core_batch_cache_owner_lists_entries_for_cache_key_without_raw_store_access() -> None:
    cache = BatchSimulationCache()
    cache.put_completion_entry(
        cache_key="ck",
        set_id="b",
        is_preview=False,
        t=[0.0],
        series={"A": [2.0]},
    )
    cache.put_completion_entry(
        cache_key="ck",
        set_id="a",
        is_preview=False,
        t=[0.0],
        series={"A": [1.0]},
    )
    cache.put_completion_entry(
        cache_key="other",
        set_id="z",
        is_preview=False,
        t=[0.0],
        series={"A": [99.0]},
    )

    entries = cache.entries_for_cache_key(cache_key="ck", is_preview=False)

    assert [set_id for set_id, _entry in entries] == ["a", "b"]
    assert [entry["series"]["A"].tolist()[0] for _set_id, entry in entries] == [1.0, 2.0]


def test_core_batch_cache_owner_reports_cached_set_membership_by_id_or_name() -> None:
    cache = BatchSimulationCache()
    cache.put_completion_entry(
        cache_key="explicit",
        set_id="set-1",
        is_preview=False,
        t=[0.0],
        series={"A": [1.0]},
    )
    cache.put_completion_entry(
        cache_key="preview",
        set_id="Set 2",
        is_preview=True,
        t=[0.0],
        series={"A": [2.0]},
    )

    assert cache.contains_set_identifier(set_id="set-1", set_name="Set 1") is True
    assert cache.contains_set_identifier(set_id="set-2", set_name="Set 2") is True
    assert cache.contains_set_identifier(set_id="set-3", set_name="Set 3") is False


def test_core_batch_cache_owner_purges_deleted_set_entries_from_both_stores() -> None:
    cache = BatchSimulationCache()
    cache.put_completion_entry(
        cache_key="explicit",
        set_id="set-1",
        is_preview=False,
        t=[0.0],
        series={"A": [1.0]},
    )
    cache.put_completion_entry(
        cache_key="explicit",
        set_id="keep",
        is_preview=False,
        t=[0.0],
        series={"A": [2.0]},
    )
    cache.put_completion_entry(
        cache_key="preview",
        set_id="Set 1",
        is_preview=True,
        t=[0.0],
        series={"A": [3.0]},
    )

    removed = cache.purge_entries_for_set_identifiers(set_ids=("set-1",), set_names=("Set 1",))

    assert removed == 2
    assert cache.contains_set_identifier(set_id="set-1", set_name="Set 1") is False
    assert cache.contains_set_identifier(set_id="keep", set_name="Keep") is True


def test_core_batch_cache_owner_reports_cache_caps_without_store_reachthrough() -> None:
    cache = BatchSimulationCache(result_cache_cap=3, preview_cache_cap=4)

    assert cache.result_cache_max_entries() == 3
    assert cache.preview_cache_max_entries() == 4

    cache.set_caps(result_cap=5, preview_cap=6)

    assert cache.result_cache_max_entries() == 5
    assert cache.preview_cache_max_entries() == 6


def test_batch_cache_owner_applies_run_start_cache_decision() -> None:
    cache = BatchSimulationCache()
    cache.active_cache_preview_scope_set_ids = ("old-preview",)
    cache.active_cache_valid_set_ids = ("old-valid",)
    cache.active_cache_invalidated_set_ids = ("old-invalid",)

    cache.apply_run_start_cache_decision(
        fast_mode=False,
        explicit_cache_valid_set_ids=("set-1", "set-2", "set-1"),
        explicit_cache_invalidated_set_ids=("set-3",),
        preview_scope_set_ids=("preview",),
    )

    assert cache.active_cache_preview_scope_set_ids is None
    assert cache.active_cache_valid_set_ids == ("set-1", "set-2")
    assert cache.active_cache_invalidated_set_ids == ("set-3",)
    assert cache.active_preview_scope_set_ids is None

    cache.apply_run_start_cache_decision(
        fast_mode=True,
        explicit_cache_valid_set_ids=("ignored",),
        explicit_cache_invalidated_set_ids=("ignored-stale",),
        preview_scope_set_ids=("preview-1", "preview-1", "preview-2"),
    )

    assert cache.active_cache_valid_set_ids == ("set-1", "set-2")
    assert cache.active_cache_invalidated_set_ids == ("set-3",)
    assert cache.active_preview_scope_set_ids == ("preview-1", "preview-2")


def test_batch_cache_owner_records_completion_cache_identity_and_entries() -> None:
    cache = BatchSimulationCache()
    entry = {
        "t": np.asarray([0.0, 1.0]),
        "series": {"A": np.asarray([1.0, 0.5])},
    }

    preview_key = cache.record_preview_completion_cache_key(
        cache_key="preview-key",
        preview_scope_set_ids=("set-1", "set-1"),
    )
    stored_key = cache.put_batch_cache_entry(
        cache_key="preview-key",
        set_id="set-1",
        entry=entry,
        is_preview=True,
    )

    assert preview_key == "preview-key"
    assert cache.active_preview_cache_key == "preview-key"
    assert cache.active_preview_scope_set_ids == ("set-1",)
    assert stored_key == "preview-key::set-1"
    assert cache.preview_cache.get(stored_key)["series"]["A"].tolist() == [1.0, 0.5]


def test_batch_cache_owner_records_run_cache_keys_and_scoped_failure_state() -> None:
    cache = BatchSimulationCache()

    assert cache.record_run_cache_key(cache_key="preview-key", fast_mode=True) == "preview-key"
    assert cache.active_preview_cache_key == "preview-key"
    assert cache.active_cache_key is None

    assert cache.record_run_cache_key(cache_key="result-key", fast_mode=False) == "result-key"
    assert cache.active_cache_key == "result-key"
    assert cache.active_cache_preview_token is None

    changed = cache.record_explicit_scoped_failure_cache_state(
        cache_key="other-key",
        explicit_cache_valid_set_ids=("stale",),
        explicit_cache_invalidated_set_ids=("ignored",),
    )

    assert changed is False
    assert cache.active_cache_valid_set_ids is None

    changed = cache.record_explicit_scoped_failure_cache_state(
        cache_key="result-key",
        explicit_cache_valid_set_ids=("set-1", "set-1"),
        explicit_cache_invalidated_set_ids=("set-2",),
    )

    assert changed is True
    assert cache.active_cache_valid_set_ids == ("set-1",)
    assert cache.active_cache_invalidated_set_ids == ("set-2",)


def test_batch_cache_owner_applies_explicit_reconciliation() -> None:
    cache = BatchSimulationCache()
    cache.active_preview_cache_key = "preview"
    cache.last_display_selection.append("set-1")
    cache.active_batch_set = "Set 1"

    cache.apply_explicit_cache_reconciliation(
        clear_active_selection_state=False,
        active_cache_key="result-key",
        active_cache_preview_token="preview-token",
        active_cache_preview_scope_set_ids=("set-1", "set-2"),
        active_cache_valid_set_ids=("set-1",),
        active_cache_invalidated_set_ids=("set-2",),
    )

    assert cache.active_cache_key == "result-key"
    assert cache.active_cache_preview_token == "preview-token"
    assert cache.active_cache_preview_scope_set_ids == ("set-1", "set-2")
    assert cache.active_cache_valid_set_ids == ("set-1",)
    assert cache.active_cache_invalidated_set_ids == ("set-2",)
    assert cache.active_preview_cache_key == "preview"
    assert cache.last_display_selection == ["set-1"]

    cache.apply_explicit_cache_reconciliation(
        clear_active_selection_state=True,
        active_cache_key="ignored",
        active_cache_preview_token="ignored",
        active_cache_preview_scope_set_ids=("ignored",),
        active_cache_valid_set_ids=("ignored",),
        active_cache_invalidated_set_ids=("ignored",),
    )

    assert cache.active_cache_key is None
    assert cache.active_preview_cache_key is None
    assert cache.last_display_selection == []
    assert cache.active_batch_set is None
