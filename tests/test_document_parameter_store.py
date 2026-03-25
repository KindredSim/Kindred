"""Unit tests for DocumentParameterStore."""

from __future__ import annotations

import pytest

from kindred.core.document_parameter_store import DocumentParameterStore


pytestmark = pytest.mark.unit


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


def test_empty_store():
    store = DocumentParameterStore()
    assert store.shared_params == {}
    assert store.set_local_overrides == {}
    assert store.effective_params("s1") == {}
    assert store.has_any_local_overrides() is False
    assert store.set_ids_with_local_overrides() == []
    assert store.schema_text == ""
    assert store.schema_id == ""


# ------------------------------------------------------------------
# Schema identity
# ------------------------------------------------------------------


def test_schema_identity():
    store = DocumentParameterStore()
    store.set_schema("A -> B; k=1")
    assert store.schema_text == "A -> B; k=1"
    assert store.schema_id != ""
    first_id = store.schema_id

    store.set_schema("A -> B; k=1")
    assert store.schema_id == first_id

    store.set_schema("A -> B; k=2")
    assert store.schema_id != first_id


# ------------------------------------------------------------------
# Shared parameter sync
# ------------------------------------------------------------------


def test_sync_shared_params():
    store = DocumentParameterStore()
    changed = store.sync_shared_params({"k1": 1.0, "k2": 2.0})
    assert changed is False  # no prior overrides, so nothing changed
    assert store.shared_params == {"k1": 1.0, "k2": 2.0}


def test_shared_params_property_returns_immutable_snapshot():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})

    exposed = store.shared_params

    with pytest.raises(TypeError):
        exposed["k1"] = 9.0

    assert dict(exposed) == {"k1": 1.0}
    assert store.shared_params == {"k1": 1.0}
    assert store.effective_params("s1") == {"k1": 1.0}


def test_sync_shared_params_filters_invalid():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0, "bad": float("nan"), "also_bad": "nope"})
    assert store.shared_params == {"k1": 1.0}


def test_sync_shared_params_prunes_matching_overrides():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    store.stage_override("s1", "k1", 2.0)
    assert store.has_local_overrides_for_set("s1") is True

    changed = store.sync_shared_params({"k1": 2.0})
    assert changed is True
    assert store.has_local_overrides_for_set("s1") is False


def test_sync_shared_params_prunes_removed_parameters():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0, "k2": 2.0})
    store.stage_override("s1", "k1", 5.0)
    store.stage_override("s1", "k2", 6.0)

    changed = store.sync_shared_params({"k1": 1.0})
    assert changed is True
    assert store.local_overrides_for_set("s1") == {"k1": 5.0}


# ------------------------------------------------------------------
# Staging per-set overrides
# ------------------------------------------------------------------


def test_stage_override_basic():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    changed = store.stage_override("s1", "k1", 2.0)
    assert changed is True
    assert store.local_overrides_for_set("s1") == {"k1": pytest.approx(2.0)}
    assert store.has_local_overrides_for_set("s1") is True
    assert store.has_any_local_overrides() is True


def test_stage_override_same_as_baseline_prunes():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    store.stage_override("s1", "k1", 2.0)
    changed = store.stage_override("s1", "k1", 1.0)
    assert changed is True
    assert store.local_overrides_for_set("s1") == {}
    assert store.has_local_overrides_for_set("s1") is False


def test_stage_override_no_change():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    store.stage_override("s1", "k1", 2.0)
    changed = store.stage_override("s1", "k1", 2.0)
    assert changed is False


def test_stage_override_multiple_sets():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    store.stage_override("s1", "k1", 2.0)
    store.stage_override("s2", "k1", 3.0)
    assert store.set_ids_with_local_overrides() == ["s1", "s2"]
    assert store.local_overrides_for_set("s1") == {"k1": pytest.approx(2.0)}
    assert store.local_overrides_for_set("s2") == {"k1": pytest.approx(3.0)}


def test_stage_override_no_baseline():
    """Staging a parameter that has no baseline should still work."""
    store = DocumentParameterStore()
    changed = store.stage_override("s1", "k1", 5.0)
    assert changed is True
    assert store.local_overrides_for_set("s1") == {"k1": pytest.approx(5.0)}


def test_stage_override_empty_set_id_is_rejected_and_shared_only_state_stays_aligned():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})

    assert store.stage_override("", "k1", 9.0) is False
    assert store.stage_override("   ", "k1", 9.0) is False
    assert store.has_any_local_overrides() is False
    assert store.local_overrides_for_set("") == {}
    assert store.has_local_overrides_for_set("") is False
    assert store.clear_local_overrides_for_set("") is False
    assert store.set_ids_with_local_overrides() == []
    assert store.effective_params("") == {"k1": 1.0}
    assert store.effective_params(None) == {"k1": 1.0}
    assert store.param_fingerprint("") == store.param_fingerprint()
    assert store.param_fingerprint(None) == store.param_fingerprint()


def test_set_local_overrides_property_returns_copy():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})

    exposed = store.set_local_overrides
    exposed[""] = {"k1": 9.0}

    assert store.set_local_overrides == {}
    assert store.has_any_local_overrides() is False
    assert store.local_overrides_for_set("") == {}
    assert store.effective_params("") == {"k1": 1.0}
    assert store.param_fingerprint("") == store.param_fingerprint()


def test_set_local_overrides_property_returns_detached_nested_copy():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    store.stage_override("s1", "k1", 2.0)

    exposed = store.set_local_overrides
    exposed["s1"]["k1"] = 9.0
    exposed[""] = {"k1": 7.0}

    assert store.local_overrides_for_set("s1") == {"k1": pytest.approx(2.0)}
    assert store.set_ids_with_local_overrides() == ["s1"]
    assert store.has_any_local_overrides() is True
    assert store.effective_params("s1") == {"k1": pytest.approx(2.0)}


# ------------------------------------------------------------------
# Effective parameter composition
# ------------------------------------------------------------------


def test_effective_params_shared_only():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0, "k2": 2.0})
    assert store.effective_params("s1") == {"k1": 1.0, "k2": 2.0}


def test_effective_params_with_overrides():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0, "k2": 2.0})
    store.stage_override("s1", "k1", 5.0)
    assert store.effective_params("s1") == {"k1": pytest.approx(5.0), "k2": 2.0}
    assert store.effective_params("s2") == {"k1": 1.0, "k2": 2.0}


def test_effective_params_empty_set_id():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    assert store.effective_params("") == {"k1": 1.0}
    assert store.effective_params() == {"k1": 1.0}


# ------------------------------------------------------------------
# Clear operations
# ------------------------------------------------------------------


def test_clear_local_overrides_for_set():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    store.stage_override("s1", "k1", 2.0)
    store.stage_override("s2", "k1", 3.0)

    result = store.clear_local_overrides_for_set("s1")
    assert result is True
    assert store.has_local_overrides_for_set("s1") is False
    assert store.has_local_overrides_for_set("s2") is True


def test_clear_local_overrides_for_nonexistent_set():
    store = DocumentParameterStore()
    assert store.clear_local_overrides_for_set("nope") is False


def test_clear_all_local_overrides():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    store.stage_override("s1", "k1", 2.0)
    store.stage_override("s2", "k1", 3.0)
    store.clear_all_local_overrides()
    assert store.has_any_local_overrides() is False
    assert store.set_local_overrides == {}


def test_clear_shared_params():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    store.clear_shared_params()
    assert store.shared_params == {}


# ------------------------------------------------------------------
# Commit / globalize
# ------------------------------------------------------------------


def test_commit_effective_as_shared():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    store.stage_override("s1", "k1", 2.0)
    store.stage_override("s2", "k1", 3.0)

    result = store.commit_effective_as_shared("s1")

    assert result == {"k1": pytest.approx(2.0)}
    assert store.shared_params == {"k1": pytest.approx(2.0)}
    assert store.has_local_overrides_for_set("s1") is False
    assert store.local_overrides_for_set("s2") == {"k1": pytest.approx(3.0)}
    assert store.has_any_local_overrides() is True
    assert store.effective_params("s1") == {"k1": pytest.approx(2.0)}
    assert store.effective_params("s2") == {"k1": pytest.approx(3.0)}


def test_commit_effective_as_shared_preserves_other_sets_local_only_overrides():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    store.stage_override("s1", "k1", 2.0)
    store.stage_override("s2", "k_local", 5.0)

    result = store.commit_effective_as_shared("s1")

    assert result == {"k1": pytest.approx(2.0)}
    assert store.shared_params == {"k1": pytest.approx(2.0)}
    assert store.local_overrides_for_set("s1") == {}
    assert store.local_overrides_for_set("s2") == {"k_local": pytest.approx(5.0)}
    assert store.effective_params("s2") == {"k1": pytest.approx(2.0), "k_local": pytest.approx(5.0)}
    assert store.has_any_local_overrides() is True


def test_commit_effective_as_shared_no_overrides():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0, "k2": 2.0})
    result = store.commit_effective_as_shared("s1")
    assert result == {"k1": 1.0, "k2": 2.0}


def test_commit_effective_as_shared_empty_set_id_is_rejected_without_mutation():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    store.stage_override("s1", "k1", 2.0)

    result_empty = store.commit_effective_as_shared("")
    result_whitespace = store.commit_effective_as_shared("   ")
    result_none = store.commit_effective_as_shared(None)

    assert result_empty == {"k1": 1.0}
    assert result_whitespace == {"k1": 1.0}
    assert result_none == {"k1": 1.0}
    assert store.shared_params == {"k1": 1.0}
    assert store.local_overrides_for_set("s1") == {"k1": pytest.approx(2.0)}
    assert store.has_local_overrides_for_set("s1") is True
    assert store.has_any_local_overrides() is True
    assert store.param_fingerprint("") == store.param_fingerprint()


# ------------------------------------------------------------------
# Fingerprinting
# ------------------------------------------------------------------


def test_param_fingerprint_deterministic():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0, "k2": 2.0})
    fp1 = store.param_fingerprint("s1")
    fp2 = store.param_fingerprint("s1")
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex


def test_param_fingerprint_differs_with_overrides():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    fp_shared = store.param_fingerprint("s1")

    store.stage_override("s1", "k1", 2.0)
    fp_overridden = store.param_fingerprint("s1")

    assert fp_shared != fp_overridden


def test_param_fingerprint_same_effective_same_hash():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 2.0})
    fp_from_shared = store.param_fingerprint("s1")

    store2 = DocumentParameterStore()
    store2.sync_shared_params({"k1": 1.0})
    store2.stage_override("s1", "k1", 2.0)
    fp_from_override = store2.param_fingerprint("s1")

    assert fp_from_shared == fp_from_override


def test_param_fingerprint_empty_set_id_uses_shared():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    store.stage_override("s1", "k1", 99.0)
    fp_empty = store.param_fingerprint("")
    fp_none = store.param_fingerprint()
    assert fp_empty == fp_none


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------


def test_reset():
    store = DocumentParameterStore()
    store.set_schema("A -> B; k=1")
    store.sync_shared_params({"k1": 1.0})
    store.stage_override("s1", "k1", 2.0)
    store.reset()
    assert store.shared_params == {}
    assert store.set_local_overrides == {}
    assert store.schema_text == ""
    assert store.schema_id == ""
