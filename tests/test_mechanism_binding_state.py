import pytest

from kindred.core.document_parameter_store import DocumentParameterStore


pytestmark = [pytest.mark.unit]


def test_document_parameter_store_keeps_binding_fingerprint_separate_from_schema_identity():
    store = DocumentParameterStore()
    store.set_schema("reaction: A -> B; k=1.0")
    schema_id = store.schema_id

    store.sync_shared_params({"k1": 1.0})
    first_fingerprint = store.param_fingerprint()
    store.sync_shared_params({"k1": 2.0})
    second_fingerprint = store.param_fingerprint()

    assert store.schema_id == schema_id
    assert second_fingerprint != first_fingerprint


def test_document_parameter_store_staged_preview_bindings_do_not_mutate_canonical_bindings():
    store = DocumentParameterStore()
    store.sync_shared_params({"k1": 1.0})
    canonical_fingerprint = store.param_fingerprint()

    assert store.stage_override("set-a", "k1", 2.0) is True

    assert store.shared_params["k1"] == pytest.approx(1.0)
    assert store.param_fingerprint() == canonical_fingerprint
    assert store.param_fingerprint("set-a") != canonical_fingerprint
