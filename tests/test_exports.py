import csv

import numpy as np
import pytest
from PySide6 import QtWidgets

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.gui.ports import (
    CompletedRunDisplayIntent,
    CompletedRunDisplayTransaction,
    CompletionDisplayEntry,
    DisplayTransitionCause,
    DisplayRefreshSource,
    DisplaySetRole,
    DisplayTransitionOutcomeKind,
    ResolvedBatchDisplayRequestEntry,
)
from tests.workflow_helpers import completion_provenance_payload

pytestmark = pytest.mark.gui


def _display_payload(*, t, series, mechanism_text: str, owned_species=None) -> dict:
    payload = {
        "t": t,
        "series": series,
        "algebra_scalars": {},
        "solver_provenance": {},
        "mechanism_text": str(mechanism_text),
        "solver_config": {},
        "warnings": [],
        "completion_provenance": completion_provenance_payload(
            t=t,
            series=series,
            mechanism_text=str(mechanism_text),
        ),
    }
    if owned_species is None:
        owned_species = tuple(str(name) for name in dict(series or {}) if str(name))
    owned_species_t = tuple(str(name) for name in (owned_species or ()) if str(name))
    if owned_species_t:
        payload["owned_species"] = owned_species_t
    return payload


@pytest.fixture(autouse=True)
def suppress_message_boxes(monkeypatch):
    """Prevent modal dialogs from blocking automated exports."""

    def _silent(*_args, **_kwargs):
        return QtWidgets.QMessageBox.StandardButton.Ok

    for attr in ("information", "warning", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, attr, _silent)


@pytest.fixture
def prepared_window(main_window, monkeypatch):
    """Load a preset, seed simulation results, and add a dataset tab."""
    main_window._load_preset_mechanism("M1")
    dsl = main_window._get_mechanism_text()
    mechanism = parse_dsl_to_mechanism(dsl)

    t = np.linspace(0.0, 5.0, 24)
    species_names = mechanism.species_names() or ["A"]
    series_matrix = []
    for idx, _ in enumerate(species_names):
        start = 1.0 / (idx + 1)
        end = 0.1 * (idx + 1)
        series_matrix.append(np.linspace(start, end, t.size))
    Y = np.vstack(series_matrix)

    monkeypatch.setattr(
        main_window._sim_ui_port.provenance,
        "integrate_ctc",
        lambda *args, **kwargs: (0.5, "mock", True, 1e-6, "tail"),
    )
    series_map = {name: Y[idx] for idx, name in enumerate(species_names)}
    owned_species = tuple(str(name) for name in species_names)
    set_id = "export-set"
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(set_id,),
        labels_by_set_id={set_id: "Export Set"},
        primary_set_id=set_id,
        cache_key="export-fixture-cache",
        run_id=1,
        request_id=1,
        owned_species_by_set_id={set_id: owned_species},
    )
    entry = CompletionDisplayEntry(
        set_id=set_id,
        label="Export Set",
        t=t,
        series=series_map,
        algebra_scalars={},
        solver_provenance=None,
        mechanism_text=dsl,
        solver_config={},
        warnings=(),
        completion_provenance=completion_provenance_payload(
            t=t,
            series=series_map,
            mechanism_text=dsl,
        ),
        owned_species=owned_species,
    )
    outcome = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(entry,),
            display_set_ids=(set_id,),
            display_primary_set_id=set_id,
            failed_set_ids=(),
        )
    )
    assert outcome.transition_outcome is not None
    assert outcome.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    dataset_panel = main_window._plot_tabs.add_dataset_tab("Dataset 1")
    first_species = species_names[0]
    dataset_panel.set_data(
        data_x=t,
        data_y=series_map[first_species],
        xlabel="Time",
        ylabel=first_species,
        all_species=series_map,
    )
    return main_window, dataset_panel, tuple(species_names), series_map, t


def test_csv_export_from_simulation_tab(tmp_path, prepared_window):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    csv_path = tmp_path / "simulation.csv"
    window._plot_tabs._tabs.setCurrentIndex(0)
    window.project_controller.handle_export_config(
        {"path": str(csv_path), "mode": "default", "scope": "all"}
    )
    assert csv_path.exists()
    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["Time (s)"] + [f"[{name}]" for name in species_names]
    assert rows[1] == [str(t[0])] + [str(series_map[name][0]) for name in species_names]


def test_csv_export_from_simulation_tab_respects_axis_scope(tmp_path, prepared_window):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    csv_path = tmp_path / "simulation-axis.csv"
    window._plot_tabs._tabs.setCurrentIndex(0)
    plot = window._plot_tabs.get_current_plot()
    first_species = species_names[0]
    plot.set_selected_series([first_species])

    window.project_controller.handle_export_config(
        {"path": str(csv_path), "mode": "default", "scope": "axis"}
    )

    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["Time (s)", f"[{first_species}]"]
    assert rows[1] == [str(t[0]), str(series_map[first_species][0])]


def test_csv_export_from_simulation_tab_without_active_transaction_rejects_stale_plot_payload(
    tmp_path,
    main_window,
):
    window = main_window
    csv_path = tmp_path / "simulation-stale.csv"
    window._plot_tabs._tabs.setCurrentIndex(0)
    plot = window._plot_tabs.get_current_plot()
    plot.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([1.0, 0.5], dtype=float)},
        label="Stale plot payload",
    )

    window.project_controller.handle_export_config(
        {"path": str(csv_path), "mode": "default", "scope": "all"}
    )

    assert not csv_path.exists()


def test_main_plot_copy_and_csv_export_omit_hidden_reference_transaction_layers(
    tmp_path,
    prepared_window,
):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    window._plot_tabs._tabs.setCurrentIndex(0)
    plot = window._plot_tabs.get_current_plot()
    first_species = species_names[0]
    assert window._batch_model.set_row_requested_show(0, False)
    outcome = window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id="export-set",
                label="Export Set",
                entry=_display_payload(
                    t=t,
                    series={first_species: np.asarray(series_map[first_species], dtype=float)},
                    mechanism_text=window._get_mechanism_text(),
                ),
                canonical_entry=_display_payload(
                    t=t,
                    series={first_species: np.asarray(series_map[first_species], dtype=float) + 10.0},
                    mechanism_text=window._get_mechanism_text(),
                ),
            ),
        ),
        prefer_set="export-set",
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )
    assert outcome.transition_outcome is not None
    assert outcome.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    active = window.results_controller.active_display_transaction()
    assert active is not None
    assert active.display_set_ids == ("export-set",)
    provenance = window._simulation_provenance_owner.last_simulation_provenance
    assert provenance["display_transaction"]["display_set_ids"] == ["export-set"]
    assert "requested_show_set_ids" not in provenance["display_transaction"]
    plot.set_reference_layers_visible(False)
    hidden_active = window.results_controller.active_display_transaction()
    assert hidden_active is not None
    visible_reference_metadata = [
        metadata
        for metadata in hidden_active.sets.values()
        if metadata.role is DisplaySetRole.REFERENCE_OVERLAY
    ]
    assert visible_reference_metadata
    assert all(not metadata.visible for metadata in visible_reference_metadata)
    hidden_reference_metadata = [
        metadata
        for metadata in hidden_active.sets.values()
        if metadata.role is DisplaySetRole.REFERENCE_OVERLAY
    ]
    assert hidden_reference_metadata
    assert all(not metadata.visible for metadata in hidden_reference_metadata)

    republished = window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id="export-set",
                label="Export Set",
                entry=_display_payload(
                    t=t,
                    series={first_species: np.asarray(series_map[first_species], dtype=float)},
                    mechanism_text=window._get_mechanism_text(),
                ),
                canonical_entry=_display_payload(
                    t=t,
                    series={first_species: np.asarray(series_map[first_species], dtype=float) + 10.0},
                    mechanism_text=window._get_mechanism_text(),
                ),
            ),
        ),
        prefer_set="export-set",
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )
    assert republished.transition_outcome is not None
    assert republished.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    republished_active = window.results_controller.active_display_transaction()
    assert republished_active is not None
    republished_reference_metadata = [
        metadata
        for metadata in republished_active.sets.values()
        if metadata.role is DisplaySetRole.REFERENCE_OVERLAY
    ]
    assert republished_reference_metadata
    assert all(not metadata.visible for metadata in republished_reference_metadata)

    csv_path = tmp_path / "simulation-hidden-reference.csv"
    window.project_controller.handle_export_config(
        {"path": str(csv_path), "mode": "default", "scope": "all"}
    )

    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert all("Export Set [ref]" not in cell for cell in rows[0])
    plot._copy_all()
    clipboard = QtWidgets.QApplication.clipboard()
    assert clipboard is not None
    assert "Export Set [ref]" not in clipboard.text()
    plot._copy_visible_data()
    assert "Export Set [ref]" not in clipboard.text()


def test_resolved_display_without_species_metadata_is_denied(
    prepared_window,
):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    window._plot_tabs._tabs.setCurrentIndex(0)
    first_species = species_names[0]
    previous_transaction = window.results_controller.active_display_transaction()
    assert previous_transaction is not None

    outcome = window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id="missing-species-export",
                label="Missing Species Export",
                entry=_display_payload(
                    t=t,
                    series={first_species: np.asarray(series_map[first_species], dtype=float)},
                    mechanism_text=window._get_mechanism_text(),
                    owned_species=(),
                ),
                canonical_entry=_display_payload(
                    t=t,
                    series={first_species: np.asarray(series_map[first_species], dtype=float) + 10.0},
                    mechanism_text=window._get_mechanism_text(),
                    owned_species=(),
                ),
            ),
        ),
        prefer_set="export-set",
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
        requested_show_set_ids=("missing-species-export", "missing-sibling-export"),
        requested_labels_by_set_id={
            "missing-species-export": "Missing Species Export",
            "missing-sibling-export": "Missing Sibling Export",
        },
        missing_intent_set_ids=("missing-sibling-export",),
    )
    assert outcome.transition_outcome is not None
    assert outcome.transition_outcome.kind is DisplayTransitionOutcomeKind.DENIED
    assert outcome.transition_outcome.cause is DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE
    assert outcome.transition_outcome.requested_show_set_ids == (
        "missing-species-export",
        "missing-sibling-export",
    )
    assert outcome.transition_outcome.display_set_ids == ()
    assert outcome.transition_outcome.unresolved_intent_set_ids == (
        "missing-sibling-export",
        "missing-species-export",
    )
    assert outcome.transition_outcome.missing_intent_set_ids == ("missing-sibling-export",)
    assert outcome.transition_outcome.semantic_unavailable_set_ids == ("missing-species-export",)
    assert window.results_controller.active_display_transaction() is previous_transaction


def test_resolved_display_rejects_owned_species_missing_from_series(
    prepared_window,
):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    window._plot_tabs._tabs.setCurrentIndex(0)
    first_species = species_names[0]
    previous_transaction = window.results_controller.active_display_transaction()
    assert previous_transaction is not None

    outcome = window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id="incomplete-species-export",
                label="Incomplete Species Export",
                entry=_display_payload(
                    t=t,
                    series={first_species: np.asarray(series_map[first_species], dtype=float)},
                    mechanism_text=window._get_mechanism_text(),
                    owned_species=(first_species, "missing_species"),
                ),
            ),
        ),
        prefer_set="incomplete-species-export",
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    assert outcome.transition_outcome is not None
    assert outcome.transition_outcome.kind is DisplayTransitionOutcomeKind.DENIED
    assert outcome.transition_outcome.cause is DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE
    assert outcome.transition_outcome.unresolved_intent_set_ids == ("incomplete-species-export",)
    assert outcome.transition_outcome.semantic_unavailable_set_ids == ("incomplete-species-export",)
    assert window.results_controller.active_display_transaction() is previous_transaction


def test_resolved_display_semantic_failure_does_not_veto_valid_sibling(prepared_window):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    first_species = species_names[0]
    previous_transaction = window.results_controller.active_display_transaction()
    assert previous_transaction is not None

    outcome = window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id="valid-resolved-sibling",
                label="Valid Resolved Sibling",
                entry=_display_payload(
                    t=t,
                    series={first_species: np.asarray(series_map[first_species], dtype=float)},
                    mechanism_text=window._get_mechanism_text(),
                    owned_species=(first_species,),
                ),
            ),
            ResolvedBatchDisplayRequestEntry(
                set_id="semantic-resolved-sibling",
                label="Semantic Resolved Sibling",
                entry=_display_payload(
                    t=t,
                    series={first_species: np.asarray(series_map[first_species], dtype=float)},
                    mechanism_text=window._get_mechanism_text(),
                    owned_species=(first_species, "missing_species"),
                ),
            ),
        ),
        prefer_set="valid-resolved-sibling",
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
        requested_show_set_ids=("valid-resolved-sibling", "semantic-resolved-sibling"),
        requested_labels_by_set_id={
            "valid-resolved-sibling": "Valid Resolved Sibling",
            "semantic-resolved-sibling": "Semantic Resolved Sibling",
        },
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert transition.requested_show_set_ids == (
        "valid-resolved-sibling",
        "semantic-resolved-sibling",
    )
    assert transition.display_set_ids == ("valid-resolved-sibling",)
    assert transition.unresolved_intent_set_ids == ("semantic-resolved-sibling",)
    assert transition.semantic_unavailable_set_ids == ("semantic-resolved-sibling",)
    active = window.results_controller.active_display_transaction()
    assert active is not None
    assert active is not previous_transaction
    assert active.display_set_ids == ("valid-resolved-sibling",)
    assert "semantic-resolved-sibling" not in active.display_set_ids


def test_cached_display_scope_exports_only_cache_owned_species_metadata(
    prepared_window,
):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    window._plot_tabs._tabs.setCurrentIndex(0)
    first_species = species_names[0]
    second_species = species_names[1]
    set_id = str(window._batch_store.set_id_for_row(0))
    set_name = str(window._batch_store.set_name_for_row(0))
    cache_key = "export-cache-owned-species"
    cache = window.simulation_controller.batch_cache
    full_series = {
        first_species: np.asarray(series_map[first_species], dtype=float),
        second_species: np.asarray(series_map[second_species], dtype=float),
    }
    cache.put_completion_entry(
        cache_key=cache_key,
        set_id=set_id,
        is_preview=False,
        t=t,
        series=full_series,
        mechanism_text=window._get_mechanism_text(),
        completion_provenance=completion_provenance_payload(
            t=t,
            series=full_series,
            mechanism_text=window._get_mechanism_text(),
        ),
        owned_species=(first_species,),
    )
    cache.apply_explicit_cache_reconciliation(
        clear_active_cache_identity_state=False,
        active_cache_key=cache_key,
        active_cache_preview_token=None,
        active_cache_preview_scope_set_ids=None,
        active_cache_valid_set_ids=(set_id,),
        active_cache_invalidated_set_ids=(),
    )

    outcome = window.results_controller.publish_cached_batch_display_scope(
        cache_key=cache_key,
        requested_show_set_ids=(set_id,),
        prefer_set=set_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    assert outcome.transition_outcome is not None
    assert outcome.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    active = window.results_controller.active_display_transaction()
    assert active is not None
    assert active.sets[f"result:{set_id}"].owned_species == (first_species,)
    assert tuple(active.sets[f"result:{set_id}"].series) == (first_species,)
    provenance = window._simulation_provenance_owner.last_simulation_provenance
    assert provenance["species_names"] == [first_species]
    assert tuple(window._simulation_provenance_owner.last_simulation_ctc) == (first_species,)
    plot = window._plot_tabs.get_current_plot()
    plot.clear()
    clipboard = QtWidgets.QApplication.clipboard()
    assert clipboard is not None
    clipboard.clear()
    plot._copy_all()
    assert f"{set_name}::{first_species}" in clipboard.text()
    headers, _rows = window.results_controller.build_main_plot_csv_export("all")
    assert headers == ["Time (s)", f"[{first_species}]"]


def test_explicit_cached_display_scope_uses_requested_cache_identity(prepared_window):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    window._plot_tabs._tabs.setCurrentIndex(0)
    first_species = species_names[0]
    set_id = str(window._batch_store.set_id_for_row(0))
    set_name = str(window._batch_store.set_name_for_row(0))
    cache = window.simulation_controller.batch_cache
    cache.put_completion_entry(
        cache_key="requested-display-cache",
        set_id=set_id,
        is_preview=False,
        t=t,
        series={first_species: np.asarray(series_map[first_species], dtype=float)},
        mechanism_text=window._get_mechanism_text(),
        completion_provenance=completion_provenance_payload(
            t=t,
            series={first_species: np.asarray(series_map[first_species], dtype=float)},
            mechanism_text=window._get_mechanism_text(),
        ),
        owned_species=(first_species,),
    )
    cache.apply_explicit_cache_reconciliation(
        clear_active_cache_identity_state=False,
        active_cache_key="active-display-cache",
        active_cache_preview_token=None,
        active_cache_preview_scope_set_ids=None,
        active_cache_valid_set_ids=("active-only-set",),
        active_cache_invalidated_set_ids=(),
    )
    assert window._batch_model.set_row_requested_show(0, False)

    outcome = window.results_controller.publish_cached_batch_display_scope(
        cache_key="requested-display-cache",
        requested_show_set_ids=(set_id,),
        prefer_set=set_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    assert outcome.transition_outcome is not None
    assert outcome.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    active = window.results_controller.active_display_transaction()
    assert active is not None
    assert active.display_set_ids == (set_id,)
    assert active.primary_display_set_id == set_id
    assert active.sets[f"result:{set_id}"].label == set_name
    provenance = window._simulation_provenance_owner.last_simulation_provenance
    assert provenance["display_transaction"]["display_set_ids"] == [set_id]
    assert "requested_show_set_ids" not in provenance["display_transaction"]


def test_cached_display_scope_records_missing_request_members(prepared_window):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    first_species = species_names[0]
    cached_id = str(window._batch_store.set_id_for_row(0))
    add_btn = window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    missing_id = str(window._batch_store.set_id_for_row(1))
    cache = window.simulation_controller.batch_cache
    cache.put_completion_entry(
        cache_key="partial-request-cache",
        set_id=cached_id,
        is_preview=False,
        t=t,
        series={first_species: np.asarray(series_map[first_species], dtype=float)},
        mechanism_text=window._get_mechanism_text(),
        completion_provenance=completion_provenance_payload(
            t=t,
            series={first_species: np.asarray(series_map[first_species], dtype=float)},
            mechanism_text=window._get_mechanism_text(),
        ),
        owned_species=(first_species,),
    )

    outcome = window.results_controller.publish_cached_batch_display_scope(
        cache_key="partial-request-cache",
        requested_show_set_ids=(cached_id, missing_id),
        prefer_set=cached_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    assert outcome.transition_outcome is not None
    assert outcome.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert outcome.transition_outcome.requested_show_set_ids == (cached_id, missing_id)
    assert outcome.transition_outcome.display_set_ids == (cached_id,)
    assert outcome.transition_outcome.unresolved_intent_set_ids == (missing_id,)
    assert outcome.transition_outcome.missing_intent_set_ids == (missing_id,)
    active = window.results_controller.active_display_transaction()
    assert active is not None
    assert active.display_set_ids == (cached_id,)
    assert missing_id not in active.display_set_ids


def test_copy_all_uses_active_transaction_without_plot_series_authority(prepared_window):
    window, _dataset_panel, species_names, _series_map, _t = prepared_window
    window._plot_tabs._tabs.setCurrentIndex(0)
    plot = window._plot_tabs.get_current_plot()
    first_species = species_names[0]
    plot.set_selected_series([first_species])
    plot.clear()
    clipboard = QtWidgets.QApplication.clipboard()
    assert clipboard is not None
    clipboard.clear()

    plot._copy_all()

    text = clipboard.text()
    assert "Export Set::Time (s)" in text
    assert f"Export Set::{first_species}" in text


def test_cached_display_semantic_failure_records_unavailable_set(prepared_window):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    window._plot_tabs._tabs.setCurrentIndex(0)
    first_species = species_names[0]
    set_id = str(window._batch_store.set_id_for_row(0))
    cache_key = "semantic-invalid-cache"
    cache = window.simulation_controller.batch_cache
    cache.put_completion_entry(
        cache_key=cache_key,
        set_id=set_id,
        is_preview=False,
        t=t,
        series={first_species: np.asarray(series_map[first_species], dtype=float)},
        mechanism_text=window._get_mechanism_text(),
        completion_provenance=completion_provenance_payload(
            t=t,
            series={first_species: np.asarray(series_map[first_species], dtype=float)},
            mechanism_text=window._get_mechanism_text(),
        ),
        owned_species=(first_species, "missing_species"),
    )

    outcome = window.results_controller.publish_cached_batch_display_scope(
        cache_key=cache_key,
        requested_show_set_ids=(set_id,),
        prefer_set=set_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    assert outcome.transition_outcome is not None
    assert outcome.transition_outcome.kind is DisplayTransitionOutcomeKind.DENIED
    assert outcome.transition_outcome.cause is DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE
    assert outcome.transition_outcome.unresolved_intent_set_ids == (set_id,)
    assert outcome.transition_outcome.semantic_unavailable_set_ids == (set_id,)


def test_cached_display_semantic_failure_does_not_veto_valid_cached_sibling(prepared_window):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    first_species = species_names[0]
    valid_id = str(window._batch_store.set_id_for_row(0))
    add_btn = window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    semantic_id = str(window._batch_store.set_id_for_row(1))
    cache_key = "semantic-invalid-cache-subset"
    cache = window.simulation_controller.batch_cache
    cache.put_completion_entry(
        cache_key=cache_key,
        set_id=valid_id,
        is_preview=False,
        t=t,
        series={first_species: np.asarray(series_map[first_species], dtype=float)},
        mechanism_text=window._get_mechanism_text(),
        completion_provenance=completion_provenance_payload(
            t=t,
            series={first_species: np.asarray(series_map[first_species], dtype=float)},
            mechanism_text=window._get_mechanism_text(),
        ),
        owned_species=(first_species,),
    )
    cache.put_completion_entry(
        cache_key=cache_key,
        set_id=semantic_id,
        is_preview=False,
        t=t,
        series={first_species: np.asarray(series_map[first_species], dtype=float)},
        mechanism_text=window._get_mechanism_text(),
        completion_provenance=completion_provenance_payload(
            t=t,
            series={first_species: np.asarray(series_map[first_species], dtype=float)},
            mechanism_text=window._get_mechanism_text(),
        ),
        owned_species=(first_species, "missing_species"),
    )

    outcome = window.results_controller.publish_cached_batch_display_scope(
        cache_key=cache_key,
        requested_show_set_ids=(valid_id, semantic_id),
        prefer_set=valid_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert transition.requested_show_set_ids == (valid_id, semantic_id)
    assert transition.display_set_ids == (valid_id,)
    assert transition.unresolved_intent_set_ids == (semantic_id,)
    assert transition.semantic_unavailable_set_ids == (semantic_id,)
    active = window.results_controller.active_display_transaction()
    assert active is not None
    assert active.display_set_ids == (valid_id,)
    assert semantic_id not in active.display_set_ids


def test_main_plot_direct_export_interfaces_require_active_display_transaction(prepared_window):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    window._plot_tabs._tabs.setCurrentIndex(0)
    first_species = species_names[0]
    plot = window._plot_tabs.get_current_plot()
    window.results_controller.clear_active_display_transaction()
    plot.set_data(
        t,
        {first_species: np.asarray(series_map[first_species], dtype=float)},
        label="Unowned Plot State",
        primary_set_id="unowned-plot-state",
        owned_species=(first_species,),
    )

    assert window.results_controller.active_display_transaction() is None
    assert plot.export_payload() is None
    with pytest.raises(ValueError, match="active simulation display transaction"):
        plot.build_visible_export("all")


def test_copy_visible_uses_visible_plot_data_not_copy_all_provider(prepared_window):
    window, _dataset_panel, species_names, _series_map, _t = prepared_window
    window._plot_tabs._tabs.setCurrentIndex(0)
    plot = window._plot_tabs.get_current_plot()
    first_species = species_names[0]
    plot.set_selected_series([first_species])
    calls: list[str] = []

    def _unexpected_copy_all_plan():
        calls.append("called")
        raise AssertionError("Copy Visible must not use the Copy All ADT provider")

    plot.set_copy_all_export_plan_provider(_unexpected_copy_all_plan)
    clipboard = QtWidgets.QApplication.clipboard()
    assert clipboard is not None
    clipboard.clear()

    plot._copy_visible_data()

    assert calls == []
    text = clipboard.text()
    assert "export-set::Time (s)" in text
    assert f"export-set::{first_species}" in text


def test_copy_all_uses_adt_displayed_truth_not_axis_or_dataset_overlay_state(prepared_window):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    assert len(species_names) >= 2
    window._plot_tabs._tabs.setCurrentIndex(0)
    plot = window._plot_tabs.get_current_plot()
    first_species = species_names[0]
    second_species = species_names[1]
    plot.set_selected_series([first_species])
    plot.set_overlay_catalog(
        {
            "Dataset 1": {
                "t": t,
                "species": {first_species: series_map[first_species]},
            }
        }
    )
    plot.overlay_panel().reconcile_selection(
        previous_selected_datasets=(),
        previous_enabled_species={},
        include_dataset_ids=("Dataset 1",),
        ordered_dataset_ids=("Dataset 1",),
        allow_default_include=True,
        emit=True,
    )
    clipboard = QtWidgets.QApplication.clipboard()
    assert clipboard is not None
    clipboard.clear()

    plot._copy_all()

    text = clipboard.text()
    assert f"Export Set::{first_species}" in text
    assert f"Export Set::{second_species}" in text
    assert "Dataset overlay:" not in text


def test_copy_visible_reference_label_suffix_is_idempotent(prepared_window):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    window._plot_tabs._tabs.setCurrentIndex(0)
    first_species = species_names[0]
    plot = window._plot_tabs.get_current_plot()
    plot.set_data(
        t,
        {first_species: np.asarray(series_map[first_species], dtype=float)},
        label="Export Set",
        primary_set_id="export-set",
        overlays=[
            {
                "label": "Export Set [ref]",
                "t": t,
                "series": {first_species: np.asarray(series_map[first_species], dtype=float) + 1.0},
                "set_id": "export-set",
                "layer_kind": "reference",
                "layer_id": "reference:export-set",
            }
        ],
        owned_species=(first_species,),
    )
    plot.set_reference_layers_visible(True)
    plot.set_selected_series([first_species])
    clipboard = QtWidgets.QApplication.clipboard()
    assert clipboard is not None
    clipboard.clear()

    plot._copy_visible_data()

    text = clipboard.text()
    assert "Export Set [ref]::Time (s)" in text
    assert f"Export Set [ref]::{first_species}" in text
    assert "Export Set [ref] [ref]" not in text


def test_csv_export_from_simulation_tab_omits_rendered_dataset_overlays(
    tmp_path,
    prepared_window,
):
    window, _dataset_panel, species_names, series_map, t = prepared_window
    first_species = species_names[0]
    csv_path = tmp_path / "simulation-overlay.csv"
    window._plot_tabs._tabs.setCurrentIndex(0)
    plot = window._plot_tabs.get_current_plot()
    plot.set_overlay_catalog(
        {
            "Dataset 1": {
                "t": t,
                "species": {first_species: series_map[first_species]},
            }
        }
    )
    plot.overlay_panel().reconcile_selection(
        previous_selected_datasets=(),
        previous_enabled_species={},
        include_dataset_ids=("Dataset 1",),
        ordered_dataset_ids=("Dataset 1",),
        allow_default_include=True,
        emit=True,
    )
    window.project_controller.handle_export_config(
        {"path": str(csv_path), "mode": "default", "scope": "all"}
    )

    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert "Dataset overlay: Dataset 1::Time (s)" not in rows[0]
    assert f"Dataset overlay: Dataset 1::{first_species}" not in rows[0]


def test_dataset_tab_csv_export(tmp_path, prepared_window):
    window, dataset_panel, _species_names, _series_map, _t = prepared_window
    csv_path = tmp_path / "dataset.csv"
    window._plot_tabs._tabs.setCurrentWidget(dataset_panel)
    window.project_controller.handle_export_config({"path": str(csv_path), "mode": "default", "scope": "all"})
    assert csv_path.exists()
