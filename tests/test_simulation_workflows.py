from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.computational_mode import GENERATED_BLOCK_END, GENERATED_BLOCK_START
from kindred.gui.controllers.batch_run_context_owner import BatchContextSeed
from kindred.gui.ports import (
    ActiveDisplayKind,
    BatchDisplayRequestCoverage,
    BatchDisplayRequestResolution,
    CompletedRunDisplayIntent,
    CompletedRunDisplayTransaction,
    CompletionDisplayEntry,
    DisplayEventKind,
    DisplayRefreshSource,
    DisplayStatus,
    DisplayTransitionCause,
    DisplayTransitionOutcome,
    DisplayTransitionOutcomeKind,
    FreshPreviewDisplayEntry,
    FreshPreviewDisplayTransaction,
    ResolvedBatchDisplayRequestEntry,
    SimulationCompletionDisplayOutcome,
)

from tests.workflow_helpers import (
    completion_provenance_payload,
    set_batch_current_and_selected_rows,
    slider_handle_center,
)


pytestmark = [pytest.mark.gui, pytest.mark.slow]


def _active_display_set_ids(main_window) -> tuple[str, ...]:
    transaction = main_window.results_controller.active_display_transaction()
    return tuple(transaction.display_set_ids) if transaction is not None else ()


def _completion_display_entry(
    *,
    set_id: str,
    label: str,
    values: list[float],
    mechanism_text: str,
) -> CompletionDisplayEntry:
    t = np.asarray([0.0, 1.0], dtype=float)
    series = {"A": np.asarray(values, dtype=float)}
    return CompletionDisplayEntry(
        set_id=str(set_id),
        label=str(label),
        t=t,
        series=series,
        algebra_scalars={},
        solver_provenance={},
        mechanism_text=str(mechanism_text),
        solver_config={},
        warnings=(),
        completion_provenance=completion_provenance_payload(
            t=t,
            series=series,
            mechanism_text=str(mechanism_text),
        ),
        owned_species=("A",),
    )


def test_main_window_override_materialization_preserves_dg_authority_without_kr(main_window):
    source = "equilibrium: A <-> B; kf=6.0; dG_eq=-1.0"
    main_window._mechanism_editor._reactions_text.setPlainText(source)
    main_window._extract_and_populate_variables()

    updated = main_window._apply_overrides_to_text(source, overrides={"kf1": 7.0})

    assert "kf=7" in updated
    assert "dG_eq=-1.0" in updated
    assert "kr=" not in updated
    parse_dsl_to_mechanism(updated, initials={})


def test_main_window_kf_edit_refreshes_derived_kr_readout_for_dg_authority(main_window):
    source = "T=298.15\nenergy=J/mol\nequilibrium: A <-> B; kf=6.0; dG_eq=0.0"
    main_window._mechanism_editor._reactions_text.setPlainText(source)
    main_window._extract_and_populate_variables()

    main_window._update_variable_in_mechanism("kf1", 12.0, commit=True)

    variables = main_window._mechanism_editor._variable_sliders.get_variables()
    assert variables["kf1"] == pytest.approx(12.0)
    assert variables["Keq1"] == pytest.approx(1.0)
    assert variables["kr1"] == pytest.approx(12.0)


def test_main_window_kf_edit_refreshes_derived_kr_readout_with_cm_std_ratio(main_window):
    source = (
        "T=298.15\n"
        "energy=J/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; dG_eq=0.0; cm_id=cm1; cm_std_ratio=0.5\n"
        f"{GENERATED_BLOCK_END}"
    )
    main_window._mechanism_editor._reactions_text.setPlainText(source)
    main_window._extract_and_populate_variables()

    main_window._update_variable_in_mechanism("kf1", 20.0, commit=True)

    variables = main_window._mechanism_editor._variable_sliders.get_variables()
    assert variables["kf1"] == pytest.approx(20.0)
    assert variables["Keq1"] == pytest.approx(1.0)
    assert variables["kr1"] == pytest.approx(40.0)


def test_computational_mode_fast_eq_override_respects_blocked_derived_kr_constraint(main_window, monkeypatch):
    source = (
        "T=298.15\n"
        "energy=J/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; dG_eq=0.0; cm_id=cm1; cm_std_ratio=1.0\n"
        f"{GENERATED_BLOCK_END}\n"
        "param kr1 = 99.0\n"
    )

    monkeypatch.setattr(
        main_window,
        "_collect_energy_overrides",
        lambda **_kwargs: [
            (
                "dG_eq1",
                1000.0,
                {"role": "dG_eq_fast", "cm_id": "cm1", "unit": "J/mol", "kf_fixed": 10.0, "std_ratio": 1.0},
            )
        ],
    )

    updated = main_window._apply_energy_overrides_to_computational_mode_fast_equilibria(source)

    assert updated == source


def test_deleting_displayed_batch_set_deauthorizes_active_completed_run_transaction(
    main_window,
    qtbot,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)

    first_id = str(main_window._batch_store.set_id_for_row(0))
    second_id = str(main_window._batch_store.set_id_for_row(1))
    first_name = str(main_window._batch_store.set_name_for_row(0))
    second_name = str(main_window._batch_store.set_name_for_row(1))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(first_id, second_id),
        labels_by_set_id={first_id: first_name, second_id: second_name},
        primary_set_id=first_id,
        cache_key="delete-displayed-batch-set",
        run_id=1,
        request_id=1,
        owned_species_by_set_id={first_id: ("A",), second_id: ("A",)},
        run_target_set_ids=(first_id, second_id),
    )
    outcome = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=first_id,
                    label=first_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
                _completion_display_entry(
                    set_id=second_id,
                    label=second_name,
                    values=[0.5, 0.2],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(first_id, second_id),
            display_primary_set_id=first_id,
            failed_set_ids=(),
        )
    )
    assert outcome.transition_outcome is not None
    assert outcome.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert first_id in _active_display_set_ids(main_window)

    set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    delete_btn = main_window.findChild(QtWidgets.QPushButton, "deleteBatchSetButton")
    assert delete_btn is not None
    delete_btn.click()

    assert first_id not in _active_display_set_ids(main_window)


def test_allow_edit_accepting_new_mechanism_deauthorizes_active_display(main_window, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="allow-edit-stale-display",
        run_id=31,
        request_id=32,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )
    published = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert main_window.results_controller.active_display_transaction() is not None

    monkeypatch.setattr(main_window, "_prompt_mechanism_edit_unlock_warning", lambda: True)
    main_window._on_mechanism_edit_lock_action_triggered(True)
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=2.0")
    main_window._on_mechanism_edit_lock_action_triggered(False)

    assert main_window.results_controller.active_display_transaction() is None
    transition = main_window.results_controller._last_display_transition_outcome
    assert transition is not None
    assert transition.active_transaction is None
    assert transition.previous_transaction is not None
    assert transition.display_status is DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE
    assert transition.cause is DisplayTransitionCause.AFFECTED_SCOPE_INTERSECTS_ACTIVE_DISPLAY


def test_empty_show_with_active_completed_run_returns_typed_deauthorization_outcome(
    main_window,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="displayed-empty-show",
        run_id=5,
        request_id=6,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )
    published = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    signal_blocker = QtCore.QSignalBlocker(main_window._batch_model)
    try:
        assert main_window._batch_model.set_row_requested_show(0, False)
    finally:
        del signal_blocker

    outcome = main_window.results_controller.refresh_display_from_request_scope(
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.CLEARED
    assert transition.active_transaction is None
    assert transition.previous_transaction is not None
    assert transition.affected_set_ids == (displayed_id,)
    assert main_window.results_controller.active_display_transaction() is None


def test_completed_run_publication_uses_captured_run_start_show_intent(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    main_window._batch_model.set_row_requested_show(0, False)
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="stale-run-start-show-intent",
        run_id=41,
        request_id=42,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )

    outcome = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert transition.requested_show_set_ids == (displayed_id,)
    assert transition.display_set_ids == (displayed_id,)
    assert main_window.results_controller.active_display_transaction() is not None


def test_stale_simulation_completion_does_not_publish_cache_or_display(main_window):
    controller = main_window.simulation_controller
    controller._active_run_id = 3
    controller._latest_sim_request_id = 5
    controller._simulation_running = True
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 2}
    controller.batch_context_owner.load_context(
        BatchContextSeed(
            active=True,
            parallel=False,
            fast_mode=False,
            run_id=3,
            request_id=5,
            runtime_input_global_epoch=0,
            runtime_input_set_epoch_by_set_id={"id1": 1},
            cache_key="workflow-stale-completion",
            queue_ids=("id1",),
            queue_names=("set1",),
            rows=(0,),
            pos=0,
            total=1,
            primary_set_id="id1",
        )
    )
    callback_identity = controller._capture_simulation_callback_identity(
        run_id=3,
        fast_mode=False,
        request_id=5,
        preview_owner_epoch=None,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="workflow-stale-completion",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
        simulation_identity={},
    )

    controller._cache_admin.publish_completion_cache_truth = MagicMock()
    controller._cache_admin.publish_completion_cache = MagicMock()
    controller.ui.results.publish_simulation_completion_result = MagicMock()
    controller.ui.results.publish_completion_intervention_annotations = MagicMock()
    controller.ui.provenance.publish_simulation_completion_provenance = MagicMock()

    controller.on_simulation_complete(
        {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "Y": np.asarray([[1.0, 0.5]], dtype=float),
            "species_names": ["A"],
            "algebra_scalars": {},
            "mechanism": None,
            "mechanism_text": "reaction: A -> B; k=1",
            "solver_config": {"solver": "BDF"},
            "fallback_occurred": False,
            "fallback_message": None,
        },
        callback_identity=callback_identity,
    )

    controller._cache_admin.publish_completion_cache_truth.assert_not_called()
    controller._cache_admin.publish_completion_cache.assert_not_called()
    controller.ui.results.publish_simulation_completion_result.assert_not_called()
    controller.ui.results.publish_completion_intervention_annotations.assert_not_called()
    controller.ui.provenance.publish_simulation_completion_provenance.assert_not_called()
    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.active is True


def test_completed_run_publication_ignores_additive_live_show_rows_outside_captured_intent(
    main_window,
    qtbot,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    first_id = str(main_window._batch_store.set_id_for_row(0))
    second_id = str(main_window._batch_store.set_id_for_row(1))
    first_name = str(main_window._batch_store.set_name_for_row(0))
    main_window._batch_model.set_row_requested_show(0, True)
    main_window._batch_model.set_row_requested_show(1, True)
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(first_id,),
        labels_by_set_id={first_id: first_name},
        primary_set_id=first_id,
        cache_key="additive-stale-run-start-show-intent",
        run_id=43,
        request_id=44,
        owned_species_by_set_id={first_id: ("A",)},
        run_target_set_ids=(first_id, second_id),
    )

    outcome = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=first_id,
                    label=first_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(first_id,),
            display_primary_set_id=first_id,
            failed_set_ids=(),
        )
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert transition.requested_show_set_ids == (first_id,)
    assert transition.display_set_ids == (first_id,)
    assert second_id not in transition.requested_show_set_ids
    assert second_id not in _active_display_set_ids(main_window)


def test_completed_run_publication_rejects_deleted_in_flight_run_target(
    main_window,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    deleted_id = str(main_window._batch_store.set_id_for_row(0))
    deleted_name = str(main_window._batch_store.set_name_for_row(0))
    set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    delete_btn = main_window.findChild(QtWidgets.QPushButton, "deleteBatchSetButton")
    assert delete_btn is not None
    delete_btn.click()
    assert main_window._batch_store.row_for_set_id(deleted_id) is None

    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(deleted_id,),
        labels_by_set_id={deleted_id: deleted_name},
        primary_set_id=deleted_id,
        cache_key="deleted-in-flight-run-target",
        run_id=45,
        request_id=46,
        owned_species_by_set_id={deleted_id: ("A",)},
        run_target_set_ids=(deleted_id,),
    )
    outcome = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=deleted_id,
                    label=deleted_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(deleted_id,),
            display_primary_set_id=deleted_id,
            failed_set_ids=(),
        )
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.FAILED
    assert transition.cause is DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE
    assert transition.active_transaction is None
    assert deleted_id in transition.affected_set_ids


def test_unavailable_non_active_show_request_preserves_valid_display(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    missing_show_id = "missing-show"
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="active-status-display",
        run_id=11,
        request_id=12,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )
    published = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED

    denied = main_window.results_controller.publish_completed_run_display_unavailable(
        cause=DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
        affected_set_ids=(missing_show_id,),
        requested_show_set_ids=(displayed_id, missing_show_id),
        requested_labels_by_set_id={
            displayed_id: displayed_name,
            missing_show_id: "Missing Show",
        },
        unresolved_intent_set_ids=(missing_show_id,),
    )

    assert denied.transition_outcome is not None
    assert denied.transition_outcome.active_transaction is not None
    assert denied.transition_outcome.previous_transaction is not None
    assert denied.transition_outcome.display_status is DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE
    assert denied.transition_outcome.requested_show_set_ids == (displayed_id, missing_show_id)
    assert denied.transition_outcome.display_set_ids == ()
    assert _active_display_set_ids(main_window) == (displayed_id,)
    assert (
        main_window._status_text_value()
        == "Result not cached; Missing Show is unavailable."
    )


def test_failed_or_semantically_unavailable_completed_member_preserves_unrelated_adt(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="failed-member-deauthorizes",
        run_id=13,
        request_id=14,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )
    published = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED

    denied = main_window.results_controller.publish_completed_run_display_unavailable(
        cause=DisplayTransitionCause.NO_DISPLAYABLE_COMPLETION_RESULTS,
        affected_set_ids=("failed-other",),
        requested_show_set_ids=(displayed_id, "failed-other"),
        requested_labels_by_set_id={
            displayed_id: displayed_name,
            "failed-other": "Failed Other",
        },
        unresolved_intent_set_ids=("failed-other",),
        failed_intent_set_ids=("failed-other",),
    )

    assert denied.transition_outcome is not None
    assert denied.transition_outcome.active_transaction is not None
    assert denied.transition_outcome.previous_transaction is not None
    assert denied.transition_outcome.display_status is DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE
    assert denied.transition_outcome.requested_show_set_ids == (displayed_id, "failed-other")
    assert denied.transition_outcome.display_set_ids == ()
    assert denied.transition_outcome.failed_intent_set_ids == ("failed-other",)
    assert _active_display_set_ids(main_window) == (displayed_id,)

    semantic_denied = main_window.results_controller.publish_completed_run_display_unavailable(
        cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
        affected_set_ids=("semantic-other",),
        requested_show_set_ids=(displayed_id, "semantic-other"),
        requested_labels_by_set_id={
            displayed_id: displayed_name,
            "semantic-other": "Semantic Other",
        },
        unresolved_intent_set_ids=("semantic-other",),
        semantic_unavailable_set_ids=("semantic-other",),
    )

    assert semantic_denied.transition_outcome is not None
    assert semantic_denied.transition_outcome.active_transaction is not None
    assert semantic_denied.transition_outcome.previous_transaction is not None
    assert semantic_denied.transition_outcome.display_status is DisplayStatus.DISPLAY_DENIED
    assert semantic_denied.transition_outcome.requested_show_set_ids == (displayed_id, "semantic-other")
    assert semantic_denied.transition_outcome.display_set_ids == ()
    assert semantic_denied.transition_outcome.semantic_unavailable_set_ids == ("semantic-other",)
    assert _active_display_set_ids(main_window) == (displayed_id,)


def test_completed_run_transaction_displays_successful_sibling_and_records_failed_target(
    main_window,
    qtbot,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    failed_id = str(main_window._batch_store.set_id_for_row(1))
    non_run_show_id = str(main_window._batch_store.set_id_for_row(2))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    failed_name = str(main_window._batch_store.set_name_for_row(1))
    main_window._batch_model.set_row_requested_show(0, True)
    main_window._batch_model.set_row_requested_show(1, True)
    main_window._batch_model.set_row_requested_show(2, True)
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id, failed_id, non_run_show_id),
        labels_by_set_id={
            displayed_id: displayed_name,
            failed_id: failed_name,
            non_run_show_id: str(main_window._batch_store.set_name_for_row(2)),
        },
        primary_set_id=displayed_id,
        cache_key="partial-composite-completed-run",
        run_id=141,
        request_id=142,
        owned_species_by_set_id={displayed_id: ("A",), failed_id: ("A",), non_run_show_id: ("A",)},
        run_target_set_ids=(displayed_id, failed_id),
    )

    outcome = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(failed_id,),
            failed_intent_set_ids=(failed_id,),
            missing_intent_set_ids=(non_run_show_id,),
            unresolved_intent_set_ids=(failed_id, non_run_show_id),
        )
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert transition.display_status is DisplayStatus.DISPLAYED_COMPLETED_RUN
    assert transition.requested_show_set_ids == (displayed_id, failed_id, non_run_show_id)
    assert transition.display_set_ids == (displayed_id,)
    assert transition.attempted_display_set_ids == (displayed_id,)
    assert transition.failed_intent_set_ids == (failed_id,)
    assert transition.missing_intent_set_ids == (non_run_show_id,)
    assert transition.unresolved_intent_set_ids == (failed_id, non_run_show_id)
    active = main_window.results_controller.active_display_transaction()
    assert active is not None
    assert active.display_set_ids == (displayed_id,)
    assert failed_id not in active.display_set_ids
    assert non_run_show_id not in active.display_set_ids
    assert not hasattr(active, "request_scope")
    assert not hasattr(active, "event")
    provenance = main_window._simulation_provenance_owner.last_simulation_provenance
    display_transaction = provenance.get("display_transaction")
    assert isinstance(display_transaction, dict)
    assert display_transaction["display_set_ids"] == [displayed_id]
    assert "requested_show_set_ids" not in display_transaction
    assert "unresolved_intent_set_ids" not in display_transaction
    assert "failed_intent_set_ids" not in display_transaction
    assert "missing_intent_set_ids" not in display_transaction
    assert main_window._status_text_value() == (
        "Displayed completed run; set2 failed and set3 needs a run."
    )

    plot = main_window._plot_tabs.get_current_plot()

    def _unexpected_missing_prompt(items):
        raise AssertionError("Copy All must not derive request-outcome prompts from ADT")

    monkeypatch.setattr(plot, "_confirm_copy_all_missing_items", _unexpected_missing_prompt)
    plot._copy_all()


def test_completed_run_transaction_displays_successful_sibling_and_records_semantic_unavailable_target(
    main_window,
    qtbot,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    semantic_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    semantic_name = str(main_window._batch_store.set_name_for_row(1))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id, semantic_id),
        labels_by_set_id={
            displayed_id: displayed_name,
            semantic_id: semantic_name,
        },
        primary_set_id=displayed_id,
        cache_key="completed-run-semantic-subset",
        run_id=143,
        request_id=144,
        owned_species_by_set_id={
            displayed_id: ("A",),
            semantic_id: ("A", "missing_species"),
        },
        run_target_set_ids=(displayed_id, semantic_id),
    )

    outcome = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
                replace(
                    _completion_display_entry(
                        set_id=semantic_id,
                        label=semantic_name,
                        values=[0.9, 0.3],
                        mechanism_text="A -> B ; k=1.0",
                    ),
                    owned_species=("A", "missing_species"),
                ),
            ),
            display_set_ids=(displayed_id, semantic_id),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert transition.requested_show_set_ids == (displayed_id, semantic_id)
    assert transition.display_set_ids == (displayed_id,)
    assert transition.unresolved_intent_set_ids == (semantic_id,)
    assert transition.semantic_unavailable_set_ids == (semantic_id,)
    active = main_window.results_controller.active_display_transaction()
    assert active is not None
    assert active.display_set_ids == (displayed_id,)
    assert semantic_id not in active.display_set_ids


def test_completed_run_provenance_uses_displayed_species_subset(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A", "B"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    t = np.asarray([0.0, 1.0], dtype=float)
    full_series = {
        "A": np.asarray([1.0, 0.4], dtype=float),
        "B": np.asarray([0.0, 0.6], dtype=float),
    }
    entry = CompletionDisplayEntry(
        set_id=displayed_id,
        label=displayed_name,
        t=t,
        series=full_series,
        algebra_scalars={},
        solver_provenance={},
        mechanism_text="A -> B ; k=1.0",
        solver_config={},
        warnings=(),
        completion_provenance=completion_provenance_payload(
            t=t,
            series=full_series,
            mechanism_text="A -> B ; k=1.0",
        ),
        owned_species=("A",),
    )
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="displayed-subset-provenance",
        run_id=31,
        request_id=32,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )

    outcome = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(entry,),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )

    assert outcome.transition_outcome is not None
    assert outcome.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    provenance = main_window._simulation_provenance_owner.last_simulation_provenance
    assert provenance["species_names"] == ["A"]
    assert provenance["num_species"] == 1
    ctc = main_window._simulation_provenance_owner.last_simulation_ctc
    assert tuple(ctc) == ("A",)


def test_invalid_completed_run_transaction_deauthorizes_stale_adt(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="invalid-completed-run-transaction",
        run_id=15,
        request_id=16,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )
    valid_entry = _completion_display_entry(
        set_id=displayed_id,
        label=displayed_name,
        values=[1.0, 0.4],
        mechanism_text="A -> B ; k=1.0",
    )
    published = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(valid_entry,),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED

    invalid = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(replace(valid_entry, completion_provenance=None),),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )

    transition = invalid.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.FAILED
    assert transition.active_transaction is None
    assert transition.previous_transaction is not None
    assert transition.previous_transaction.display_set_ids == (displayed_id,)
    assert transition.display_status is DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE
    assert main_window.results_controller.active_display_transaction() is None
    assert _active_display_set_ids(main_window) == ()


def test_empty_completed_run_transaction_preserves_requested_scope(main_window, qtbot):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    first_id = str(main_window._batch_store.set_id_for_row(0))
    second_id = str(main_window._batch_store.set_id_for_row(1))
    first_name = str(main_window._batch_store.set_name_for_row(0))
    second_name = str(main_window._batch_store.set_name_for_row(1))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(first_id, second_id),
        labels_by_set_id={first_id: first_name, second_id: second_name},
        primary_set_id=first_id,
        cache_key="empty-completed-run-transaction",
        run_id=23,
        request_id=24,
        owned_species_by_set_id={first_id: ("A",), second_id: ("A",)},
        run_target_set_ids=(first_id, second_id),
    )

    invalid = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(),
            display_set_ids=(first_id,),
            display_primary_set_id=first_id,
            failed_set_ids=(),
        )
    )

    transition = invalid.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.FAILED
    assert transition.requested_show_set_ids == (first_id, second_id)
    assert dict(transition.requested_labels_by_set_id) == {
        first_id: first_name,
        second_id: second_name,
    }
    assert transition.display_set_ids == ()
    assert transition.attempted_display_set_ids == (first_id,)
    assert transition.affected_set_ids == (first_id,)
    assert transition.unresolved_intent_set_ids == (first_id, second_id)
    assert transition.missing_intent_set_ids == (first_id, second_id)


def test_fresh_preview_display_records_transaction_request_scope_not_live_show(main_window, qtbot):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    live_show_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    assert main_window._batch_model.set_row_requested_show(0, False)
    assert main_window.results_controller._ui.requested_show_batch_set_ids() == [live_show_id]

    t = np.asarray([0.0, 1.0], dtype=float)
    series = {"A": np.asarray([1.0, 0.6], dtype=float)}
    outcome = main_window.results_controller.publish_fresh_preview_display(
        FreshPreviewDisplayTransaction(
            entries=(
                FreshPreviewDisplayEntry(
                    set_id=displayed_id,
                    label=displayed_name,
                    t=t,
                    series=series,
                    algebra_scalars={},
                    solver_provenance={},
                    completion_provenance=completion_provenance_payload(
                        t=t,
                        series=series,
                        mechanism_text="A -> B ; k=1.0",
                    ),
                    owned_species=("A",),
                    workspace_preview_provenance={"source": "fresh-preview-regression"},
                ),
            ),
            display_set_ids=(displayed_id,),
            target_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            cache_key="fresh-preview-transaction-request-scope",
            display_source=DisplayRefreshSource.SLIDER_REPLAY,
            requested_show_set_ids=(displayed_id,),
            requested_labels_by_set_id={displayed_id: displayed_name},
            request_id=31,
            run_id=32,
        )
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert transition.requested_show_set_ids == (displayed_id,)
    assert transition.display_set_ids == (displayed_id,)
    assert live_show_id not in transition.requested_show_set_ids
    active = main_window.results_controller.active_display_transaction()
    assert active is not None
    assert active.display_set_ids == (displayed_id,)


def test_queued_fresh_preview_flush_records_queued_scope_not_live_show(main_window, qtbot):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    live_show_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    request_id = main_window.simulation_controller.next_slider_preview_request_id()
    main_window.simulation_controller._claim_preview_ownership(
        request_id=request_id,
        target_set_ids=(displayed_id,),
    )
    t = np.asarray([0.0, 1.0], dtype=float)
    series = {"A": np.asarray([1.0, 0.6], dtype=float)}
    main_window.simulation_controller.queue_slider_plot_update(
        set_id=displayed_id,
        cache_key="queued-fresh-preview-transaction-scope",
        request_id=request_id,
        run_id=None,
        slider_triggered=True,
        fresh_preview_entry=FreshPreviewDisplayEntry(
            set_id=displayed_id,
            label=displayed_name,
            t=t,
            series=series,
            algebra_scalars={},
            solver_provenance={},
            completion_provenance=completion_provenance_payload(
                t=t,
                series=series,
                mechanism_text="A -> B ; k=1.0",
            ),
            owned_species=("A",),
            workspace_preview_provenance={"source": "queued-fresh-preview-regression"},
        ),
    )
    assert main_window._batch_model.set_row_requested_show(0, False)
    assert main_window.results_controller._ui.requested_show_batch_set_ids() == [live_show_id]

    displayed = main_window.simulation_controller._flush_slider_plot_updates()

    assert displayed is True
    active = main_window.results_controller.active_display_transaction()
    assert active is not None
    assert active.display_set_ids == (displayed_id,)
    transition = main_window.results_controller._last_display_transition_outcome
    assert transition is not None
    assert transition.requested_show_set_ids == (displayed_id,)
    assert dict(transition.requested_labels_by_set_id) == {
        displayed_id: displayed_name,
    }
    assert live_show_id not in transition.requested_show_set_ids


def test_invalid_completed_run_request_preserves_unrelated_adt(main_window, qtbot):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    failed_request_id = str(main_window._batch_store.set_id_for_row(1))
    missing_request_id = "missing-invalid-completed-run-sibling"
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    failed_request_name = str(main_window._batch_store.set_name_for_row(1))
    displayed_entry = _completion_display_entry(
        set_id=displayed_id,
        label=displayed_name,
        values=[1.0, 0.4],
        mechanism_text="A -> B ; k=1.0",
    )
    published = main_window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id=displayed_id,
                label=displayed_name,
                entry=displayed_entry.to_display_payload(),
            ),
        ),
        prefer_set=displayed_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    previous_transaction = main_window.results_controller.active_display_transaction()
    assert previous_transaction is not None
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(failed_request_id, missing_request_id),
        labels_by_set_id={
            failed_request_id: failed_request_name,
            missing_request_id: "Missing Invalid Sibling",
        },
        primary_set_id=failed_request_id,
        cache_key="unrelated-invalid-completed-run-transaction",
        run_id=25,
        request_id=26,
        owned_species_by_set_id={failed_request_id: ("A",)},
        run_target_set_ids=(failed_request_id,),
    )

    invalid = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                replace(
                    displayed_entry,
                    set_id=failed_request_id,
                    label=failed_request_name,
                    completion_provenance=None,
                ),
            ),
            display_set_ids=(failed_request_id,),
            display_primary_set_id=failed_request_id,
            failed_set_ids=(),
        )
    )

    transition = invalid.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.FAILED
    assert transition.active_transaction is previous_transaction
    assert transition.previous_transaction is previous_transaction
    assert transition.requested_show_set_ids == (failed_request_id, missing_request_id)
    assert transition.display_set_ids == ()
    assert transition.attempted_display_set_ids == (failed_request_id,)
    assert transition.affected_set_ids == (failed_request_id,)
    assert transition.unresolved_intent_set_ids == (failed_request_id,)
    assert main_window.results_controller.active_display_transaction() is previous_transaction
    assert _active_display_set_ids(main_window) == (displayed_id,)


def test_invalid_completed_run_request_records_missing_sibling_without_display(
    main_window,
    qtbot,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    failed_request_id = str(main_window._batch_store.set_id_for_row(0))
    missing_request_id = "missing-invalid-no-display-sibling"
    failed_request_name = str(main_window._batch_store.set_name_for_row(0))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(failed_request_id, missing_request_id),
        labels_by_set_id={
            failed_request_id: failed_request_name,
            missing_request_id: "Missing Invalid No Display Sibling",
        },
        primary_set_id=failed_request_id,
        cache_key="invalid-completed-run-no-display-transaction",
        run_id=27,
        request_id=28,
        owned_species_by_set_id={failed_request_id: ("A",)},
        run_target_set_ids=(failed_request_id,),
    )

    invalid = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                replace(
                    _completion_display_entry(
                        set_id=failed_request_id,
                        label=failed_request_name,
                        values=[1.0, 0.4],
                        mechanism_text="A -> B ; k=1.0",
                    ),
                    completion_provenance=None,
                ),
            ),
            display_set_ids=(failed_request_id,),
            display_primary_set_id=failed_request_id,
            failed_set_ids=(),
            missing_intent_set_ids=(missing_request_id,),
            unresolved_intent_set_ids=(failed_request_id, missing_request_id),
        )
    )

    transition = invalid.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.FAILED
    assert transition.requested_show_set_ids == (failed_request_id, missing_request_id)
    assert transition.display_set_ids == ()
    assert transition.attempted_display_set_ids == (failed_request_id,)
    assert transition.affected_set_ids == (failed_request_id, missing_request_id)
    assert transition.unresolved_intent_set_ids == (failed_request_id, missing_request_id)
    assert transition.missing_intent_set_ids == (missing_request_id,)
    assert main_window.results_controller.active_display_transaction() is None
    assert (
        main_window._status_text_value()
        == "Not displayed; 1 result needs a run and set1 is unavailable."
    )


def test_direct_completion_without_species_provenance_deauthorizes_stale_adt(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="direct-missing-provenance",
        run_id=17,
        request_id=18,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )
    published = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED

    denied = main_window.results_controller.publish_direct_completion_result(
        t=np.asarray([0.0, 1.0], dtype=float),
        series={"A": np.asarray([2.0, 1.0], dtype=float)},
        batch_set=displayed_name,
        batch_set_id=displayed_id,
        algebra_scalars={},
        solver_provenance={},
        direct_completion_provenance={},
    )

    transition = denied.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.FAILED
    assert transition.active_transaction is None
    assert transition.previous_transaction is not None
    assert transition.display_status is DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE
    assert main_window.results_controller.active_display_transaction() is None


def test_denied_resolved_request_reports_request_outcome_without_reusing_active_status(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="denied-resolved-request",
        run_id=21,
        request_id=22,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )
    published = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED

    candidate_entry = _completion_display_entry(
        set_id="candidate",
        label="Candidate",
        values=[0.8, 0.3],
        mechanism_text="A -> B ; k=1.0",
    )
    denied = main_window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id="candidate",
                label="Candidate",
                entry=candidate_entry.to_display_payload(),
            ),
        ),
    )

    assert denied.transition_outcome is not None
    assert denied.transition_outcome.kind is DisplayTransitionOutcomeKind.DENIED
    assert denied.transition_outcome.display_status is DisplayStatus.DISPLAY_DENIED
    assert denied.transition_outcome.affected_set_ids == ("candidate",)
    assert denied.transition_outcome.unresolved_intent_set_ids == ("candidate",)
    assert _active_display_set_ids(main_window) == (displayed_id,)
    assert (
        main_window._status_text_value()
        == "Display unchanged; Candidate is unavailable."
    )


def test_denied_resolved_request_records_semantic_unavailable_sibling(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="denied-resolved-semantic-request",
        run_id=23,
        request_id=24,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )
    published = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED

    denied = main_window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id="semantic-denied",
                label="Semantic Denied",
                entry=replace(
                    _completion_display_entry(
                        set_id="semantic-denied",
                        label="Semantic Denied",
                        values=[0.8, 0.3],
                        mechanism_text="A -> B ; k=1.0",
                    ),
                    owned_species=("A", "missing_species"),
                ).to_display_payload(),
            ),
        ),
    )

    transition = denied.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.DENIED
    assert transition.display_status is DisplayStatus.DISPLAY_DENIED
    assert transition.requested_show_set_ids == ("semantic-denied",)
    assert transition.display_set_ids == ()
    assert transition.unresolved_intent_set_ids == ("semantic-denied",)
    assert transition.semantic_unavailable_set_ids == ("semantic-denied",)
    assert _active_display_set_ids(main_window) == (displayed_id,)
    assert (
        main_window._status_text_value()
        == "Display unchanged; Semantic Denied has no displayable result."
    )


def test_denied_cached_publication_reports_requested_outcome(main_window, qtbot):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    candidate_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="completed-run-active-for-denied-cache",
        run_id=121,
        request_id=122,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )
    published = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    t = np.asarray([0.0, 1.0], dtype=float)
    series = {"A": np.asarray([0.8, 0.2], dtype=float)}
    cache_key = "denied-cached-publication-request"
    main_window.simulation_controller.batch_cache.put_completion_entry(
        cache_key=cache_key,
        set_id=candidate_id,
        is_preview=False,
        t=t,
        series=series,
        mechanism_text="A -> B ; k=1.0",
        completion_provenance=completion_provenance_payload(
            t=t,
            series=series,
            mechanism_text="A -> B ; k=1.0",
        ),
        owned_species=("A",),
    )

    denied = main_window.results_controller.publish_cached_batch_display_scope(
        cache_key=cache_key,
        requested_show_set_ids=(candidate_id,),
        prefer_set=candidate_id,
    )

    transition = denied.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.DENIED
    assert transition.affected_set_ids == (candidate_id,)
    assert transition.unresolved_intent_set_ids == (candidate_id,)
    assert _active_display_set_ids(main_window) == (displayed_id,)
    assert (
        main_window._status_text_value()
        == "Display unchanged; set2 is unavailable."
    )


def test_completed_run_denied_refresh_records_full_requested_show_scope(main_window, qtbot):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    missing_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="completed-run-active-denied-refresh",
        run_id=131,
        request_id=132,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )
    published = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    main_window._batch_model.set_row_requested_show(0, True)
    main_window._batch_model.set_row_requested_show(1, True)
    set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])

    denied = main_window.results_controller.refresh_display_from_request_scope(
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    transition = denied.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.DENIED
    assert transition.affected_set_ids == (displayed_id, missing_id)
    assert transition.unresolved_intent_set_ids == (missing_id,)
    assert transition.missing_intent_set_ids == (missing_id,)
    assert _active_display_set_ids(main_window) == (displayed_id,)
    assert (
        main_window._status_text_value()
        == "Result not cached; set2 needs a run."
    )


def test_completed_run_denied_refresh_with_active_focus_records_missing_requested_member(
    main_window,
    qtbot,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    missing_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id,),
        labels_by_set_id={displayed_id: displayed_name},
        primary_set_id=displayed_id,
        cache_key="completed-run-active-focus-denied-refresh",
        run_id=133,
        request_id=134,
        owned_species_by_set_id={displayed_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )
    published = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
        )
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    main_window._batch_model.set_row_requested_show(0, True)
    main_window._batch_model.set_row_requested_show(1, True)
    set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    denied = main_window.results_controller.refresh_display_from_request_scope(
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    transition = denied.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.DENIED
    assert transition.affected_set_ids == (displayed_id, missing_id)
    assert transition.unresolved_intent_set_ids == (missing_id,)
    assert transition.missing_intent_set_ids == (missing_id,)
    assert _active_display_set_ids(main_window) == (displayed_id,)
    assert (
        main_window._status_text_value()
        == "Result not cached; set2 needs a run."
    )


def test_unavailable_additive_show_request_preserves_resolved_display(main_window, qtbot):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    missing_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    entry = _completion_display_entry(
        set_id=displayed_id,
        label=displayed_name,
        values=[1.0, 0.4],
        mechanism_text="A -> B ; k=1.0",
    )
    published = main_window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id=displayed_id,
                label=displayed_name,
                entry=entry.to_display_payload(),
            ),
        ),
        prefer_set=displayed_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    main_window._batch_model.set_row_requested_show(0, True)
    main_window._batch_model.set_row_requested_show(1, True)

    denied = main_window.results_controller.refresh_display_from_request_scope(
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    assert denied.transition_outcome is not None
    assert denied.transition_outcome.kind is DisplayTransitionOutcomeKind.DENIED
    assert denied.transition_outcome.display_status is DisplayStatus.DISPLAY_DENIED
    assert denied.transition_outcome.affected_set_ids == (displayed_id, missing_id)
    assert denied.transition_outcome.unresolved_intent_set_ids == (missing_id,)
    assert denied.transition_outcome.missing_intent_set_ids == (missing_id,)
    assert _active_display_set_ids(main_window) == (displayed_id,)
    active = main_window.results_controller.active_display_transaction()
    assert active is not None
    assert active.status is DisplayStatus.DISPLAYED_RESOLVED_RESULT
    assert (
        main_window._status_text_value()
        == "Result not cached; 1 result needs a run."
    )
    assert missing_id not in active.display_set_ids


def test_cached_additive_show_request_keeps_existing_displayed_member_available(main_window, qtbot):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    cached_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    cached_name = str(main_window._batch_store.set_name_for_row(1))
    displayed_entry = _completion_display_entry(
        set_id=displayed_id,
        label=displayed_name,
        values=[1.0, 0.4],
        mechanism_text="A -> B ; k=1.0",
    )
    published = main_window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id=displayed_id,
                label=displayed_name,
                entry=displayed_entry.to_display_payload(),
            ),
        ),
        prefer_set=displayed_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    t = np.asarray([0.0, 1.0], dtype=float)
    series = {"A": np.asarray([2.0, 1.0], dtype=float)}
    cache_key = "additive-cached-sibling"
    main_window.simulation_controller.batch_cache.put_completion_entry(
        cache_key=cache_key,
        set_id=cached_id,
        is_preview=False,
        t=t,
        series=series,
        mechanism_text="A -> B ; k=1.0",
        completion_provenance=completion_provenance_payload(
            t=t,
            series=series,
            mechanism_text="A -> B ; k=1.0",
        ),
        owned_species=("A",),
    )

    outcome = main_window.results_controller.publish_cached_batch_display_scope(
        cache_key=cache_key,
        requested_show_set_ids=(displayed_id, cached_id),
        prefer_set=cached_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert transition.requested_show_set_ids == (displayed_id, cached_id)
    assert set(transition.display_set_ids) == {displayed_id, cached_id}
    assert transition.unresolved_intent_set_ids == ()
    assert transition.missing_intent_set_ids == ()
    active = main_window.results_controller.active_display_transaction()
    assert active is not None
    assert active.primary_display_set_id == cached_id
    assert set(active.display_set_ids) == {displayed_id, cached_id}
    assert active.sets[f"result:{displayed_id}"].label == displayed_name
    assert active.sets[f"result:{cached_id}"].label == cached_name
    assert main_window._status_text_value() == "Displayed cached result."


def test_cached_additive_show_request_excludes_invalidated_active_member(main_window, qtbot):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    cached_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    cached_name = str(main_window._batch_store.set_name_for_row(1))
    displayed_entry = _completion_display_entry(
        set_id=displayed_id,
        label=displayed_name,
        values=[1.0, 0.4],
        mechanism_text="A -> B ; k=1.0",
    )
    published = main_window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id=displayed_id,
                label=displayed_name,
                entry=displayed_entry.to_display_payload(),
            ),
        ),
        prefer_set=displayed_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    t = np.asarray([0.0, 1.0], dtype=float)
    series = {"A": np.asarray([2.0, 1.0], dtype=float)}
    cache_key = "additive-invalidated-active-sibling"
    main_window.simulation_controller.batch_cache.put_completion_entry(
        cache_key=cache_key,
        set_id=cached_id,
        is_preview=False,
        t=t,
        series=series,
        mechanism_text="A -> B ; k=1.0",
        completion_provenance=completion_provenance_payload(
            t=t,
            series=series,
            mechanism_text="A -> B ; k=1.0",
        ),
        owned_species=("A",),
    )
    main_window.simulation_controller.batch_cache.apply_explicit_cache_reconciliation(
        clear_active_cache_identity_state=False,
        active_cache_key=cache_key,
        active_cache_preview_token=None,
        active_cache_preview_scope_set_ids=None,
        active_cache_valid_set_ids=(cached_id,),
        active_cache_invalidated_set_ids=(displayed_id,),
    )

    outcome = main_window.results_controller.publish_cached_batch_display_scope(
        cache_key=cache_key,
        requested_show_set_ids=(displayed_id, cached_id),
        prefer_set=cached_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert transition.requested_show_set_ids == (displayed_id, cached_id)
    assert transition.display_set_ids == (cached_id,)
    assert transition.unresolved_intent_set_ids == (displayed_id,)
    assert transition.missing_intent_set_ids == (displayed_id,)
    active = main_window.results_controller.active_display_transaction()
    assert active is not None
    assert active.primary_display_set_id == cached_id
    assert active.display_set_ids == (cached_id,)
    assert f"result:{displayed_id}" not in active.sets
    assert active.sets[f"result:{cached_id}"].label == cached_name


def test_cached_refresh_clears_active_only_invalidated_display(main_window, qtbot):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    displayed_entry = _completion_display_entry(
        set_id=displayed_id,
        label=displayed_name,
        values=[1.0, 0.4],
        mechanism_text="A -> B ; k=1.0",
    )
    published = main_window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id=displayed_id,
                label=displayed_name,
                entry=displayed_entry.to_display_payload(),
            ),
        ),
        prefer_set=displayed_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    previous = main_window.results_controller.active_display_transaction()
    assert previous is not None
    cache_key = "active-only-invalidated-cached-refresh"
    main_window.simulation_controller.batch_cache.apply_explicit_cache_reconciliation(
        clear_active_cache_identity_state=False,
        active_cache_key=cache_key,
        active_cache_preview_token=None,
        active_cache_preview_scope_set_ids=None,
        active_cache_valid_set_ids=(),
        active_cache_invalidated_set_ids=(displayed_id,),
    )

    outcome = main_window.results_controller.publish_cached_batch_display_scope(
        cache_key=cache_key,
        requested_show_set_ids=(displayed_id,),
        prefer_set=displayed_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.DENIED
    assert transition.active_transaction is None
    assert transition.previous_transaction is previous
    assert transition.requested_show_set_ids == (displayed_id,)
    assert transition.display_set_ids == ()
    assert transition.unresolved_intent_set_ids == (displayed_id,)
    assert transition.missing_intent_set_ids == (displayed_id,)
    assert main_window.results_controller.active_display_transaction() is None
    assert _active_display_set_ids(main_window) == ()


def test_failed_cached_additive_metadata_attempt_deauthorizes_possibly_mutated_display(
    main_window,
    qtbot,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    cached_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    published = main_window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id=displayed_id,
                label=displayed_name,
                entry=_completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ).to_display_payload(),
            ),
        ),
        prefer_set=displayed_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    previous = main_window.results_controller.active_display_transaction()
    assert previous is not None
    t = np.asarray([0.0, 1.0], dtype=float)
    series = {"A": np.asarray([2.0, 1.0], dtype=float)}
    cache_key = "failed-additive-cached-sibling"
    main_window.simulation_controller.batch_cache.put_completion_entry(
        cache_key=cache_key,
        set_id=cached_id,
        is_preview=False,
        t=t,
        series=series,
        mechanism_text="A -> B ; k=1.0",
        completion_provenance=completion_provenance_payload(
            t=t,
            series=series,
            mechanism_text="A -> B ; k=1.0",
        ),
        owned_species=("A",),
    )
    monkeypatch.setattr(
        main_window.results_controller,
        "_apply_cached_batch_plot_metadata",
        lambda **kwargs: "metadata failed",
    )

    outcome = main_window.results_controller.publish_cached_batch_display_scope(
        cache_key=cache_key,
        requested_show_set_ids=(displayed_id, cached_id),
        prefer_set=cached_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.FAILED
    assert transition.active_transaction is None
    assert transition.previous_transaction is previous
    assert transition.display_set_ids == ()
    assert transition.attempted_display_set_ids == (cached_id, displayed_id)
    assert transition.unresolved_intent_set_ids == (cached_id, displayed_id)
    assert transition.missing_intent_set_ids == ()
    assert main_window.results_controller.active_display_transaction() is None
    assert _active_display_set_ids(main_window) == ()
    assert main_window._status_text_value() == "Display failed; set2 and set1 are unavailable."


def test_cached_additive_missing_sibling_preserves_existing_display(main_window, qtbot):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    missing_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    published = main_window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id=displayed_id,
                label=displayed_name,
                entry=_completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ).to_display_payload(),
            ),
        ),
        prefer_set=displayed_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    previous = main_window.results_controller.active_display_transaction()
    assert previous is not None

    outcome = main_window.results_controller.publish_cached_batch_display_scope(
        cache_key="missing-additive-cached-sibling",
        requested_show_set_ids=(displayed_id, missing_id),
        prefer_set=missing_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.DENIED
    assert transition.active_transaction is previous
    assert transition.requested_show_set_ids == (displayed_id, missing_id)
    assert transition.display_set_ids == ()
    assert transition.unresolved_intent_set_ids == (missing_id,)
    assert transition.missing_intent_set_ids == (missing_id,)
    assert main_window.results_controller.active_display_transaction() is previous
    assert _active_display_set_ids(main_window) == (displayed_id,)
    assert main_window._status_text_value() == (
        "Result not cached; set2 needs a run."
    )


def test_partial_workspace_preview_publication_records_full_requested_scope(main_window, qtbot, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    missing_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    main_window._batch_model.set_row_requested_show(0, True)
    main_window._batch_model.set_row_requested_show(1, True)
    entry = _completion_display_entry(
        set_id=displayed_id,
        label=displayed_name,
        values=[1.0, 0.4],
        mechanism_text="A -> B ; k=1.0",
    )

    monkeypatch.setattr(
        main_window.results_controller,
        "_ui",
        replace(
            main_window.results_controller._ui,
            workspace_display_request_resolution=lambda requested: BatchDisplayRequestResolution(
                resolved_entries=(
                    ResolvedBatchDisplayRequestEntry(
                        set_id=displayed_id,
                        label=displayed_name,
                        entry=entry.to_display_payload(),
                        workspace_preview_provenance={"preview": "dirty-workspace"},
                    ),
                ),
                unavailable_cause=DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                coverage=BatchDisplayRequestCoverage.INCOMPLETE,
                has_workspace_display_request=True,
                has_resolved_workspace_preview=True,
                focused_uses_workspace_controls=True,
                focused_has_resolved_entry=True,
            ),
        ),
    )

    published = main_window.results_controller.refresh_display_from_request_scope(
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    transition = published.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert transition.missing_intent_set_ids == (missing_id,)
    assert transition.unresolved_intent_set_ids == (missing_id,)
    active = main_window.results_controller.active_display_transaction()
    assert active is not None
    assert active.kind is ActiveDisplayKind.WORKSPACE_PREVIEW
    assert active.display_set_ids == (displayed_id,)
    provenance = main_window._simulation_provenance_owner.last_simulation_provenance
    assert provenance["display_transaction"]["display_set_ids"] == [displayed_id]
    assert "requested_show_set_ids" not in provenance["display_transaction"]
    assert "missing_intent_set_ids" not in provenance["display_transaction"]
    assert main_window._status_text_value() == "Displayed workspace preview; set2 needs a run."


def test_failed_completed_run_display_attempt_preserves_request_outcome(main_window, qtbot, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    missing_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(displayed_id, missing_id),
        labels_by_set_id={
            displayed_id: displayed_name,
            missing_id: str(main_window._batch_store.set_name_for_row(1)),
        },
        primary_set_id=displayed_id,
        cache_key="failed-display-attempt-cache",
        run_id=151,
        request_id=152,
        owned_species_by_set_id={displayed_id: ("A",), missing_id: ("A",)},
        run_target_set_ids=(displayed_id,),
    )
    monkeypatch.setattr(
        main_window.results_controller,
        "_apply_completed_run_plot_metadata",
        lambda **kwargs: "metadata failed",
    )

    outcome = main_window.results_controller.publish_completed_run_display_transaction(
        CompletedRunDisplayTransaction(
            intent=intent,
            completion_entries=(
                _completion_display_entry(
                    set_id=displayed_id,
                    label=displayed_name,
                    values=[1.0, 0.4],
                    mechanism_text="A -> B ; k=1.0",
                ),
            ),
            display_set_ids=(displayed_id,),
            display_primary_set_id=displayed_id,
            failed_set_ids=(),
            missing_intent_set_ids=(missing_id,),
            unresolved_intent_set_ids=(missing_id,),
        )
    )

    transition = outcome.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.FAILED
    assert transition.requested_show_set_ids == (displayed_id, missing_id)
    assert transition.display_set_ids == ()
    assert transition.attempted_display_set_ids == (displayed_id,)
    assert transition.affected_set_ids == (displayed_id,)
    assert transition.unresolved_intent_set_ids == (missing_id,)
    assert transition.missing_intent_set_ids == (missing_id,)
    assert main_window.results_controller.active_display_transaction() is None


def test_failed_partial_workspace_preview_attempt_keeps_display_failure_outcome(main_window, qtbot, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    missing_id = str(main_window._batch_store.set_id_for_row(1))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    main_window._batch_model.set_row_requested_show(0, True)
    main_window._batch_model.set_row_requested_show(1, True)
    entry = _completion_display_entry(
        set_id=displayed_id,
        label=displayed_name,
        values=[1.0, 0.4],
        mechanism_text="A -> B ; k=1.0",
    )
    monkeypatch.setattr(
        main_window.results_controller,
        "_ui",
        replace(
            main_window.results_controller._ui,
            workspace_display_request_resolution=lambda requested: BatchDisplayRequestResolution(
                resolved_entries=(
                    ResolvedBatchDisplayRequestEntry(
                        set_id=displayed_id,
                        label=displayed_name,
                        entry=entry.to_display_payload(),
                        workspace_preview_provenance={"preview": "dirty-workspace"},
                    ),
                ),
                unavailable_cause=DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                coverage=BatchDisplayRequestCoverage.INCOMPLETE,
                has_workspace_display_request=True,
                has_resolved_workspace_preview=True,
                focused_uses_workspace_controls=True,
                focused_has_resolved_entry=True,
            ),
        ),
    )
    monkeypatch.setattr(
        main_window.results_controller,
        "_apply_resolved_batch_plot_metadata",
        lambda **kwargs: "metadata failed",
    )

    failed = main_window.results_controller.refresh_display_from_request_scope(
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )

    transition = failed.transition_outcome
    assert transition is not None
    assert transition.kind is DisplayTransitionOutcomeKind.FAILED
    assert transition.cause is DisplayTransitionCause.DISPLAY_MUTATION_FAILED
    assert transition.requested_show_set_ids == (displayed_id, missing_id)
    assert transition.display_set_ids == ()
    assert transition.attempted_display_set_ids == (displayed_id,)
    assert transition.unresolved_intent_set_ids == (missing_id,)
    assert transition.missing_intent_set_ids == (missing_id,)
    assert main_window._status_text_value() == (
        "Display failed; set2 needs a run."
    )


def test_terminal_scoped_finalization_forwards_missing_request_members(main_window, qtbot, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)
    first_id = str(main_window._batch_store.set_id_for_row(0))
    second_id = str(main_window._batch_store.set_id_for_row(1))
    intent = CompletedRunDisplayIntent(
        requested_show_set_ids=(first_id, second_id),
        labels_by_set_id={
            first_id: str(main_window._batch_store.set_name_for_row(0)),
            second_id: str(main_window._batch_store.set_name_for_row(1)),
        },
        primary_set_id=first_id,
        cache_key="terminal-scoped-missing",
        run_id=171,
        request_id=172,
        owned_species_by_set_id={first_id: ("A",), second_id: ("A",)},
        run_target_set_ids=(first_id, second_id),
    )
    owner = main_window.simulation_controller._batch_context_owner
    owner.load_context(
        BatchContextSeed(
            active=True,
            run_id=171,
            request_id=172,
            cache_key="terminal-scoped-missing",
            queue_ids=(first_id, second_id),
            queue_names=("Set 1", "Set 2"),
            completed_run_display_intent=intent,
        )
    )

    class _UnavailableCapture:
        def __init__(self) -> None:
            self.kwargs = None

        def publish_completed_run_display_transaction(self, transaction):
            raise AssertionError(f"unexpected transaction: {transaction!r}")

        def publish_completed_run_display_unavailable(self, **kwargs):
            self.kwargs = dict(kwargs)
            return SimulationCompletionDisplayOutcome(
                transition_outcome=DisplayTransitionOutcome(
                    kind=DisplayTransitionOutcomeKind.FAILED,
                    active_transaction=None,
                    previous_transaction=None,
                    display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
                    affected_set_ids=tuple(kwargs.get("affected_set_ids") or ()),
                    unresolved_intent_set_ids=tuple(kwargs.get("unresolved_intent_set_ids") or ()),
                    missing_intent_set_ids=tuple(kwargs.get("missing_intent_set_ids") or ()),
                    event_kind=DisplayEventKind.COMPLETED_RUN_COVERAGE_UNAVAILABLE,
                    cause=kwargs.get("cause"),
                )
            )

    capture = _UnavailableCapture()
    monkeypatch.setattr(main_window.simulation_controller, "_completion_publication_owner", capture)

    transition = main_window.simulation_controller._finalize_scoped_batch_success_subset(
        owner._current_context()
    )

    assert transition is not None
    assert capture.kwargs is not None
    assert capture.kwargs["unresolved_intent_set_ids"] == (first_id, second_id)
    assert capture.kwargs["missing_intent_set_ids"] == (first_id, second_id)
    assert transition.missing_intent_set_ids == (first_id, second_id)


def test_workspace_preview_clear_uses_adt_display_scope_after_identity_changes(
    main_window,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    entry = _completion_display_entry(
        set_id=displayed_id,
        label=displayed_name,
        values=[1.0, 0.4],
        mechanism_text="A -> B ; k=1.0",
    )
    published = main_window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id=displayed_id,
                label=displayed_name,
                entry=entry.to_display_payload(),
                workspace_preview_provenance={"preview": "old-workspace"},
            ),
        ),
        prefer_set=displayed_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    active = main_window.results_controller.active_display_transaction()
    assert active is not None
    assert active.kind is ActiveDisplayKind.WORKSPACE_PREVIEW
    assert active.display_set_ids == (displayed_id,)
    monkeypatch.setattr(
        main_window.results_controller,
        "current_workspace_preview_identity_payload",
        lambda *, set_id: {"preview": f"changed-{set_id}"},
    )

    cleared = main_window.results_controller.clear_display_if_workspace_previews_were_displayed(
        (displayed_id,)
    )

    assert cleared is True
    assert main_window.results_controller.active_display_transaction() is None
    assert _active_display_set_ids(main_window) == ()


def test_deleting_displayed_resolved_set_deauthorizes_active_transaction(
    main_window,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    displayed_id = str(main_window._batch_store.set_id_for_row(0))
    displayed_name = str(main_window._batch_store.set_name_for_row(0))
    entry = _completion_display_entry(
        set_id=displayed_id,
        label=displayed_name,
        values=[1.0, 0.4],
        mechanism_text="A -> B ; k=1.0",
    )
    published = main_window.results_controller.publish_resolved_batch_display_request(
        resolved_entries=(
            ResolvedBatchDisplayRequestEntry(
                set_id=displayed_id,
                label=displayed_name,
                entry=entry.to_display_payload(),
            ),
        ),
        prefer_set=displayed_id,
        display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
    )
    assert published.transition_outcome is not None
    assert published.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    assert _active_display_set_ids(main_window) == (displayed_id,)
    set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    delete_btn = main_window.findChild(QtWidgets.QPushButton, "deleteBatchSetButton")
    assert delete_btn is not None
    delete_btn.click()

    assert main_window.results_controller.active_display_transaction() is None
    assert _active_display_set_ids(main_window) == ()


def test_species_mode_slider_overlay_commit_and_reset_follow_transaction_boundaries(
    main_window,
    qtbot,
    monkeypatch,
):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")
    assert model.setData(model.index(1, 1), "0.25")
    assert model.setData(model.index(1, 2), "0.75")

    set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])
    main_window.set_slider_edit_target_set_ids(
        [
            str(main_window.batch_set_id_for_row(0) or ""),
            str(main_window.batch_set_id_for_row(1) or ""),
        ]
    )

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    qtbot.addWidget(main_window)
    main_window.show()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    commit_btn = main_window.findChild(QtWidgets.QPushButton, "commitSliderOverridesButton")
    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert slider_a is not None
    assert commit_btn is not None
    assert reset_btn is not None

    press_pos = slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    staged_row_0 = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    staged_row_1 = main_window._preview_session.preview_initials_for_row(1, main_window.batch_initials_for_row(1))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(1.0, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(0.25, rel=1e-6, abs=1e-9)
    assert float(staged_row_0["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(staged_row_1["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)

    qtbot.mouseClick(commit_btn, QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()

    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert commit_btn.isEnabled() is False
    assert reset_btn.isEnabled() is False

    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(0)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    dirty_row_0 = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    dirty_row_1 = main_window._preview_session.preview_initials_for_row(1, main_window.batch_initials_for_row(1))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(dirty_row_0["A"]) != pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(dirty_row_1["A"]) != pytest.approx(2.5, rel=1e-6, abs=1e-9)

    qtbot.mouseClick(reset_btn, QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()

    reset_row_0 = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    reset_row_1 = main_window._preview_session.preview_initials_for_row(1, main_window.batch_initials_for_row(1))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(reset_row_0["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(reset_row_1["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
