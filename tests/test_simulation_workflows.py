from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.computational_mode import GENERATED_BLOCK_END, GENERATED_BLOCK_START
from kindred.gui.controllers.batch_run_context_owner import BatchContextSeed
from kindred.gui.ports import (
    CompletionDisplayEntry,
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
