from copy import deepcopy
import math
import os

import pytest
from PySide6 import QtCore, QtWidgets

from kindred.core.batch_initial_conditions import BatchInitialConditionsStore, BatchSet
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.gui.fitting.runtime_inputs import FittingRuntimeInputs
from kindred.gui.main_window import MainWindow
from kindred.gui.tutorial_manager import launch_tutorial
from kindred.gui.widgets.tutorial_overlay import TutorialOverlay

pytestmark = [pytest.mark.gui, pytest.mark.slow]


def _visible_project_state(window):
    return deepcopy(window.serialize_project_state())


def _serialized_batch_set(project_state, name: str) -> dict:
    for entry in project_state["batch_initial_conditions"]["sets"]:
        if entry["name"] == str(name):
            return dict(entry)
    raise AssertionError(f"Missing batch initial-condition set {name!r}")


def _assert_latest_runtime_inputs_match_window(window, notifications) -> None:
    assert notifications
    runtime_inputs = notifications[-1]
    assert isinstance(runtime_inputs, FittingRuntimeInputs)
    assert runtime_inputs.generation > 0
    project_state = window.serialize_project_state()
    assert runtime_inputs.to_stamp_payload() == {
        "evaluator": {
            "temperature_K": float(window.temperature_spinbox_value()),
            "use_sparse_jacobian": bool(project_state["use_sparse_jacobian"]),
            "wegscheider_cyclicity_enabled": bool(
                project_state["wegscheider_cyclicity_enabled"]
            ),
        },
        "batch_runtime_lane_budget": int(project_state["batch_runtime_lane_budget"]),
    }


@pytest.fixture(scope="session")
def qt_app():
    """Ensure a QApplication exists for GUI-driven tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtCore.QStandardPaths.setTestModeEnabled(True)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    QtCore.QCoreApplication.setApplicationName("KindredTests")
    QtCore.QCoreApplication.setOrganizationName("KindredTests")
    return app


def test_project_payload_without_state_network_clears_existing_state_network(main_window, qt_app):
    stale_state_network_text = "\n".join(
        [
            "state: S_old, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS_old, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "state: P_old, kind=GS, energy=-1, energy_unit=kJ/mol, degeneracy=1",
            "edge: S_old,TS_old",
            "edge: TS_old,P_old",
        ]
    )
    replacement_mechanism = "reaction: ProjectA -> ProjectB; k=0.4"

    seed_payload = dict(main_window.serialize_project_state())
    seed_payload["mechanism_source"] = MechanismAuthoringSource.from_parts(
        reactions_text="reaction: OldA -> OldB; k=1",
        state_network_dsl=stale_state_network_text,
    ).to_payload()
    main_window.apply_project_payload(seed_payload, record_undo=False)
    assert main_window.mechanism_state_network_dsl_raw() == stale_state_network_text

    payload = dict(main_window.serialize_project_state())
    payload["mechanism_source"] = {
        "reactions_text": replacement_mechanism,
        "state_network_dsl": "",
    }
    payload["notes"] = ""
    payload["batch_initial_conditions"] = {}

    main_window.apply_project_payload(payload, record_undo=False)
    qt_app.processEvents()

    assert main_window.mechanism_reactions_text_raw() == replacement_mechanism
    assert main_window.mechanism_state_network_dsl_raw() == ""
    assert "# State Network" not in main_window.get_mechanism_text()
    assert stale_state_network_text not in main_window.get_mechanism_text()


def test_project_round_trip_includes_all_dsl(tmp_path, monkeypatch, qt_app):
    """Saving then loading restores the full project source and migrates inline initials."""
    reactions_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "let rate_ratio = [B] / max([A], 1e-6)",
        ]
    )
    notes_text = "These are free-form notes (never parsed).\n"
    state_network_text = "\n".join(
        [
            "state: S1, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "edge: S1,TS1",
        ]
    )

    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)

    window = MainWindow()
    try:
        source = MechanismAuthoringSource.from_parts(
            reactions_text=reactions_text,
            state_network_dsl=state_network_text,
        )
        seed_payload = dict(window.serialize_project_state())
        seed_payload["notes"] = notes_text
        seed_payload["mechanism_source"] = source.to_payload()
        window.apply_project_payload(seed_payload, record_undo=False)

        project_path = tmp_path / "round_trip.kin"
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(project_path), "Kindred Project (*.kin)"),
        )
        window.project_controller.save_project()
    finally:
        window.close()

    loader = MainWindow()
    try:
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(project_path), "Kindred Project (*.kin)"),
        )
        loader.project_controller.load_project()

        expected_reactions_text = "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "# Initial Conditions moved to Batch Initial Conditions table (set1): A=1, B=0. Table values are authoritative.",
                "# Algebra",
                "let rate_ratio = [B] / max([A], 1e-6)",
            ]
        )
        loaded_state = loader.serialize_project_state()
        assert loaded_state["mechanism_source"] == {
            "reactions_text": expected_reactions_text,
            "state_network_dsl": state_network_text,
        }
        assert loaded_state["notes"] == notes_text
        assert _serialized_batch_set(loaded_state, "set1")["values"] == {"A": "1", "B": "0"}
    finally:
        loader.close()


def test_project_load_with_saved_batch_initials_strips_mechanism_text_initials(main_window, qt_app):
    saved_store = BatchInitialConditionsStore([BatchSet("set1", {"A": "5.0", "B": "7.0"})])
    saved_store.set_species(["A", "B"])
    payload = dict(main_window.serialize_project_state())
    payload["mechanism_source"] = MechanismAuthoringSource.from_parts(
        reactions_text="\n".join(
            [
                "reaction: A -> B; k=1",
                "set1 = {",
                "  [A] = 1.0",
                "  [B] = 0.0",
                "}",
            ]
        ),
        state_network_dsl="",
    ).to_payload()
    payload["batch_initial_conditions"] = saved_store.as_serializable()

    main_window.apply_project_payload(payload, record_undo=False)
    qt_app.processEvents()

    assert main_window.mechanism_reactions_text_raw() == "reaction: A -> B; k=1"
    assert _serialized_batch_set(main_window.serialize_project_state(), "set1")["values"] == {
        "A": "5.0",
        "B": "7.0",
    }


def test_project_apply_persists_fit_runtime_settings_after_project_settings(
    main_window,
    qt_app,
):
    notifications: list[dict[str, object]] = []

    class _FitWindow:
        def objectName(self):
            return "fit-window"

        def apply_runtime_inputs(self, *, runtime_inputs):
            evaluator = runtime_inputs.evaluator
            notifications.append(
                {
                    "temperature_K": float(evaluator.temperature_K),
                    "use_sparse_jacobian": bool(evaluator.use_sparse_jacobian),
                    "wegscheider_cyclicity_enabled": bool(
                        evaluator.wegscheider_cyclicity_enabled
                    ),
                    "batch_runtime_lane_budget": int(
                        runtime_inputs.batch_runtime_lane_budget
                    ),
                }
            )

    main_window.fitting_runtime_input_publisher.register_window(_FitWindow())

    original_state = main_window.serialize_project_state()
    original_temperature = float(original_state["temperature_K"])
    original_sparse = bool(original_state["use_sparse_jacobian"])
    original_wegscheider = bool(original_state["wegscheider_cyclicity_enabled"])
    original_lane_budget = int(original_state["batch_runtime_lane_budget"])

    payload = dict(main_window.serialize_project_state())
    payload["use_sparse_jacobian"] = not original_sparse
    payload["wegscheider_cyclicity_enabled"] = not original_wegscheider
    payload["batch_runtime_lane_budget"] = original_lane_budget + 1
    payload["temperature_K"] = original_temperature

    main_window.apply_project_payload(payload, record_undo=False)
    qt_app.processEvents()

    applied_state = main_window.serialize_project_state()
    assert float(applied_state["temperature_K"]) == original_temperature
    assert bool(applied_state["use_sparse_jacobian"]) is (not original_sparse)
    assert bool(applied_state["wegscheider_cyclicity_enabled"]) is (not original_wegscheider)
    assert int(applied_state["batch_runtime_lane_budget"]) == original_lane_budget + 1
    assert notifications == [
        {
            "temperature_K": original_temperature,
            "use_sparse_jacobian": not original_sparse,
            "wegscheider_cyclicity_enabled": not original_wegscheider,
            "batch_runtime_lane_budget": original_lane_budget + 1,
        }
    ]


def test_project_load_abort_after_dirty_reactions_apply_publishes_runtime_inputs(
    main_window,
    monkeypatch,
    qt_app,
):
    notifications = []

    class _FitWindow:
        def objectName(self):
            return "fit-window"

        def apply_runtime_inputs(self, *, runtime_inputs):
            notifications.append(runtime_inputs)

    main_window.fitting_runtime_input_publisher.register_window(_FitWindow())
    monkeypatch.setattr(
        main_window._reactions_draft_resolution_owner,
        "_prompt_draft_decision",
        lambda *args, **kwargs: "apply",
    )
    main_window._mechanism_editor.set_reactions_text("reaction: DraftA -> DraftB; k=2")
    main_window._refresh_reactions_draft_state()

    payload = dict(main_window.serialize_project_state())
    payload["mechanism_source"] = MechanismAuthoringSource.from_parts(
        reactions_text="\n".join(
            [
                "reaction: ReplacementA -> ReplacementB; k=1",
                "set1 = {",
                "  [ReplacementA] = 1.0",
                "  unsupported initial condition text",
                "}",
            ]
        ),
        state_network_dsl="",
    ).to_payload()
    payload["batch_initial_conditions"] = BatchInitialConditionsStore().as_serializable()

    with pytest.raises(ValueError, match="Initial Conditions could not be reconciled"):
        main_window.apply_project_payload(payload, record_undo=False)
    qt_app.processEvents()

    assert main_window.mechanism_reactions_text_raw() == "reaction: DraftA -> DraftB; k=2"
    _assert_latest_runtime_inputs_match_window(main_window, notifications)


def test_reactions_draft_materialization_failure_publishes_committed_source(
    main_window,
    monkeypatch,
    qt_app,
):
    notifications = []

    class _FitWindow:
        def objectName(self):
            return "fit-window"

        def apply_runtime_inputs(self, *, runtime_inputs):
            notifications.append(runtime_inputs)

    main_window.fitting_runtime_input_publisher.register_window(_FitWindow())
    main_window._mechanism_editor.set_reactions_text("reaction: DraftA -> DraftB; k=2")

    def _raise_after_source_commit(*args, **kwargs):
        raise RuntimeError("forced materialization failure")

    monkeypatch.setattr(
        main_window._initial_conditions_source_acceptance_owner,
        "apply_prepared_plan",
        _raise_after_source_commit,
    )

    with pytest.raises(RuntimeError, match="forced materialization failure"):
        main_window._apply_reactions_draft_with_transition()
    qt_app.processEvents()

    assert main_window.mechanism_reactions_text_raw() == "reaction: DraftA -> DraftB; k=2"
    _assert_latest_runtime_inputs_match_window(main_window, notifications)


def test_authoritative_rewrite_draft_materialization_failure_publishes_once(
    main_window,
    monkeypatch,
    qt_app,
):
    notifications = []

    class _FitWindow:
        def objectName(self):
            return "fit-window"

        def apply_runtime_inputs(self, *, runtime_inputs):
            notifications.append(runtime_inputs)

    main_window.fitting_runtime_input_publisher.register_window(_FitWindow())
    monkeypatch.setattr(
        main_window._reactions_draft_resolution_owner,
        "_prompt_draft_decision",
        lambda *args, **kwargs: "apply",
    )
    main_window._mechanism_editor.set_reactions_text("reaction: DraftA -> DraftB; k=2")
    main_window._refresh_reactions_draft_state()

    def _raise_after_source_commit(*args, **kwargs):
        raise RuntimeError("forced materialization failure")

    monkeypatch.setattr(
        main_window._initial_conditions_source_acceptance_owner,
        "apply_prepared_plan",
        _raise_after_source_commit,
    )

    with pytest.raises(RuntimeError, match="forced materialization failure"):
        main_window.rewrite_authoritative_mechanism_source(
            action_label="Rewrite after draft",
            description="Rewrite after draft",
            build_source=lambda source: source.with_reactions_text(
                "reaction: RewriteA -> RewriteB; k=3"
            ),
            record_undo=False,
            transition_source="test_rewrite_after_draft",
        )
    qt_app.processEvents()

    assert main_window.mechanism_reactions_text_raw() == "reaction: DraftA -> DraftB; k=2"
    assert len(notifications) == 1
    _assert_latest_runtime_inputs_match_window(main_window, notifications)


def test_invalid_project_mechanism_source_is_rejected_without_changing_project_state(
    main_window,
):
    original_state = _visible_project_state(main_window)
    payload = dict(main_window.serialize_project_state())
    payload["mechanism_source"] = {
        "reactions_text": "reaction: ReplacementA -> ReplacementB; k=1",
    }

    with pytest.raises(ValueError, match="state_network_dsl"):
        main_window.apply_project_payload(payload, record_undo=False)

    assert _visible_project_state(main_window) == original_state


def test_project_mechanism_source_rejects_unknown_source_fields_without_changing_project_state(
    main_window,
):
    original_state = _visible_project_state(main_window)
    payload = dict(main_window.serialize_project_state())
    payload["mechanism_source"] = {
        "reactions_text": "reaction: ReplacementA -> ReplacementB; k=1",
        "state_network_dsl": "",
        "unexpected_source_field": "rejected",
    }

    with pytest.raises(ValueError, match="unexpected_source_field"):
        main_window.apply_project_payload(payload, record_undo=False)

    assert _visible_project_state(main_window) == original_state


def test_project_source_initial_condition_reconciliation_failure_preserves_current_project(
    main_window,
):
    original_state = _visible_project_state(main_window)
    payload = dict(main_window.serialize_project_state())
    payload["mechanism_source"] = MechanismAuthoringSource.from_parts(
        reactions_text="\n".join(
            [
                "reaction: ReplacementA -> ReplacementB; k=1",
                "set1 = {",
                "  [ReplacementA] = 1.0",
                "  unsupported initial condition text",
                "}",
            ]
        ),
        state_network_dsl="",
    ).to_payload()
    payload["batch_initial_conditions"] = BatchInitialConditionsStore().as_serializable()
    payload["notes"] = "Replacement project notes"

    with pytest.raises(ValueError, match="Initial Conditions could not be reconciled"):
        main_window.apply_project_payload(payload, record_undo=False)

    assert _visible_project_state(main_window) == original_state


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("notes", {}, "notes"),
        ("batch_initial_conditions", [], "batch_initial_conditions"),
        ("num_points", "100", "num_points"),
        ("temperature_K", "298.15", "temperature_K"),
        ("rtol", math.nan, "rtol"),
    ],
)
def test_invalid_current_project_payload_fields_are_rejected_without_changing_project_state(
    main_window,
    field,
    bad_value,
    message,
):
    original_state = _visible_project_state(main_window)
    payload = dict(main_window.serialize_project_state())
    payload["mechanism_source"] = {
        "reactions_text": "reaction: ReplacementA -> ReplacementB; k=1",
        "state_network_dsl": "",
    }
    payload[field] = bad_value

    with pytest.raises((TypeError, ValueError), match=message):
        main_window.apply_project_payload(payload, record_undo=False)

    assert _visible_project_state(main_window) == original_state


def test_project_state_rejects_unknown_top_level_field_without_changing_project_state(
    main_window,
):
    original_state = _visible_project_state(main_window)
    payload = main_window.serialize_project_state()
    assert "unexpected_project_field" not in payload

    legacy_payload = dict(payload)
    legacy_payload["unexpected_project_field"] = "ignored-by-no-one"

    with pytest.raises(ValueError, match="unexpected_project_field"):
        main_window.apply_project_payload(legacy_payload, record_undo=False)

    assert _visible_project_state(main_window) == original_state


def test_public_project_apply_rejects_invalid_payload_without_changing_project_state(
    main_window,
):
    original_state = _visible_project_state(main_window)
    payload = main_window.serialize_project_state()
    payload["project_schema_version"] = 5

    with pytest.raises(ValueError, match="project_schema_version"):
        main_window.apply_project_payload(payload, record_undo=False)

    assert _visible_project_state(main_window) == original_state


def test_new_project_uses_complete_current_project_payload(main_window, monkeypatch, qt_app):
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Discard,
    )

    seed_payload = dict(main_window.serialize_project_state())
    seed_payload["mechanism_source"] = MechanismAuthoringSource.from_parts(
        reactions_text="reaction: ExistingA -> ExistingB; k=1",
        state_network_dsl="",
    ).to_payload()
    main_window.apply_project_payload(seed_payload, record_undo=False)
    main_window.apply_solver_runtime_state(solver="Radau", rtol=1e-6, atol=1e-12)
    main_window.config_controller.update_user_preference("solver", "BDF")

    main_window.project_controller.new_project()
    qt_app.processEvents()

    payload = main_window.serialize_project_state()
    assert set(payload) >= {"version", "solver_method", "solver_warning"}
    assert payload["solver"] == "BDF"
    assert payload["solver_method"] == "BDF"
    assert payload["solver_warning"] is None
    assert payload["mechanism_source"] == {"reactions_text": "", "state_network_dsl": ""}
    assert main_window.mechanism_reactions_text_raw() == ""
    assert main_window.mechanism_state_network_dsl_raw() == ""


def test_profile_activation_updates_widgets(tmp_path, monkeypatch, qt_app):
    """Activating a profile updates solver/grid widgets and persists selection."""
    templates_dir = tmp_path / "templates_profile"

    def _fake_templates_dir(_self):
        return templates_dir

    monkeypatch.setattr(
        "kindred.config.templates.TemplateManager._get_templates_directory",
        _fake_templates_dir,
    )
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)

    window = MainWindow()
    try:
        profile = window._profile_manager.get_profile("Fast")
        assert profile is not None, "Expected bundled 'Fast' profile"

        window._activate_profile("Fast")

        assert window._num_points_spinbox.value() == profile.grid_n
        assert window._initial_solver == profile.solver_method
        assert f"Solver: {profile.solver_method}" in window._solver_summary_label.text()

        stored = window._settings_owner.qsettings.value("profiles/active", "", type=str)
        assert stored == "Fast"
    finally:
        window._settings_owner.qsettings.clear()
        window.close()


def test_tutorial_overlay_highlights_widget(tmp_path, monkeypatch, qt_app):
    """Launching tutorials finds actual widgets described in the steps."""
    templates_dir = tmp_path / "templates_tutorial"

    def _fake_templates_dir(_self):
        return templates_dir

    monkeypatch.setattr(
        "kindred.config.templates.TemplateManager._get_templates_directory",
        _fake_templates_dir,
    )
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)

    window = MainWindow()
    overlay: TutorialOverlay | None = None
    try:
        overlay = launch_tutorial(window, "getting_started")
        assert isinstance(overlay, TutorialOverlay)

        QtWidgets.QApplication.processEvents()
        overlay._next_btn.click()  # Move to Mechanism Editor step
        QtWidgets.QApplication.processEvents()

        highlight_rect = overlay._get_highlight_rect()
        assert highlight_rect is not None and not highlight_rect.isNull()
    finally:
        if overlay is not None:
            overlay._on_skip()
        window.close()
