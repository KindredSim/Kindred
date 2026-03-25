import numpy as np
import pytest
from PySide6 import QtWidgets

from kindred.gui.main_window import MainWindow

pytestmark = [pytest.mark.gui, pytest.mark.slow]


def _run_fake_simulation(window: MainWindow, monkeypatch):
    """Feed deterministic data through the simulation completion path."""
    t = np.linspace(0.0, 5.0, 12)
    species_a = np.linspace(1.0, 0.2, t.size)
    species_b = np.linspace(0.0, 0.6, t.size)
    payload = {
        "t": t,
        "Y": np.vstack([species_a, species_b]),
        "species_names": ["A", "B"],
        "mechanism_text": window._get_mechanism_text(),
        "solver_config": {"solver": "LSODA", "rtol": 1e-6, "atol": 1e-12},
        "mechanism": None,
    }

    monkeypatch.setattr(
        window._sim_ui_port.provenance,
        "integrate_ctc",
        lambda *args, **kwargs: (0.5, "mock", True, 1e-6, "tail"),
    )
    window.set_data(
        t,
        {"A": species_a, "B": species_b},
        label="Results",
        overlays=[],
    )
    window.set_last_simulation_ctc({"A": 0.5, "B": 0.5})
    return payload


def test_main_window_title_and_menu_entries(main_window):
    """Smoke-test that the window title and primary menus exist."""
    assert main_window.windowTitle() == "Kindred"
    menu_titles = [action.text() for action in main_window.menuBar().actions()]
    expected = {
        "&File",
        "&Edit",
        "&View",
        "&Profiles",
        "E&xamples",
        "&Simulation",
        "&Fitting",
        "&Tools",
        "&Help",
    }
    assert expected.issubset(set(menu_titles))


def test_fitting_menu_does_not_expose_stale_external_diagnostics_action(main_window):
    from PySide6 import QtGui

    action = main_window.findChild(QtGui.QAction, "fittingDiagnosticsAction")
    assert action is None


def test_advanced_dsl_toggle_removed(main_window):
    """Advanced DSL is always on and should not appear as a toggle in the UI."""
    checkboxes = main_window.findChildren(QtWidgets.QCheckBox)
    assert all("Advanced DSL" not in cb.text() for cb in checkboxes)


def test_loading_preset_populates_editor(main_window):
    """Ensure preset DSL text loads without prompting."""
    main_window._load_preset_mechanism("M1")
    text = main_window._mechanism_editor._reactions_text.toPlainText()
    assert text.strip(), "Preset M1 should populate the mechanism editor"
    assert ("reaction:" in text.lower()) or ("->" in text), "Preset should include reaction text"


def test_arrhenius_mechanism_runs_by_default(main_window):
    """Arrhenius-style DSL parses and simulates without any toggle."""
    dsl_text = "\n".join([
        "reaction: A -> B; Ea=55, A=1e12, energy=kJ/mol",
        "initial: A=1.0",
        "initial: B=0.0",
        "T=320",
    ])
    result = main_window._run_dataset_simulation(dsl_text)
    assert result["t"].size > 0
    species = result["species"]
    assert "A" in species and "B" in species
    assert species["B"][-1] > species["B"][0]


def test_simulation_completion_updates_state(main_window, monkeypatch):
    """Mock a simulation completion and ensure data + CTC caches update."""
    main_window._load_preset_mechanism("M1")
    _run_fake_simulation(main_window, monkeypatch)
    visible = set(main_window._plot_tabs._main_plot.visible_series())
    assert {"A", "B"}.issubset(visible)
    assert set(main_window._last_simulation_ctc) == {"A", "B"}
    assert main_window._last_simulation_ctc["A"] == pytest.approx(0.5)
    assert main_window._last_simulation_ctc["B"] == pytest.approx(0.5)


def test_species_registry_detection(main_window):
    """Detect species and initial concentrations for a simple DSL snippet."""
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A + B -> C; k=0.5\n"
        "initial: A=1.0\n"
        "initial: B=0.25\n"
    )
    entries, error_message = main_window._gather_species_registry_entries()
    assert error_message is None
    names = {name for name, _ in entries}
    assert names == {"A", "B", "C"}
    initials = {name: value for name, value in entries}
    assert initials["A"] == pytest.approx(1.0)
    assert initials["B"] == pytest.approx(0.25)
    assert initials["C"] == pytest.approx(0.0)


def test_species_registry_accepts_named_inline_initial_set_blocks(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A + B -> C; k=0.5\n"
        "\n"
        "Set B = {\n"
        "[A] = 1.0\n"
        "[B] = 0.25\n"
        "}\n"
    )

    entries, error_message = main_window._gather_species_registry_entries()

    assert error_message is None
    names = {name for name, _ in entries}
    assert names == {"A", "B", "C"}


def test_color_roster_refresh_handles_malformed_named_set_block(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\n"
        "\n"
        "Set B = {\n"
        "[A] = 1.0\n"
    )

    roster = main_window._current_mechanism_species_roster_for_colors()

    assert roster is None


def test_species_registry_reports_parse_failure(main_window):
    """Invalid DSL surfaces a helpful error instead of crashing."""
    main_window._mechanism_editor._reactions_text.setPlainText("this line does not parse")
    entries, error_message = main_window._gather_species_registry_entries()
    assert entries == []
    assert error_message is not None
    assert "DSL parse error" in error_message


def test_species_registry_malformed_named_set_returns_parse_error(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\n"
        "\n"
        "Set B = {\n"
        "[A] = 1.0\n"
    )

    entries, error_message = main_window._gather_species_registry_entries()

    assert entries == []
    assert error_message is not None
    assert "DSL parse error" in error_message


def test_mechanism_editor_validation_accepts_named_inline_initial_set_blocks(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\n"
        "\n"
        "Set B = {\n"
        "[A] = 1.0\n"
        "}\n"
    )

    main_window._mechanism_editor._validate_dsl()

    label = main_window._mechanism_editor._validation_label.text()
    assert label.startswith("✓ Valid:")
    assert "2 species" in label
    assert "1 reactions" in label


def test_mechanism_editor_validation_rejects_unsupported_let_named_set_blocks(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\n"
        "\n"
        "let baseline = {\n"
        "}\n"
    )

    main_window._mechanism_editor._validate_dsl()

    label = main_window._mechanism_editor._validation_label.text()
    assert label.startswith("✗ Error:")
    assert "unrecognized line" in label
    assert "let baseline = {" in label
