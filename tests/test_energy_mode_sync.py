import math
import re

import numpy as np
import pytest
from PySide6 import QtWidgets

from kindred.core.constants import R
from kindred.core.simulator.dsl import parse_dsl_to_mechanism


def _state_network_dsl(*, energy_directive: str | None, T: float, energies: dict[str, float]) -> str:
    header = []
    if energy_directive:
        header.append(f"energy={energy_directive}")
    header.append(f"T={T}")
    states = [
        f"state: A, kind=GS, energy={energies['A']}, degeneracy=1",
        f"state: TS1, kind=TS, energy={energies['TS1']}, degeneracy=1",
        f"state: B, kind=GS, energy={energies['B']}, degeneracy=1",
        "edge: A,TS1",
        "edge: TS1,B",
    ]
    return "\n".join(header + states)


@pytest.mark.unit
def test_energy_mode_parsing_exposes_temperature_and_default_energy_unit_and_K_relation():
    dsl = _state_network_dsl(
        energy_directive=None,
        T=200.0,
        energies={"A": 0.0, "TS1": 50.0, "B": -10.0},  # default kJ/mol
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})

    # In energy mode, temperature comes from DSL T=...
    assert mech.metadata.get("temperature_K") == pytest.approx(200.0)

    # Global energy unit defaults to kJ/mol unless overridden by DSL header.
    assert mech.metadata.get("energy_unit") == "kJ/mol"

    # State-network structure should be retained for downstream UI sync/provenance.
    assert "state_network" in (mech.metadata or {})

    assert len(mech.equilibria) == 1
    eq = mech.equilibria[0]

    dG_eq_J_per_mol = -10.0 * 1000.0  # -10 kJ/mol
    expected_K = math.exp(-dG_eq_J_per_mol / (R * 200.0))
    assert float(eq.Keq) == pytest.approx(expected_K, rel=1e-12)


@pytest.mark.unit
def test_energy_mode_kcal_matches_kj_after_conversion():
    T = 298.15
    dsl_kj = _state_network_dsl(
        energy_directive="kJ/mol",
        T=T,
        energies={"A": 0.0, "TS1": 50.0, "B": -10.0},
    )

    kj_to_kcal = 1.0 / 4.184
    dsl_kcal = _state_network_dsl(
        energy_directive="kcal/mol",
        T=T,
        energies={
            "A": 0.0,
            "TS1": 50.0 * kj_to_kcal,
            "B": -10.0 * kj_to_kcal,
        },
    )

    mech_kj = parse_dsl_to_mechanism(dsl_kj, initials={})
    mech_kcal = parse_dsl_to_mechanism(dsl_kcal, initials={})

    assert mech_kcal.metadata.get("energy_unit") == "kcal/mol"

    eq_kj = mech_kj.equilibria[0]
    eq_kcal = mech_kcal.equilibria[0]

    assert float(eq_kcal.Keq) == pytest.approx(float(eq_kj.Keq), rel=1e-9)
    assert float(eq_kcal.kf) == pytest.approx(float(eq_kj.kf), rel=1e-9)
    assert float(eq_kcal.kr) == pytest.approx(float(eq_kj.kr), rel=1e-9)


@pytest.mark.gui
def test_gui_energy_mode_temperature_comes_from_dsl(main_window, monkeypatch):
    main_window._temperature_spinbox.setValue(298.15)
    main_window._mechanism_editor._reactions_text.setPlainText("T=200.0")
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(
        "\n".join(
            [
                "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
                "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
                "state: B, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
                "edge: A,TS1",
                "edge: TS1,B",
            ]
        )
    )

    dsl = main_window._get_mechanism_text()
    mech = parse_dsl_to_mechanism(dsl, initials={})
    species_names = mech.species_names()
    assert species_names == ["A", "B"]

    t = np.linspace(0.0, 1.0, 5)
    Y = np.vstack([np.linspace(1.0, 0.5, t.size), np.linspace(0.0, 0.5, t.size)])

    monkeypatch.setattr(
        main_window._sim_ui_port.provenance,
        "integrate_ctc",
        lambda *args, **kwargs: (0.0, "mock", True, 1e-6, "tail"),
    )

    payload = {
        "t": t,
        "Y": Y,
        "species_names": species_names,
        "mechanism": mech,
        "mechanism_text": dsl,
        "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
    }
    main_window.simulation_controller.on_simulation_complete(payload)

    indicator = main_window._temperature_mode_indicator.text()
    assert "200.00" in indicator
    assert not main_window._temperature_spinbox.isEnabled()


def _prime_energy_mode_sliders(main_window, monkeypatch, *, T: float = 200.0) -> str:
    main_window._temperature_spinbox.setValue(298.15)
    main_window._mechanism_editor._reactions_text.setPlainText(f"T={T}")
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(
        "\n".join(
            [
                "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
                "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
                "state: B, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
                "edge: A,TS1",
                "edge: TS1,B",
            ]
        )
    )

    dsl = main_window._get_mechanism_text()
    mech = parse_dsl_to_mechanism(dsl, initials={})
    species_names = mech.species_names()
    assert species_names == ["A", "B"]

    t = np.linspace(0.0, 1.0, 5)
    Y = np.vstack([np.linspace(1.0, 0.5, t.size), np.linspace(0.0, 0.5, t.size)])

    monkeypatch.setattr(
        main_window._sim_ui_port.provenance,
        "integrate_ctc",
        lambda *args, **kwargs: (0.0, "mock", True, 1e-6, "tail"),
    )

    payload = {
        "t": t,
        "Y": Y,
        "species_names": species_names,
        "mechanism": mech,
        "mechanism_text": dsl,
        "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
    }
    main_window.simulation_controller.on_simulation_complete(payload)
    return dsl


def _prime_energy_mode_sliders_in_reaction_editor(main_window, monkeypatch, *, T: float = 200.0) -> str:
    """
    Prime energy-mode sliders using a semicolon-style state-network DSL embedded directly in the
    visible Reaction DSL editor.
    """
    main_window._temperature_spinbox.setValue(298.15)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl("")
    dsl = "\n".join(
        [
            f"T={T}",
            "state: name=A; kind=GS; energy=0.0; degeneracy=1",
            "state: name=B; kind=GS; energy=5.0; degeneracy=1",
            "state: name=TS1; kind=TS; energy=20.0; degeneracy=1",
            "edge: A,TS1",
            "edge: B,TS1",
        ]
    )
    main_window._mechanism_editor._reactions_text.setPlainText(dsl)

    mech = parse_dsl_to_mechanism(dsl, initials={})
    species_names = mech.species_names()
    assert set(species_names) == {"A", "B"}

    t = np.linspace(0.0, 1.0, 5)
    Y = np.vstack([np.linspace(1.0, 0.5, t.size), np.linspace(0.0, 0.5, t.size)])

    monkeypatch.setattr(
        main_window._sim_ui_port.provenance,
        "integrate_ctc",
        lambda *args, **kwargs: (0.0, "mock", True, 1e-6, "tail"),
    )

    payload = {
        "t": t,
        "Y": Y,
        "species_names": list(species_names),
        "mechanism": mech,
        "mechanism_text": dsl,
        "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
    }
    main_window.simulation_controller.on_simulation_complete(payload)
    return dsl


@pytest.mark.gui
def test_gui_energy_slider_updates_worker_mechanism_text_and_commits_state_network_dsl(main_window, monkeypatch):
    _prime_energy_mode_sliders(main_window, monkeypatch, T=200.0)

    from kindred.core.simulation_plan import SimulationPlan
    from kindred.gui.simulation_worker import ContainedSimulationWorker

    monkeypatch.setattr(ContainedSimulationWorker, "start", lambda self: None)

    var_eq = "dG_eq__TS1__A__B"
    sliders = main_window._mechanism_editor._variable_sliders
    preview = main_window._preview_session
    assert sliders.has_variable(var_eq)

    main_window._on_slider_drag_started(var_eq)
    eq_slider = sliders._sliders[var_eq]
    eq_slider.setValue(sliders._value_to_slider_pos(var_eq, -5.0))
    timer = getattr(preview, "_variable_update_timer", None)
    if timer is not None:
        timer.stop()

    main_window.simulation_controller.launch_pending_slider_preview_replay()
    worker = main_window.simulation_controller.run_state.simulation_worker
    worker_plan = getattr(worker, "_simulation_plan", getattr(worker, "_simulation_plan_payload", {}))
    worker_text = SimulationPlan.from_payload(worker_plan).to_execution_request().mechanism_text
    assert re.search(r"^state:\s*B,.*\benergy=-5\b", worker_text, flags=re.MULTILINE)

    main_window._on_slider_drag_finished(var_eq)
    release_timer = getattr(preview, "_slider_release_commit_timer", None)
    if release_timer is not None:
        release_timer.stop()
    main_window.simulation_controller._pending_slider_simulation = False
    preview._pending_slider_values.clear()
    preview._slider_release_in_progress = False
    preview._slider_release_primary_name = ""
    preview._suppress_slider_refresh = False

    # Override mode: persistence into editor DSL is explicit via the Commit button (no auto-commit on release).
    main_window._mechanism_editor._commit_slider_overrides_btn.click()
    QtWidgets.QApplication.processEvents()
    timer = getattr(preview, "_variable_update_timer", None)
    if timer is not None:
        timer.stop()

    state_dsl = main_window._mechanism_editor._state_network_editor.get_state_network_dsl()
    assert re.search(r"^state:\s*B,.*\benergy=-5\b", state_dsl, flags=re.MULTILINE)

    main_window.simulation_controller.cleanup_worker_safely(
        main_window.simulation_controller.run_state.simulation_worker,
        "simulation worker (test)",
    )
    main_window.simulation_controller.run_state.simulation_worker = None


@pytest.mark.gui
def test_gui_energy_slider_refreshes_derived_K_immediately(main_window, monkeypatch):
    _prime_energy_mode_sliders(main_window, monkeypatch, T=200.0)

    captured = {}

    def _capture(params):
        captured["params"] = dict(params)

    monkeypatch.setattr(main_window._plot_tabs._main_plot, "update_parameters", _capture)

    var_eq = "dG_eq__TS1__A__B"
    preview = main_window._preview_session
    main_window._on_slider_drag_started(var_eq)
    main_window._on_variable_changed(var_eq, -5.0)
    timer = getattr(preview, "_variable_update_timer", None)
    if timer is not None:
        timer.stop()

    assert "params" in captured
    params = captured["params"]
    K_key = "Keq (A→B via TS1)"
    assert K_key in params
    dG_eq_J_per_mol = -5.0 * 1000.0
    expected_K = math.exp(-dG_eq_J_per_mol / (R * 200.0))
    assert float(params[K_key][0]) == pytest.approx(expected_K, rel=1e-12)


@pytest.mark.gui
def test_gui_energy_slider_updates_reaction_dsl_and_worker_and_persists_on_run(main_window, monkeypatch):
    _prime_energy_mode_sliders_in_reaction_editor(main_window, monkeypatch, T=200.0)

    from kindred.core.simulation_plan import SimulationPlan
    from kindred.gui.simulation_worker import ContainedSimulationWorker

    monkeypatch.setattr(ContainedSimulationWorker, "start", lambda self: None)

    meta_map = dict(main_window.variable_metadata() or {})
    var_eq = next(
        name
        for name, meta in meta_map.items()
        if isinstance(meta, dict) and meta.get("type") == "energy" and meta.get("role") == "dG_eq"
    )
    meta = meta_map[var_eq]
    reactant = str(meta.get("reactant"))
    product = str(meta.get("product"))
    assert {reactant, product} == {"A", "B"}

    base_energies = {"A": 0.0, "B": 5.0, "TS1": 20.0}  # kJ/mol
    initial = float((main_window.variable_slider_values() or {}).get(var_eq))
    new_val = initial - 1.0
    expected_product_energy = base_energies[reactant] + new_val

    sliders = main_window._mechanism_editor._variable_sliders
    preview = main_window._preview_session
    assert sliders.has_variable(var_eq)
    main_window._on_slider_drag_started(var_eq)
    eq_slider = sliders._sliders[var_eq]
    eq_slider.setValue(sliders._value_to_slider_pos(var_eq, float(new_val)))
    timer = getattr(preview, "_variable_update_timer", None)
    if timer is not None:
        timer.stop()

    main_window.simulation_controller.launch_pending_slider_preview_replay()
    worker = main_window.simulation_controller.run_state.simulation_worker
    worker_plan = getattr(worker, "_simulation_plan", getattr(worker, "_simulation_plan_payload", {}))
    worker_text = SimulationPlan.from_payload(worker_plan).to_execution_request().mechanism_text
    m = re.search(
        rf"^state:.*\bname\s*=\s*{re.escape(product)}\b.*\benergy\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        worker_text,
        flags=re.MULTILINE,
    )
    assert m, "Expected product state line in worker mechanism text"
    assert float(m.group(1)) == pytest.approx(expected_product_energy, rel=1e-12)

    main_window._on_slider_drag_finished(var_eq)
    release_timer = getattr(preview, "_slider_release_commit_timer", None)
    if release_timer is not None:
        release_timer.stop()
    main_window.simulation_controller._pending_slider_simulation = False
    preview._pending_slider_values.clear()
    preview._slider_release_in_progress = False
    preview._slider_release_primary_name = ""
    preview._suppress_slider_refresh = False

    # Override mode: persistence into editor DSL is explicit via the Commit button (no auto-commit on release).
    main_window._mechanism_editor._commit_slider_overrides_btn.click()
    QtWidgets.QApplication.processEvents()
    timer = getattr(preview, "_variable_update_timer", None)
    if timer is not None:
        timer.stop()

    editor_text = main_window._mechanism_editor._reactions_text.toPlainText()
    m2 = re.search(
        rf"^state:.*\bname\s*=\s*{re.escape(product)}\b.*\benergy\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        editor_text,
        flags=re.MULTILINE,
    )
    assert m2, "Expected product state line in Reaction DSL editor text"
    assert float(m2.group(1)) == pytest.approx(expected_product_energy, rel=1e-12)

    mech_after = parse_dsl_to_mechanism(editor_text, initials={})
    payload = {
        "t": np.linspace(0.0, 1.0, 5),
        "Y": np.vstack([np.linspace(1.0, 0.5, 5), np.linspace(0.0, 0.5, 5)]),
        "species_names": mech_after.species_names(),
        "mechanism": mech_after,
        "mechanism_text": editor_text,
        "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
    }
    # Manual "Run Simulation" paths clear the slider-triggered flag before completion refresh.
    preview._slider_triggered_simulation = False
    main_window.simulation_controller.on_simulation_complete(payload, fast_mode=False)
    current_vars = main_window._mechanism_editor._variable_sliders.get_variables()
    assert float(current_vars[var_eq]) == pytest.approx(new_val, rel=1e-12)

    main_window.simulation_controller.cleanup_worker_safely(
        main_window.simulation_controller.run_state.simulation_worker,
        "simulation worker (test)",
    )
    main_window.simulation_controller.run_state.simulation_worker = None
