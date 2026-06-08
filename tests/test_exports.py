import csv

import numpy as np
import pytest
from PySide6 import QtWidgets

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.gui.ports import (
    CompletedRunDisplayIntent,
    CompletedRunDisplayTransaction,
    CompletionDisplayEntry,
    DisplayTransitionOutcomeKind,
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
        display_species=tuple(str(name) for name in species_names),
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

def test_dataset_tab_csv_export(tmp_path, prepared_window):
    window, dataset_panel, _species_names, _series_map, _t = prepared_window
    csv_path = tmp_path / "dataset.csv"
    window._plot_tabs._tabs.setCurrentWidget(dataset_panel)
    window.project_controller.handle_export_config({"path": str(csv_path), "mode": "default", "scope": "all"})
    assert csv_path.exists()
