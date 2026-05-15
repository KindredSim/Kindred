import numpy as np
import pytest
from PySide6 import QtWidgets

from kindred.core.simulator.dsl import parse_dsl_to_mechanism

pytestmark = pytest.mark.gui


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
    main_window.results_controller.set_data(t, series_map, label="Results", overlays=[])
    dataset_panel = main_window._plot_tabs.add_dataset_tab("Dataset 1")
    first_species = species_names[0]
    dataset_panel.set_data(
        data_x=t,
        data_y=series_map[first_species],
        xlabel="Time",
        ylabel=first_species,
        all_species=series_map,
    )
    return main_window, dataset_panel


def test_csv_export_from_simulation_tab(tmp_path, prepared_window):
    window, _dataset_panel = prepared_window
    csv_path = tmp_path / "simulation.csv"
    window._plot_tabs._tabs.setCurrentIndex(0)
    window.project_controller.handle_export_config(
        {"path": str(csv_path), "mode": "default", "scope": "all"}
    )
    assert csv_path.exists()


def test_dataset_tab_csv_export(tmp_path, prepared_window):
    window, dataset_panel = prepared_window
    csv_path = tmp_path / "dataset.csv"
    window._plot_tabs._tabs.setCurrentWidget(dataset_panel)
    window.project_controller.handle_export_config({"path": str(csv_path), "mode": "default", "scope": "all"})
    assert csv_path.exists()
