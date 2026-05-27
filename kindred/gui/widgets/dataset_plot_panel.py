"""
Dataset Plot Panel

A simplified plot panel for dataset tabs that shows experimental data with
simulation overlay capability. Uses the main mechanism (no duplication).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Callable, List, Tuple, Iterator

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QCheckBox,
    QScrollArea
)
from PySide6.QtCore import Qt, Signal

from kindred.core.analysis.global_fit_projection import FitRenderDatasetProjection
from kindred.gui.plot_config import get_plot_panel_class

logger = logging.getLogger(__name__)


class DatasetPlotPanel(QWidget):
    """
    Simplified plot panel for dataset tabs.

    Features:
    - Full PyQtGraphPlotPanel with all interactive features
    - Species selection dropdown
    - Simulate button (uses main mechanism + DSL-defined initials)
    - Dataset shown as scatter points
    - Simulation results shown as lines

    No mechanism duplication - uses main window's mechanism.
    """

    # Signal emitted when user wants to run simulation
    simulateRequested = Signal()

    def __init__(
        self,
        dataset_name: str = "",
        parent: Optional[QWidget] = None
    ):
        """
        Initialize dataset plot panel.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset being displayed
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self._dataset_name = dataset_name
        self._dataset_data: Dict[str, np.ndarray] = {}
        self._species_checkboxes: Dict[str, QCheckBox] = {}
        self._model_t: Optional[np.ndarray] = None
        self._model_series: Dict[str, np.ndarray] = {}
        self._x_label = "Time"
        self._y_label = "Concentration"
        self._simulation_callback: Optional[Callable] = None

        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Top control bar
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(8, 8, 8, 0)
        controls_layout.setSpacing(8)

        # Dataset name
        name_label = QLabel(f"Dataset: {self._dataset_name}")
        name_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        controls_layout.addWidget(name_label)

        controls_layout.addStretch()

        # Simulate button
        self._simulate_btn = QPushButton("▶ Simulate")
        simulate_font = self._simulate_btn.font()
        simulate_font.setBold(True)
        self._simulate_btn.setFont(simulate_font)
        self._simulate_btn.clicked.connect(self._on_simulate_clicked)
        controls_layout.addWidget(self._simulate_btn)

        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("font-size: 9pt; font-style: italic; margin-left: 10px;")
        controls_layout.addWidget(self._status_label)

        layout.addLayout(controls_layout)

        self._species_strip = QWidget(self)
        species_strip_layout = QVBoxLayout(self._species_strip)
        species_strip_layout.setContentsMargins(8, 0, 8, 0)
        species_strip_layout.setSpacing(4)

        species_header = QLabel("Visible species")
        species_header.setStyleSheet("font-weight: bold;")
        species_strip_layout.addWidget(species_header)

        self._species_scroll = QScrollArea(self._species_strip)
        self._species_scroll.setWidgetResizable(True)
        self._species_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._species_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._species_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._species_scroll.setMaximumHeight(84)

        self._species_checkboxes_container = QWidget()
        self._species_checkboxes_layout = QHBoxLayout(self._species_checkboxes_container)
        self._species_checkboxes_layout.setContentsMargins(0, 0, 0, 0)
        self._species_checkboxes_layout.setSpacing(10)
        self._species_checkboxes_layout.addStretch(1)
        self._species_scroll.setWidget(self._species_checkboxes_container)
        species_strip_layout.addWidget(self._species_scroll)
        layout.addWidget(self._species_strip, stretch=0)

        plot_panel_class = get_plot_panel_class()
        self._plot_panel = plot_panel_class()
        layout.addWidget(self._plot_panel, stretch=1)

    def set_simulation_callback(self, callback: Callable):
        """
        Set callback function to run simulation.

        Callback should accept (mechanism_text: str) and return
        dict with keys 't' and 'species'.
        """
        self._simulation_callback = callback

    def _on_simulate_clicked(self):
        """Handle simulate button click."""
        self.simulateRequested.emit()

    def _on_species_checkbox_toggled(self, species_name: str, checked: bool):
        """Handle species checkbox toggle."""
        _ = species_name
        _ = checked
        self._render_dataset_layers()

    def set_data(
        self,
        data_x: np.ndarray,
        data_y: np.ndarray,
        confidence_upper: Optional[np.ndarray] = None,
        confidence_lower: Optional[np.ndarray] = None,
        xlabel: str = "Time",
        ylabel: str = "Concentration",
        all_species: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Set dataset data for visualization and simulation overlays.

        Parameters
        ----------
        data_x, data_y : np.ndarray
            Experimental data points
        confidence_upper, confidence_lower : np.ndarray, optional
            Confidence interval bands
        xlabel, ylabel : str
            Axis labels
        all_species : dict, optional
            All species data {name: y_data}
        """
        # Store the dataset data
        self._dataset_data = {"t": np.asarray(data_x, dtype=float)}
        self._x_label = str(xlabel or "Time")
        self._y_label = "Concentration"

        if all_species:
            # Multi-species dataset
            for species_name, y_data in all_species.items():
                self._dataset_data[str(species_name)] = np.asarray(y_data, dtype=float)
        else:
            # Single species dataset
            self._dataset_data[str(ylabel)] = np.asarray(data_y, dtype=float)

        self._model_t = None
        self._model_series = {}

        # Create species checkboxes
        self._create_species_checkboxes()

        self._render_dataset_layers()

    def reset_for_tab_close(self) -> None:
        """Clear panel-owned and backend-owned state before tab removal."""
        self._dataset_data = {}
        self._model_t = None
        self._model_series = {}
        self._clear_species_checkboxes()
        self._status_label.setText("Ready")
        self._plot_panel.clear()

    def _clear_species_checkboxes(self) -> None:
        """Remove all species checkbox widgets and clear the registry."""
        while self._species_checkboxes_layout.count():
            item = self._species_checkboxes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._species_checkboxes.clear()

    def _create_species_checkboxes(self):
        """Create checkboxes for each species in the dataset."""
        self._clear_species_checkboxes()

        # Create checkbox for each species
        species_names = [k for k in self._dataset_data.keys() if k != 't']

        for species_name in species_names:
            checkbox = QCheckBox(species_name)
            checkbox.setChecked(True)  # All visible by default
            checkbox.toggled.connect(
                lambda checked, name=species_name: self._on_species_checkbox_toggled(name, checked)
            )
            self._species_checkboxes[species_name] = checkbox
            insert_at = max(0, self._species_checkboxes_layout.count() - 1)
            self._species_checkboxes_layout.insertWidget(insert_at, checkbox)

    def _visible_species_names(self) -> List[str]:
        return [
            name
            for name, checkbox in (self._species_checkboxes or {}).items()
            if bool(getattr(checkbox, "isChecked", lambda: False)())
        ]

    def apply_fit_render_projection(self, projection: FitRenderDatasetProjection) -> None:
        """Apply a typed fitting render projection as the dataset-tab fit overlay."""
        if not isinstance(projection, FitRenderDatasetProjection):
            return
        if projection.status != "ok":
            return
        self._model_t = np.asarray(projection.model_x, dtype=float)
        self._model_series = {
            str(name): np.asarray(values, dtype=float)
            for name, values in projection.model_series.items()
        }
        self._render_dataset_layers()

    def _render_dataset_layers(self) -> None:
        """Delegate dataset-tab rendering to the backend public contract."""
        if not self._dataset_data:
            return
        t = np.asarray(self._dataset_data.get("t"), dtype=float).reshape(-1)
        dataset_series = {
            str(name): np.asarray(values, dtype=float).reshape(-1)
            for name, values in self._dataset_data.items()
            if name != "t"
        }
        try:
            self._plot_panel.render_dataset_layers(
                data_t=t,
                dataset_series=dataset_series,
                model_t=self._model_t,
                model_series=dict(self._model_series) if self._model_series else None,
                visible_species=self._visible_species_names(),
                xlabel=self._x_label,
                ylabel=self._y_label,
            )
            logger.info(
                "Rendered dataset layers for %s (%s species, %s model series)",
                self._dataset_name,
                len(dataset_series),
                len(self._model_series),
            )
        except Exception as exc:
            logger.error("Failed to render dataset layers: %s", exc)

    def plot_simulation_results(self, t: np.ndarray, species_data: Dict[str, np.ndarray]):
        """
        Plot simulation results as lines on top of data points.

        Called externally after simulation completes.
        """
        self._plot_simulation_results(t, species_data)

    def _plot_simulation_results(self, t: np.ndarray, species_data: Dict[str, np.ndarray]):
        """Plot simulation results as lines for all species."""
        self._model_t = np.asarray(t, dtype=float)
        self._model_series = {
            str(name): np.asarray(values, dtype=float)
            for name, values in (species_data or {}).items()
            if str(name)
        }
        self._render_dataset_layers()
        self._status_label.setText("Simulation complete")

    def set_status(self, message: str):
        """Update status label."""
        self._status_label.setText(message)

    def get_dataset_data(self) -> Dict[str, np.ndarray]:
        """Get the dataset data."""
        return self._dataset_data

    def export_payload(self) -> Optional[Dict[str, object]]:
        """
        Return a standardized payload for CSV exports.

        Returns
        -------
        dict or None
            {'t': np.ndarray, 'series': Dict[str, np.ndarray]}
        """
        if not self._dataset_data or 't' not in self._dataset_data:
            return None
        t = np.asarray(self._dataset_data.get('t'))
        species = {
            name: np.asarray(values)
            for name, values in self._dataset_data.items()
            if name != 't'
        }
        if t.size == 0 or not species:
            return None
        return {"t": t, "series": species}

    def build_visible_export(self, scope: str) -> Tuple[List[str], List[List[object]]]:
        payload = self.export_payload()
        if not payload:
            raise ValueError("No dataset data available to export.")

        t_values = np.asarray(payload.get("t"), dtype=float).reshape(-1)
        series = payload.get("series") or {}
        if t_values.size == 0 or not isinstance(series, dict) or not series:
            raise ValueError("No dataset data available to export.")

        species_names = list(series.keys())
        if str(scope or "") == "axis":
            visible = [
                name
                for name, cb in (self._species_checkboxes or {}).items()
                if bool(getattr(cb, "isChecked", lambda: False)())
            ]
            species_names = [name for name in species_names if name in visible]

        if not species_names:
            raise ValueError("No series selected to export.")

        arrays = []
        for name in species_names:
            arr = np.asarray(series[name], dtype=float).reshape(-1)
            if arr.shape[0] != t_values.shape[0]:
                raise ValueError(
                    f"Series '{name}' length ({arr.shape[0]}) does not match time grid ({t_values.shape[0]})."
                )
            arrays.append(arr)

        header = ["Time"] + species_names

        def row_iter() -> Iterator[List[object]]:
            for idx in range(t_values.shape[0]):
                yield [t_values[idx]] + [arr[idx] for arr in arrays]

        return header, row_iter()
