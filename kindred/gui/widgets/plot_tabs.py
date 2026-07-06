# kindred/gui/widgets/plot_tabs.py
"""Tabbed plot widget for multiple datasets with fit metrics."""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple, Dict, Any, Sequence

from PySide6 import QtWidgets

from kindred.core.analysis.global_fit_projection import FitRenderDatasetProjection
from kindred.gui.plot_config import get_plot_panel_class
# Direct imports required to avoid circular dependency with widgets/__init__.py
from kindred.gui.widgets.grid_plot_view import GridPlotView
from kindred.gui.widgets.dataset_plot_panel import DatasetPlotPanel

logger = logging.getLogger(__name__)

__all__ = ["PlotTabsWidget"]


class PlotTabsWidget(QtWidgets.QWidget):
    """
    Tabbed plot area for multiple datasets with fit metrics.

    Features:
    - Default "Simulation" tab with the configured PyQtGraph backend
    - "All Datasets" grid view for comparing multiple datasets
    - Dynamic dataset tabs with DatasetPlotPanel (enhanced plotting, species selection)
    - Dataset residual MSE displayed neutrally; R² uses color-coded tab indicators.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        main_plot_embed_analysis_tabs: bool = True,
    ):
        """
        Initialize plot tabs widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget for different datasets/views
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._tabs.setTabPosition(QtWidgets.QTabWidget.North)
        self._tabs.setTabsClosable(True)  # Allow tab closing for memory management
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        layout.addWidget(self._tabs)

        # Add default "Simulation" tab (not closeable)
        # Use PyQtGraph plot backend (GPU-accelerated)
        PlotPanelClass = get_plot_panel_class()
        self._main_plot = PlotPanelClass(
            embed_analysis_tabs=bool(main_plot_embed_analysis_tabs),
            workspace_splitter_object_name="mainPlotWorkspaceSplitter",
            enable_axis_inversion_actions=True,
            enable_reference_layer_toggle_action=True,
            enable_copy_visible_data_action=True,
        )
        self._main_plot.setObjectName("plotPanel")
        self._main_plot_analysis_widget = None
        if not bool(main_plot_embed_analysis_tabs):
            detach = getattr(self._main_plot, "detach_analysis_tabs_for_dock", None)
            if callable(detach):
                self._main_plot_analysis_widget = detach()
        sim_index = self._tabs.addTab(self._main_plot, "Simulation")
        self._tabs.tabBar().setTabButton(sim_index, QtWidgets.QTabBar.RightSide, None)  # Remove close button
        logger.debug(f"Main plot using {PlotPanelClass.__name__}")

        # Add "All Datasets" grid view tab (not closeable)
        self._grid_view = GridPlotView()
        self._grid_tab_index = self._tabs.addTab(self._grid_view, "All Datasets")
        self._tabs.tabBar().setTabButton(self._grid_tab_index, QtWidgets.QTabBar.RightSide, None)  # Remove close button

        # Track dataset plots for grid view updates
        self._dataset_plots: List[Tuple[str, DatasetPlotPanel]] = []
        self._dataset_tab_names: Dict[int, str] = {}

        # P1 FIX: Track tab colors to maintain complete stylesheet
        # Maps tab_index -> color for proper multi-tab coloring
        self._tab_colors = {}

    def main_plot_analysis_widget(self) -> Optional[QtWidgets.QWidget]:
        widget = getattr(self, "_main_plot_analysis_widget", None)
        if widget is not None:
            return widget
        getter = getattr(self._main_plot, "analysis_tabs_widget", None)
        if callable(getter):
            return getter()
        return getattr(self._main_plot, "_details_tabs", None)

    def main_plot_workspace_splitter(self) -> Optional[QtWidgets.QSplitter]:
        getter = getattr(self._main_plot, "workspace_splitter", None)
        if callable(getter):
            return getter()
        return getattr(self._main_plot, "_main_splitter", None)

    def _analysis_tabs_widget(self) -> Optional[QtWidgets.QTabWidget]:
        widget = self.main_plot_analysis_widget()
        if isinstance(widget, QtWidgets.QTabWidget):
            return widget
        return None

    def available_analysis_surfaces(self) -> List[str]:
        tabs = self._analysis_tabs_widget()
        if tabs is None:
            return []
        return [str(tabs.tabText(index)) for index in range(tabs.count())]

    def focus_analysis_surface(self, surface_name: str) -> bool:
        tabs = self._analysis_tabs_widget()
        if tabs is None:
            return False
        target_name = str(surface_name).strip()
        if not target_name:
            return False
        for index in range(tabs.count()):
            if tabs.tabText(index) != target_name:
                continue
            tabs.setCurrentIndex(index)
            current_widget = tabs.currentWidget()
            if current_widget is not None:
                current_widget.setFocus()
            else:
                tabs.setFocus()
            return True
        return False

    def add_dataset_tab(
        self,
        name: str,
        residual_mse: Optional[float] = None,
        r_squared: Optional[float] = None,
    ):
        """
        Add a new tab for a dataset with enhanced plotting.

        Parameters
        ----------
        name : str
            Dataset name
        residual_mse : float, optional
            Raw residual mean square for the fit
        r_squared : float, optional
            R-squared goodness of fit

        Returns
        -------
        DatasetPlotPanel
            The plot panel for this dataset with full plotting features

        """
        # Use DatasetPlotPanel for dataset tabs (simplified, no mechanism duplication)
        plot = DatasetPlotPanel(dataset_name=name)

        tab_label, color, tooltip = self._fit_quality_tab_state(
            name,
            residual_mse=residual_mse,
            r_squared=r_squared,
        )

        tab_index = self._tabs.addTab(plot, tab_label)
        self._tabs.setTabToolTip(tab_index, tooltip)

        # Apply color to tab
        self._set_tab_color(tab_index, color)

        # Track this dataset
        self._dataset_plots.append((name, plot))
        self._dataset_tab_names[id(plot)] = name

        return plot

    def sync_dataset_tab(
        self,
        name: str,
        *,
        t,
        data_y,
        ylabel: str = "Concentration",
        xlabel: str = "Time",
        x_units: object = None,
        y_units: object = "M",
        all_species: Optional[Dict[str, Any]] = None,
        observations: Optional[Dict[str, Any]] = None,
        residual_mse: Optional[float] = None,
        r_squared: Optional[float] = None,
        fit_render_projection: Optional[FitRenderDatasetProjection] = None,
    ) -> DatasetPlotPanel:
        """
        Create or update a dataset tab using PlotTabsWidget-owned tab lookup and index bookkeeping.
        """
        panel = next((plot for dataset_name, plot in self._dataset_plots if dataset_name == name), None)
        if panel is None:
            panel = self.add_dataset_tab(name, residual_mse=residual_mse, r_squared=r_squared)

        panel.set_data(
            t,
            data_y,
            xlabel=xlabel,
            ylabel=ylabel,
            x_units=x_units,
            y_units=y_units,
            all_species=all_species,
            observations=observations,
        )
        if fit_render_projection is not None:
            panel.apply_fit_render_projection(fit_render_projection)

        tab_index = self._tabs.indexOf(panel)
        if tab_index != -1:
            self.update_tab_fit_quality(
                tab_index,
                residual_mse=residual_mse,
                r_squared=r_squared,
            )
        return panel

    def sync_dataset_grid(self, dataset_entries: Sequence[Dict[str, Any]]) -> None:
        """Replace dataset grid contents using the PlotTabsWidget-owned grid view."""
        normalized: List[Dict[str, Any]] = []
        for entry in dataset_entries or []:
            if not isinstance(entry, dict):
                continue
            normalized.append(dict(entry))
        self._grid_view.set_datasets(normalized)

    def remove_dataset_tab(self, name: str):
        """
        Remove an existing dataset tab by name, if present.

        Parameters
        ----------
        name : str
            Dataset name to remove
        """
        for dataset_name, panel in list(self._dataset_plots):
            if dataset_name != name:
                continue

            tab_index = self._tabs.indexOf(panel)
            if tab_index != -1:
                self._on_tab_close_requested(tab_index)
            break

    def update_tab_fit_quality(
        self,
        tab_index: int,
        residual_mse: Optional[float] = None,
        r_squared: Optional[float] = None
    ):
        """
        Update fit quality indicators for an existing tab.

        Parameters
        ----------
        tab_index : int
            Index of tab to update
        residual_mse : float, optional
            Raw residual mean square for the fit
        r_squared : float, optional
            R-squared goodness of fit
        """
        panel = self._tabs.widget(tab_index)
        base_name = self._dataset_tab_names.get(id(panel)) if isinstance(panel, DatasetPlotPanel) else None
        if not base_name:
            base_name = self._tabs.tabText(tab_index)

        new_label, color, tooltip = self._fit_quality_tab_state(
            base_name,
            residual_mse=residual_mse,
            r_squared=r_squared,
        )

        self._tabs.setTabText(tab_index, new_label)
        self._tabs.setTabToolTip(tab_index, tooltip)
        self._set_tab_color(tab_index, color)

    @staticmethod
    def _finite_metric(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            metric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(metric) or math.isinf(metric):
            return None
        return metric

    def _fit_quality_tab_state(
        self,
        base_name: str,
        *,
        residual_mse: Optional[float],
        r_squared: Optional[float],
    ) -> Tuple[str, str, str]:
        base = str(base_name)
        mse_val = self._finite_metric(residual_mse)
        r2_val = self._finite_metric(r_squared)
        tooltip_parts = [base]
        if mse_val is not None:
            label = f"{base} (MSE={mse_val:.3e})"
            tooltip_parts.append(f"MSE = {mse_val:.3e}")
            if r2_val is not None:
                tooltip_parts.append(f"R² = {r2_val:.3f}")
                color = self._get_color_for_r_squared(r2_val)
            else:
                color = "#666666"
        elif r2_val is not None:
            label = f"{base} (R²={r2_val:.3f})"
            tooltip_parts.append(f"R² = {r2_val:.3f}")
            color = self._get_color_for_r_squared(r2_val)
        else:
            label = base
            color = "#888888"
        return label, color, "\n".join(part for part in tooltip_parts if part)

    def show_batch_results(
        self,
        batch_results: Sequence[Dict[str, Any]],
        preferred_species: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Populate the grid view with multiple simulation results and switch to it.

        Parameters
        ----------
        batch_results : Sequence[dict]
            Iterable of entries with keys 't', 'series', and 'parameters'.
        preferred_species : Sequence[str], optional
            Species names to keep highlighted if available.
        """
        if not batch_results:
            logger.warning("No results to display.")
            return

        self._grid_view.clear_datasets()

        for idx, entry in enumerate(batch_results, start=1):
            t_values = entry.get("t")
            species_map = entry.get("series") or entry.get("species")
            if t_values is None or not species_map:
                logger.debug("Skipping incomplete entry at index %d", idx)
                continue

            parameters = entry.get("parameters", {})
            param_fragments = [f"{name}={value:.4g}" for name, value in parameters.items()]
            label_suffix = "; ".join(param_fragments) if param_fragments else "Parameters unavailable"
            dataset_name = f"Run {idx}: {label_suffix}"

            try:
                first_species = next(iter(species_map.keys()))
            except StopIteration:
                logger.debug("Entry %d lacks species data", idx)
                continue

            current_species = None
            if preferred_species:
                for name in preferred_species:
                    if name in species_map:
                        current_species = name
                        break
            if current_species is None:
                current_species = first_species

            self._grid_view.add_dataset(
                dataset_name,
                t_values,
                species_map[current_species],
                all_species=species_map,
                current_species=current_species,
            )

        if preferred_species:
            self._grid_view.set_species_selection(preferred_species)

        self._tabs.setCurrentWidget(self._grid_view)

    def _get_color_for_r_squared(self, r_squared: float) -> str:
        """
        Get color code based on R-squared value.

        P1 FIX: Added validation for NaN, out-of-bounds, or infinite values.

        Parameters
        ----------
        r_squared : float
            R-squared value (should be 0-1)

        Returns
        -------
        str
            HTML color code
        """
        # Validate input
        if not isinstance(r_squared, (int, float)):
            return "#888888"  # Gray for invalid type

        if math.isnan(r_squared) or math.isinf(r_squared):
            return "#888888"  # Gray for NaN/inf

        if r_squared > 1:
            return "#888888"  # Gray for out of bounds

        # Color based on quality
        if r_squared > 0.95:
            return "#2a5"  # Green (good fit)
        elif r_squared > 0.85:
            return "#da5"  # Yellow (moderate fit)
        else:
            return "#d44"  # Red (poor fit)

    def _apply_tab_color_stylesheet(self) -> None:
        tab_bar = self._tabs.tabBar()
        stylesheet_parts = []
        for idx, tab_color in self._tab_colors.items():
            css_index = idx + 1
            stylesheet_parts.append(f"""
                QTabBar::tab:nth-child({css_index}) {{
                    background-color: {tab_color};
                    color: white;
                    font-weight: bold;
                    padding: 6px 12px;
                    margin: 2px;
                    border-radius: 4px;
                }}
                QTabBar::tab:nth-child({css_index}):selected {{
                    background-color: {tab_color};
                    border: 2px solid white;
                }}
                QTabBar::tab:nth-child({css_index}):hover {{
                    background-color: {tab_color};
                    opacity: 0.8;
                }}
            """)

        complete_stylesheet = "\n".join(stylesheet_parts)
        tab_bar.setStyleSheet(complete_stylesheet)

    def _rebase_tab_colors_after_removal(self, removed_index: int) -> None:
        rebased: Dict[int, str] = {}
        for idx, color in self._tab_colors.items():
            if idx == removed_index:
                continue
            rebased[idx - 1 if idx > removed_index else idx] = color
        self._tab_colors = rebased
        self._apply_tab_color_stylesheet()

    def _set_tab_color(self, tab_index: int, color: str):
        """
        Apply color to a tab.

        P1 FIX: This now maintains state for all tab colors and rebuilds
        the complete stylesheet, fixing the bug where only the last tab
        was colored (nth-child selector was being replaced each time).

        Parameters
        ----------
        tab_index : int
            Index of tab to color
        color : str
            HTML color code (e.g., "#2a5")
        """
        self._tab_colors[tab_index] = color
        self._apply_tab_color_stylesheet()

    def _on_tab_close_requested(self, index: int):
        """
        Handle tab close request - clear data and remove tab.

        Parameters
        ----------
        index : int
            Index of tab to close

        Notes
        -----
        Prevents closing of main "Simulation" and "All Datasets" tabs.
        Clears plot data before removing tab to free memory.
        """
        # Don't allow closing the main simulation tab (index 0) or grid view tab
        if index <= 1:
            return

        # Get the widget before removing
        widget = self._tabs.widget(index)

        # Delegate close/reset cleanup to the dataset panel owner.
        if isinstance(widget, DatasetPlotPanel):
            widget.reset_for_tab_close()

        # Remove from dataset tracking
        name_to_remove = None
        for name, plot in self._dataset_plots:
            if plot == widget:
                name_to_remove = name
                break

        if name_to_remove:
            removed = [p for n, p in self._dataset_plots if n == name_to_remove]
            self._dataset_plots = [(n, p) for n, p in self._dataset_plots if n != name_to_remove]
            for panel in removed:
                self._dataset_tab_names.pop(id(panel), None)

        # Remove the tab
        self._tabs.removeTab(index)
        self._rebase_tab_colors_after_removal(index)

    def get_current_plot(self) -> QtWidgets.QWidget:
        """Get the currently visible plot."""
        return self._tabs.currentWidget()
