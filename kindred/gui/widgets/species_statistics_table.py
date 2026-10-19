# kindred/gui/widgets/species_statistics_table.py
"""Species statistics summary table widget."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from kindred.core.quantity_units import (
    MOLAR,
    SECOND,
    Unit,
    convert_value,
    format_unit_for_display,
)
from kindred.gui.ui_helpers import SCIENTIFIC_NAME_HTML_ROLE, ScientificNameHeaderView
from kindred.core.simulator.solver_types import SimulationPresentationBoundary

__all__ = ["SpeciesStatisticsTable"]


def _display_statistic(value: float, source: Unit, target: Unit) -> str:
    try:
        return f"{convert_value(value, source, target):.6g}"
    except ValueError:
        return "Unavailable"


class SpeciesStatisticsTable(QtWidgets.QTableWidget):
    """
    Bottom panel showing simulation-owned species statistics.

    Displays per-species statistics including:
    - Species name
    - Final concentration at t_end
    - Maximum concentration reached
    - Time at which maximum occurs
    - CTC (cumulative temporal concentration) - the signed integral of the
      concentration minus its final value over the simulated span

    The CTC is the result's own value, which results with interventions other
    than temperature changes do not carry.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """
        Initialize species statistics table.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

        # Configure table
        self.setColumnCount(5)
        self.setHorizontalHeader(
            ScientificNameHeaderView(QtCore.Qt.Orientation.Horizontal, self)
        )
        self.setHorizontalHeaderLabels([
            "Species", "Final", "Max", "t@Max", "CTC"
        ])

        # Add tooltips to explain each column
        header = self.horizontalHeader()
        self.horizontalHeaderItem(0).setToolTip("Species name.")
        self.horizontalHeaderItem(1).setToolTip("Concentration at the last returned simulation sample.")
        self.horizontalHeaderItem(2).setToolTip("Maximum concentration reached.")
        self.horizontalHeaderItem(3).setToolTip("Time at which maximum concentration occurs.")
        self.horizontalHeaderItem(4).setToolTip(
            "Cumulative temporal concentration, the integral over the simulated "
            "time of the concentration minus its final concentration. Unavailable "
            "for simulations with interventions other than temperature changes."
        )

        header.setStretchLastSection(True)
        for col in range(self.columnCount()):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        self.setAlternatingRowColors(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def sizeHintForColumn(self, column: int) -> int:
        width = super().sizeHintForColumn(column)
        # Qt also accounts for the header, so "Species" always fits.
        return min(width, 140) if column == 0 else width

    def update_results(
        self,
        canonical_time: np.ndarray,
        species_data: Mapping[str, np.ndarray],
        *,
        species_ctc: Mapping[str, float] | None,
        presentation_boundaries: tuple[SimulationPresentationBoundary, ...],
        concentration_display_unit: Unit,
        time_display_unit: Unit,
    ) -> None:
        """
        Update table with simulation results.

        Parameters
        ----------
        canonical_time : np.ndarray
            Canonical ``SimulationResultValue.t`` array. Display X is invalid
            at this scientific-statistics boundary.
        species_data : dict
            Dictionary of {species_name: concentration_array}
        species_ctc : Mapping, optional
            Each species' cumulative temporal concentration in M·s, absent for
            results with interventions other than temperature changes.
        presentation_boundaries : tuple, optional
            Every retained intervention boundary state of the primary result;
            these contribute to Max and t@Max without becoming table rows.
        """
        ctc_unit = concentration_display_unit.multiply(time_display_unit)
        for column, label, unit in (
            (1, "Final", concentration_display_unit),
            (2, "Max", concentration_display_unit),
            (3, "t@Max", time_display_unit),
            (4, "CTC", ctc_unit),
        ):
            header_item = self.horizontalHeaderItem(column)
            unit_text = unit.symbol
            header_item.setText(f"{label} ({unit_text})")
            header_item.setData(
                SCIENTIFIC_NAME_HTML_ROLE,
                f"{label} ({format_unit_for_display(unit)})",
            )
        projected_roster = tuple(sorted(species_data))
        calculated: list[tuple[str, ...]] = []
        for species_name in projected_roster:
            concentrations = species_data[species_name]
            final_conc = concentrations[-1]
            max_conc = np.max(concentrations)
            idx_max = np.argmax(concentrations)
            max_time = canonical_time[idx_max]
            for boundary in presentation_boundaries:
                for state in boundary.states:
                    boundary_value = dict(state.concentrations).get(species_name)
                    if boundary_value is not None and boundary_value > max_conc:
                        max_conc = boundary_value
                        max_time = boundary.time
            final_text = _display_statistic(final_conc, MOLAR, concentration_display_unit)
            maximum_text = _display_statistic(max_conc, MOLAR, concentration_display_unit)
            time_text = _display_statistic(max_time, SECOND, time_display_unit)
            ctc_text = "—"
            if species_ctc is not None:
                ctc_text = _display_statistic(species_ctc[species_name], MOLAR.multiply(SECOND), ctc_unit)
            calculated.append(
                (
                    species_name,
                    final_text,
                    maximum_text,
                    time_text,
                    ctc_text,
                )
            )
        rows = tuple(calculated)

        names = tuple(row[0] for row in rows)
        current_names = tuple(
            self.item(row, 0).text() if self.item(row, 0) is not None else ""
            for row in range(self.rowCount())
        )
        if current_names != names:
            self.setRowCount(len(rows))
            for row in range(len(rows)):
                for column in range(self.columnCount()):
                    item = QtWidgets.QTableWidgetItem()
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.setItem(row, column, item)
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.item(row, column).setText(value)
