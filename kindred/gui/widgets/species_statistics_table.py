# kindred/gui/widgets/species_statistics_table.py
"""Species statistics summary table widget."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from kindred.core.intervention_schedule_compiler import (
    InterventionExecutionContract,
)
from kindred.core.results import integrate_ctc
from kindred.core.simulator.solver_types import SimulationPresentationBoundary

__all__ = ["SpeciesStatisticsTable"]

class SpeciesStatisticsTable(QtWidgets.QTableWidget):
    """
    Bottom panel showing simulation-owned species statistics.

    Displays per-species statistics including:
    - Species name
    - Final concentration at t_end
    - Maximum concentration reached
    - Time at which maximum occurs
    - CTC (Concentration-Time Curve) - fixed-policy scalar integral

    CTC uses one deterministic native-grid trapezoid policy. A nonfinite
    numerical result is rendered as unavailable by the typed result owner.
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

        self._statistics_cache: tuple[
            object,
            tuple[str, ...],
            bool,
            tuple[tuple[str, ...], ...],
        ] | None = None

        # Configure table
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels([
            "Species", "Final", "Max", "t@Max", "CTC"
        ])

        # Add tooltips to explain each column
        header = self.horizontalHeader()
        header.setToolTip("Species: Chemical species name")
        self.horizontalHeaderItem(0).setToolTip("Species name")
        self.horizontalHeaderItem(1).setToolTip("Final concentration at t_end")
        self.horizontalHeaderItem(2).setToolTip("Maximum concentration reached")
        self.horizontalHeaderItem(3).setToolTip("Time at which maximum concentration occurs")
        self.horizontalHeaderItem(4).setToolTip(
            "Concentration-Time Curve (fixed-policy scalar):\n"
            "• Species → 0: ∫C dt\n"
            "• Species → C_final: ∫|C - C_final| dt\n"
            "Nonfinite numerical results are unavailable."
        )

        header.setStretchLastSection(False)
        for col in range(self.columnCount()):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)  # Species
        self.setColumnWidth(0, 140)
        self.setAlternatingRowColors(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def update_results(
        self,
        canonical_time: np.ndarray,
        species_data: Mapping[str, np.ndarray],
        *,
        intervention_execution: InterventionExecutionContract,
        presentation_boundaries: tuple[SimulationPresentationBoundary, ...] = (),
        accepted_result_key: object | None = None,
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
        intervention_execution : InterventionExecutionContract
            Accepted Run intervention truth. Any intervention makes CTC
            inapplicable for the complete result.
        presentation_boundaries : tuple, optional
            Exact intervention pre/post values retained by the primary result;
            these contribute to Max and t@Max without becoming table rows.
        accepted_result_key : object, optional
            Accepted-result identity used with the projected species roster
            for the single-entry statistics cache.
        """
        if not isinstance(
            intervention_execution,
            InterventionExecutionContract,
        ):
            raise TypeError(
                "intervention_execution must be an "
                "InterventionExecutionContract."
            )
        boundaries = tuple(presentation_boundaries)
        if any(
            not isinstance(boundary, SimulationPresentationBoundary)
            for boundary in boundaries
        ):
            raise TypeError(
                "presentation_boundaries must contain only "
                "SimulationPresentationBoundary values."
            )
        ctc_applicable = intervention_execution.is_empty
        time = np.asarray(canonical_time, dtype=float)
        if (
            time.ndim != 1
            or time.size == 0
            or np.any(~np.isfinite(time))
            or (time.size > 1 and np.any(np.diff(time) <= 0.0))
        ):
            raise ValueError(
                "Scientific statistics require finite, strictly increasing "
                "canonical simulation time."
            )
        normalized: dict[str, np.ndarray] = {}
        for raw_name, raw_values in species_data.items():
            name = str(raw_name or "").strip()
            values = np.asarray(raw_values, dtype=float)
            if (
                not name
                or values.ndim != 1
                or values.size != time.size
                or np.any(~np.isfinite(values))
            ):
                raise ValueError(
                    "Scientific statistics series must be named, finite, and "
                    "aligned with canonical simulation time."
                )
            if name in normalized:
                raise ValueError(
                    "Scientific statistics series names must be unique after "
                    "normalization."
                )
            normalized[name] = values
        projected_roster = tuple(sorted(normalized))
        cached = self._statistics_cache
        if (
            cached is not None
            and accepted_result_key is not None
            and cached[0] == accepted_result_key
            and cached[1] == projected_roster
            and cached[2] is ctc_applicable
        ):
            rows = cached[3]
        else:
            calculated: list[tuple[str, ...]] = []
            for species_name in projected_roster:
                concentrations = normalized[species_name]
                final_conc = concentrations[-1]
                max_conc = np.max(concentrations)
                idx_max = np.argmax(concentrations)
                max_time = time[idx_max]
                for boundary in boundaries:
                    for boundary_values in (boundary.pre_state, boundary.post_state):
                        boundary_value = dict(boundary_values).get(species_name)
                        if boundary_value is not None and boundary_value > max_conc:
                            max_conc = boundary_value
                            max_time = boundary.time
                ctc_text = "—"
                if ctc_applicable:
                    integrand = (
                        concentrations
                        if final_conc == 0.0
                        else np.abs(concentrations - final_conc)
                    )
                    ctc = integrate_ctc(time, integrand)
                    ctc_text = (
                        f"{ctc.value:.6g}"
                        if ctc.available
                        else "Unavailable"
                    )
                calculated.append(
                    (
                        species_name,
                        f"{final_conc:.6g}",
                        f"{max_conc:.6g}",
                        f"{max_time:.6g}",
                        ctc_text,
                    )
                )
            rows = tuple(calculated)
            if accepted_result_key is not None:
                self._statistics_cache = (
                    accepted_result_key,
                    projected_roster,
                    ctc_applicable,
                    rows,
                )
            else:
                self._statistics_cache = None

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
