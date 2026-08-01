# kindred/gui/widgets/species_statistics_table.py
"""Species statistics summary table widget."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from kindred.core.results import integrate_ctc

__all__ = ["SpeciesStatisticsTable"]

class SpeciesStatisticsTable(QtWidgets.QTableWidget):
    """
    Bottom panel showing final concentrations, max values, equilibrium times, and fit stats.

    Displays per-species statistics including:
    - Species name
    - Final concentration at t_end
    - Maximum concentration reached
    - Time at which maximum occurs
    - Result-level residual mean square from fitting
    - CTC (Concentration-Time Curve) - baseline-corrected integral

    The CTC calculation uses smart logic:
    - For species converging to ~0: CTC = ∫C dt (area under curve)
    - For species converging to C_final: CTC = ∫|C - C_final| dt (deviation from equilibrium)

    This prevents infinite growth for species with non-zero equilibrium values.
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

        # Configure table
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels([
            "Species", "Final", "Max", "t@Max", "Result MSE", "CTC"
        ])

        # Add tooltips to explain each column
        header = self.horizontalHeader()
        header.setToolTip("Species: Chemical species name")
        self.horizontalHeaderItem(0).setToolTip("Species name")
        self.horizontalHeaderItem(1).setToolTip("Final concentration at t_end")
        self.horizontalHeaderItem(2).setToolTip("Maximum concentration reached")
        self.horizontalHeaderItem(3).setToolTip("Time at which maximum concentration occurs")
        self.horizontalHeaderItem(4).setToolTip("Result-level raw residual mean square from fitting")
        self.horizontalHeaderItem(5).setToolTip(
            "Concentration-Time Curve (baseline-corrected):\n"
            "• Species → 0: ∫C dt (area under curve)\n"
            "• Species → C_final: ∫|C - C_final| dt (deviation from equilibrium)"
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

    def update_results(self, t: np.ndarray, species_data: Dict[str, np.ndarray],
                       residual_mse: Optional[float] = None):
        """
        Update table with simulation results.

        Parameters
        ----------
        t : np.ndarray
            Time array
        species_data : dict
            Dictionary of {species_name: concentration_array}
        residual_mse : float, optional
            Result-level raw residual mean square for fitting results
        """
        # Clear existing rows
        self.setRowCount(0)

        # Add row for each species
        for i, (species_name, concentrations) in enumerate(sorted(species_data.items())):
            self.insertRow(i)

            # Column 0: Species name
            _item0 = QtWidgets.QTableWidgetItem(species_name)
            _item0.setTextAlignment(QtCore.Qt.AlignCenter)
            self.setItem(i, 0, _item0)

            # Column 1: Final concentration
            final_conc = concentrations[-1]
            _item1 = QtWidgets.QTableWidgetItem(f"{final_conc:.6g}")
            _item1.setTextAlignment(QtCore.Qt.AlignCenter)
            self.setItem(i, 1, _item1)

            # Column 2: Maximum concentration
            max_conc = np.max(concentrations)
            _item2 = QtWidgets.QTableWidgetItem(f"{max_conc:.6g}")
            _item2.setTextAlignment(QtCore.Qt.AlignCenter)
            self.setItem(i, 2, _item2)

            # Column 3: Time at maximum (t@Max)
            idx_max = np.argmax(concentrations)
            t_at_max = t[idx_max]
            _item3 = QtWidgets.QTableWidgetItem(f"{t_at_max:.6g}")
            _item3.setTextAlignment(QtCore.Qt.AlignCenter)
            self.setItem(i, 3, _item3)

            # Column 4: result-level residual MSE (only if provided, from fitting)
            if residual_mse is not None:
                _item4 = QtWidgets.QTableWidgetItem(f"{residual_mse:.6g}")
            else:
                _item4 = QtWidgets.QTableWidgetItem("—")
            _item4.setTextAlignment(QtCore.Qt.AlignCenter)
            self.setItem(i, 4, _item4)

            # Column 5: CTC (Concentration-Time Curve)
            # Baseline-corrected integral measuring deviation from equilibrium:
            # - Species → 0: ∫C dt (area under curve)
            # - Species → C_final ≠ 0: ∫|C - C_final| dt (deviation from baseline)

            threshold = max(
                1e-10,
                0.01 * float(np.max(np.abs(concentrations))),
            )
            integrand = (
                concentrations
                if abs(float(final_conc)) < threshold
                else np.abs(concentrations - final_conc)
            )
            ctc, _method, _uniform, _eps, _tail = integrate_ctc(
                t,
                integrand,
                uniformity_eps=1e-6,
                tail_strategy="38",
            )

            _item5 = QtWidgets.QTableWidgetItem(f"{ctc:.6g}")
            _item5.setTextAlignment(QtCore.Qt.AlignCenter)
            self.setItem(i, 5, _item5)

        # Force immediate visual update for WSL/X11 environments
        self.viewport().repaint()
