"""
Fitting diagnostics dialog for parameter optimization analysis.

Provides statistical diagnostics and visualizations for fitting results:
- Parameter correlation matrix (heatmap)
- Residual analysis (Q-Q plot, histogram, residuals vs fitted)
- Confidence intervals (approximate)
- Convergence history plot
- Goodness-of-fit statistics (R², χ², RMSE)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

logger = logging.getLogger(__name__)

__all__ = ["FittingDiagnosticsDialog"]


class FittingDiagnosticsDialog(QtWidgets.QDialog):
    """
    Dialog for displaying fitting diagnostics and statistics.

    Features:
    - Parameter correlation matrix heatmap
    - Residual plots (Q-Q, histogram, vs fitted)
    - Goodness-of-fit statistics
    - Convergence history
    - Approximate confidence intervals

    The dialog uses PyQtGraph for plotting when available, with fallback to text-based display.
    """

    def __init__(self, fit_result: Dict[str, Any], parent: Optional[QtWidgets.QWidget] = None):
        """
        Initialize fitting diagnostics dialog.

        Parameters
        ----------
        fit_result : dict
            Fitting result dictionary containing:
            - 'parameters': Dict[str, float] - Fitted parameters
            - 'residuals': np.ndarray - Residuals (observed - predicted)
            - 'predicted': np.ndarray - Predicted values
            - 'observed': np.ndarray - Observed values
            - 'jacobian': Optional[np.ndarray] - Jacobian matrix at solution
            - 'covariance': Optional[np.ndarray] - Covariance matrix
            - 'success': bool - Whether fit converged
            - 'message': str - Optimizer message
            - 'nfev': int - Number of function evaluations
            - 'cost': float - Final cost function value
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self.fit_result = fit_result
        self.setWindowTitle("Fitting Diagnostics")
        self.setModal(False)  # Allow user to interact with main window
        self.resize(900, 700)

        layout = QtWidgets.QVBoxLayout(self)

        # Title
        title = QtWidgets.QLabel("Fitting Diagnostics")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Tab widget for different diagnostic views
        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs)

        # Tab 1: Summary statistics
        self._create_summary_tab()

        # Tab 2: Parameter diagnostics
        self._create_parameter_tab()

        # Tab 3: Residual analysis
        self._create_residual_tab()

        # Tab 4: Convergence history
        self._create_convergence_tab()

        # Close button
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)

    def _create_summary_tab(self):
        """Create summary statistics tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Goodness-of-fit statistics
        stats_group = QtWidgets.QGroupBox("Goodness of Fit")
        stats_layout = QtWidgets.QFormLayout(stats_group)

        residuals = self.fit_result.get('residuals', np.array([]))
        observed = self.fit_result.get('observed', np.array([]))

        if len(residuals) > 0 and len(observed) > 0:
            # Calculate statistics
            n = len(residuals)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((observed - np.mean(observed))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))

            # Chi-squared (normalized by variance)
            if len(observed) > 0:
                variance = np.var(observed)
                chi_squared = ss_res / variance if variance > 0 else np.inf
            else:
                chi_squared = np.inf

            stats_layout.addRow("R² (coefficient of determination):", QtWidgets.QLabel(f"{r_squared:.6f}"))
            stats_layout.addRow("RMSE (root mean squared error):", QtWidgets.QLabel(f"{rmse:.6e}"))
            stats_layout.addRow("MAE (mean absolute error):", QtWidgets.QLabel(f"{mae:.6e}"))
            stats_layout.addRow("χ² (chi-squared):", QtWidgets.QLabel(f"{chi_squared:.6e}"))
            stats_layout.addRow("Number of data points:", QtWidgets.QLabel(f"{n}"))
        else:
            stats_layout.addRow(QtWidgets.QLabel("No residual data available"))

        layout.addWidget(stats_group)

        # Optimizer information
        opt_group = QtWidgets.QGroupBox("Optimizer Information")
        opt_layout = QtWidgets.QFormLayout(opt_group)

        opt_layout.addRow("Success:", QtWidgets.QLabel(
            "✓ Yes" if self.fit_result.get('success', False) else "✗ No"
        ))
        opt_layout.addRow("Message:", QtWidgets.QLabel(str(self.fit_result.get('message', 'N/A'))))
        opt_layout.addRow("Function evaluations:", QtWidgets.QLabel(str(self.fit_result.get('nfev', 'N/A'))))
        opt_layout.addRow("Final cost:", QtWidgets.QLabel(f"{self.fit_result.get('cost', 0.0):.6e}"))

        layout.addWidget(opt_group)

        # Parameter values
        params_group = QtWidgets.QGroupBox("Fitted Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)

        params_table = QtWidgets.QTableWidget()
        params_table.setColumnCount(2)
        params_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        params_table.horizontalHeader().setStretchLastSection(True)

        parameters = self.fit_result.get('parameters', {})
        params_table.setRowCount(len(parameters))
        for i, (name, value) in enumerate(parameters.items()):
            params_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            params_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{value:.6e}"))

        params_layout.addWidget(params_table)
        layout.addWidget(params_group)

        layout.addStretch()
        self._tabs.addTab(tab, "Summary")

    def _create_parameter_tab(self):
        """Create parameter diagnostics tab (correlation matrix, confidence intervals)."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Try to calculate correlation matrix
        covariance = self.fit_result.get('covariance', None)
        parameters = self.fit_result.get('parameters', {})
        param_names = list(parameters.keys())

        if covariance is not None and len(param_names) > 0:
            # Calculate correlation matrix from covariance
            std_devs = np.sqrt(np.diag(covariance))
            correlation = covariance / np.outer(std_devs, std_devs)

            # Correlation matrix heatmap (using PyQtGraph if available)
            try:
                import pyqtgraph as pg

                corr_label = QtWidgets.QLabel("Parameter Correlation Matrix:")
                corr_label.setStyleSheet("font-weight: bold;")
                layout.addWidget(corr_label)

                # Create image view for heatmap
                image_view = pg.ImageView()
                image_view.setImage(correlation, axes={'x': 1, 'y': 0})
                image_view.setMinimumHeight(300)

                # Set colormap (red-white-blue)
                colors = [
                    (0, 0, 255),    # Blue for -1
                    (255, 255, 255),  # White for 0
                    (255, 0, 0)     # Red for +1
                ]
                cmap = pg.ColorMap(pos=[-1, 0, 1], color=colors)
                image_view.setColorMap(cmap)

                layout.addWidget(image_view)

                # Add parameter labels as legend
                legend_text = "Parameters: " + ", ".join([f"{i}: {name}" for i, name in enumerate(param_names)])
                legend = QtWidgets.QLabel(legend_text)
                legend.setWordWrap(True)
                layout.addWidget(legend)

            except ImportError:
                # Fallback to text display
                corr_text = QtWidgets.QPlainTextEdit()
                corr_text.setReadOnly(True)
                corr_text.setMaximumHeight(200)

                text = "Parameter Correlation Matrix:\n\n"
                text += "        " + " ".join([f"{name:>8s}" for name in param_names]) + "\n"
                for i, name in enumerate(param_names):
                    text += f"{name:>8s} " + " ".join([f"{correlation[i, j]:>8.4f}" for j in range(len(param_names))]) + "\n"

                corr_text.setPlainText(text)
                layout.addWidget(corr_text)

            # Confidence intervals (approximate, 95%)
            ci_group = QtWidgets.QGroupBox("Approximate 95% Confidence Intervals")
            ci_layout = QtWidgets.QFormLayout(ci_group)

            z_score = 1.96  # 95% confidence
            for i, (name, value) in enumerate(parameters.items()):
                stderr = std_devs[i]
                ci_lower = value - z_score * stderr
                ci_upper = value + z_score * stderr
                ci_text = f"{value:.6e} ± {z_score * stderr:.6e}  [{ci_lower:.6e}, {ci_upper:.6e}]"
                ci_layout.addRow(f"{name}:", QtWidgets.QLabel(ci_text))

            layout.addWidget(ci_group)

        else:
            warning = QtWidgets.QLabel(
                "Covariance matrix not available.\n\n"
                "Parameter correlations and confidence intervals cannot be computed.\n"
                "This typically occurs when the Jacobian was not calculated by the optimizer."
            )
            warning.setWordWrap(True)
            warning.setStyleSheet("font-weight: bold; padding: 10px;")
            layout.addWidget(warning)

        layout.addStretch()
        self._tabs.addTab(tab, "Parameters")

    def _create_residual_tab(self):
        """Create residual analysis tab (Q-Q plot, histogram, residuals vs fitted)."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        residuals = self.fit_result.get('residuals', np.array([]))
        predicted = self.fit_result.get('predicted', np.array([]))

        if len(residuals) == 0:
            warning = QtWidgets.QLabel("No residual data available for analysis.")
            warning.setStyleSheet("font-weight: bold; padding: 10px;")
            layout.addWidget(warning)
            self._tabs.addTab(tab, "Residuals")
            return

        # Use PyQtGraph if available
        try:
            import pyqtgraph as pg

            # Residuals vs Fitted plot
            layout.addWidget(QtWidgets.QLabel("<b>Residuals vs Fitted Values</b>"))

            residuals_plot = pg.PlotWidget()
            residuals_plot.setLabel('left', 'Residuals')
            residuals_plot.setLabel('bottom', 'Fitted Values')
            residuals_plot.plot(predicted, residuals, pen=None, symbol='o', symbolSize=5, symbolBrush='b')
            residuals_plot.addLine(y=0, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine))
            residuals_plot.showGrid(x=True, y=True, alpha=0.3)
            residuals_plot.setMinimumHeight(200)
            layout.addWidget(residuals_plot)

            # Histogram of residuals
            layout.addWidget(QtWidgets.QLabel("<b>Residual Histogram</b>"))

            hist_plot = pg.PlotWidget()
            hist_plot.setLabel('left', 'Frequency')
            hist_plot.setLabel('bottom', 'Residuals')

            # Calculate histogram
            y, x = np.histogram(residuals, bins=30)
            hist_plot.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
            hist_plot.showGrid(x=True, y=True, alpha=0.3)
            hist_plot.setMinimumHeight(200)
            layout.addWidget(hist_plot)

            # Q-Q plot (normal probability plot)
            layout.addWidget(QtWidgets.QLabel("<b>Q-Q Plot (Normal)</b>"))

            qq_plot = pg.PlotWidget()
            qq_plot.setLabel('left', 'Sample Quantiles')
            qq_plot.setLabel('bottom', 'Theoretical Quantiles (Normal)')

            # Calculate Q-Q plot points
            sorted_residuals = np.sort(residuals)
            n = len(sorted_residuals)
            theoretical_quantiles = np.array([np.percentile(np.random.normal(0, 1, 10000), 100 * (i + 0.5) / n)
                                             for i in range(n)])

            qq_plot.plot(theoretical_quantiles, sorted_residuals, pen=None, symbol='o', symbolSize=5, symbolBrush='b')

            # Add reference line (y=x)
            min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
            max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
            qq_plot.plot([min_val, max_val], [min_val, max_val], pen=pg.mkPen('r', style=Qt.PenStyle.DashLine))

            qq_plot.showGrid(x=True, y=True, alpha=0.3)
            qq_plot.setMinimumHeight(200)
            layout.addWidget(qq_plot)

        except ImportError:
            # Fallback to text-based statistics
            stats_text = QtWidgets.QPlainTextEdit()
            stats_text.setReadOnly(True)

            text = "Residual Statistics:\n\n"
            text += f"Mean: {np.mean(residuals):.6e}\n"
            text += f"Std Dev: {np.std(residuals):.6e}\n"
            text += f"Min: {np.min(residuals):.6e}\n"
            text += f"Max: {np.max(residuals):.6e}\n"
            text += f"Median: {np.median(residuals):.6e}\n"
            text += "\nQuantiles:\n"
            text += f"  25%: {np.percentile(residuals, 25):.6e}\n"
            text += f"  50%: {np.percentile(residuals, 50):.6e}\n"
            text += f"  75%: {np.percentile(residuals, 75):.6e}\n"

            stats_text.setPlainText(text)
            layout.addWidget(stats_text)

        self._tabs.addTab(tab, "Residuals")

    def _create_convergence_tab(self):
        """Create convergence history tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Check if convergence history is available
        history = self.fit_result.get('history', None)

        if history is None:
            warning = QtWidgets.QLabel(
                "Convergence history not available.\n\n"
                "The optimizer did not record iteration history.\n"
                "Enable history recording in the fitting settings to see convergence plots."
            )
            warning.setWordWrap(True)
            warning.setStyleSheet("font-weight: bold; padding: 10px;")
            layout.addWidget(warning)
        else:
            # Plot convergence history
            try:
                import pyqtgraph as pg

                layout.addWidget(QtWidgets.QLabel("<b>Cost Function vs Iteration</b>"))

                conv_plot = pg.PlotWidget()
                conv_plot.setLabel('left', 'Cost Function')
                conv_plot.setLabel('bottom', 'Iteration')

                iterations = np.arange(len(history))
                conv_plot.plot(iterations, history, pen=pg.mkPen('b', width=2))
                conv_plot.showGrid(x=True, y=True, alpha=0.3)
                conv_plot.setMinimumHeight(300)

                layout.addWidget(conv_plot)

            except ImportError:
                # Fallback to text display
                history_text = QtWidgets.QPlainTextEdit()
                history_text.setReadOnly(True)

                text = "Convergence History:\n\n"
                text += "Iteration    Cost Function\n"
                text += "-" * 40 + "\n"
                for i, cost in enumerate(history):
                    text += f"{i:>9d}    {cost:.6e}\n"

                history_text.setPlainText(text)
                layout.addWidget(history_text)

        layout.addStretch()
        self._tabs.addTab(tab, "Convergence")
