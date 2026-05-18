"""
Tutorial management system.

Provides built-in tutorials covering key Kindred workflows:
- Getting Started (basic simulation)
- Mechanism Language (DSL syntax)
- Batch Initial Conditions
- Interactive Sliders
- Data Import and Overlay
- Parameter Fitting
- Right-click Copy
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from PySide6 import QtWidgets

# Direct import required to avoid circular dependency with widgets/__init__.py
from kindred.gui.widgets.tutorial_overlay import TutorialOverlay, TutorialStep

logger = logging.getLogger(__name__)

__all__ = ["TutorialManager", "launch_tutorial"]


class TutorialManager:
    """
    Manages tutorial content and progression.

    Features:
    - Built-in tutorial definitions
    - Tutorial selection dialog
    - Progress tracking
    - Tutorial completion statistics
    """

    # Tutorial definitions
    TUTORIALS: Dict[str, Dict] = {
        "getting_started": {
            "title": "Getting Started with Kindred",
            "description": "Create a simple mechanism and run your first simulation.",
            "duration": "3 minutes",
            "steps": [
                TutorialStep(
                    title="Welcome to Kindred",
                    instruction=(
                        "Kindred is a chemical kinetics simulator and fitting tool. "
                        "This tutorial walks you through your first simulation."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="The Mechanism Editor",
                    instruction=(
                        "The <b>Reactions</b> tab is where you write your chemical "
                        "mechanism. Kindred uses a simple text-based syntax."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Enter a mechanism",
                    instruction=(
                        "Type these two reactions into the editor:<br><br>"
                        "<code>A -&gt; B ; k=1.0</code><br>"
                        "<code>B -&gt; C ; k=0.5</code><br><br>"
                        "Each line defines a reaction with its rate constant."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Check for the green tick",
                    instruction=(
                        "Below the editor you will see a validation indicator. "
                        "A green tick means the mechanism parsed successfully."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Set initial concentrations",
                    instruction=(
                        "The <b>Initial Conditions</b> panel holds starting "
                        "concentrations. The default set starts every species at zero. "
                        "Set <b>A</b> to <code>1.0</code> by editing its cell."
                    ),
                    target_widget="batchDock",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Run the simulation",
                    instruction=(
                        "Click <b>Run Selected</b> to solve the ODEs and plot "
                        "concentration vs. time."
                    ),
                    target_widget="runSelectedButton",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="Read the plot",
                    instruction=(
                        "The plot shows species concentrations over time. "
                        "You should see A decaying, B rising then falling, and C "
                        "accumulating. Congratulations on your first simulation!"
                    ),
                    target_widget="plotPanel",
                    arrow_direction="right",
                ),
            ],
        },
        "mechanism_language": {
            "title": "Mechanism Language",
            "description": "Learn the reaction syntax, rate constants, equilibria, and algebra.",
            "duration": "5 minutes",
            "steps": [
                TutorialStep(
                    title="Reaction syntax basics",
                    instruction=(
                        "Reactions use arrows to show direction:<br>"
                        "<code>A -&gt; B ; k=1.0</code> (irreversible)<br>"
                        "<code>A &lt;-&gt; B ; kf=1.0 ; kr=0.5</code> (reversible)<br><br>"
                        "You can also write <code>=&gt;</code> or <code>&lt;=&gt;</code>."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Stoichiometric coefficients",
                    instruction=(
                        "Place coefficients before species names:<br>"
                        "<code>2A + B -&gt; C ; k=0.01</code><br><br>"
                        "An optional <code>*</code> is allowed: <code>2*A</code> "
                        "is the same as <code>2A</code>."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Numeric rate constants",
                    instruction=(
                        "Rate constants on reaction lines must be numeric values:<br>"
                        "<code>k=1.5</code>, <code>kf=1e5</code>, <code>kr=0.002</code><br><br>"
                        "Symbolic relationships between rates are handled separately "
                        "with <code>param</code> declarations in the Reactions text."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Equilibrium lines",
                    instruction=(
                        "Use the <code>equilibrium:</code> keyword for fast equilibria:<br>"
                        "<code>equilibrium: A &lt;-&gt; B ; K=2.5 ; kf=10.0</code><br><br>"
                        "A reversible arrow is required. Provide <code>kf=</code> plus exactly one "
                        "of <code>kr=</code>, <code>K=</code>, or <code>dG_eq=</code>."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Global directives",
                    instruction=(
                        "Set global conditions at the top of your mechanism:<br>"
                        "<code>energy=kJ/mol</code> (or <code>kcal/mol</code>)<br>"
                        "<code>T=310</code> (temperature in Kelvin)<br>"
                        "<code>[A]=1.0</code> (initial concentration)"
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Algebra declarations",
                    instruction=(
                        "Add <code>param</code> and <code>let</code> declarations in the "
                        "Reactions text for derived parameters and observables:<br>"
                        "<code>param scale = 2.0</code> (adjustable parameter)<br>"
                        "<code>param k2 = k1 * scale</code> (derived constraint)<br>"
                        "<code>let total = [A] + [B]</code> (observable)"
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="param vs let",
                    instruction=(
                        "<code>param</code> defines kinetic parameters evaluated "
                        "<b>before</b> the solver runs. It cannot reference species "
                        "concentrations.<br><br>"
                        "<code>let</code> defines observables that can read species data "
                        "like <code>[A]</code> and <code>[A]_0</code>."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Language summary",
                    instruction=(
                        "You now know the core syntax. The <b>Help</b> tab in the "
                        "mechanism editor has a quick-reference with examples."
                    ),
                    arrow_direction="none",
                ),
            ],
        },
        "batch_initial_conditions": {
            "title": "Initial Conditions",
            "description": "Create sets and control starting concentrations.",
            "duration": "3 minutes",
            "steps": [
                TutorialStep(
                    title="What are sets?",
                    instruction=(
                        "Each <b>set</b> is a row of starting concentrations "
                        "for your species. You can define multiple sets to compare "
                        "different conditions in one run."
                    ),
                    target_widget="batchDock",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Add a set",
                    instruction=(
                        "Click <b>Add Set</b> to create a new row. "
                        "Each row starts with all concentrations at zero."
                    ),
                    target_widget="addBatchSetButton",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="Edit concentrations",
                    instruction=(
                        "Click a cell in the table to type a concentration value. "
                        "You can also paste a block of values from a spreadsheet."
                    ),
                    target_widget="batchTable",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Select sets to run",
                    instruction=(
                        "Click rows to select which sets to simulate. "
                        "Hold <b>Ctrl</b> (or <b>Cmd</b>) to select multiple rows."
                    ),
                    target_widget="batchTable",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Run selected sets",
                    instruction=(
                        "Click <b>Run Selected</b> to simulate only the selected "
                        "sets. Unselected sets are not affected."
                    ),
                    target_widget="runSelectedButton",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="Sets summary",
                    instruction=(
                        "Sets let you explore how different starting conditions "
                        "affect your kinetics without changing the mechanism itself."
                    ),
                    arrow_direction="none",
                ),
            ],
        },
        "interactive_sliders": {
            "title": "Interactive Sliders",
            "description": "Adjust parameters with live preview and apply changes.",
            "duration": "4 minutes",
            "steps": [
                TutorialStep(
                    title="Before you start",
                    instruction=(
                        "Sliders require a mechanism with adjustable rate "
                        "constants (for example <code>k=1.0</code>). If the "
                        "slider pane is not visible below the editor, close "
                        "this tutorial, enter a mechanism, and relaunch."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Slider pane appears",
                    instruction=(
                        "After entering a valid mechanism with adjustable "
                        "parameters, the <b>slider pane</b> appears below "
                        "the Reactions editor."
                    ),
                    target_widget="mechanismSliderPane",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Choose visible sliders",
                    instruction=(
                        "Sliders start hidden. Click <b>Visible sliders</b> "
                        "and check the parameters you want to adjust."
                    ),
                    target_widget="sliderVisibilityPickerButton",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="Drag for live preview",
                    instruction=(
                        "Drag any slider to see a <b>live preview</b> of how the "
                        "simulation changes. The plot updates automatically."
                    ),
                    target_widget="unifiedSliderSurface",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Preview vs applied state",
                    instruction=(
                        "Slider adjustments are <b>preview only</b>. They do not "
                        "change the canonical mechanism until you explicitly apply them. "
                        "The preview is shown as an overlay on the plot."
                    ),
                    target_widget="unifiedSliderSurface",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Apply slider values",
                    instruction=(
                        "After changing a slider value, <b>Apply</b> activates. "
                        "Click it to promote the current slider values into the "
                        "canonical mechanism and update the DSL text."
                    ),
                    target_widget="commitSliderOverridesButton",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="Reset sliders",
                    instruction=(
                        "Click <b>Reset</b> to discard your slider adjustments "
                        "and return to the canonical mechanism values."
                    ),
                    target_widget="resetSliderOverridesButton",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="Fine adjustment mode",
                    instruction=(
                        "Toggle <b>Fine</b> mode for more precise slider control "
                        "when you need small adjustments."
                    ),
                    target_widget="mechanismSliderPane",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Sliders summary",
                    instruction=(
                        "Sliders give you instant visual feedback on how parameters "
                        "affect your kinetics. Apply when you find values you like."
                    ),
                    arrow_direction="none",
                ),
            ],
        },
        "data_import": {
            "title": "Data Import and Overlay",
            "description": "Load experimental data and overlay it on simulations.",
            "duration": "3 minutes",
            "steps": [
                TutorialStep(
                    title="Supported file formats",
                    instruction=(
                        "Kindred imports experimental data from <b>CSV</b> and "
                        "<b>Excel (.xlsx)</b> files. Each file should have a time "
                        "column and one or more concentration columns."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Open the import dialog",
                    instruction=(
                        "Click the <b>Load</b> button in the Data panel (or use "
                        "<b>File &gt; Load Data</b>) to open the import dialog."
                    ),
                    target_widget="loadDataButton",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="Configure the import",
                    instruction=(
                        "The import dialog lets you select columns, set units, and "
                        "choose which sheets to import. Each Excel sheet is imported "
                        "independently."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="View imported data",
                    instruction=(
                        "After import, your datasets appear in the <b>Data</b> panel. "
                        "Each dataset is mapped to a set for simulation."
                    ),
                    target_widget="dataPanel",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Overlay on the plot",
                    instruction=(
                        "Imported data is automatically overlaid on the simulation "
                        "plot, making it easy to compare model predictions with "
                        "experimental measurements."
                    ),
                    target_widget="plotPanel",
                    arrow_direction="right",
                ),
                TutorialStep(
                    title="Data import summary",
                    instruction=(
                        "With data loaded, you can visually compare your model "
                        "to experiments. This is the foundation for parameter fitting."
                    ),
                    arrow_direction="none",
                ),
            ],
        },
        "parameter_fitting": {
            "title": "Parameter Fitting",
            "description": "Fit rate constants to experimental data using Global Fit.",
            "duration": "5 minutes",
            "steps": [
                TutorialStep(
                    title="Parameter fitting overview",
                    instruction=(
                        "Kindred can automatically optimise rate constants and other "
                        "parameters to match experimental data. This tutorial assumes "
                        "you already have a mechanism and loaded data."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Prepare your mechanism",
                    instruction=(
                        "Write your mechanism in the <b>Reactions</b> tab with numeric "
                        "rate constants as starting guesses. The fitter will adjust "
                        "these values."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Load experimental data",
                    instruction=(
                        "Make sure your experimental data is loaded in the "
                        "<b>Data</b> panel. Each dataset should be mapped to "
                        "a set."
                    ),
                    target_widget="dataPanel",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Open Global Fit",
                    instruction=(
                        "Go to <b>Fitting &gt; Global Fit</b> (or press "
                        "<b>Ctrl+Shift+F</b>) to open the fitting window."
                    ),
                    target_widget="globalFitAction",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="Select fit targets",
                    instruction=(
                        "In the fitting window, choose which species to fit "
                        "and set parameter bounds (min, max, initial guess). "
                        "Realistic bounds improve convergence."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Run the fit",
                    instruction=(
                        "Click <b>Run Fit</b> to start the optimiser. "
                        "Progress is shown live with parameter values and "
                        "residual plots updating in real time."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Review results",
                    instruction=(
                        "When the fit completes, review optimised parameter values, "
                        "uncertainties, and goodness-of-fit statistics inside the "
                        "fitting window."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Apply to project",
                    instruction=(
                        "Use <b>Apply to Project</b> to promote fitted values back "
                        "into your mechanism. You can apply parameters only, initial "
                        "conditions only, or both."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Fitting summary",
                    instruction=(
                        "Tips for good fits: use physically reasonable bounds, "
                        "check residual plots for systematic errors, and try "
                        "different initial guesses if the fit does not converge."
                    ),
                    arrow_direction="none",
                ),
            ],
        },
        "right_click_copy": {
            "title": "Right-click Copy",
            "description": "Copy plot data to the clipboard for use in spreadsheets.",
            "duration": "1 minute",
            "steps": [
                TutorialStep(
                    title="Copy data from the plot",
                    instruction=(
                        "You can copy simulation results directly from the plot "
                        "without exporting to a file."
                    ),
                    target_widget="plotPanel",
                    arrow_direction="right",
                ),
                TutorialStep(
                    title="Right-click the plot",
                    instruction=(
                        "Right-click anywhere on the <b>plot area</b> to open "
                        "the context menu with copy options."
                    ),
                    target_widget="plotViewport",
                    arrow_direction="right",
                ),
                TutorialStep(
                    title="Paste into a spreadsheet",
                    instruction=(
                        "After copying, paste into Excel, Google Sheets, or any "
                        "text editor. The data is tab-separated for easy import."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Copy summary",
                    instruction=(
                        "Right-click copy is the fastest way to get simulation "
                        "data into a report or spreadsheet."
                    ),
                    arrow_direction="none",
                ),
            ],
        },
    }

    @staticmethod
    def get_tutorial_list() -> List[Dict]:
        """Get list of available tutorials with metadata."""
        return [
            {
                "id": tutorial_id,
                "title": tutorial["title"],
                "description": tutorial["description"],
                "duration": tutorial["duration"],
            }
            for tutorial_id, tutorial in TutorialManager.TUTORIALS.items()
        ]

    @staticmethod
    def get_tutorial_steps(tutorial_id: str) -> Optional[List[TutorialStep]]:
        """Get steps for a tutorial."""
        tutorial = TutorialManager.TUTORIALS.get(tutorial_id)
        if tutorial:
            return tutorial["steps"]
        return None


def launch_tutorial(parent: QtWidgets.QWidget, tutorial_id: str):
    """
    Launch a tutorial overlay.

    Parameters
    ----------
    parent : QWidget
        Parent widget (usually main window)
    tutorial_id : str
        Tutorial ID (e.g., 'getting_started')
    """
    steps = TutorialManager.get_tutorial_steps(tutorial_id)
    if not steps:
        logger.error(f"Tutorial not found: {tutorial_id}")
        QtWidgets.QMessageBox.warning(
            parent,
            "Tutorial Not Found",
            f"The tutorial '{tutorial_id}' could not be found."
        )
        return

    overlay = TutorialOverlay(parent, steps)
    overlay.tutorialCompleted.connect(lambda: logger.info(f"Tutorial completed: {tutorial_id}"))
    overlay.tutorialSkipped.connect(lambda: logger.info(f"Tutorial skipped: {tutorial_id}"))
    overlay.show()
    overlay.raise_()

    logger.info(f"Tutorial launched: {tutorial_id}")
    return overlay
