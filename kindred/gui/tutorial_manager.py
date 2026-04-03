"""
Tutorial management system.

Provides built-in tutorials covering key Kindred workflows:
- Getting Started (basic simulation)
- Parameter Fitting (experimental data fitting)
- Temperature Schedules (time-varying kinetics)
- State Networks (transition state theory)
- Advanced Features (equilibria, algebra)
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
            "description": "Learn how to create a simple reaction mechanism and run your first simulation.",
            "duration": "5 minutes",
            "steps": [
                TutorialStep(
                    title="Welcome to Kindred!",
                    instruction=(
                        "Kindred is a chemical kinetics simulation and fitting tool. "
                        "This tutorial will guide you through creating your first simulation.<br><br>"
                        "<b>What you'll learn:</b><ul>"
                        "<li>Writing reaction mechanisms in DSL</li>"
                        "<li>Setting initial conditions</li>"
                        "<li>Running simulations</li>"
                        "<li>Viewing results</li></ul>"
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="The Mechanism Editor",
                    instruction=(
                        "The <b>Reactions tab</b> is where you define your chemical mechanism using Kindred's DSL syntax.<br><br>"
                        "DSL stands for Domain-Specific Language - it's a simple, readable format for writing reactions."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Write Your First Reaction",
                    instruction=(
                        "Let's write a simple reaction: <code>A → B</code><br><br>"
                        "In the Reactions tab, type:<br>"
                        "<code>reaction: A -> B; k=0.5</code><br><br>"
                        "This defines a first-order reaction with rate constant k=0.5 s⁻¹."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Set Initial Conditions",
                    instruction=(
                        "Now add initial concentrations below your reaction:<br>"
                        "<code>initial: A=1.0</code><br>"
                        "<code>initial: B=0.0</code><br><br>"
                        "This sets [A]₀ = 1.0 M and [B]₀ = 0.0 M."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Run the Simulation",
                    instruction=(
                        "Click the <b>Run Selected</b> button (or press <b>Ctrl+R</b>) to simulate your mechanism.<br><br>"
                        "Kindred will solve the differential equations and plot concentration vs time."
                    ),
                    target_widget="runSelectedButton",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="View Results",
                    instruction=(
                        "The <b>Plot</b> panel shows your simulation results.<br><br>"
                        "You should see [A] decreasing exponentially while [B] increases to 1.0 M.<br><br>"
                        "You can export results via <b>File → Export CSV...</b>."
                    ),
                    target_widget="plotPanel",
                    arrow_direction="right",
                ),
                TutorialStep(
                    title="Congratulations!",
                    instruction=(
                        "You've completed your first Kindred simulation! 🎉<br><br>"
                        "<b>Next steps:</b><ul>"
                        "<li>Try more complex reactions (bimolecular, reversible)</li>"
                        "<li>Explore equilibria and thermodynamics</li>"
                        "<li>Learn about parameter fitting</li></ul><br>"
                        "Check out other tutorials from <b>Help → Interactive Tutorials...</b>."
                    ),
                    arrow_direction="none",
                ),
            ],
        },
        "parameter_fitting": {
            "title": "Parameter Fitting Workflow",
            "description": "Learn how to fit rate constants to experimental data.",
            "duration": "10 minutes",
            "steps": [
                TutorialStep(
                    title="Parameter Fitting in Kindred",
                    instruction=(
                        "Kindred can automatically fit rate constants and other parameters to experimental data.<br><br>"
                        "<b>What you'll learn:</b><ul>"
                        "<li>Loading experimental data</li>"
                        "<li>Opening the fitting window</li>"
                        "<li>Configuring bounds</li>"
                        "<li>Running optimization</li>"
                        "<li>Viewing diagnostics</li></ul>"
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Load Experimental Data",
                    instruction=(
                        "First, load your experimental data from a CSV file.<br><br>"
                        "Go to <b>File → Load Data...</b> and select a CSV with time and concentration columns."
                    ),
                    target_widget="loadDataAction",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="The Data Panel",
                    instruction=(
                        "The <b>Data</b> panel shows your loaded datasets.<br><br>"
                        "You can preview the data and confirm the dataset you want to fit."
                    ),
                    target_widget="dataPanel",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Define Your Mechanism",
                    instruction=(
                        "Write the mechanism you want to fit in the <b>Reactions</b> tab.<br><br>"
                        "Use parameter names (like <code>k1</code>, <code>k2</code>) instead of numeric values:<br>"
                        "<code>reaction: A -> B; k=k1</code>"
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Open the Fitting Window",
                    instruction=(
                        "Click <b>Fitting → Global Fit...</b> (or <b>Ctrl+Shift+F</b>) to open the fitting window.<br><br>"
                        "Kindred will automatically detect parameter names from your mechanism."
                    ),
                    target_widget="globalFitAction",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="Configure Bounds",
                    instruction=(
                        "In the fitting window's <b>Parameters</b> tab, set realistic bounds for each parameter:<br><br>"
                        "• <b>Value</b>: Starting guess<br>"
                        "• <b>Min/Max</b>: Search bounds<br><br>"
                        "Good bounds improve convergence and prevent unphysical values."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Run Fit",
                    instruction=(
                        "Click <b>Run Fit</b> inside the fitting window to start optimization.<br><br>"
                        "Kindred uses scipy's least-squares optimizer to minimize residuals."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Review Results",
                    instruction=(
                        "Review the fit inside the fitting window itself.<br>"
                        "• Optimized parameter values<br>"
                        "• Uncertainties (standard errors)<br>"
                        "• Goodness-of-fit statistics (R², χ²)<br><br>"
                        "Nothing updates the project until you use <b>Apply to Project</b>."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Apply to Project",
                    instruction=(
                        "Use the fitting window's <b>Apply to Project</b> control when you are ready to promote results.<br><br>"
                        "Choose whether to apply:<br>"
                        "• Parameters only<br>"
                        "• Initial conditions only<br>"
                        "• Parameters and initial conditions"
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Fitting Complete!",
                    instruction=(
                        "You now know how to fit kinetic models to data! 🎉<br><br>"
                        "<b>Tips:</b><ul>"
                        "<li>Use physically reasonable bounds</li>"
                        "<li>Check residual plots for systematic errors</li>"
                        "<li>Try different initial guesses if fit fails</li>"
                        "</ul>"
                    ),
                    arrow_direction="none",
                ),
            ],
        },
        "temperature_schedules": {
            "title": "Temperature Schedules",
            "description": "Create time-varying temperature profiles for non-isothermal kinetics.",
            "duration": "7 minutes",
            "steps": [
                TutorialStep(
                    title="Time-Varying Temperature",
                    instruction=(
                        "Kindred supports piecewise constant temperature schedules for simulating:<br>"
                        "• Temperature ramps<br>"
                        "• Step changes<br>"
                        "• Cyclic heating/cooling<br>"
                        "• Arbitrary temperature profiles"
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Temperature Schedule Editor",
                    instruction=(
                        "Open <b>Tools → Temperature Schedule...</b> to launch the visual editor.<br><br>"
                        "The editor provides:<br>"
                        "• Table-based interval editing<br>"
                        "• Live preview plot<br>"
                        "• Template presets<br>"
                        "• DSL export"
                    ),
                    target_widget="temperatureScheduleAction",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="Create a Schedule",
                    instruction=(
                        "Try a template like <b>Linear Ramp (25°C → 100°C)</b>.<br><br>"
                        "You can customize intervals by editing the table:<br>"
                        "• Start/End Time (seconds)<br>"
                        "• Temperature (Kelvin)"
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Preview and Export",
                    instruction=(
                        "The preview plot shows your temperature profile.<br><br>"
                        "Click <b>OK</b> to export DSL syntax like:<br>"
                        "<code>temp_step: t=[0,25,50,...], T=[298.15,316.9,...]</code><br><br>"
                        "This gets inserted into your mechanism."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Temperature-Dependent Rates",
                    instruction=(
                        "Reactions with Eyring or Arrhenius parameters automatically use the temperature schedule:<br><br>"
                        "<code>reaction: A -> B; dG_act=75.0</code><br><br>"
                        "Rate constants will vary with temperature during simulation."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Temperature Schedules Mastered!",
                    instruction=(
                        "You can now simulate non-isothermal kinetics! 🔥<br><br>"
                        "<b>Applications:</b><ul>"
                        "<li>Temperature-programmed reactions</li>"
                        "<li>Thermal analysis (DSC, TGA)</li>"
                        "<li>Reaction calorimetry</li>"
                        "<li>Process optimization</li></ul>"
                    ),
                    arrow_direction="none",
                ),
            ],
        },
        "state_networks": {
            "title": "State Networks (Transition State Theory)",
            "description": "Model reactions using energy landscapes and transition states.",
            "duration": "8 minutes",
            "steps": [
                TutorialStep(
                    title="Transition State Theory in Kindred",
                    instruction=(
                        "State networks provide an alternative way to define mechanisms based on energy landscapes.<br><br>"
                        "<b>Concepts:</b><ul>"
                        "<li>States (energy minima)</li>"
                        "<li>Transition states (energy barriers)</li>"
                        "<li>Automatic rate calculation via Eyring equation</li></ul>"
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Access State Network Editor",
                    instruction=(
                        "Go to <b>Edit → State Network Editor...</b> to open the visual editor.<br><br>"
                        "State networks are defined using tables for states and edges. Changes are "
                        "automatically included in the mechanism when you run a simulation."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Define States",
                    instruction=(
                        "Click <b>Add State</b> to insert a row in the States table. "
                        "Each state has columns:<br>"
                        "• <b>Name</b> (e.g., 'A', 'B')<br>"
                        "• <b>Type</b> (GS for ground state, TS for transition state)<br>"
                        "• <b>Energy</b> (relative free energy)<br>"
                        "• <b>Unit</b> (kJ/mol, kcal/mol, or J/mol)<br>"
                        "• <b>Degeneracy</b> (statistical weight)"
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Add Edges",
                    instruction=(
                        "Click <b>Add Edge</b> to connect states in the Edges table. "
                        "Each edge has columns:<br>"
                        "• <b>State A</b>: source state name<br>"
                        "• <b>State B</b>: target state name<br><br>"
                        "Kindred automatically calculates forward and reverse rates using Eyring theory."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Accept and Run",
                    instruction=(
                        "Click <b>OK</b> to accept your state network changes.<br><br>"
                        "The state network is automatically included when running simulations. "
                        "Kindred generates Eyring-based rate constants from the energy landscape "
                        "you defined."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="State Networks Complete!",
                    instruction=(
                        "You can now use transition state theory in Kindred! ⚛️<br><br>"
                        "<b>When to use state networks:</b><ul>"
                        "<li>Reactions with known barriers</li>"
                        "<li>Computational chemistry data</li>"
                        "<li>Complex equilibrium networks</li></ul>"
                    ),
                    arrow_direction="none",
                ),
            ],
        },
        "advanced_features": {
            "title": "Advanced Features",
            "description": "Explore equilibria, algebraic expressions, and multiple datasets.",
            "duration": "10 minutes",
            "steps": [
                TutorialStep(
                    title="Beyond Basic Reactions",
                    instruction=(
                        "Kindred supports advanced modeling features:<br><br>"
                        "• <b>Equilibria</b>: Fast reversible reactions<br>"
                        "• <b>Algebra</b>: Custom expressions in <code># Algebra</code> blocks<br>"
                        "• <b>Multiple datasets</b>: Comparison and fitting"
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Equilibria",
                    instruction=(
                        "Define fast equilibria using equilibrium constants:<br>"
                        "<code>equilibrium: A <-> B; K=2.5</code> (or <code>A <=> B</code>)<br><br>"
                        "Or thermodynamic parameters:<br>"
                        "<code>equilibrium: A <=> B; dG0=-2.3; T=298.15</code><br><br>"
                        "Kindred handles equilibria efficiently in the ODE system."
                    ),
                    target_widget="mechanismEditor",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Algebraic Expressions",
                    instruction=(
                        "Define algebraic expressions inside the Reactions editor using a <code># Algebra</code> section:<br>"
                        "• Derived series: <code>let Total = [A] + [B]</code><br>"
                        "• Scalar parameters: <code>param scale = 1.0</code><br>"
                        "• Fluxes: <code>let flux = k1 * [A]</code><br><br>"
                        "These are evaluated dynamically during simulation."
                    ),
                    arrow_direction="none",
                ),
                TutorialStep(
                    title="Multiple Datasets",
                    instruction=(
                        "Load multiple datasets to compare experiments:<br><br>"
                        "• Different conditions<br>"
                        "• Replicate measurements<br>"
                        "• Different species<br><br>"
                        "Use the grid view for side-by-side comparison."
                    ),
                    target_widget="dataPanel",
                    arrow_direction="left",
                ),
                TutorialStep(
                    title="Simulation Settings",
                    instruction=(
                        "Fine-tune simulations in <b>Simulation → Simulation Settings...</b>:<br><br>"
                        "• Solver method (LSODA, Radau, BDF)<br>"
                        "• Tolerances (rtol, atol)<br>"
                        "• Cache management<br>"
                        "• Worker parallelism"
                    ),
                    target_widget="simulationSettingsAction",
                    arrow_direction="top",
                ),
                TutorialStep(
                    title="All Features Unlocked!",
                    instruction=(
                        "You're now a Kindred power user! 🚀<br><br>"
                        "<b>Resources:</b><ul>"
                        "<li><b>F1</b>: Documentation</li>"
                        "<li><b>Ctrl+?</b>: Keyboard shortcuts</li>"
                        "<li><b>Help → About</b>: Version and license</li></ul><br>"
                        "Happy modeling!"
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
