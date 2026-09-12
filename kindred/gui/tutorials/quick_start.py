"""A first run, both copy actions, and a saved project."""

from dataclasses import replace

from PySide6 import QtWidgets

from kindred.gui.tutorials.controller import Step
from kindred.gui.tutorials.overlay import Highlight
from kindred.gui.tutorials.practice_lesson import PracticeLesson


class QuickStart(PracticeLesson):
    title = "Quick Start"
    description = "Run a small model, inspect the curves, copy the results, and save your project."
    _result_steps = frozenset({"read_curves", "copy_plotted"})

    def _time_ready(self) -> bool:
        panel = self.bundle.simulation_panel
        settings = self.tab.resolved_execution_settings(self.preset_set)
        try:
            visible_start = float(panel.t_start_edit.text())
            visible_end = float(panel.t_end_edit.text())
        except ValueError:
            return False
        return settings.t_start == visible_start == 0 and settings.t_end == visible_end == 5

    def _solver_points_ready(self) -> bool:
        panel = self.bundle.simulation_panel
        settings = self.tab.resolved_execution_settings(self.preset_set)
        try:
            visible_points = int(panel.num_points_spinbox.cleanText())
        except ValueError:
            return False
        return (
            panel.num_points_spinbox.hasAcceptableInput()
            and visible_points == settings.integration_points == 100
            and panel.solver_method_combo.currentText() == settings.solver
        )

    def _steps(self) -> tuple[Step, ...]:
        create_tab, load_preset = self._setup_steps(introduce_workspace=True)
        return (
            Step(
                "welcome",
                "Welcome to Kindred",
                "<p>Kindred is an application for modeling <b>chemical reaction kinetics</b>. You can:</p>"
                "<ul><li>Build reaction mechanisms and simulate their behavior.</li>"
                "<li>Compare simulations with experimental data.</li>"
                "<li>Fit kinetic parameters to measurements.</li></ul>"
                "<p>This tutorial introduces the workspace through practical exercises. "
                "The next page outlines what you will learn.</p>",
                lambda: None,
                link=("Kindred on GitHub", "https://github.com/kindredsim/kindred"),
            ),
            Step(
                "learn",
                "What You Will Learn",
                "<p>In Quick Start, you will:</p>"
                "<ul><li>Load a preset mechanism and choose initial conditions.</li>"
                "<li>Set the simulation time and run the model.</li>"
                "<li>Copy selected curves with <b>Copy Plotted Data</b>, then compare the output with <b>Copy All</b>.</li>"
                "<li>Save your work.</li></ul>"
                "<p><b>Advance Automatically</b> is enabled by default. It moves on after you complete an automatically checked action. "
                "Uncheck it to choose Next yourself. Steps that are already complete when you reach them wait for Next.</p>"
                "<p><b>Manual checks</b> ask you to inspect the result and choose Continue.</p>"
                "<ul><li><b>Previous</b> revisits instructions without undoing work.</li>"
                "<li><b>Skip Step</b> moves on without checking completion.</li>"
                "<li><b>−</b> minimizes the tutorial. <b>×</b> closes it.</li></ul>",
                lambda: None,
            ),
            create_tab,
            Step(
                "name_tab",
                "Optional: Rename the Tab",
                "<p>To rename the tab, right-click its name above the plot and choose <b>Rename…</b>. "
                "Enter any name you like and confirm.</p>"
                "<p>You can also keep the current name. Click <b>Next</b> when you are ready to continue.</p>",
                lambda: Highlight(self._dialog(QtWidgets.QInputDialog))
                if self._dialog(QtWidgets.QInputDialog) is not None
                else self._tab_target(),
            ),
            load_preset,
            Step(
                "initial_conditions",
                "Choose Initial Conditions",
                "<p><b>Initial Conditions</b> contains the starting concentrations for each simulation, "
                "and loading M1 supplied the values in <b>preset_set1</b>.</p>"
                "<p>Select only <b>preset_set1</b>. Check that <b>A = 1</b> and <b>B = 0</b>.</p>"
                "<p>Selected rows choose what to run. The <b>Plot</b> checkbox chooses which results to display.</p>"
                "<p>For more details, see <b>Reactions and Initial Conditions</b> in <b>Help → Interactive Tutorials</b>.</p>",
                lambda: self._table_target(),
                lambda: self._values(self.preset_set, 1.0)
                and self.bundle.initial_actions.selected_run_set_ids() == (self.preset_set,),
                "The initial conditions are ready.",
            ),
            Step(
                "time_range",
                "Set the Time Range",
                "<p><i>t</i><sub>start</sub> is the simulation start time. <i>t</i><sub>end</sub> is the end time.</p>"
                "<p>Set <i>t</i><sub>start</sub> to <b>0</b> and <i>t</i><sub>end</sub> to <b>5</b> seconds. "
                "Leave either value unchanged if it is already correct.</p>"
                "<p>Press <b>Enter</b> after each edit.</p>",
                lambda: tuple(
                    self._panel(widget, "batch_initial_conditions")
                    for widget in (self.bundle.simulation_panel.t_start_edit, self.bundle.simulation_panel.t_end_edit)
                ),
                self._time_ready,
                "The simulation time is ready.",
            ),
            Step(
                "solver_points",
                "Solver and Output Points",
                "<p><b>Solver</b> selects the numerical method used to calculate the simulation. "
                "The default in Kindred is <b>BDF</b>. <b>Radau</b> is particularly suited to strongly stiff systems "
                "and can offer better robustness, potentially at higher computational cost. Consider switching to Radau "
                "for mechanisms with fast and slow processes occurring together, such as rapid equilibration alongside "
                "much slower reactions. Very different rate constants can indicate this, but their units, reaction orders, "
                "and the species concentrations also matter. There is no universal rate-constant cutoff for stiffness. "
                "For more details on stiff problems and numerical solvers, see <b>(Reference to be added soon)</b>.</p>"
                "<p><b>Points</b> sets the number of evenly spaced output times, including "
                "<i>t</i><sub>start</sub> and <i>t</i><sub>end</sub>. It does not set the internal integration steps.</p>"
                "<p>Use either solver for this exercise. Set <b>Points</b> to <b>100</b> and press <b>Enter</b>.</p>",
                lambda: tuple(
                    self._panel(widget, "batch_initial_conditions")
                    for widget in (self.bundle.simulation_panel.solver_method_combo, self.bundle.simulation_panel.num_points_spinbox)
                ),
                self._solver_points_ready,
                "The solver and output points are ready.",
            ),
            replace(
                self._first_run_step(),
                instruction="<p>Keep only <b>preset_set1</b> selected, then click <b>Run</b> in either <b>Initial Conditions</b> "
                "or the <b>Mechanism</b> panel. Both buttons run the same selected initial-condition sets.</p>"
                "<p>Wait for the simulation to finish. The results will appear in the main plot.</p>",
            ),
            Step(
                "read_curves",
                "Look at the Plot",
                "<p>Find <b>A</b>, <b>B</b>, and <b>total</b> in the legend. The names and colors match the plotted curves.</p>"
                "<p>The <b>X</b> and <b>Y</b> controls above the plot let you choose what to display and change the display units.</p>"
                "<p>Right-click inside the plot to display an options menu.</p>"
                "<p>If the legend is hidden, choose <b>Appearance → Show Legend</b>. "
                "Choose <b>Auto Range</b> in the main plot menu if you need to restore the view.</p>",
                self._plot_target,
            ),
            Step(
                "copy_plotted",
                "Copy Your Results",
                "<p>Open the <b>Y</b> selector above the plot. Uncheck <b>B</b> and <b>total</b>, "
                "leaving only <b>A</b> selected, then close the selector.</p>"
                "<p>Right-click the plot and choose <b>Copy Plotted Data</b>.</p>"
                "<p><b>Copy Plotted Data</b> copies the curves shown in the plot, using the current display and processing settings. "
                "Curves hidden from the plot are excluded.</p>"
                "<p>The data is copied as <b>tab-separated values (TSV)</b>, including column headers. "
                "Select an empty cell in a spreadsheet and paste. The values will appear in separate rows and columns. "
                "Find the time and A columns. B and total should be absent.</p>"
                "<p>Now leave only A selected, right-click the plot, and choose <b>Copy All</b>. Paste into a second empty sheet.</p>"
                "<p><b>Copy All</b> includes all available results for the current mechanism tab, including hidden results, "
                "with every recorded point in standard units: seconds and M.</p>"
                "<p>Compare the two outputs. B and total appear in Copy All even though they remain hidden from the plot. "
                "Then choose <b>Continue</b>.</p>",
                self._plot_target,
                manual=True,
            ),
            self._save_step(),
            replace(
                self._finish_step(),
                instruction="<p>Your practice tabs stay in the project. You can keep exploring or save your work.</p>"
                "<p>To resume saved work, choose <b>File → Open Project...</b> and open your saved <b>.kin file</b>.</p>"
                "<p>Choose <b>Finish</b> to close this lesson. Open <b>Help → Interactive Tutorials</b> to choose another. "
                "Each lesson includes its own setup.</p>",
            ),
        )
