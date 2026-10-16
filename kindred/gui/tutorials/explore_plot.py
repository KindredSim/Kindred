"""Plot selection, navigation, and appearance."""

from kindred.core.project_presentation_state import SimulationSeriesAxisSelection
from kindred.gui.tutorials.controller import Step
from kindred.gui.tutorials.practice_lesson import PracticeLesson


class ExplorePlot(PracticeLesson):
    title = "Explore the Plot"
    description = (
        "Choose what to plot, adjust the axes, and customize the plot appearance."
    )
    _result_steps = frozenset(
        {
            "plot_quantities",
            "invert_selection",
            "plot_observable",
            "plot_x",
            "plot_time",
            "plot_units",
            "plot_navigation",
            "plot_ranges",
            "plot_mouse",
            "plot_grid",
            "plot_style",
            "plot_log_axes",
            "plot_invert_axes",
        }
    )

    def _steps(self) -> tuple[Step, ...]:
        plot = self._plot_target
        return (
            *self._setup_steps(),
            self._prepare_run_step(),
            self._first_run_step(),
            Step(
                "plot_quantities",
                "Choose the Quantities",
                "<p>Open the <b>Y</b> selector above the plot. It lets you choose several quantities.</p>"
                "<p>A label such as <b>A +2</b> means A plus two other selected quantities. It is a selection summary.</p>"
                "<p>Select only <b>A</b>. Uncheck B and total, then close the selector.</p>",
                plot,
                lambda: self.registry.presentation(self.tab_id).selected_y_series
                == (SimulationSeriesAxisSelection("A"),),
                "Only A is selected for Y.",
            ),
            Step(
                "invert_selection",
                "Invert the Y Selection",
                "<p>With only A selected, open Y and click <b>Invert Selection</b>. B and total are now selected.</p>"
                "<p>Click it again to return to A alone. This swaps checked and unchecked quantities. "
                "It does not reverse the axis direction.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_observable",
                "Plot the Observable",
                "<p>Open Y and select <b>A</b>, <b>B</b>, and <b>total</b>. Close the selector.</p>"
                "<p>The user-defined observable appears alongside the species. Use the legend to identify it.</p>",
                plot,
                lambda: set(self.registry.presentation(self.tab_id).selected_y_series)
                == {
                    SimulationSeriesAxisSelection(name) for name in ("A", "B", "total")
                },
                "A, B, and total are selected.",
            ),
            Step(
                "plot_x",
                "Choose the X Quantity",
                "<p>The <b>X</b> selector can use time, a species concentration, or an observable.</p>"
                "<p>Change X from <b>t</b> to <b>B</b>. The selected Y quantities are now plotted against B. "
                "Since B is also selected for Y, the plot includes B against B.</p>",
                plot,
                lambda: self.registry.presentation(self.tab_id).selected_x_series
                == SimulationSeriesAxisSelection("B"),
                "B is selected for X.",
            ),
            Step(
                "plot_time",
                "Return to a Time Plot",
                "<p>Set X back to <b>t</b>. Changing axes does not require another simulation.</p>",
                plot,
                lambda: self.registry.presentation(self.tab_id).selected_x_series
                == SimulationSeriesAxisSelection("t"),
                "Time is selected for X.",
            ),
            Step(
                "plot_units",
                "Change the Display Units",
                "<p>Keep X at t and select only A for Y. Change the X <b>Unit</b> from <b>s</b> to <b>min</b>. "
                "Change the Y Unit from <b>M</b> to <b>mM</b>. Read the new axis labels.</p>"
                "<p>The results are the same. Only the units used to display and copy the selected curves change. "
                "Unit choices depend on the selected quantities.</p>"
                "<p>Restore <b>s</b> and <b>M</b>.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_navigation",
                "Zoom and Pan",
                "<ol><li>Right-click the plot and enable <b>Zoom</b>. Hold the left mouse button and drag a rectangle around part of the data.</li>"
                "<li>Turn Zoom off, then drag with the left mouse button to move the view.</li>"
                "<li>Hold the right mouse button and drag to adjust the axis scales. A right-click without dragging opens the menu.</li></ol>"
                "<p>Scrolling also zooms. Finish with Zoom off and <b>Auto Range</b> enabled in the plot menu.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_ranges",
                "Fit Once or Keep Adjusting",
                "<p>Choose <b>Fit Once</b>. It fits the displayed data and leaves both axes manual. "
                "You can now enter limits in the row above the plot.</p>"
                "<p>Choose <b>Axes → Y Axis → Auto Range</b>. Y becomes automatic while X stays manual. "
                "The labels beside the limits show which mode each axis uses.</p>"
                "<p>Finish by enabling <b>Auto Range</b> in the plot menu. Both axes will continue to adjust as data changes.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_mouse",
                "Lock One Direction",
                "<p>Turn off <b>Axes → X Axis → Mouse Navigation</b>. Drag the plot and scroll. "
                "Mouse movement changes Y while X stays fixed.</p>"
                "<p>This locks mouse control, not automatic adjustments. The Y axis has the same option.</p>"
                "<p>Restore <b>Mouse Navigation</b> on both axes and enable the main <b>Auto Range</b> action.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_grid",
                "Adjust the Grid",
                "<p>Open <b>Appearance</b> in the plot menu. <b>Show Grid</b> switches both directions together. "
                "<b>X Grid</b> shows vertical lines and <b>Y Grid</b> shows horizontal lines.</p>"
                "<p>Turn off X Grid and notice that Show Grid is no longer checked. Enable Show Grid to restore both.</p>"
                "<p>Try <b>Grid Opacity</b>. Lower values make the lines fainter. Restore both grid directions "
                "and set the opacity to <b>25%</b>.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_style",
                "Show Points and Adjust Visibility",
                "<p>Enable <b>Appearance → Simulation Sample Markers</b>. The points mark the samples supplied to the curves. "
                "Downsampling changes which points are shown. Experimental markers have separate settings.</p>"
                "<p>Lower <b>Simulation Opacity</b>, then restore <b>100%</b>. Reference and Preview curves keep their usual "
                "relative transparency when you adjust this setting.</p>"
                "<p>Try <b>Appearance → Show Legend</b> off and on. Finish with the legend visible and simulation markers off.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_log_axes",
                "Try Log Scales",
                "<p>Enable <b>Axes → X Axis → Log Scale</b>. Only positive values can appear, so the starting time of zero "
                "is not shown. Turn it off and try Log Scale under <b>Y Axis</b>.</p>"
                "<p>Manual log limits must be positive. Enabling a log scale with incompatible manual limits fits that axis "
                "again where data is available and leaves it manual.</p>"
                "<p>Turn Log Scale off on both axes when finished.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_invert_axes",
                "Reverse Axis Directions",
                "<p>Choose <b>Axes → X Axis → Invert Axis</b> to reverse the displayed direction. "
                "Turn it off, then try Invert Axis under Y Axis.</p>"
                "<p>This changes the view without changing the simulation results. "
                "Finish with both inversion options off and Auto Range enabled in the plot menu.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_annotations",
                "Other Plot Options",
                "<p>The plot menu also includes <b>Transform</b> options and controls for intervention indicators "
                "and labels under <b>Appearance</b>. M1 has no interventions, so those indicators and labels "
                "do not appear in this example.</p>",
                plot,
            ),
            Step(
                "finish",
                "Lesson Complete",
                "<p>You have explored how to choose what appears on each axis, change display units, zoom and pan, "
                "adjust axis ranges, and customize the plot appearance.</p>"
                "<p>These controls let you examine the same simulation results in different ways without running the model again.</p>"
                "<p>Choose <b>Finish</b> to close this lesson.</p>",
                lambda: None,
            ),
        )
