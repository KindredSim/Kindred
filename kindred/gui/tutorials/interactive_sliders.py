"""Try two parameter changes, compare previews, and apply a settled result."""

from dataclasses import replace
from math import isclose

from kindred.core.project_presentation_state import SimulationSeriesAxisSelection
from kindred.gui.tutorials.controller import Step
from kindred.gui.tutorials.practice_lesson import PracticeLesson


class InteractiveSliders(PracticeLesson):
    title = "Interactive Sliders"
    description = "Try two reaction rates, compare Preview with Reference, and decide which changes to keep."
    _preset_name = "M5"
    _practice_species = ("A", "B", "C")
    _preset_rates = {"kf1": 1.5, "kr1": 0.25, "k2": 0.1}
    _extra_steps = frozenset(
        {"rename_extra", "configure_extra", "run_new_set", "slider_targets", "preview_both", "understand_apply", "apply_preview"}
    )
    _result_steps = frozenset(
        {
            "show_slider", "preview_target", "preview_first", "preview_reference",
            "preview_second", "reset_trial", "drag_slider", "fine_slider",
            "slider_targets", "preview_both", "understand_apply", "apply_preview",
        }
    )

    def blocked(self):
        missing = super().blocked()
        if missing is not None or self.tab_id is None:
            return missing
        name = self.step.name
        if name in {"preview_first", "preview_second", "drag_slider", "fine_slider", "preview_both"}:
            sliders = self.bundle.editor.variable_sliders_widget()
            if not all(sliders.variable_visible(key) for key in ("kf1", "k2")):
                return self._missing("Show both rate sliders before adjusting them.", "show_slider")
        if name in {"preview_first", "preview_reference", "preview_second", "drag_slider", "fine_slider"}:
            if self.bundle.slider_binding.resolve_target_set_ids() != (self.preset_set,):
                return self._missing("Choose preset_set1 as the only slider target.", "preview_target")
        if name in {"slider_targets", "preview_both", "understand_apply", "apply_preview"}:
            repo = self.registry.tab_runtime(self.tab_id).result_repository
            if repo.current_result(self.extra_set) is None:
                return self._missing("Run Higher A before previewing both sets.", "run_new_set")
        if name in {"preview_both", "understand_apply", "apply_preview"}:
            if set(self.bundle.slider_binding.resolve_target_set_ids()) != {self.preset_set, self.extra_set}:
                return self._missing("Choose both practice sets as slider targets.", "slider_targets")
        if name == "preview_reference" and not self._preview_ready(0.75, 0.1, (self.preset_set,)):
            return self._missing("Finish the first rate trial before comparing it with Reference.", "preview_first")
        if name == "understand_apply" and not self._preview_ready(1.125, 0.05, (self.preset_set, self.extra_set)):
            return self._missing("Finish both rate changes before reviewing Apply.", "preview_both")
        return None

    def _preview_ready(self, first: float, second: float, identities: tuple) -> bool:
        if self.jobs.busy:
            return False
        repo = self.registry.tab_runtime(self.tab_id).result_repository
        for identity in identities:
            result = repo.current_result(identity)
            if identity not in repo.preview_set_ids() or result is None:
                return False
            parameters = {name: value for name, value, _unit in result.parameters}
            if any(
                key not in parameters or not isclose(parameters[key], value, rel_tol=1e-8)
                for key, value in (("kf1", first), ("kr1", 0.25), ("k2", second))
            ):
                return False
        return True

    def _slider_summary_target(self):
        controls = self.bundle.slider_panel.layout().itemAt(0).layout()
        return self._panel(controls.itemAtPosition(1, 0).widget(), "interactive_sliders")

    def _steps(self) -> tuple[Step, ...]:
        create_tab, load_preset = self._setup_steps()
        return (
            create_tab,
            replace(
                load_preset,
                instruction="<p>Open <b>Examples → Preset Mechanisms → M5: A &lt;-&gt; B, B -&gt; C</b>.</p>"
                "<p>M5 contains <b>A ⇌ B</b> with <b>kf1 = 1.5</b> and <b>kr1 = 0.25</b>, "
                "and <b>B → C</b> with <b>k2 = 0.1</b>. We will change kf1 and k2 with sliders "
                "and keep kr1 at 0.25 throughout this lesson.</p>"
                "<p>The preset adds <b>preset_set1</b> with <b>A = 1</b>, <b>B = 0</b>, and <b>C = 0</b>. "
                "Use that set for the first run. The original set1 row can stay.</p>",
                success="M5 and the starting values are ready.",
            ),
            replace(
                self._prepare_run_step(),
                instruction="<p>In <b>Initial Conditions</b>, select only <b>preset_set1</b>. "
                "Check <b>A = 1</b>, <b>B = 0</b>, and <b>C = 0</b>.</p>"
                "<p>Set <i>t</i><sub>start</sub> to <b>0</b>, <i>t</i><sub>end</sub> to <b>5</b> seconds, "
                "and <b>Points</b> to <b>100</b>. Press <b>Enter</b> after each edit. Either solver works for this exercise.</p>",
            ),
            Step(
                "run_first", "Run M5",
                "<p>Keep the supplied rates: <b>kf1 = 1.5</b>, <b>kr1 = 0.25</b>, and <b>k2 = 0.1</b>.</p>"
                "<p>Select only <b>preset_set1</b>, then click <b>Run</b> in either <b>Initial Conditions</b> "
                "or the <b>Mechanism</b> panel. Wait for the curves.</p>",
                self._run_targets,
                lambda: self._values(self.preset_set, 1.0)
                and self._run_ok({"kf1": 1.5, "kr1": 0.25, "k2": 0.1}, (self.preset_set,)),
                "The supplied rates have been used in the completed run.",
            ),
            Step(
                "show_slider", "Show Both Rate Sliders",
                "<p>If the <b>Interactive Sliders panel</b> is hidden, open "
                "<b>View → Panels → Interactive Sliders</b>. You can also click <b>Show this control</b> below.</p>"
                "<p>In the Interactive Sliders panel, open <b>Visible sliders</b>. "
                "Check <b>kf1 — A → B</b> and <b>k2 — B → C</b>, then close the menu.</p>"
                "<p><b>kf1</b> controls the forward rate of A ⇌ B and <b>k2</b> controls B → C. "
                "You can drag either slider or type a value. Leave <b>kr1 = 0.25</b>.</p>",
                self._slider_target,
                lambda: all(self.bundle.editor.variable_sliders_widget().variable_visible(key) for key in ("kf1", "k2")),
            ),
            Step(
                "preview_target", "Choose the Set to Preview",
                "<p>In the <b>Initial Conditions panel</b>, find the <b>Slider</b> column. "
                "Checking a box pins that set as a slider target, so slider changes continue to affect it when you click another row.</p>"
                "<p>For now, leave all Slider checkboxes unchecked and click <b>preset_set1</b>. "
                "The current row is a slider target even when its checkbox is unchecked.</p>"
                "<p>In the <b>Interactive Sliders panel</b>, find preset_set1 beside <b>Initial-condition sets</b>.</p>",
                lambda: (
                    *self._table_column_targets(self.bundle.batch_model.slider_column()),
                    self._table_target(self.preset_set),
                    self._slider_summary_target(),
                ),
                lambda: not self.registry.presentation(self.tab_id).pinned_slider_set_ids
                and self.bundle.slider_binding.resolve_target_set_ids() == (self.preset_set,),
            ),
            Step(
                "preview_first", "Try a Different First Rate",
                "<p>Enter <b>0.75</b> in the <b>kf1</b> value field. Keep <b>k2</b> at <b>0.1</b>. "
                "Press <b>Enter</b> after an edit and wait for the Preview to finish.</p>"
                "<p>The <b>accepted model</b> is the version currently kept in this mechanism tab. "
                "Slider changes create a trial Preview. They become part of the accepted model when you click <b>Apply</b>.</p>"
                "<p>The proposed forward rate is now 0.75. The accepted forward rate is still 1.5.</p>",
                self._slider_target,
                lambda: self._preview_ready(0.75, 0.1, (self.preset_set,)),
                "The Preview uses kf1 = 0.75, kr1 = 0.25, and k2 = 0.1.",
            ),
            Step(
                "preview_reference", "Compare Preview with Reference",
                "<p><b>Preview</b> shows the proposed change. <b>Reference</b> shows the accepted model, "
                "with <b>kf1 = 1.5</b> and <b>k2 = 0.1</b>.</p>"
                "<p>Compare the curves. If Reference is hidden, right-click the plot and enable "
                "<b>Appearance → Show Reference</b>. Use the legend to tell them apart.</p>",
                self._plot_target, manual=True,
            ),
            Step(
                "preview_second", "Change the Second Rate Too",
                "<p>Keep <b>kf1 = 0.75</b> and enter <b>0.2</b> in the <b>k2</b> value field. "
                "Press <b>Enter</b> and wait for the Preview.</p>"
                "<p>The Preview now includes both proposed changes. Changing the second slider keeps the trial value of the first. "
                "Reference still uses the original pair, kf1 = 1.5 and k2 = 0.1.</p>",
                self._slider_target,
                lambda: self._preview_ready(0.75, 0.2, (self.preset_set,)),
                "The Preview includes both changed rates.",
            ),
            Step(
                "reset_trial", "Discard Both Changes",
                "<p>Click <b>Reset</b> in the <b>Interactive Sliders panel</b>. Both sliders return to the accepted values: "
                "<b>kf1 = 1.5</b> and <b>k2 = 0.1</b>.</p>"
                "<p>Reset discards the trial without changing the accepted model.</p>",
                self._slider_target, manual=True,
            ),
            Step(
                "drag_slider", "Drag to Explore",
                "<p>Drag the <b>kf1</b> handle a short distance with the left mouse button, then try <b>k2</b>. "
                "Watch the Preview change.</p>"
                "<p>Release each handle and wait for the completed Preview. There is no exact value to reach yet.</p>",
                lambda: (self._slider_target(), self._plot_target()), manual=True,
            ),
            Step(
                "fine_slider", "Make Smaller Adjustments",
                "<p>Enable <b>Fine</b> and drag either handle again. Mouse dragging is ten times slower. "
                "With a slider focused, try the Left and Right arrow keys for smaller keyboard adjustments.</p>"
                "<p>Turn Fine off and click <b>Reset</b> before continuing. "
                "Restore <b>kf1 = 1.5</b> and <b>k2 = 0.1</b> for the next run.</p>",
                self._slider_target, manual=True,
            ),
            self._add_set_step(),
            self._rename_extra_step(),
            replace(
                self._configure_extra_step(),
                instruction="<p>In <b>Higher A</b>, enter <b>A = 2</b>. Keep <b>B = 0</b> and <b>C = 0</b>. "
                "Press <b>Enter</b> after each edit.</p>"
                "<p>Leave preset_set1 at A = 1, B = 0, and C = 0.</p>",
            ),
            Step(
                "run_new_set", "Run Higher A",
                "<p>Select only <b>Higher A</b> and click <b>Run</b> in either panel. "
                "Keep the supplied rates, <b>kf1 = 1.5</b> and <b>k2 = 0.1</b>.</p>"
                "<p>This gives the second set results to compare with its Preview.</p>",
                lambda: (self._table_target(self.extra_set), *self._run_targets()),
                lambda: self._values(self.extra_set, 2.0)
                and self._run_ok({"kf1": 1.5, "kr1": 0.25, "k2": 0.1}, (self.extra_set,)),
            ),
            Step(
                "slider_targets", "Preview Both Sets",
                "<p>Check <b>Slider</b> for <b>Higher A</b>, then click the <b>preset_set1</b> row. "
                "Both names should appear beside <b>Initial-condition sets</b> in the <b>Interactive Sliders panel</b>.</p>"
                "<p>Check <b>Plot</b> for both sets. In the <b>Y</b> selector above the plot, select only <b>B</b>.</p>"
                "<p>The current row and pinned rows choose the slider targets. Plot chooses which results you see.</p>",
                lambda: (self._initial_target(), self._slider_target(), self._plot_target()),
                lambda: self.extra_set in self.registry.presentation(self.tab_id).pinned_slider_set_ids
                and self.bundle.batch_model.set_id_for_live_index(self.bundle.batch_table.currentIndex()) == self.preset_set
                and set(self.bundle.slider_binding.resolve_target_set_ids()) == {self.preset_set, self.extra_set}
                and set(self.registry.presentation(self.tab_id).plot_set_ids) >= {self.preset_set, self.extra_set}
                and self.registry.presentation(self.tab_id).selected_y_series == (SimulationSeriesAxisSelection("B"),),
            ),
            Step(
                "preview_both", "Choose Both Rates to Keep",
                "<p>Enter <b>1.125</b> for <b>kf1</b> and <b>0.05</b> for <b>k2</b>. Press <b>Enter</b> after each edit.</p>"
                "<p>Wait for both previews to finish. Both sets use the same proposed rates, "
                "with different starting concentrations.</p>",
                self._slider_target,
                lambda: self._preview_ready(1.125, 0.05, (self.preset_set, self.extra_set)),
                "Both sets have completed previews using the chosen rates.",
            ),
            Step(
                "understand_apply", "What Does Apply Do?",
                "<p>The accepted rates are <b>kf1 = 1.5</b> and <b>k2 = 0.1</b>. "
                "The proposed rates are <b>kf1 = 1.125</b> and <b>k2 = 0.05</b>. The reverse rate remains <b>kr1 = 0.25</b>.</p>"
                "<p><b>Apply</b> makes both proposed rates part of the accepted model. It updates Reactions "
                "and keeps the completed Preview results for the targeted sets. You do not need to run those simulations again.</p>"
                "<p>The rates are shared by all sets in the tab. Results for other sets become out of date "
                "and need a new run. Here, both practice sets are targets.</p>"
                "<p>Apply does not save the project file. Use <b>File → Save Project</b> to save your work.</p>",
                self._slider_target,
            ),
            Step(
                "apply_preview", "Keep Both Changes",
                "<p>Click <b>Apply</b> in the <b>Interactive Sliders panel</b>. If Preview is still computing, wait and click Apply again.</p>"
                "<p>Check Reactions for <b>kf = 1.125</b> and <b>kr = 0.25</b> on A ⇌ B, and <b>k = 0.05</b> on B → C. "
                "The completed curves now belong to the accepted model. Later trials start from these rates, "
                "and Reset returns to them.</p>"
                "<p>Apply uses the accepted source behind the Preview. It replaces pending edits in Reactions. "
                "Finish a source edit with Run before beginning a slider trial if you want that edit included.</p>",
                lambda: (self._slider_target(), self._source_target()), manual=True,
            ),
            Step(
                "finish", "Lesson Complete",
                "<p>You have tried two parameter changes together, compared Preview with Reference, "
                "and used sliders with more than one initial-condition set.</p>"
                "<p>Use <b>Reset</b> to discard a trial or <b>Apply</b> to keep the changes and completed results.</p>"
                "<p>Choose <b>Finish</b> to close this lesson.</p>",
                lambda: None,
            ),
        )
