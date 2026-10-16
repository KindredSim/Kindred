"""Reactions, drafts, and sets of starting concentrations."""

from kindred.gui.tutorials.controller import Step
from kindred.gui.tutorials.overlay import Highlight
from kindred.gui.tutorials.practice_lesson import PracticeLesson


class ReactionsAndInitialConditions(PracticeLesson):
    title = "Reactions and Initial Conditions"
    description = "Read a reaction, change its rate, and compare two sets of starting concentrations."
    _extra_steps = frozenset(
        {
            "rename_extra",
            "configure_extra",
            "run_new_set",
            "compare_sets",
            "copy_selected_set",
        }
    )
    _result_steps = frozenset({"edit_rate"})

    def blocked(self):
        missing = super().blocked()
        if missing is not None or self.tab_id is None:
            return missing
        if self.step.name in {"move_set_up", "move_set_down"}:
            if self.original_set not in {
                item.set_id for item in self.tab.initial_condition_sets
            }:
                return (
                    "The original set1 row was removed. Use Skip Step to continue without this reordering exercise.",
                    "",
                    None,
                )
        if self.step.name in {"compare_sets", "copy_selected_set"}:
            repo = self.registry.tab_runtime(self.tab_id).result_repository
            if repo.current_result(self.preset_set) is None:
                return self._missing(
                    "Run preset_set1 with the edited mechanism before comparing results.",
                    "run_edited",
                )
            if repo.current_result(self.extra_set) is None:
                return self._missing(
                    "Run Higher A before comparing both sets.", "run_new_set"
                )
        return None

    def _preset_before_original(self) -> bool:
        identities = [item.set_id for item in self.tab.initial_condition_sets]
        return identities.index(self.preset_set) < identities.index(self.original_set)

    def _delete_unused_targets(self) -> tuple[Highlight, ...]:
        if self.original_set not in self.bundle.batch_model.set_ids():
            return ()
        row = self._table_target(self.original_set)
        if self.bundle.initial_actions.selected_run_set_ids() != (self.original_set,):
            return (row,)
        return (
            row,
            self._panel(self.bundle.simulation_panel.delete_batch_set_btn, "batch_initial_conditions"),
        )

    def _steps(self) -> tuple[Step, ...]:
        return (
            *self._setup_steps(),
            Step(
                "read_mechanism",
                "Read the Reaction",
                "<p>Find <b>A -&gt; B ; kf = 1.0</b> in <b>Reactions</b>.</p>"
                "<p><b>-&gt;</b> is the syntax for an irreversible reaction. The semicolon separates the reaction "
                "from its parameter settings. <b>kf</b> specifies the forward rate constant.</p>"
                "<p>Lines beginning with <b>#</b> are comments. They help explain the model and are not executed.</p>",
                self._source_target,
            ),
            Step(
                "read_observable",
                "Read a Calculated Quantity",
                "<p>Find <b>let total = [A] + [B]</b> below the reaction.</p>"
                "<p>A <b>user-defined observable</b> is a calculated quantity that you can plot or use as a fitting target.</p>"
                "<ul><li><b>let</b> introduces the definition.</li><li><b>total</b> is the name chosen for this observable.</li>"
                "<li><b>[A]</b> and <b>[B]</b> refer to the concentrations of A and B at simulation time <i>t</i>. "
                "Brackets distinguish these references from parameter names.</li></ul>"
                "<p>The <b>Help</b> tab beside Reactions lists supported expressions "
                "under <b>Parameters and Observables</b>.</p>",
                self._source_target,
            ),
            Step(
                "initial_rows",
                "What Is a Set?",
                "<p>Each row in <b>Initial Conditions</b> stores a named set of starting concentrations. "
                "Run different sets with the same reactions to obtain separate results.</p>"
                "<p>The fresh tab supplied <b>set1</b> with zero concentrations. Loading M1 added <b>preset_set1</b> "
                "with A = 1 and B = 0. Keep preset_set1 for this lesson.</p>"
                "<p>You can reorder rows with <b>Move Up</b> and <b>Move Down</b>. "
                "You can remove unused rows with <b>Delete Set(s)</b>, but every tab must keep at least one set.</p>",
                self._initial_target,
            ),
            Step(
                "initial_sources",
                "Where Starting Values Come From",
                "<p>You can enter concentrations directly in <b>Initial Conditions</b>, load declarations with a preset or template, "
                "or create a set from suitable starting measurements when mapping experimental data.</p>"
                "<p>You can also paste initial-condition declarations into Reactions. Run checks that source and imports "
                "the declarations before simulating. Writing them in Reactions is optional.</p>",
                lambda: (self._source_target(), self._initial_target()),
            ),
            Step(
                "read_initial_provenance",
                "Understand the Provenance Comments",
                "<p>Find the comments beginning with <b># Kindred IC provenance |</b>. "
                "<b>IC</b> means initial conditions. <b>Provenance</b> means where the values came from.</p>"
                "<p>These comments record the starting values imported from M1. They are not live controls "
                "and do not change when you edit the table. Edit current concentrations in <b>Initial Conditions</b>.</p>",
                self._source_target,
            ),
            Step(
                "move_set_up",
                "Move a Set Up",
                "<p>Select only <b>preset_set1</b> and click <b>Move Up</b> to place it above <b>set1</b>. "
                "If it is already above set1, choose Next.</p>"
                "<p>This changes the row order without changing the concentrations. You can skip these row-management exercises.</p>",
                lambda: (
                    self._table_target(self.preset_set),
                    self._panel(
                        self.bundle.simulation_panel.move_batch_up_btn,
                        "batch_initial_conditions",
                    ),
                ),
                self._preset_before_original,
                "preset_set1 is above set1.",
            ),
            Step(
                "move_set_down",
                "Move a Set Down",
                "<p>Keep only <b>preset_set1</b> selected and click <b>Move Down</b> to place it below <b>set1</b>. "
                "If it is already below set1, choose Next.</p>",
                lambda: (
                    self._table_target(self.preset_set),
                    self._panel(
                        self.bundle.simulation_panel.move_batch_down_btn,
                        "batch_initial_conditions",
                    ),
                ),
                lambda: not self._preset_before_original(),
                "preset_set1 is below set1.",
            ),
            Step(
                "delete_unused_set",
                "Delete an Unused Set",
                "<p>Select only the original <b>set1</b> row. Click <b>Delete Set(s)</b> and confirm the deletion.</p>"
                "<p>Keep <b>preset_set1</b> for the simulation. A tab must retain at least one set. "
                "If set1 is already absent, choose Next. Use Skip Step if you want to keep it.</p>",
                self._delete_unused_targets,
                lambda: self.original_set
                not in {item.set_id for item in self.tab.initial_condition_sets},
                "set1 has been removed.",
            ),
            self._prepare_run_step(),
            self._first_run_step(),
            Step(
                "edit_rate",
                "Edit Reactions as a Draft",
                "<p>In Reactions, change <b>kf = 1.0</b> to <b>kf = 0.5</b>.</p>"
                "<p>Typing creates a <b>draft</b> automatically. The visible text can be incomplete while you edit. "
                "The <b>accepted model</b> is the version currently kept in this mechanism tab. "
                "The existing curves still describe that version while you edit.</p>"
                "<p><b>Run</b> checks the draft before calculating new results. Invalid text stays available so you can correct it. "
                "<b>Save Project</b> also checks pending source edits.</p>"
                "<p><b>Revert</b> beside Reactions restores the accepted text. For this exercise, keep the draft at 0.5 "
                "and choose Continue.</p>",
                self._source_target,
                manual=True,
            ),
            Step(
                "run_edited",
                "Run the Changed Model",
                "<p>Keep <b>kf = 0.5</b> in Reactions. In Initial Conditions, check <b>A = 1</b> and <b>B = 0</b> for <b>preset_set1</b>.</p>"
                "<p>Select only that row and click <b>Run</b>. Kindred checks the edited source before simulating. "
                "Correct any reported source error and try again.</p>",
                lambda: (self._table_target(self.preset_set), *self._run_targets()),
                lambda: self._values(self.preset_set, 1.0)
                and self._run_ok({"k1": 0.5}, (self.preset_set,)),
                "The changed model has finished running.",
            ),
            self._add_set_step(),
            self._rename_extra_step(),
            self._configure_extra_step(),
            Step(
                "run_new_set",
                "Run the Second Starting Scenario",
                "<p>Select only <b>Higher A</b> and click <b>Run</b>. Keep <b>kf = 0.5</b>.</p>"
                "<p>Selecting rows chooses what to run. <b>Plot</b> and <b>Slider</b> checkboxes have different jobs. "
                "To run several sets together, select their rows with Ctrl-click, or Cmd-click on macOS.</p>",
                lambda: (self._table_target(self.extra_set), *self._run_targets()),
                lambda: self._values(self.extra_set, 2.0)
                and self._run_ok({"k1": 0.5}, (self.extra_set,)),
                "Higher A has its own results.",
            ),
            Step(
                "compare_sets",
                "Compare the Two Sets",
                "<p>In the <b>Initial Conditions panel</b>, check <b>Plot</b> for <b>preset_set1</b> and <b>Higher A</b>.</p>"
                "<p>Uncheck Plot for Higher A, then check it again. This hides and restores available results without another run. "
                "It does not select the row for Run.</p>"
                "<p>Starting a run automatically checks Plot for the sets being run, even if you unchecked them earlier.</p>",
                lambda: self._table_column_targets(self.bundle.batch_model.plot_column()),
                manual=True,
            ),
            Step(
                "copy_selected_set",
                "Copy Results from One Set",
                "<p>In the <b>Initial Conditions panel</b>, leave <b>Plot</b> checked only for <b>preset_set1</b>.</p>"
                "<p>Right-click the plot and choose <b>Copy Plotted Data</b>. Paste into an empty spreadsheet.</p>"
                "<p>Check that the copied results include preset_set1 and exclude Higher A.</p>"
                "<p>Check Plot for Higher A again, then choose <b>Continue</b>.</p>",
                lambda: (*self._table_column_targets(self.bundle.batch_model.plot_column()), self._plot_target()),
                manual=True,
            ),
            self._finish_step(),
        )
