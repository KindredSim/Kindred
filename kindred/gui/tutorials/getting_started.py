"""Getting Started adapts to existing GUI owners; it never extends their API."""

from __future__ import annotations

from math import isclose

from PySide6 import QtWidgets

from kindred.core.project_presentation_state import SimulationSeriesAxisSelection
from kindred.gui.tutorials.controller import Lesson, Step
from kindred.gui.tutorials.overlay import Highlight, menu_target, scroll_into_view
from kindred.gui.widgets.template_manager_dialog import (
    TemplateEditorDialog,
    TemplateManagerDialog,
)


class GettingStarted(Lesson):
    title = "Getting Started"
    description = (
        "Load a preset, run and modify a model, compare starting conditions, "
        "explore sliders, create and reuse your own template, and save your project. "
        "You work in fresh mechanism tabs and keep them when you finish."
    )

    def __init__(
        self, *, workspace, registry, surfaces, examples_menu, project_controller, jobs
    ):
        super().__init__()
        self.workspace, self.registry, self.surfaces = workspace, registry, surfaces
        self.examples_menu, self.jobs = examples_menu, jobs
        self.practice = self.reuse = None
        self.original_set = None
        self.preset_set = self.extra_set = self.reuse_set = None
        self._tabs_before = set()
        self._sets_before = set()
        self._run_baselines = {}
        self.observe(project_controller.projectStateChanged, self._project_changed)
        self.observe(jobs.run_dispatched, self._run_dispatched)
        self.observe(jobs.results_changed)
        self.observe(jobs.state_changed)
        self.steps = self._steps()

    @property
    def tab_id(self):
        return (
            self.reuse
            if self.step.name in self._reuse_steps and self.reuse
            else self.practice
        )

    @property
    def bundle(self):
        return self.surfaces[self.tab_id]

    @property
    def tab(self):
        return self.registry.tab(self.tab_id)

    def enter(self, position: int) -> None:
        super().enter(position)
        if self.step.name in {"create_tab", "create_reuse_tab"}:
            self._tabs_before = {tab.tab_id for tab in self.registry.document.tabs}
        if (
            self.step.name == "add_set"
            and self.extra_set is None
            and self.practice is not None
        ):
            self._sets_before = {
                item.set_id for item in self.tab.initial_condition_sets
            }

    # Explicit exercise groups describe resource needs, independent of display order.
    _reuse_steps = {
        "create_reuse_tab",
        "apply_template",
        "reuse_values",
        "run_reused",
        "save_project",
        "finish",
    }
    _workspace_steps = {"welcome", "learn", "create_tab", "save_project", "finish"}
    _plot_steps = {
        "read_curves",
        "plot_y",
        "plot_units",
        "plot_observable",
        "plot_x",
        "plot_time",
        "plot_navigation",
        "plot_legend",
        "plot_log_axes",
        "plot_invert_axes",
    }
    _preset_steps = {
        "read_mechanism",
        "initial_sources",
        "read_initial_provenance",
        "read_observable",
        "initial_rows",
        "move_set_up",
        "move_set_down",
        "delete_unused_set",
        "select_preset_set",
        "time_range",
        "solver_points",
        "run_first",
        "edit_rate",
        "run_edited",
        "add_set",
        "rename_set",
        "set_more_a",
        "select_new_set",
        "run_new_set",
        "plot_sets",
        "copy_all",
        "simplify_plot",
        "copy_plotted",
        "pin_slider_set",
        "show_slider",
        "preview_target",
        "preview_quarter",
        "preview_reference",
        "reset_trial",
        "drag_slider",
        "fine_slider",
        "preview_three_quarters",
        "apply_preview",
    } | _plot_steps
    _comparison_steps = {
        "rename_set",
        "set_more_a",
        "select_new_set",
        "run_new_set",
        "plot_sets",
        "copy_all",
        "simplify_plot",
        "copy_plotted",
        "pin_slider_set",
        "drag_slider",
        "fine_slider",
        "preview_three_quarters",
        "apply_preview",
    }
    _run_steps = {"run_first", "run_edited", "run_new_set", "run_reused"}

    def _missing(self, message, step):
        return (
            message
            + " Return to its setup step, or use Skip Step to move on without doing this exercise.",
            "Return to setup",
            lambda: self.revisitRequested.emit(step),
        )

    def blocked(self):
        name = self.step.name
        live = {tab.tab_id for tab in self.registry.document.tabs}
        if any(
            identity is not None and identity not in live
            for identity in (self.practice, self.reuse)
        ):
            self.ended.emit(
                "A lesson tab was closed. Restart the lesson to create fresh practice work."
            )
            return None
        if name in self._workspace_steps:
            return None
        if self.practice is None:
            return self._missing("This exercise needs a practice tab.", "create_tab")
        if (
            name in self._reuse_steps
            and name != "create_reuse_tab"
            and self.reuse is None
        ):
            return self._missing(
                "This exercise needs a tab for template reuse.", "create_reuse_tab"
            )
        creating_reuse = (
            name == "create_reuse_tab"
            and self.reuse is None
            and self.registry.active_tab_id not in self._tabs_before
        )
        if self.registry.active_tab_id != self.tab_id and not creating_reuse:
            return (
                "This lesson is paused while another mechanism tab is active.",
                "Return to the lesson tab",
                lambda: self.workspace.tabActivated.emit(self.tab_id),
            )
        sets = {item.set_id for item in self.tab.initial_condition_sets}
        if name in self._preset_steps and (
            self.preset_set is None or self.preset_set not in sets
        ):
            self.preset_set = None
            return self._missing(
                "This exercise needs the preset's starting conditions.", "load_preset"
            )
        if name in {"move_set_up", "move_set_down"} and self.original_set not in sets:
            return (
                "The original set1 row has already been removed. Use Skip Step to continue without this reordering exercise.",
                "",
                None,
            )
        if (
            name in self._plot_steps
            and self.registry.tab_runtime(self.tab_id).result_repository.current_result(
                self.preset_set
            )
            is None
        ):
            return self._missing(
                "These plot exercises need a completed simulation.", "run_first"
            )
        if name in self._comparison_steps and (
            self.extra_set is None or self.extra_set not in sets
        ):
            self.extra_set = None
            return self._missing(
                "This exercise needs the additional starting-condition set.", "add_set"
            )
        if name in {
            "select_new_set", "run_new_set", "plot_sets", "copy_all",
            "simplify_plot", "copy_plotted",
        }:
            repo = self.registry.tab_runtime(self.tab_id).result_repository
            if repo.current_result(self.preset_set) is None:
                return self._missing(
                    "This exercise uses the preset's existing simulation.", "run_edited"
                )
            if (
                name in {"plot_sets", "copy_all", "simplify_plot", "copy_plotted"}
                and repo.current_result(self.extra_set) is None
            ):
                return self._missing(
                    "This exercise needs results for the new set.", "run_new_set"
                )
        if name in {
            "drag_slider",
            "fine_slider",
            "preview_three_quarters",
            "apply_preview",
        } and set(self.bundle.slider_binding.resolve_target_set_ids()) != {
            self.preset_set,
            self.extra_set,
        }:
            return self._missing(
                "This exercise previews both starting-condition sets.", "pin_slider_set"
            )
        if name in {"reuse_values", "run_reused"} and self.reuse_set not in sets:
            self.reuse = self.reuse_set = None
            return self._missing(
                "The reused model's starting-condition set was removed.",
                "create_reuse_tab",
            )
        return None

    def working(self) -> bool:
        return self.jobs.busy and self.jobs.origin_tab_id == self.tab_id

    def _new_tab(self, reuse=False) -> bool:
        attribute = "reuse" if reuse else "practice"
        if getattr(self, attribute) is None:
            identity = self.registry.active_tab_id
            if identity in self._tabs_before or identity not in self.surfaces:
                return False
            setattr(self, attribute, identity)
            if reuse:
                self.reuse_set = self.tab.initial_condition_sets[0].set_id
            else:
                self.original_set = self.tab.initial_condition_sets[0].set_id
            bundle = self.bundle
            for signal in (
                bundle.editor_binding.draftStateChanged,
                bundle.initial_actions.initialValuesChanged,
                bundle.initial_actions.selectedRunSetIdsChanged,
                bundle.slider_binding.draftStateChanged,
                bundle.slider_binding.targetSetIdsChanged,
            ):
                self.observe(signal)
        return True

    def _loaded(self) -> bool:
        if self.tab.prepared_dsl is None:
            return False
        if self.preset_set is None:
            self.preset_set = next(
                (
                    item.set_id
                    for item in self.tab.initial_condition_sets
                    if item.name == "preset_set1"
                ),
                None,
            )
        return self.preset_set is not None and self._values(self.preset_set, 1.0)

    def _project_changed(self, replaced: bool) -> None:
        if replaced:
            self.ended.emit(
                "The project was replaced. Start another lesson from Help when ready."
            )
        else:
            self.changed.emit()

    def _run_dispatched(self, tab_id, set_ids) -> None:
        if tab_id == self.tab_id and self.step.name in self._run_steps:
            repo = self.registry.tab_runtime(tab_id).result_repository
            self._run_baselines[self.step.name] = {
                identity: repo.current_result(identity) for identity in set_ids
            }
        self.changed.emit()

    def _values(self, set_id, amount: float) -> bool:
        item = next(
            (s for s in self.tab.initial_condition_sets if s.set_id == set_id), None
        )
        if item is None or self.registry.tab_runtime(
            self.tab_id
        ).initial_text_by_set_id.get(set_id):
            return False
        return dict(item.values) == {"A": amount, "B": 0.0}

    def _run_ok(self, rate: float, identities: tuple) -> bool:
        baseline = self._run_baselines.get(self.step.name, {})
        runtime = self.registry.tab_runtime(self.tab_id)
        if (
            self.jobs.busy
            or runtime.has_dirty_drafts
            or set(baseline) != set(identities)
        ):
            return False
        repo = runtime.result_repository
        for identity in identities:
            result = repo.current_result(identity)
            if (
                result is None
                or result is baseline[identity]
                or identity in repo.preview_set_ids()
            ):
                return False
            value = next(
                (value for name, value, _unit in result.parameters if name == "k1"),
                None,
            )
            if value is None or not isclose(value, rate, rel_tol=1e-8):
                return False
        return bool(identities)

    def _added_set(self) -> bool:
        if self.extra_set is None:
            self.extra_set = next(
                (
                    item.set_id
                    for item in self.tab.initial_condition_sets
                    if item.set_id not in self._sets_before
                ),
                None,
            )
        return self.extra_set is not None

    def _preset_before_original(self) -> bool:
        ids = [item.set_id for item in self.tab.initial_condition_sets]
        return ids.index(self.preset_set) < ids.index(self.original_set)

    def _panel(self, widget, identity: str) -> Highlight:
        def reveal():
            self.workspace.reveal_dock(identity)
            scroll_into_view(widget)

        return Highlight(widget, reveal=reveal)

    def _tab_target(self) -> Highlight:
        # The existing central layout owns the tab strip. Highlight the containing
        # workspace rather than reaching into its private tab-bar implementation.
        return Highlight(self.workspace.centralWidget())

    def _source_target(self) -> Highlight:
        return self._panel(self.bundle.editor, "mechanism")

    def _table_target(self, set_id=None, species=None) -> Highlight:
        table, model = self.bundle.batch_table, self.bundle.batch_model

        def index():
            ids = model.set_ids()
            row = ids.index(set_id) if set_id in ids else 0
            column = (
                model.species_names().index(species) + 1
                if species in model.species_names()
                else 0
            )
            return model.index(row, column)

        def reveal():
            self.workspace.reveal_dock("batch_initial_conditions")
            table.scrollTo(index())

        return Highlight(
            table.viewport(),
            (lambda: table.visualRect(index())) if set_id else None,
            reveal,
        )

    def _plot_target(self) -> Highlight:
        return Highlight(self.workspace.plot_for_tab(self.tab_id))

    def _initial_target(self) -> Highlight:
        return self._panel(self.bundle.simulation_panel, "batch_initial_conditions")

    def _run_target(self) -> Highlight:
        return self._panel(
            self.bundle.simulation_panel.run_btn, "batch_initial_conditions"
        )

    def _preset_target(self) -> Highlight:
        return menu_target(self.examples_menu, lambda action: action.data() == "M1")

    def _examples_target(self) -> Highlight:
        return menu_target(self.examples_menu)

    @staticmethod
    def _dialog(kind):
        dialog = QtWidgets.QApplication.activeModalWidget()
        return dialog if isinstance(dialog, kind) else None

    def _manager_target(self) -> Highlight:
        dialog = self._dialog((TemplateManagerDialog, TemplateEditorDialog))
        return Highlight(dialog) if dialog is not None else self._examples_target()

    def _slider_target(self) -> Highlight:
        return self._panel(self.bundle.slider_panel, "interactive_sliders")

    def _steps(self) -> tuple[Step, ...]:
        table = self._table_target
        initial = self._initial_target
        source, run, plot = self._source_target, self._run_target, self._plot_target
        manager, slider = self._manager_target, self._slider_target
        return (
            Step(
                "welcome",
                "Welcome to Kindred",
                "<p>Kindred is an application for modelling <b>chemical reaction kinetics</b>. You can:</p>"
                "<ul><li>Build reaction mechanisms and simulate their behavior.</li>"
                "<li>Compare simulations with experimental data.</li>"
                "<li>Fit kinetic parameters to measurements.</li>"
                "</ul>"
                "<h3>About This Tutorial</h3>"
                "<p>This tutorial introduces Kindred’s workspace through practical exercises. The next page outlines what you will learn.</p>",
                lambda: None,
                link=("Kindred on GitHub", "https://github.com/kindredsim/kindred"),
            ),
            Step(
                "learn",
                "What You Will Learn",
                "<p>In this tutorial, you will learn to:</p>"
                "<ul><li>Load a <b>preset mechanism</b>.</li>"
                "<li>Run a model, explore its plots, and modify it.</li>"
                "<li>Use sets of <b>initial conditions</b>.</li>"
                "<li>Create and reuse <b>mechanism templates</b>.</li>"
                "<li>Save a Kindred project as a <b>.kin file</b>.</li>"
                "</ul>"
                "<h3>Using the Tutorial</h3>"
                "<p><b>Advance Automatically</b> is enabled by default and moves on after you complete an automatically checked action. Untick it if you prefer to choose <b>Next</b> yourself. A step that is already complete when you reach it waits for Next.</p>"
                "<p><b>Manual checks</b> ask you to inspect the result and choose Continue.</p>"
                "<ul><li><b>Previous</b> revisits instructions without undoing work.</li>"
                "<li><b>Skip Step</b> moves on without verifying completion.</li>"
                '<li><b>"−"</b> minimizes the tutorial. <b>"×"</b> closes it.</li>'
                "</ul>",
                lambda: None,
            ),
            Step(
                "create_tab",
                "Create a New Mechanism Tab",
                "<h3>Projects and Tabs</h3>"
                "<p>Kindred saves projects as <b>.kin files</b>. A fresh workspace opens a new project with a tab named <b>Mechanism 1</b>.</p>"
                "<p>A project can contain several mechanism tabs and always has at least one. Each tab has its own:</p>"
                "<ul><li>Reaction mechanism and notes.</li>"
                "<li>Initial-condition sets and simulation settings.</li>"
                "<li>Simulation results.</li>"
                "</ul>"
                "<p><b>Imported datasets</b> belong to the project and are shared across its tabs.</p>"
                "<h3>Create a Tab</h3>"
                '<p>Click <b>"+"</b> beside the mechanism tabs, above the main plot.</p>'
                "<p>Presets and templates load into the selected tab. Use this new tab for the tutorial exercises.</p>",
                self._tab_target,
                self._new_tab,
            ),
            Step(
                "name_tab",
                "Name the Tab",
                "<p>A descriptive name helps you identify a mechanism when several tabs are open.</p>"
                "<ol><li>Right-click the new tab’s name above the main plot and choose <b>Rename…</b>.</li>"
                "<li>Enter <b>Getting Started</b> and confirm.</li>"
                "</ol>",
                lambda: Highlight(self._dialog(QtWidgets.QInputDialog))
                if self._dialog(QtWidgets.QInputDialog) is not None
                else self._tab_target(),
                lambda: self.tab.name == "Getting Started",
            ),
            Step(
                "load_preset",
                "Load a Preset",
                "<p>A preset provides a mechanism and starting values that you can explore and modify.</p>"
                "<h3>Load M1</h3>"
                "<p>Open <b>Examples → Preset Mechanisms → M1</b>.</p>"
                "<p>Follow the highlighted menu entries. There is no need to press Next between menus.</p>",
                self._preset_target,
                self._loaded,
                "Accepted source and starting values are available. Check that you selected M1.",
            ),
            Step(
                "read_mechanism",
                "Read the Reaction",
                "<p>M1 models a <b>simple first-order isomerization</b> using a single irreversible elementary reaction. Look at this line in <b>Reactions</b>:</p>"
                "<p><b>A -&gt; B ; kf = 1.0</b></p>"
                "<ul><li><b>A</b> is the reactant and <b>B</b> is the product.</li>"
                '<li><b>"-&gt;"</b> indicates an irreversible reaction: A converts into B.</li>'
                '<li><b>";"</b> separates the reaction from its parameter settings.</li>'
                "<li><b>kf</b> specifies the <b>forward reaction rate constant</b>, here set to <b>1.0</b>.</li>"
                "</ul>",
                source,
            ),
            Step(
                "initial_sources",
                "Where Initial Conditions Come From",
                "<p>Initial conditions specify the species concentrations at the start of a simulation. You can supply them in several ways:</p>"
                "<ul><li><b>Enter them directly</b> in the Initial Conditions table. Use Add Set when you need another set.</li>"
                "<li><b>Load them with a mechanism.</b> Presets and user-defined templates can include starting-condition declarations. These are useful for distributing a model with starting values.</li>"
                "<li><b>Use experimental data.</b> Mapping an imported dataset can create a set from suitable measured starting concentrations. <b>Experimental Data Tutorial (To be Added soon)</b></li></ul>"
                "<p>You can also paste starting-condition declarations together with mechanism text into <b>Reactions</b>. When you click <b>Run</b>, Kindred checks the mechanism text and imports those declarations into the table before simulating. Writing initial conditions in Reactions is optional.</p>",
                lambda: (source(), initial()),
            ),
            Step(
                "read_initial_provenance",
                "Understand the Provenance Comments",
                "<p><b>IC</b> means <b>initial conditions</b>: the concentrations at the start of a simulation. <b>Provenance</b> records where those values came from.</p>"
                "<p>When Kindred loaded M1, it imported the starting concentrations into the <b>Initial Conditions</b> table as <b>preset_set1</b>. It retained the original declarations as comments with this prefix:</p>"
                "<p><b># Kindred IC provenance |</b></p>"
                '<p><b>"#"</b> starts a comment. Comment lines provide information and are not executed as mechanism instructions. Headings such as <b># Algebra</b> are comments too.</p>'
                "<p>The provenance comments record the imported values, <b>[A] = 1.0</b> and <b>[B] = 0.0</b>. They do not update when the table changes. Edit starting concentrations in the <b>Initial Conditions</b> table.</p>",
                source,
            ),
            Step(
                "read_observable",
                "Understand an Observable",
                "<p>The final line defines a <b>user-defined observable</b>: a calculated quantity that can be plotted and used as a <b>fitting target</b> against experimental data.</p>"
                "<p><b>let total = [A] + [B]</b></p>"
                "<ul><li><b>let</b> introduces the observable definition.</li>"
                "<li><b>total</b> is the name chosen for this observable.</li>"
                "<li><b>[A]</b> and <b>[B]</b> refer to the current concentrations of species <b>A</b> and <b>B</b>.</li>"
                "<li>The expression <b>[A] + [B]</b> adds those concentrations at each time point.</li>"
                "</ul>"
                "<p>Open the <b>Help</b> tab beside Reactions for supported expressions and functions, including the <b>Parameters and Observables</b> section.</p>",
                source,
            ),
            Step(
                "initial_rows",
                "Understand the Two Rows",
                "<p>Every mechanism tab must contain at least one <b>initial-condition set</b>, even before you enter any concentrations. A fresh tab supplies <b>set1</b>, initially with zero concentrations.</p>"
                "<p>Loading M1 added <b>preset_set1</b>. You can now delete the unused <b>set1</b> row or leave it in place.</p>"
                "<p>The next exercises use <b>Move Up</b>, <b>Move Down</b>, and <b>Delete Set(s)</b>. You can skip them and still use <b>preset_set1</b> for the simulation.</p>",
                table,
            ),
            Step(
                "move_set_up",
                "Move a Set Up",
                "<ol><li>Select only <b>preset_set1</b> in Initial Conditions.</li>"
                "<li>Click <b>Move Up</b> to place it above <b>set1</b>. If it is already above <b>set1</b>, choose Next.</li></ol>"
                "<p>These controls change the row order, without changing the concentrations in either set.</p>",
                initial,
                self._preset_before_original,
                "preset_set1 is above the original row.",
            ),
            Step(
                "move_set_down",
                "Move a Set Down",
                "<p>With only <b>preset_set1</b> selected, click <b>Move Down</b> to place it below <b>set1</b>. If it is already below <b>set1</b>, choose Next.</p>",
                initial,
                lambda: not self._preset_before_original(),
                "preset_set1 is below the original row.",
            ),
            Step(
                "delete_unused_set",
                "Delete an Unused Set",
                "<ol><li>Select only the original <b>set1</b> row.</li>"
                "<li>Click <b>Delete Set(s)</b> and confirm the deletion.</li></ol>"
                "<p>Keep <b>preset_set1</b> for the simulation. A tab must always retain at least one set.</p>"
                "<p>If <b>set1</b> is already absent, choose Next. You can also use Skip Step to leave it in place.</p>",
                initial,
                lambda: self.original_set
                not in {item.set_id for item in self.tab.initial_condition_sets},
                "The original row has been removed. preset_set1 remains available.",
            ),
            Step(
                "select_preset_set",
                "Choose the Preset's Starting Conditions",
                "<p>Selected rows determine which starting-condition sets <b>Run</b> uses.</p>"
                "<h3>Check the Preset Row</h3>"
                "<ol><li>Find <b>preset_set1</b> in <b>Initial Conditions</b>. If it is not already selected, select it.</li>"
                "<li>Check that <b>A = 1</b> and <b>B = 0</b>.</li>"
                "</ol>"
                "<p>Select only <b>preset_set1</b> for this run. Any other rows can remain unselected.</p>",
                initial,
                lambda: self._values(self.preset_set, 1.0)
                and self.bundle.initial_actions.selected_run_set_ids()
                == (self.preset_set,),
            ),
            Step(
                "time_range",
                "Set the Time Range",
                "<ul><li><i>t</i><sub>start</sub> is the time your simulation starts. Its default is <b>0</b>.</li>"
                "<li><i>t</i><sub>end</sub> is the time it ends.</li></ul>"
                "<h3>Set the Time Range</h3>"
                "<p>In <b>Initial Conditions</b>, check that <i>t</i><sub>start</sub> is <b>0</b>. Change it only if needed. Set <i>t</i><sub>end</sub> to <b>5</b> seconds if it is not already 5.</p>"
                "<p>Press <b>Enter</b> after each edit.</p>",
                initial,
                lambda: not self.bundle.simulation_panel.has_run_defaults_draft
                and self.tab.resolved_execution_settings(self.preset_set).t_start == 0
                and self.tab.resolved_execution_settings(self.preset_set).t_end == 5,
            ),
            Step(
                "solver_points",
                "Solver and Output Points",
                "<p><b>Solver</b> selects the numerical method used to calculate the simulation. <b>BDF</b> is Kindred’s default. <b>Radau</b> is particularly suited to strongly stiff systems and can offer better robustness, potentially at higher computational cost. Consider switching to Radau for mechanisms with fast and slow processes occurring together, such as rapid equilibration alongside much slower reactions. Very different rate constants can indicate this, but their units, reaction orders, and the species concentrations also matter. There is no universal rate-constant cutoff for stiffness. For more details on stiff problems and numerical solvers, see <b>(Reference to be added soon)</b>.</p>"
                "<p><b>Points</b> sets the number of evenly spaced output times, including <i>t</i><sub>start</sub> and <i>t</i><sub>end</sub>. It does not set the solver’s internal integration steps.</p>"
                "<p>Use either solver for this exercise. Set <b>Points</b> to <b>100</b> if it is not already 100. Confirm these settings before continuing.</p>",
                initial,
                lambda: not self.bundle.simulation_panel.has_run_defaults_draft
                and self.tab.resolved_execution_settings(
                    self.preset_set
                ).integration_points
                == 100,
                "Simulation settings are ready with 100 output points.",
            ),
            Step(
                "run_first",
                "Run Your First Simulation",
                "<p>Click <b>Run</b> in <b>Initial Conditions</b> and wait for the simulation to finish.</p>"
                "<p>You can also click <b>Run</b> in the <b>Mechanism</b> panel. Both buttons run the same selected initial-condition sets.</p>"
                "<p>The results will appear as curves in the main plot.</p>",
                initial,
                lambda: self._values(self.preset_set, 1.0)
                and self._run_ok(1.0, (self.preset_set,)),
                "Simulation completed. The curves are ready to inspect.",
            ),
            Step(
                "read_curves",
                "Explore the Plot",
                "<p>Your first simulation is now available in the main plot.</p>"
                "<ul><li>The <b>legend</b> identifies the plotted quantities by name and color.</li>"
                "<li>The controls above the plot select the <b>X</b> and <b>Y</b> quantities and their display units.</li></ul>"
                "<p>Find A, B, and total in the legend. If the legend is hidden, right-click the plot and enable <b>Show Legend</b>.</p>"
                "<p>The next exercises use these results to explore the plot controls before changing the model.</p>",
                plot,
            ),
            Step(
                "plot_y",
                "Choose What to Plot",
                "<p>Open the <b>Y</b> selector above the plot. It lets you choose several quantities.</p>"
                "<p>A label such as <b>“A +2”</b> means A plus two other selected quantities. It is a selection summary, not an algebraic expression.</p>"
                "<p>For now, select only <b>A</b>. Uncheck B and total, then close the selector. The plot should show A alone.</p>",
                plot,
                lambda: self.registry.presentation(self.tab_id).selected_y_series
                == (SimulationSeriesAxisSelection("A"),),
                "Only A is selected for Y.",
            ),
            Step(
                "plot_units",
                "Change the Display Units",
                "<p>Keep <b>X: t</b> and only <b>A</b> selected for Y.</p>"
                "<ol><li>Use the <b>Unit</b> selector beside X to change seconds (<b>s</b>) to minutes (<b>min</b>). Look at the axis labels, then restore <b>s</b>.</li>"
                "<li>Change the Y unit from <b>M</b> to <b>mM</b>, then restore <b>M</b>.</li></ol>"
                "<p>Display units change how the same results are shown. You do not need to run the simulation again.</p>"
                "<p>Unit choices depend on the selected quantities. Y units are available when all selected Y quantities are concentrations.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_observable",
                "Plot the Observable",
                "<p>Open the <b>Y</b> selector again and select <b>A</b>, <b>B</b>, and <b>total</b>. Close the selector.</p>"
                "<p>The user-defined observable appears alongside the species. Use the legend to identify it.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_x",
                "Choose the X Quantity",
                "<p>The <b>X</b> selector above the plot can use time, a species concentration, or an observable.</p>"
                "<p>Change X from <b>t</b> to <b>B</b>. You are now viewing the selected Y quantities against B instead of time.</p>"
                "<p>If <b>B</b> remains selected for Y, the plot also includes <b>B against B</b>.</p>",
                plot,
                lambda: self.registry.presentation(self.tab_id).selected_x_series
                == SimulationSeriesAxisSelection("B"),
                "B is selected for X.",
            ),
            Step(
                "plot_time",
                "Return to a Time Plot",
                "<p>Set the <b>X</b> selector back to <b>t</b>.</p>"
                "<p>This restores a plot against time for the next exercises. Changing axes does not require another simulation.</p>",
                plot,
                lambda: self.registry.presentation(self.tab_id).selected_x_series
                == SimulationSeriesAxisSelection("t"),
                "Time is selected for X.",
            ),
            Step(
                "plot_navigation",
                "Zoom, Pan, and Restore the View",
                "<ol><li>Right-click inside the plot and enable <b>Zoom</b>. Hold the <b>left mouse button</b> and drag a rectangle around part of the plotted data.</li>"
                "<li>Right-click again and disable <b>Zoom</b>. Hold the <b>left mouse button</b> and drag to move the view (pan).</li>"
                "<li>Hold the <b>right mouse button</b> and drag to adjust the axis scales. A right-click without dragging opens the menu.</li>"
                "<li>Enable <b>Auto Range</b> in the same menu to fit the visible data again.</li></ol>"
                "<p>Finish with Auto Range enabled so the next results remain visible.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_legend",
                "Show or Hide the Legend",
                "<p>The <b>legend</b> identifies the plotted curves. You can hide it when you need more space.</p>"
                "<ol><li>Right-click inside the plot and turn <b>Show Legend</b> off.</li>"
                "<li>Open the menu again and turn <b>Show Legend</b> on.</li></ol>"
                "<p>Leave the legend visible for the following exercises.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_log_axes",
                "Try Logarithmic Axes",
                "<p>The plot menu can display either axis on a logarithmic scale. Only positive values can appear on that axis.</p>"
                "<ol><li>Right-click the plot and enable <b>Log X</b>. Look at the time-axis labels. The starting time of zero is not shown.</li>"
                "<li>Turn <b>Log X</b> off, then try <b>Log Y</b>.</li>"
                "<li>Turn <b>Log Y</b> off again.</li></ol>"
                "<p>Leave both axes on their ordinary linear scales.</p>",
                plot,
                manual=True,
            ),
            Step(
                "plot_invert_axes",
                "Reverse Axis Directions",
                "<p>Inverting an axis reverses its displayed direction without changing the simulation results.</p>"
                "<ol><li>Right-click the plot and enable <b>Invert X Axis</b>. Notice that time now increases in the opposite direction.</li>"
                "<li>Turn it off, then try <b>Invert Y Axis</b>.</li>"
                "<li>Turn <b>Invert Y Axis</b> off again.</li></ol>"
                "<p>Leave both inversion options off and <b>Auto Range</b> on.</p>",
                plot,
                manual=True,
            ),
            Step(
                "edit_rate",
                "Edit the Reaction Rate",
                "<p>Reducing the reaction rate constant will slow the conversion of A into B.</p>"
                "<h3>Edit Reactions</h3>"
                "<p>Change <b>kf = 1.0</b> to <b>kf = 0.5</b>.</p>"
                "<p>The edit remains a draft until Kindred checks it when you run the model in the next step.</p>",
                source,
                manual=True,
            ),
            Step(
                "run_edited",
                "Run the Edited Model",
                "<ol><li>Keep <b>preset_set1</b> selected in Initial Conditions.</li>"
                "<li>Click <b>Run</b>.</li>"
                "</ol>"
                "<p>Kindred checks the edited mechanism before simulating it. If an error is reported, correct the text and try Run again.</p>"
                "<p>For this exercise, keep <b>A -&gt; B ; kf = 0.5</b>.</p>",
                run,
                lambda: self._values(self.preset_set, 1.0)
                and self._run_ok(0.5, (self.preset_set,)),
            ),
            Step(
                "add_set",
                "Add Another Starting-condition Set",
                "<p>You can run the same mechanism with several sets of starting concentrations.</p>"
                "<p>Click <b>Add Set</b> in <b>Initial Conditions</b> to add another row.</p>",
                lambda: self._panel(
                    self.bundle.simulation_panel.add_batch_set_btn,
                    "batch_initial_conditions",
                ),
                self._added_set,
            ),
            Step(
                "rename_set",
                "Rename the New Set",
                "<p>Give the new set a name you can recognize in the table and plot legend.</p>"
                "<ol><li>Double-click the new row's name in the <b>Set</b> column.</li>"
                "<li>Enter <b>Higher A</b> and press <b>Enter</b>.</li></ol>",
                initial,
                lambda: self.tab.initial_condition_set(self.extra_set).name
                == "Higher A",
                "The new set is named Higher A.",
            ),
            Step(
                "set_more_a",
                "Give the New Set More A",
                "<h3>Edit the New Row</h3>"
                "<ol><li>Enter <b>A = 2</b> and leave <b>B = 0</b>.</li>"
                "<li>Press <b>Enter</b> to commit the edit.</li>"
                "</ol>"
                "<p>Keep the <b>preset_set1</b> row at <b>A = 1</b> and <b>B = 0</b>.</p>",
                lambda: self._table_target(self.extra_set, "A"),
                lambda: self._values(self.extra_set, 2.0)
                and self._values(self.preset_set, 1.0),
            ),
            Step(
                "select_new_set",
                "Select the New Set to Run",
                "<p><b>preset_set1</b> already has results for <b>kf = 0.5</b>. Only the new set needs a simulation.</p>"
                "<p>Click the new row so that it is the only selected row. <b>Run</b> uses the selected rows, independently of the <b>Slider</b> and <b>Plot</b> checkboxes.</p>"
                "<p>When you want to run several sets together, use <b>Ctrl-click</b> to select their rows (<b>Cmd-click</b> on macOS).</p>",
                initial,
                lambda: self.bundle.initial_actions.selected_run_set_ids()
                == (self.extra_set,),
            ),
            Step(
                "run_new_set",
                "Run the New Starting Conditions",
                "<p>Click <b>Run</b> with only the new row selected. Wait for its simulation to finish.</p>"
                "<p>The existing results for <b>preset_set1</b> remain available. You do not need to run that set again to display it alongside the new result.</p>",
                initial,
                lambda: self._values(self.extra_set, 2.0)
                and self._run_ok(0.5, (self.extra_set,)),
            ),
            Step(
                "plot_sets",
                "Choose Which Sets to Display",
                "<p>The <b>Plot</b> checkbox controls whether a set's available results appear in the plot. It does not select the row for <b>Run</b>.</p>"
                "<p>Starting a run automatically checks <b>Plot</b> for each set being run, even if you previously unchecked it.</p>"
                "<ol><li><b>Plot</b> should already be checked for <b>preset_set1</b> and the new set after their runs. If either is unchecked, check it now. Use the legend to identify their curves.</li>"
                "<li>Uncheck <b>Plot</b> for the new set and watch its curves disappear.</li>"
                "<li>Check it again to restore those curves without running another simulation.</li></ol>",
                lambda: (initial(), plot()),
                manual=True,
            ),
            Step(
                "copy_all",
                "Copy All Results",
                "<p>You can copy Kindred’s results directly to your clipboard, and then paste directly into a spreadsheet for analysis, reporting, or making your own figures.</p>"
                "<ol><li>Keep <b>Plot</b> checked for both <b>preset_set1</b> and <b>Higher A</b>.</li>"
                "<li>Right-click inside the plot and choose <b>Copy All</b>.</li>"
                "<li>Paste into an empty sheet in Excel, LibreOffice Calc, or another spreadsheet application.</li>"
                "<li>Find the results for both sets and the columns for <b>A</b>, <b>B</b>, and <b>total</b>.</li></ol>"
                "<p><b>Copy All</b> includes all available results for the current mechanism tab—even sets and quantities hidden from the plot. It preserves full resolution and ignores plot downsampling and display-unit choices. Time is copied in seconds and species concentrations in M.</p>"
                "<p>Inspect the pasted data, then choose <b>Continue</b>.</p>",
                lambda: (initial(), plot()),
                manual=True,
            ),
            Step(
                "simplify_plot",
                "Invert the Y Selection",
                "<p><b>Invert selection</b> switches checked quantities off and unchecked quantities on. It changes which curves appear, not the axis direction.</p>"
                "<ol><li>Open the <b>Y</b> selector. <b>A</b>, <b>B</b>, and <b>total</b> should all be checked. Check any that are not.</li>"
                "<li>Click <b>Invert selection</b> to clear them, then check <b>A</b>.</li>"
                "<li>Click <b>Invert selection</b> again. Only <b>B</b> and <b>total</b> are now selected.</li>"
                "<li>Click it once more to return to <b>A only</b>, then close the selector.</li></ol>"
                "<p>In <b>Initial Conditions</b>, leave <b>Plot</b> checked only for <b>preset_set1</b>. <b>Plot</b> chooses the sets and <b>Y</b> chooses the quantities.</p>",
                lambda: (initial(), plot()),
                manual=True,
            ),
            Step(
                "copy_plotted",
                "Copy Plotted Data",
                "<p>Your plot should now show only <b>A</b> from <b>preset_set1</b>. You can copy this selection without exporting the other results.</p>"
                "<ol><li>Check that only <b>A</b> is selected for Y and only <b>preset_set1</b> has <b>Plot</b> checked.</li>"
                "<li>Right-click inside the plot and choose <b>Copy Plotted Data</b>.</li>"
                "<li>Paste into another empty sheet and compare it with your earlier <b>Copy All</b> output.</li></ol>"
                "<p><b>Copy Plotted Data</b> follows the plotted sets and quantities, selected X axis, display units, and plot downsampling. Here, it copies the plotted A series from preset_set1.</p>"
                "<p>Zooming does <b>not</b> crop the copied data: points outside the current viewport are included.</p>"
                "<p><b>Copy All</b> would still include both sets and their other quantities, even though they are now hidden.</p>"
                "<p>Inspect the difference, then choose <b>Continue</b>.</p>",
                lambda: (initial(), plot()),
                manual=True,
            ),
            Step(
                "show_slider",
                "Show the Rate Slider",
                "<p><b>Interactive Sliders</b> lets you try parameter changes before applying them to the mechanism.</p>"
                "<ol><li>Open <b>Visible sliders</b> in that panel.</li>"
                "<li>Check <b>k1 — A → B</b>, then close the menu.</li>"
                "</ol>"
                "<p>You can adjust a slider by dragging it or typing in its value field.</p>",
                slider,
                lambda: self.bundle.editor.variable_sliders_widget().variable_visible(
                    "k1"
                ),
            ),
            Step(
                "preview_target",
                "Choose the Preview Target",
                "<p>The <b>focused set</b> is the table’s <b>current row</b>. Click a row to make it current. Several rows can be selected for <b>Run</b>, but only one is current.</p>"
                "<p>Sliders automatically target the current set. Its <b>Slider</b> checkbox does not need to be checked. Those checkboxes pin additional targets, which you will try later.</p>"
                "<ol><li>Leave all <b>Slider</b> checkboxes unchecked.</li>"
                "<li>Click the <b>preset_set1</b> row.</li>"
                "<li>Find its name beside <b>Initial-condition sets</b> in Interactive Sliders.</li></ol>"
                "<p>Choosing another row changes this target. Moving to the slider controls does not change the current set.</p>",
                lambda: (initial(), slider()),
                lambda: not self.registry.presentation(
                    self.tab_id
                ).pinned_slider_set_ids
                and self.bundle.slider_binding.resolve_target_set_ids()
                == (self.preset_set,),
            ),
            Step(
                "preview_quarter",
                "Try a Rate of 0.25",
                "<ol><li>In <b>Interactive Sliders</b>, enter <b>0.25</b> in the <b>k1</b> value field.</li>"
                "<li>Press <b>Enter</b> and wait for the exact Preview.</li>"
                "</ol>"
                "<p>This previews the proposed rate without applying it to the mechanism.</p>",
                slider,
                manual=True,
            ),
            Step(
                "preview_reference",
                "Preview and Reference",
                "<ul><li><b>Preview</b> shows the proposed parameter change.</li>"
                "<li><b>Reference</b> shows the accepted model.</li>"
                "</ul>"
                "<p>Compare their curves. If Reference is hidden, right-click the plot and enable <b>Show Reference</b>.</p>",
                plot,
            ),
            Step(
                "reset_trial",
                "Reset This Trial",
                "<p>Click <b>Reset</b> in <b>Interactive Sliders</b>.</p>"
                "<p>This discards the slider changes. The accepted reaction rate constant remains <b>0.5</b>.</p>",
                slider,
                manual=True,
            ),
            Step(
                "pin_slider_set",
                "Pin an Additional Slider Target",
                "<p>The <b>Slider</b> checkbox keeps a set included when you change the current row. Slider targets are the <b>current row plus any checked Slider rows</b>.</p>"
                "<ol><li>Check <b>Slider</b> for the new set, then click the <b>preset_set1</b> row.</li>"
                "<li>Check that both set names appear beside <b>Initial-condition sets</b> in Interactive Sliders.</li>"
                "<li>Check <b>Plot</b> for both sets and keep only <b>A</b> selected for Y.</li></ol>"
                "<p><b>Slider</b> chooses preview targets. <b>Plot</b> controls which results you see.</p>",
                lambda: (initial(), slider()),
                lambda: self.extra_set
                in self.registry.presentation(self.tab_id).pinned_slider_set_ids
                and self.bundle.batch_model.set_id_for_live_index(
                    self.bundle.batch_table.currentIndex()
                )
                == self.preset_set
                and set(self.bundle.slider_binding.resolve_target_set_ids())
                == {self.preset_set, self.extra_set}
                and set(self.registry.presentation(self.tab_id).plot_set_ids)
                == {self.preset_set, self.extra_set}
                and self.registry.presentation(self.tab_id).selected_y_series
                == (SimulationSeriesAxisSelection("A"),),
            ),
            Step(
                "drag_slider",
                "Drag the Rate Slider",
                "<p>Try changing <b>k1</b> by dragging its slider while both sets are preview targets.</p>"
                "<ol><li>Hold the <b>left mouse button</b> on the slider handle and drag it a short distance.</li>"
                "<li>Watch the value and the plotted previews respond as you move.</li>"
                "<li>Release the handle and wait for the completed previews. Use the legend to identify the two sets.</li></ol>"
                "<p>There is no exact value to reach here. These changes remain a preview until you choose <b>Apply</b>.</p>",
                lambda: (slider(), plot()),
                manual=True,
            ),
            Step(
                "fine_slider",
                "Make Fine Slider Adjustments",
                "<p><b>Fine</b> makes mouse dragging ten times slower, keeping the cursor attached to the slider handle. It also provides smaller keyboard adjustments.</p>"
                "<ol><li>Enable <b>Fine</b> in Interactive Sliders.</li>"
                "<li>Drag the <b>k1</b> handle a short distance and release it.</li>"
                "<li>With the slider focused, press the <b>Left</b> or <b>Right</b> arrow key to make small adjustments. Watch the value change.</li>"
                "<li>Turn <b>Fine</b> off when finished.</li></ol>"
                "<p>The next step uses the value field to enter an exact value before applying it.</p>",
                slider,
                manual=True,
            ),
            Step(
                "preview_three_quarters",
                "Try a Rate of 0.75",
                "<ol><li>Change <b>k1</b> to <b>0.75</b> in the slider value field.</li>"
                "<li>Press <b>Enter</b>.</li>"
                "<li>Wait for both previews to finish, then use the legend to identify each set.</li>"
                "</ol>"
                "<p>Both sets preview the same proposed parameter change. You will apply it in the next step.</p>",
                slider,
                manual=True,
            ),
            Step(
                "apply_preview",
                "Apply the Change",
                "<p>Click <b>Apply</b> in <b>Interactive Sliders</b>.</p>"
                "<p>This applies the Preview’s rate and results to the accepted model.</p>"
                "<p>If Preview is still computing, wait for it to finish and click Apply again.</p>",
                slider,
                manual=True,
            ),
            Step(
                "open_templates",
                "Open Template Manager",
                "<p>A <b>mechanism template</b> stores reaction source for reuse. A project file saves the wider workspace setup.</p>"
                "<p>Open <b>Examples → Template Manager</b>.</p>",
                manager,
                lambda: self._dialog(TemplateManagerDialog) is not None,
            ),
            Step(
                "create_template",
                "Create from Your Current Mechanism",
                "<ol><li>In Template Manager, choose <b>Create from Current</b>.</li>"
                "<li>Review the mechanism text in the dialog that opens.</li>"
                "</ol>"
                "<p>The next step saves this source as a template.</p>",
                manager,
                lambda: self._dialog(TemplateEditorDialog) is not None,
            ),
            Step(
                "save_template",
                "Name and Create the Template",
                "<ol><li>Enter a name, such as <b>My first model</b>, and a short description. If the name already exists, choose another.</li>"
                "<li>Choose <b>Create</b>.</li>"
                "<li>Dismiss the success message.</li>"
                "<li>Check that the new entry appears in <b>Template Manager</b>.</li>"
                "</ol>"
                "<p>Confirm the template was created before continuing.</p>",
                manager,
                manual=True,
            ),
            Step(
                "close_templates",
                "Return to the Workspace",
                "<p>Close <b>Template Manager</b> to return to the workspace.</p>"
                "<p><b>Create from Current</b> captures mechanism source. It does not save the whole project or copy the concentrations from the Initial Conditions table.</p>",
                manager,
                lambda: self._dialog((TemplateManagerDialog, TemplateEditorDialog))
                is None,
            ),
            Step(
                "create_reuse_tab",
                "Create a Tab for Template Reuse",
                '<p>Click <b>"+"</b> beside the mechanism tabs to create another tab.</p>'
                "<p>You will apply your template in this new tab.</p>",
                self._tab_target,
                lambda: self._new_tab(reuse=True),
            ),
            Step(
                "apply_template",
                "Apply Your Saved Template",
                "<ol><li>Open <b>Examples → Template Manager</b>.</li>"
                "<li>Select the template you just created.</li>"
                "<li>Choose <b>Apply</b>.</li>"
                "</ol>"
                "<p>Check that the manager closes and the mechanism source appears in the new tab.</p>",
                manager,
                manual=True,
            ),
            Step(
                "reuse_values",
                "Set Starting Values for the Reused Model",
                "<p>A new tab starts with zero concentrations. Applying a template created from mechanism source does not restore the previous table values.</p>"
                "<h3>Set the Starting Concentrations</h3>"
                "<ol><li>In the <b>set1</b> row, enter <b>A = 1</b> and <b>B = 0</b>.</li>"
                "<li>Press <b>Enter</b> after editing.</li>"
                "<li>Select that row for Run.</li>"
                "</ol>",
                lambda: self._table_target(self.reuse_set, "A"),
                lambda: self._values(self.reuse_set, 1.0)
                and self.bundle.initial_actions.selected_run_set_ids()
                == (self.reuse_set,),
            ),
            Step(
                "run_reused",
                "Run the Reused Model",
                "<p>Click <b>Run</b> and wait for the simulation to finish.</p>"
                "<p>The reused model should run from <b>A = 1</b> and <b>B = 0</b>, with the rate constant you applied earlier: <b>0.75</b>.</p>",
                run,
                lambda: self._values(self.reuse_set, 1.0)
                and self._run_ok(0.75, (self.reuse_set,)),
            ),
            Step(
                "save_project",
                "Save the Project",
                "<p>A <b>.kin project</b> saves the mechanism tabs and their setup.</p>"
                "<ol><li>Choose <b>File → Save</b>, or press <b>Ctrl+S</b> (<b>Cmd+S</b> on macOS).</li>"
                "<li>Choose a filename and destination, then save.</li>"
                "<li>Confirm the file was saved before continuing. If you cancelled, try Save again.</li>"
                "</ol>",
                lambda: self._tab_target(),
                manual=True,
            ),
            Step(
                "finish",
                "Getting Started Complete",
                "<p>You have reached the end of Getting Started.</p>"
                "<h3>Returning to Your Work</h3>"
                "<ul><li>Use <b>File → Open</b> to open a saved <b>.kin project</b>.</li>"
                "<li>Use <b>Examples → Template Manager</b> to apply a saved template to another model.</li>"
                "</ul>"
                "<p>Choose <b>Finish</b> to close the tutorial.</p>",
                lambda: None,
            ),
        )
