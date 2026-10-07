"""Shared practice setup and observations for the focused lessons."""

from __future__ import annotations

from math import isclose

from PySide6 import QtCore, QtWidgets

from kindred.gui.parameter_enumeration import enumerate_parameter_coordinates
from kindred.gui.tutorials.controller import Lesson, Step
from kindred.gui.tutorials.overlay import Highlight, menu_target, scroll_into_view


class PracticeLesson(Lesson):
    """Observe user-led work in fresh practice tabs without changing the model."""

    _preset_name = "M1"
    _practice_species = ("A", "B")
    # Demonstrated rates distinguish the preset from neighbours that share its
    # starting concentrations (e.g. M2 for M1, M6 for M5).
    _preset_rates: dict[str, float] = {"k1": 1.0}

    _reuse_steps: frozenset[str] = frozenset()
    _result_steps: frozenset[str] = frozenset()
    _extra_steps: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        workspace,
        registry,
        surfaces,
        examples_menu,
        file_menu,
        project_controller,
        jobs,
    ):
        super().__init__()
        self.workspace, self.registry, self.surfaces = workspace, registry, surfaces
        self.examples_menu, self.jobs = examples_menu, jobs
        self.file_menu = file_menu
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

    def _missing(self, message, step):
        return (
            message
            + " Return to setup, or use Skip Step to continue without this exercise.",
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
                "A practice tab was closed. Start the lesson again to create fresh practice work."
            )
            return ("The practice tab was closed.", "", None)
        if name in {"welcome", "learn", "create_tab", "save_project", "finish"}:
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
                "The lesson is paused while another mechanism tab is active.",
                "Return to the lesson tab",
                lambda: self.workspace.tabActivated.emit(self.tab_id),
            )
        if name in {"name_tab", "load_preset"}:
            return None
        sets = {item.set_id for item in self.tab.initial_condition_sets}
        if name in self._reuse_steps:
            if name != "create_reuse_tab" and self.reuse_set not in sets:
                self.reuse = self.reuse_set = None
                return self._missing(
                    "The set used for template reuse was removed.", "create_reuse_tab"
                )
            return None
        if self.preset_set is None or self.preset_set not in sets:
            self.preset_set = None
            return self._missing(
                f"Load {self._preset_name} to supply the mechanism and starting values.", "load_preset"
            )
        if name in self._extra_steps and (
            self.extra_set is None or self.extra_set not in sets
        ):
            self.extra_set = None
            return self._missing("This exercise needs a second set.", "add_set")
        if name in self._extra_steps and name not in {
            "rename_extra",
            "configure_extra",
        }:
            if self.tab.initial_condition_set(self.extra_set).name != "Higher A":
                return self._missing(
                    "Name the new set Higher A before this exercise.", "rename_extra"
                )
        if name in self._result_steps:
            repo = self.registry.tab_runtime(self.tab_id).result_repository
            if repo.current_result(self.preset_set) is None:
                return self._missing(
                    "Run the practice model before exploring the results.", "run_first"
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
        return (
            self.preset_set is not None
            and self._values(self.preset_set, 1.0)
            and self._preset_rates_loaded()
        )

    def _preset_rates_loaded(self) -> bool:
        values = {
            item.name: float(item.value)
            for item in enumerate_parameter_coordinates(self.tab.prepared_dsl)
        }
        return all(
            name in values and isclose(values[name], rate, rel_tol=1e-8)
            for name, rate in self._preset_rates.items()
        )

    def _project_changed(self, replaced: bool) -> None:
        if replaced:
            self.ended.emit(
                "The project was replaced. Start another lesson from Help when ready."
            )
        else:
            self.changed.emit()

    def _run_dispatched(self, tab_id, set_ids) -> None:
        if tab_id == self.tab_id and self.step.name in {
            "run_first",
            "run_edited",
            "run_new_set",
        }:
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
        return dict(item.values) == {
            species: amount if species == "A" else 0.0
            for species in self._practice_species
        }

    def _run_ok(self, rates: dict[str, float], identities: tuple) -> bool:
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
            parameters = {name: value for name, value, _unit in result.parameters}
            if any(
                name not in parameters or not isclose(parameters[name], rate, rel_tol=1e-8)
                for name, rate in rates.items()
            ):
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

    def _panel(self, widget, identity: str) -> Highlight:
        def reveal():
            self.workspace.reveal_dock(identity)
            scroll_into_view(widget)

        return Highlight(widget, reveal=reveal)

    def _tab_target(self) -> Highlight:
        # The central layout places the tab strip before the plot stack.
        return Highlight(self.workspace.centralWidget().layout().itemAt(0).widget())

    def _save_target(self) -> Highlight:
        dialog = self._dialog(QtWidgets.QFileDialog)
        return (
            Highlight(dialog)
            if dialog is not None
            else menu_target(
                self.file_menu,
                lambda action: action.objectName() == "saveProjectAction",
            )
        )

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

    def _table_column_targets(self, column: int) -> tuple[Highlight, ...]:
        table, model = self.bundle.batch_table, self.bundle.batch_model
        header = table.horizontalHeader()

        def column_rect(widget):
            return QtCore.QRect(
                header.sectionViewportPosition(column),
                0,
                header.sectionSize(column),
                widget.height(),
            )

        def reveal():
            self.workspace.reveal_dock("batch_initial_conditions")
            table.scrollTo(
                model.index(0, column),
                QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
            )

        return (
            Highlight(header.viewport(), lambda: column_rect(header.viewport()), reveal),
            Highlight(table.viewport(), lambda: column_rect(table.viewport()), reveal),
        )

    def _plot_target(self) -> Highlight:
        return Highlight(self.workspace.plot_for_tab(self.tab_id))

    def _initial_target(self) -> Highlight:
        return self._panel(self.bundle.simulation_panel, "batch_initial_conditions")

    def _run_targets(self) -> tuple[Highlight, ...]:
        return (
            self._panel(self.bundle.simulation_panel.run_btn, "batch_initial_conditions"),
            self._panel(self.bundle.editor.run_btn, "mechanism"),
        )

    def _preset_target(self) -> Highlight:
        return menu_target(self.examples_menu, lambda action: action.data() == self._preset_name)

    @staticmethod
    def _dialog(kind):
        dialog = QtWidgets.QApplication.activeModalWidget()
        return dialog if isinstance(dialog, kind) else None

    def _slider_target(self) -> Highlight:
        return self._panel(self.bundle.slider_panel, "interactive_sliders")

    def _setup_steps(self, *, introduce_workspace: bool = False) -> tuple[Step, ...]:
        introduction = (
            (
                "<p>A mechanism tab contains a reaction mechanism and the settings for running simulations. "
                "A fresh workspace contains a tab named <b>Mechanism 1</b>.</p>"
            )
            if introduce_workspace
            else (
                f"<p>{self.description}</p>"
                "<p><b>Previous</b> revisits instructions without undoing work. <b>Skip Step</b> moves on without checking completion. "
                "Turn off <b>Advance Automatically</b> to choose when to move on. "
                "<b>−</b> minimizes the tutorial and <b>×</b> closes it. Your practice work stays in the project.</p>"
            )
        )
        return (
            Step(
                "create_tab",
                "Create a Practice Tab",
                introduction
                + "<p>Click <b>+</b> beside the mechanism tabs, above the plot. Use the new tab for these exercises. "
                "Presets and templates load into the selected tab.</p>",
                self._tab_target,
                self._new_tab,
                "The practice tab is ready.",
            ),
            Step(
                "load_preset",
                "Load a Preset",
                "<p>A preset provides a mechanism and starting values that you can use and modify.</p>"
                "<p>Open <b>Examples → Preset Mechanisms → M1: A -&gt; B</b>.</p>"
                "<p>Follow the highlighted menu entries. There is no need to press Next between menus.</p>"
                "<p>M1 adds <b>preset_set1</b> with starting concentrations "
                "<b>A = 1</b> and <b>B = 0</b>. A set is a named group of starting values for one simulation.</p>"
                "<p>The original <b>set1</b> row can stay. Use preset_set1.</p>",
                self._preset_target,
                self._loaded,
                "M1 and the starting values are ready.",
            ),
        )

    def _prepare_run_step(self) -> Step:
        return Step(
            "prepare_run",
            "Choose the Starting Values and Time",
            "<p>In <b>Initial Conditions</b>, select only <b>preset_set1</b>. Check <b>A = 1</b> and <b>B = 0</b>.</p>"
            "<p>Set <b><i>t</i><sub>start</sub></b> to <b>0</b>, <b><i>t</i><sub>end</sub></b> to <b>5</b> seconds, and <b>Points</b> to <b>100</b>. "
            "Press <b>Enter</b> after each edit.</p>"
            "<p><b><i>t</i><sub>start</sub></b> and <b><i>t</i><sub>end</sub></b> set the simulation interval. <b>Points</b> sets the number of evenly spaced "
            "output samples, including both endpoints. It does not set the internal integration steps. "
            "Intervention boundary rows are additional.</p>"
            "<p><b>Solver</b> selects the numerical method. Either available solver works for this exercise.</p>"
            "<p>Selected rows choose what to run. The <b>Plot</b> checkbox chooses which results to show.</p>",
            self._initial_target,
            lambda: self._values(self.preset_set, 1.0)
            and self.bundle.initial_actions.selected_run_set_ids() == (self.preset_set,)
            and not self.bundle.simulation_panel.has_run_defaults_draft
            and self.tab.resolved_execution_settings(self.preset_set).t_start == 0
            and self.tab.resolved_execution_settings(self.preset_set).t_end == 5
            and self.tab.resolved_execution_settings(self.preset_set).integration_points
            == 100,
            "The starting values and simulation time are ready.",
        )

    def _first_run_step(self) -> Step:
        return Step(
            "run_first",
            "Run the Model",
            "<p>Keep only <b>preset_set1</b> selected, then click <b>Run</b> in <b>Initial Conditions</b>.</p>"
            "<p>Wait for the simulation to finish. The results will appear as curves in the main plot.</p>"
            "<p>You can also click <b>Run</b> in the <b>Mechanism</b> panel. Both buttons run the same selected initial-condition sets.</p>",
            self._run_targets,
            lambda: self._values(self.preset_set, 1.0)
            and self._run_ok({"k1": 1.0}, (self.preset_set,)),
            "The simulation is complete.",
        )

    def _add_set_step(self) -> Step:
        return Step(
            "add_set",
            "Add a Second Set",
            "<p>Click <b>Add Set</b> in <b>Initial Conditions</b>. The new row will let you try a different starting concentration "
            "with the same reactions.</p>",
            lambda: self._panel(
                self.bundle.simulation_panel.add_batch_set_btn,
                "batch_initial_conditions",
            ),
            self._added_set,
            "A second practice set is available.",
        )

    def _rename_extra_step(self) -> Step:
        return Step(
            "rename_extra",
            "Name the New Set",
            "<p>Double-click the name in the new row, enter <b>Higher A</b>, and press <b>Enter</b>.</p>"
            "<p>Set names identify results in the table and plot legend.</p>",
            lambda: self._table_target(self.extra_set),
            lambda: self.tab.initial_condition_set(self.extra_set).name == "Higher A",
            "The new set is named Higher A.",
        )

    def _configure_extra_step(self) -> Step:
        return Step(
            "configure_extra",
            "Give the New Set More A",
            "<p>In the new row, enter <b>A = 2</b> and <b>B = 0</b>. Press <b>Enter</b> after each edit. "
            "Leave preset_set1 at A = 1 and B = 0.</p>",
            lambda: self._table_target(self.extra_set, "A"),
            lambda: self._values(self.extra_set, 2.0)
            and self._values(self.preset_set, 1.0),
            "The starting concentrations are ready.",
        )

    def _save_step(self) -> Step:
        return Step(
            "save_project",
            "Save Your Project",
            "<p>A <b>.kin project</b> stores your mechanism tabs, reaction text, notes, initial conditions, "
            "simulation settings, imported datasets, and plot appearance settings. A project can contain several mechanism tabs. "
            "Imported datasets are shared across those tabs.</p>"
            "<p>Choose <b>File → Save Project</b>.</p>"
            "<p>For a new project, choose a filename and destination. "
            "If the project is already saved, this updates that file. "
            "Use <b>Save Project As…</b> when you want a separate copy.</p>"
            "<p>Check that saving completed. If you canceled the dialog, try again before choosing <b>Continue</b>.</p>",
            self._save_target,
            manual=True,
        )

    def _finish_step(self) -> Step:
        return Step(
            "finish",
            "Lesson Complete",
            "<p>Your practice tabs stay in the project. You can keep exploring or save your work.</p>"
            "<p>Use <b>File → Open Project…</b> to reopen a saved <b>.kin file</b>. "
            "Open <b>Examples → Template Manager…</b> to reuse a saved mechanism template.</p>"
            "<p>Choose <b>Finish</b> to close this lesson. Open <b>Help → Interactive Tutorials…</b> to choose another. "
            "Each lesson includes its own setup.</p>",
            lambda: None,
        )
