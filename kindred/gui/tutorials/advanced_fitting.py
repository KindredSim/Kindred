"""The Advanced Fitting lessons: M11 fitted to the sheets of one workbook, set up once for the series."""

from dataclasses import dataclass, replace
from functools import cache
from math import isclose

from PySide6 import QtCore, QtWidgets

from kindred.core.simulator.dsl_parse import prepare_dsl_request_set
from kindred.gui.fitting.constants import DIFFERENTIAL_EVOLUTION_METHOD
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.parameter_enumeration import enumerate_parameter_coordinates
from kindred.gui.tutorials.controller import Step
from kindred.gui.tutorials.overlay import Highlight
from kindred.gui.tutorials.practice_lesson import IMPORT_DIALOGS, PracticeLesson
from kindred.gui.widgets.dataset_mapping_batch_dialog import DatasetMappingBatchDialog
from kindred.gui.widgets.import_config_dialog import ImportConfigDialog
from kindred.io.resources import get_preset_mechanism_source

_FILE = "advanced_fitting.xlsx"
_THREE_RUNS = ((0.1, 0.2, 0.2), (0.2, 0.2, 0.2), (0.4, 0.2, 0.2))
# The starting A, B, and C that the row at time 0 of each sheet records.
_STARTS = {
    **{f"{group}{number}": starts for group in ("bounds", "base", "stock", "k2runs")
       for number, starts in enumerate(_THREE_RUNS, 1)},
    **{f"profile{number}": starts for number, starts in enumerate((
        (0.2, 0.2, 0.2), (0.1, 0.2, 0.2), (0.2, 0.1, 0.2), (0.2, 0.2, 0.1),
        (0.4, 0.2, 0.2), (0.2, 0.4, 0.2), (0.2, 0.2, 0.4),
    ), 1)},
}
_WATER = 0.05


def _columns(sheet: str) -> tuple[str, ...]:
    """The columns of a sheet: bounds1-3 record A and total_PB over time, base1-3 record PB and PH
    instead of their sum, and the others also record B, C, and Int."""
    if sheet.startswith("bounds"):
        return ("A", "B", "C", "Water", "total_PB")
    if sheet.startswith("base"):
        return ("A", "B", "C", "Int", "Water", "PB", "PH")
    return ("A", "B", "C", "Int", "Water", "total_PB")


# Every sheet is recorded in seconds and M, without a units row.
_UNITS = {sheet: ("s", dict.fromkeys(_columns(sheet), "M")) for sheet in _STARTS}
_DE_EVALUATIONS = 3000
# The species that lessons 2 to 5 compare: every species measured over time.
_OVER_TIME = frozenset({"A", "B", "C", "Int", "total_PB"})


@dataclass(frozen=True)
class _Row:
    """A row of the Species Table: whether the fit compares the species, its Vary IC choice and
    Lower, and whether it varies with other datasets."""

    included: bool
    vary: str | None
    lower: str
    shared: bool


@cache
def _preset_equations(preset: str):
    """The preset's equations, which change with its reactions but not with its values."""
    return prepare_dsl_request_set(get_preset_mechanism_source(preset).reactions_text).symbolic_equations_text()


class AdvancedFittingLesson(PracticeLesson):
    """A lesson of the series. The first one started runs the shared setup; the others find
    its tab and start after it. A lesson applies its fit only to a tab copy, so the setup
    tab keeps the rate constants of M11."""

    _preset_name = "M11"
    _preset_rates = {"kf1": 1.0, "kr1": 0.01, "k2": 0.01, "k3": 100.0}
    _needs_preset_set = False
    _sample_folder = "AdvancedFitting"
    _sample_file = _FILE
    _sheet_units = _UNITS
    # The values the lesson fits.
    _fitted: tuple[str, ...] = ()
    # The sheets the lesson fits, as the lesson names them, and the species it compares.
    _sheets: tuple[str, ...] = ()
    _named = ""
    _targets: frozenset[str] = _OVER_TIME
    # The first step after the shared setup.
    _first_step = ""
    # Steps taught inside the Fitting Window, and every step that needs the workbook mapped.
    _fit_steps: frozenset[str] = frozenset()
    _mapped_steps: frozenset[str] = frozenset()

    def __init__(self, **owners):
        super().__init__(**owners)
        # The Fitting Window last seen, and the tabs present when it opened.
        self._seen_fit: FittingWindow | None = None
        self._tabs_at_open: set[str] = set()
        # Whether the Fitting Window applied the lesson's final fit.
        self._applied = False
        # The Species Table rows last seen for each dataset the open Fitting Window showed.
        self._rows_seen: dict[str, dict[str, _Row]] = {}
        # Whether the open Fitting Window was reset to a clean setup.
        self._reset_seen = False
        self._take_over()

    def _mechanism_loaded(self) -> bool:
        """M11 with its reactions and its rate constants, so that a tab the user changed is not taken for the
        setup."""
        return super()._mechanism_loaded() and (
            self.tab.prepared_dsl.symbolic_equations_text() == _preset_equations(self._preset_name)
        )

    def _take_over(self) -> None:
        """Adopt the active tab when it holds the shared setup, otherwise the first tab that does:
        M11 with the rate constants of the preset, and every sheet mapped to its own set from its
        data. A tab with fitted rate constants does not qualify."""
        document = self.registry.document
        everything = {record.dataset_id for record in document.datasets}
        active = self.registry.active_tab_id
        for identity in sorted((tab.tab_id for tab in document.tabs), key=lambda identity: identity != active):
            # A candidate is read as the practice tab, with the datasets mapped to it as the ones
            # the setup imported.
            self.practice = identity
            self._datasets_before = everything - {dataset_id for dataset_id, _link in self.tab.dataset_links}
            if self._mechanism_loaded() and self._mapped():
                bundle = self.bundle
                for signal in (
                    bundle.editor_binding.draftStateChanged,
                    bundle.initial_actions.initialValuesChanged,
                    bundle.initial_actions.selectedRunSetIdsChanged,
                    bundle.slider_binding.draftStateChanged,
                    bundle.slider_binding.targetSetIdsChanged,
                ):
                    signal.connect(self.changed)
                self.position = next(index for index, step in enumerate(self.steps) if step.name == self._first_step)
                return
        self.practice = None
        # Datasets present when the lesson started.
        self._datasets_before = everything

    def blocked(self):
        self._observe_fit()
        missing = super().blocked()
        if missing is not None:
            return missing
        name = self.step.name
        if not self._workbook() and (
            name in self._mapped_steps
            or name == "accept_mapping"
            or (name == "import_sheets" and self._dialog(ImportConfigDialog) is None)
        ):
            return self._missing(
                "This exercise needs every sheet of advanced_fitting.xlsx imported with all its columns in seconds and M.",
                "open_workbook",
            )
        if name in self._mapped_steps and not self._mapped():
            return self._missing(
                "This exercise needs each workbook dataset mapped to its own set from its data.", "accept_mapping"
            )
        if name in self._fit_steps and self._fit_window() is None and not self._applied:
            message, label, _ = self._missing("This exercise needs the Fitting Window open.", "open_fit")
            return message, label, self._reopen_fit
        return None

    def _fit_closed(self, *_args) -> None:
        self._seen_fit = None
        if self.step.name in self._fit_steps and not self._applied:
            self._reopen_fit()

    def _reset_requested(self, target: str, part: str) -> None:
        """The open Fitting Window reset its whole setup to the clean one."""
        self._reset_seen = self._reset_seen or (target, part) == ("initial", "both")

    # -- observations ----------------------------------------------------------
    @staticmethod
    def _starts(run: str) -> dict[str, float]:
        a, b, c = _STARTS[run]
        return {"A": a, "B": b, "C": c, "Int": 0.0, "Water": _WATER, "PB": 0.0, "PH": 0.0, "BOH": 0.0}

    @classmethod
    def _starts_from(cls, values: dict[str, float], run: str) -> bool:
        return all(
            isclose(values.get(species, -1.0), value, abs_tol=1e-12) for species, value in cls._starts(run).items()
        )

    def _mapped(self, tab=None) -> bool:
        """Each sheet maps its columns by name to its own set of the tab, the lesson tab unless given,
        which starts from its row at time 0."""
        found = self._workbook()
        tab = self.tab if tab is None else tab
        sets = {item.set_id: item for item in tab.initial_condition_sets}
        for run, dataset_id in found.items():
            link = tab.dataset_link(dataset_id)
            if (
                link is None
                or link.set_id not in sets
                or dict(link.observation_mapping) != self._expected_mapping(dataset_id)
                or not self._starts_from(dict(sets[link.set_id].values), run)
            ):
                return False
        return bool(found)

    def _observe_fit(self) -> None:
        """Record whether an open Fitting Window of the lesson tab applied the lesson's final fit. The
        window blocks every other edit, so while it is open a new tab can only come from Apply Fit to
        Tab Copy, which needs a current result."""
        window = self._fit_window()
        if window is None or self.tab_id is None or self.registry.active_tab_id != self.tab_id:
            return
        tabs = {tab.tab_id for tab in self.registry.document.tabs}
        if window is not self._seen_fit:
            self._seen_fit, self._tabs_at_open, self._rows_seen, self._reset_seen = window, tabs, {}, False
            window.destroyed.connect(self._fit_closed)
            window.resetRequested.connect(self._reset_requested)
        shown = self._shown_rows(window)
        if shown is not None:
            dataset, rows = shown
            before = self._rows_seen.get(dataset, {})
            if any(rows[species].shared != row.shared for species, row in before.items() if species in rows):
                # Vary Global can also add other datasets to a group or take them out of it, so their
                # rows count again only once each is shown.
                self._rows_seen = {}
            self._rows_seen[dataset] = rows
        # A copy of an earlier result does not count, now or later.
        new, self._tabs_at_open = tabs - self._tabs_at_open, tabs
        self._applied = self._applied or self._fit_applied(bool(new))

    def _fit_applied(self, new_tab: bool) -> bool:
        """Whether the open Fitting Window just applied the lesson's final fit; a new tab is a copy
        with a fit applied."""
        return new_tab and self._final_setup()

    def _datasets(self) -> set[str]:
        """The datasets of the sheets the lesson fits."""
        # Mapped and fit steps require the workbook.
        found = self._workbook()
        return {found[sheet] for sheet in self._sheets}

    def _included_only(self) -> bool:
        """The open Fitting Window includes exactly the datasets the lesson fits."""
        # Fit-step checks run only while the window is open (or after Apply short-circuits them).
        datasets = self._fit_window().findChild(QtWidgets.QListWidget, "global_fit_unified_dataset_list")
        wanted = self._datasets()
        if not wanted:
            return False
        checked = {
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in (datasets.item(row) for row in range(datasets.count()))
            if item.checkState() == QtCore.Qt.CheckState.Checked
        }
        return checked == wanted

    @staticmethod
    def _shown_rows(window: FittingWindow) -> tuple[str, dict[str, _Row]] | None:
        """The dataset the Species Table shows and its rows by species. The table shows one dataset
        at a time, and only that one can be changed or copied to the others."""
        datasets = window.findChild(QtWidgets.QListWidget, "global_fit_unified_dataset_list")
        table = window.findChild(QtWidgets.QTableWidget, "global_fit_unified_species_table")
        current = datasets.currentItem()
        if current is None:
            return None
        headers = [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
        include, species, vary, lower, shared = (
            headers.index(title) for title in ("Include", "Species", "Vary IC", "Lower", "Vary Global")
        )

        def checked(row: int, column: int) -> bool:
            item = table.item(row, column)
            return (
                bool(item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                and item.checkState() == QtCore.Qt.CheckState.Checked
            )

        rows = {}
        for row in range(table.rowCount()):
            # Every species row has all its cells and a Vary IC combo.
            name, choice, bound = table.item(row, species), table.cellWidget(row, vary), table.item(row, lower)
            rows[name.text()] = _Row(
                checked(row, include),
                choice.currentData(),
                bound.text(),
                checked(row, shared),
            )
        return current.data(QtCore.Qt.ItemDataRole.UserRole), rows

    def _targets_chosen(self) -> bool:
        """Each dataset the lesson fits was last seen comparing exactly the lesson's species."""
        datasets = self._datasets()
        return bool(datasets) and all(
            frozenset(species for species, row in self._rows_seen.get(dataset, {}).items() if row.included)
            == self._targets
            for dataset in datasets
        )

    def _roster_table(self) -> QtWidgets.QTableWidget:
        return self._fit_window().findChild(QtWidgets.QTableWidget, "fit_parameter_roster")

    def _roster(self) -> dict[str, dict[str, QtWidgets.QTableWidgetItem]]:
        """Parameter -> its row of the open Fitting Window's parameter table, by column title."""
        roster = self._roster_table()
        headers = [roster.horizontalHeaderItem(column).text() for column in range(roster.columnCount())]
        rows = {}
        for row in range(roster.rowCount()):
            cells = {header: roster.item(row, column) for column, header in enumerate(headers)}
            name = cells["Parameter"]
            # Rows of single datasets follow their parameter row.
            if name.data(QtCore.Qt.ItemDataRole.UserRole + 1) is None:
                rows[name.data(QtCore.Qt.ItemDataRole.UserRole)] = cells
        return rows

    def _varied(self, names: tuple[str, ...] | None = None) -> bool:
        """The open Fitting Window varies these values, the lesson's unless given, and nothing else."""
        roster = self._roster()
        varied = {
            name for name, cells in roster.items()
            if cells["Vary"].checkState() == QtCore.Qt.CheckState.Checked
        }
        return varied == set(self._fitted if names is None else names)

    def _lesson_fit(self, names: tuple[str, ...] | None = None) -> bool:
        """The open Fitting Window has a current result for the lesson's datasets and species, with
        these values varied, the lesson's unless given."""
        return self._included_only() and self._targets_chosen() and self._varied(names) and self._fit_result()

    def _final_setup(self) -> bool:
        """The open Fitting Window holds the setup whose result the lesson applies to a tab copy."""
        return False

    # -- targets ---------------------------------------------------------------
    def _parameters_tab(self) -> tuple[Highlight, ...]:
        """The Parameters tab of the open Fitting Window."""
        window = self._fit_window()
        if window is None:
            return ()
        bar = window.findChild(QtWidgets.QTabBar, "global_fit_top_tabs")
        index = next(index for index in range(bar.count()) if bar.tabText(index) == "Parameters")
        return (Highlight(bar, lambda: bar.tabRect(index)),)

    def _setup(self) -> tuple[Step, ...]:
        create_tab, load_preset = self._setup_steps()
        return (
            replace(
                create_tab,
                instruction=create_tab.instruction
                + "<p>The Advanced Fitting lessons share one setup. Once a tab holds it, the other lessons of the "
                "series start on that tab after the setup.</p>",
            ),
            replace(
                load_preset,
                title="Load M11",
                instruction="<p>Open <b>Examples → Preset Mechanisms → M11: PB cascade</b>.</p>"
                "<p>Follow the highlighted menu entries. There is no need to click <b>Next</b> between menus.</p>"
                "<p>M11 has three steps: A + B ⇌ Int + Water with the rate constants <b>kf1</b> and <b>kr1</b>, "
                "Int + C → PB with <b>k2</b>, and PB + Water → PH + BOH with <b>k3</b>. The let quantity "
                "<b>total_PB = [PB] + [PH]</b> is the measured sum of PB and PH. In these lessons, the sets come "
                "from the data.</p>",
                complete=self._mechanism_loaded,
                success="M11 is loaded.",
            ),
            Step(
                "save_file",
                "Save the Tutorial File",
                "<p>The Advanced Fitting lessons use <b>advanced_fitting.xlsx</b>, an Excel workbook supplied with "
                "Kindred.</p>"
                "<p>Click <b>Save Tutorial File…</b> and choose a folder. Kindred creates an "
                "<b>AdvancedFitting</b> folder there and copies the workbook into it.</p>",
                lambda: None,
                # A tab that already holds the setup needs no file.
                lambda: self._saved() or self._mapped(),
                "The workbook is ready.",
                waiting="Save the tutorial file to continue.",
                action=("Save Tutorial File…", self._save_file),
            ),
            Step(
                "about_workbook",
                "What Is in the Workbook",
                "<p>advanced_fitting.xlsx holds 19 runs of M11, one per sheet, in a group for each lesson:</p>"
                "<ul><li><b>bounds1</b> to <b>bounds3</b>: Bounds and Solvers</li>"
                "<li><b>base1</b> to <b>base3</b>: Calculated Quantities and Constraints, and Fitting Against a "
                "Measured Quantity</li>"
                "<li><b>stock1</b> to <b>stock3</b>: Varying Initial Conditions</li>"
                "<li><b>k2runs1</b> to <b>k2runs3</b>: Per-Dataset Rate Constants</li>"
                "<li><b>profile1</b> to <b>profile7</b>: Profile Likelihood</li></ul>"
                "<p>Every sheet runs from 0 to 1800 s. bounds1 to bounds3 record A and total_PB over time, base1 "
                "to base3 record A, B, C, Int, PB, and PH, and the other sheets record A, B, C, Int, and total_PB. "
                "The row at time 0 gives the starting concentrations of A, B, C, and Water.</p>"
                "<p>The data were simulated from M11, in most sheets with rate constants or starting concentrations "
                "that differ from those in Reactions, plus about 1% noise. Each lesson ends with how its data were "
                "simulated.</p>"
                "<p>Every sheet is recorded in seconds and M, the default units, and names its columns exactly as "
                "M11 does. Every column therefore maps automatically.</p>",
                lambda: None,
            ),
            Step(
                "open_workbook",
                "Open the Workbook",
                "<p>Choose <b>File → Import Data…</b>, or click <b>Import Data…</b> in the <b>Data</b> panel. "
                "Open <b>advanced_fitting.xlsx</b> from the AdvancedFitting folder.</p>",
                self._open_targets,
                lambda: self._import_dialog_open() or bool(self._workbook()),
                "advanced_fitting.xlsx is open in the import dialog or already imported.",
                dialog=IMPORT_DIALOGS,
            ),
            Step(
                "import_sheets",
                "Import Every Sheet",
                "<p>All 19 sheets are checked under <b>Sheets</b>. Each is read in seconds and M, as recorded.</p>"
                "<p>Click <b>Import</b>.</p>",
                lambda: (),
                lambda: bool(self._workbook()),
                "The workbook is imported.",
                dialog=IMPORT_DIALOGS,
            ),
            Step(
                "accept_mapping",
                "Accept the Mapping",
                "<p>The mapping review maps the columns of each dataset to their species and to total_PB.</p>"
                "<p><b>Initial-Condition Sets</b> previews a new set for each dataset, starting from its row at "
                "time 0.</p>"
                "<p>Click <b>OK</b>. Each dataset gets a new set named after it.</p>",
                self._review_targets,
                self._mapped,
                "Each sheet is mapped to its own new set.",
                waiting="Click OK in the mapping review. If it was closed, choose each workbook dataset in "
                "Dataset in the Data panel, click Change IC Set…, then Create IC Set from Data…, and click OK.",
                dialog=DatasetMappingBatchDialog,
            ),
        )

    def _finish(self, truth: str) -> Step:
        return replace(
            self._finish_step(),
            instruction=truth
            + "<p>Your practice tabs stay in the project. The other Advanced Fitting lessons start on the setup "
            "tab while it keeps the rate constants of M11 and its mappings.</p>"
            "<p>Click <b>Finish</b> to close this lesson. Open <b>Help → Interactive Tutorials…</b> to choose "
            "another.</p>",
            dialog=FittingWindow,
        )

    def _open_fit_step(self, about: str) -> Step:
        return Step(
            "open_fit",
            "Open the Fitting Window",
            about + "<p>Choose <b>Fitting → Global Fit…</b>.</p>",
            lambda: (),
            lambda: self._fit_window() is not None or self._applied,
            "The Fitting Window is open.",
            dialog=FittingWindow,
        )

    def _reset_target(self) -> tuple[Highlight, ...]:
        """Reset Setup… in the open Fitting Window."""
        window = self._fit_window()
        button = None if window is None else self._shown(
            window, QtWidgets.QPushButton, lambda button: button.text() == "Reset Setup…"
        )
        return () if button is None else (Highlight(button),)

    def _reset_setup_step(self) -> Step:
        """The lessons of the series share one tab, whose Fitting Window opens with the setup it last had."""
        return Step(
            "reset_setup",
            "Start from a Clean Setup",
            "<p>The Fitting Window keeps the setup of a mechanism tab while Kindred stays open, and the Advanced "
            "Fitting lessons share this tab. Choices from another lesson, or from before the window was closed, "
            "can still be here.</p>"
            "<p>Click <b>Reset Setup…</b>. Keep <b>Clean setup from the mechanism tab</b> and <b>Both</b> "
            "selected, and click <b>Reset</b>.</p>",
            self._reset_target,
            lambda: self._applied or self._reset_seen,
            "The setup starts clean from the mechanism tab.",
            waiting="Click Reset Setup…, keep Clean setup from the mechanism tab and Both, and click Reset.",
            dialog=FittingWindow,
        )

    def _choose_datasets_step(self) -> Step:
        return Step(
            "choose_datasets",
            "Choose the Datasets",
            f"<p>On <b>Data and Targets</b>, check {self._named} in <b>Datasets</b>, and leave the other "
            "datasets unchecked.</p>",
            lambda: (),
            lambda: self._applied or self._included_only(),
            f"{self._named} are included.",
            waiting=f"Check {self._named} only.",
            dialog=FittingWindow,
        )

    def _choose_targets_step(self) -> Step:
        """Compare every species measured over time in the lesson's sheets."""
        first, *others = self._sheets
        rest = f"{others[0]} and {others[1]}" if len(others) == 2 else f"{others[0]} to {others[-1]}"
        shown = " and ".join(f"<b>{sheet}</b>" for sheet in others) if len(others) == 2 else (
            f"<b>{others[0]}</b> to <b>{others[-1]}</b>"
        )
        return Step(
            "choose_targets",
            "Choose the Targets",
            "<p>The <b>Species Table</b> shows the dataset selected in <b>Datasets</b>. When <b>Include</b> is "
            "checked in a row, the fit compares that species with its mapped column. <b>Include</b> starts "
            "checked for every species with a mapped column.</p>"
            "<p>Water holds a value only at time 0, its starting concentration, so it has no measurements to fit. "
            "Compare only species measured over time.</p>"
            f"<p>Select <b>{first}</b>, uncheck <b>Include</b> for Water, and click <b>Copy Targets to "
            "Compatible</b>. It copies these choices to every other dataset that has all the checked species. "
            f"Then select {shown} to confirm that only A, B, C, Int, and total_PB are included.</p>",
            lambda: (),
            self._targets_chosen,
            f"Only A, B, C, Int, and total_PB are compared in {self._named}.",
            waiting=f"Uncheck Include for Water in {first}, click Copy Targets to Compatible, and select {rest} to "
            "confirm their targets.",
            dialog=FittingWindow,
        )


class BoundsAndSolvers(AdvancedFittingLesson):
    title = "Advanced Fitting 1: Bounds and Solvers"
    description = (
        "Fit rate constants whose true values lie far from the guesses: recognize a fit stopped by its bounds, "
        "widen the bounds on a log scale, and search the whole range with Differential Evolution."
    )
    _fitted = ("kf1", "kr1", "k2")
    _sheets = ("bounds1", "bounds2", "bounds3")
    _named = "bounds1, bounds2, and bounds3"
    _first_step = "open_fit"
    _fit_steps = frozenset({
        "reset_setup", "choose_datasets", "choose_targets", "vary_rates", "run_fit", "read_results", "set_bounds",
        "run_wide", "use_de", "apply_copy",
    })
    _mapped_steps = _fit_steps | {"open_fit"}
    # The species with measurements over time, the only ones the fit compares.
    _targets = frozenset({"A", "total_PB"})

    def _bounds_widened(self) -> bool:
        """kf1, kr1, and k2 range from 1e-4 to 1e3 on a log scale."""
        roster = self._roster()

        def widened(cells) -> bool:
            try:
                minimum, maximum = float(cells["Minimum"].text()), float(cells["Maximum"].text())
            except ValueError:
                return False
            return (
                isclose(minimum, 1e-4)
                and isclose(maximum, 1e3)
                and cells["Log Scale"].checkState() == QtCore.Qt.CheckState.Checked
            )

        return all(name in roster and widened(roster[name]) for name in self._fitted)

    def _de_chosen(self) -> bool:
        """The open Fitting Window uses Differential Evolution with at least the lesson's evaluations."""
        window = self._fit_window()
        method = window.findChild(QtWidgets.QComboBox, "global_fit_method")
        # The label stays found while another page of the window is shown.
        label = next(
            label for label in window.findChildren(QtWidgets.QLabel) if label.text() == "Maximum evaluations:"
        )
        try:
            evaluations = int(label.buddy().text())
        except ValueError:
            return False
        # A typed number applies on Enter or focus loss; until then the box is modified.
        return (
            method.currentData() == DIFFERENTIAL_EVOLUTION_METHOD
            and not label.buddy().isModified()
            and evaluations >= _DE_EVALUATIONS
        )

    def _final_setup(self) -> bool:
        return self._bounds_widened() and self._de_chosen()

    def _bounds_target(self) -> tuple[Highlight, ...]:
        """Set Bounds for Parameters…, or the Parameters tab while another page is shown."""
        window = self._fit_window()
        button = None if window is None else self._shown(
            window, QtWidgets.QPushButton, lambda button: button.text() == "Set Bounds for Parameters…"
        )
        if button is not None:
            return (Highlight(button),)
        return self._parameters_tab()

    def _steps(self) -> tuple[Step, ...]:
        return (
            *self._setup(),
            self._open_fit_step(
                "<p>This lesson fits kf1, kr1, and k2 of M11 to <b>bounds1</b>, <b>bounds2</b>, and "
                "<b>bounds3</b>. Their true values lie far from those in Reactions, as when a mechanism meets a "
                "new system for the first time.</p>"
            ),
            self._reset_setup_step(),
            self._choose_datasets_step(),
            Step(
                "choose_targets",
                "Choose the Targets",
                "<p>The <b>Species Table</b> shows the dataset selected in <b>Datasets</b>. When <b>Include</b> "
                "is checked in a row, the fit compares that species with its mapped column. <b>Include</b> starts "
                "checked for every species with a mapped column.</p>"
                "<p>B, C, and Water hold a value only at time 0, the starting concentration, so they have no "
                "measurements to fit. Compare only species measured over time.</p>"
                "<p>Select <b>bounds1</b>, uncheck <b>Include</b> for B, C, and Water, and click <b>Copy Targets "
                "to Compatible</b>. It copies these choices to every other dataset that has all the checked "
                "species. Then select <b>bounds2</b> and <b>bounds3</b> to confirm that only A and total_PB are "
                "included.</p>",
                lambda: (),
                lambda: self._applied or self._targets_chosen(),
                "Only A and total_PB are compared in bounds1, bounds2, and bounds3.",
                waiting="Uncheck Include for B, C, and Water in bounds1, click Copy Targets to Compatible, and "
                "select bounds2 and bounds3 to confirm their targets.",
                dialog=FittingWindow,
            ),
            Step(
                "vary_rates",
                "Vary the Rate Constants",
                "<p>Click <b>Parameters</b> and check <b>Vary</b> for <b>kf1</b>, <b>kr1</b>, and <b>k2</b>. "
                "Leave <b>k3</b> and <b>concentration_floor</b> unchecked.</p>"
                "<p>For rate constants, <b>Minimum</b> and <b>Maximum</b> start at 1/100 and 100 times the value in "
                "Reactions. "
                "kr1 and k2, at 0.01 in Reactions, can therefore range only from 0.0001 to 1.</p>",
                lambda: (),
                lambda: self._applied or self._varied(),
                "kf1, kr1, and k2 vary.",
                waiting="Check Vary for kf1, kr1, and k2 only.",
                dialog=FittingWindow,
            ),
            Step(
                "run_fit",
                "Run the Fit",
                "<p>Click <b>Run Fit</b> and wait for the fit to finish.</p>",
                lambda: (),
                lambda: self._applied or self._lesson_fit(),
                "The fit is complete.",
                waiting="Include only bounds1, bounds2, and bounds3. Compare only A and total_PB. Vary kf1, kr1, and "
                "k2. Then click Run Fit and wait for the fit to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "read_results",
                "Read the Results",
                "<p>The fit converged, but kr1 and k2 both ended at <b>1</b>, their Maximum.</p>"
                "<p>Click <b>Summary</b>. <b>Bound Position</b> reads <b>at maximum</b> for kr1 and k2. When a fit "
                "converges with a value at its Minimum or Maximum, it can strongly indicate that the value was "
                "constrained by the bounds, and a better fit value may lie outside the range searched.</p>",
                lambda: (),
                dialog=FittingWindow,
            ),
            Step(
                "set_bounds",
                "Widen the Bounds",
                "<p>On <b>Parameters</b>, click <b>Set Bounds for Parameters…</b>. Check kf1, kr1, and k2, enter "
                "<b>1e-4</b> in <b>Minimum</b> and <b>1e3</b> in <b>Maximum</b>, check <b>Log Scale</b>, and click "
                "<b>OK</b>.</p>"
                "<p>The range now spans seven powers of ten. <b>Log Scale</b> makes the fit search the logarithm "
                "of each value, so every power of ten gets the same room: a step from 0.001 to 0.01 counts as much "
                "as one from 100 to 1000. Use it when a range spans several powers of ten.</p>"
                "<p>Changing the setup marks the shown result as out of date until the fit runs again.</p>",
                self._bounds_target,
                lambda: self._applied or self._bounds_widened(),
                "kf1, kr1, and k2 range from 1e-4 to 1e3 on a log scale.",
                waiting="Set Minimum 1e-4, Maximum 1e3, and Log Scale for kf1, kr1, and k2.",
                dialog=FittingWindow,
            ),
            Step(
                "run_wide",
                "Run the Fit Again",
                "<p>Click <b>Run Fit</b>. <b>Initial Guess</b> holds the previous result, and the fit now searches "
                "the wider range.</p>"
                "<p>In <b>Summary</b>, Bound Position now reads <b>within bounds</b> for every value.</p>",
                lambda: (),
                lambda: self._applied or (self._bounds_widened() and self._lesson_fit()),
                "The fit with the wider bounds is complete.",
                waiting="Click Run Fit with the wider bounds and wait for the fit to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "use_de",
                "Search the Whole Range",
                "<p><b>Trust Region Reflective</b>, the default <b>Optimization method</b>, is a local method: it "
                "starts from <b>Initial Guess</b> and finds a minimum near it, which may not be the best fit in the "
                "range. <b>Differential Evolution</b> is a global method: it searches the whole range between "
                "Minimum and Maximum, at the cost of many more evaluations. Use it when no reliable estimates are "
                "known.</p>"
                "<p>In the <b>Parameters</b> tab, under <b>Fit Settings</b>, choose <b>Differential Evolution</b> in "
                f"Optimization method, set <b>Maximum evaluations</b> to <b>{_DE_EVALUATIONS}</b>, and click <b>Run "
                "Fit</b>. With the default 1000, it stops early, and Results reads <b>Fit finished · Optimizer did not "
                "converge</b>.</p>"
                "<p>It finds the same values as the previous fit, so here the minimum Trust Region Reflective found "
                "is also the best fit in the range.</p>",
                lambda: (),
                lambda: self._applied or (self._final_setup() and self._lesson_fit()),
                "The Differential Evolution fit is complete.",
                waiting=f"Choose Differential Evolution, set Maximum evaluations to {_DE_EVALUATIONS}, and click "
                "Run Fit.",
                dialog=FittingWindow,
            ),
            Step(
                "apply_copy",
                "Apply the Fit to a Tab Copy",
                "<p>Click <b>Apply Fit</b> and choose <b>Apply Fit to Tab Copy</b>. Kindred adds a copy of this "
                "tab, named after it with <b>— Fit</b> added, with the fitted values in Reactions.</p>"
                "<p>This tab keeps the rate constants of M11, so the other Advanced Fitting lessons can start "
                "from it.</p>",
                lambda: (),
                lambda: self._applied,
                "The fitted copy is in the project.",
                waiting="Click Apply Fit and choose Apply Fit to Tab Copy.",
                dialog=FittingWindow,
            ),
            self._finish(
                "<p>bounds1 to bounds3 were simulated with <b>kf1 = 0.05</b>, <b>kr1 = 3</b>, and "
                "<b>k2 = 2</b> M<sup>−1</sup> s<sup>−1</sup>. Close the <b>Fitting Window</b> and compare them with "
                "the fitted values in Reactions of the copy.</p>"
            ),
        )


@cache
def _spreadsheet_dialogs():
    """The Create from Data dialogs. Their module loads the spreadsheet engine, so it is imported when a
    lesson needs it, as the main window imports it when it opens them."""
    from kindred.gui.widgets.spreadsheet_editor import SpreadsheetEditorDialog, SpreadsheetSelectionDialog

    return SpreadsheetSelectionDialog, SpreadsheetEditorDialog


class CalculatedQuantitiesAndConstraints(AdvancedFittingLesson):
    title = "Advanced Fitting 2: Calculated Quantities and Constraints"
    description = (
        "On a copy of the setup tab, build a measured sum from two columns with Create from Data, write one rate "
        "constant as a fraction of another, and fit that fraction."
    )
    _fitted = ("kf1", "kr1", "a")
    _bases = ("base1", "base2", "base3")
    _first_step = "create_reuse_tab"
    _fit_steps = frozenset({"choose_datasets", "choose_targets", "vary_rates", "bound_a", "run_fit", "apply_project"})
    _mapped_steps = _fit_steps | {"create_reuse_tab", "add_constraint", "create_data", "open_fit"}
    # Every step after the setup works on the copy.
    _reuse_steps = _mapped_steps

    def blocked(self):
        name = self.step.name
        if name in self._reuse_steps and name != "create_reuse_tab" and self.reuse is None:
            return self._missing("This exercise needs a copy of the setup tab.", "create_reuse_tab")
        return super().blocked()

    def _copy_made(self) -> bool:
        """The active tab is a new tab with the Reactions and mappings of the setup tab, so a copy of the
        setup tab rather than of a tab with fitted values."""
        identity = self.registry.active_tab_id
        if self.reuse is None and (
            identity in self._tabs_before
            or self.registry.tab(identity).mechanism_source.reactions_text != self.tab.mechanism_source.reactions_text
            or not self._mapped(self.registry.tab(identity))
        ):
            return False
        return self._new_tab(reuse=True)

    def _constrained(self) -> bool:
        """The copy keeps the reactions of M11 and writes k2 from a = 0.01 and kf1, so k2 is no longer a
        value of its own."""
        prepared = self.tab.prepared_dsl
        if prepared is None or prepared.symbolic_equations_text() != _preset_equations(self._preset_name):
            return False
        values = {item.name: item.value for item in enumerate_parameter_coordinates(prepared)}
        dependencies = prepared.parameter_algebra_spec_for_execution().evaluation_dependencies
        return (
            "k2" not in values
            and isclose(values.get("a", -1.0), 0.01)
            and set(dependencies.get("k2", ())) == {"a", "kf1"}
        )

    def _editor_open(self) -> bool:
        """The Create from Data editor, which the main window owns, is open."""
        editor = _spreadsheet_dialogs()[1]
        return any(
            widget.isVisible()
            for widget in self.workspace.window().findChildren(
                editor, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly
            )
        )

    def _processed(self) -> dict[str, str]:
        """base run -> the newest dataset Create from Data made from it for the copy only: its total_PB is
        the sum of its PB and PH and maps to the let total_PB, and its set starts from its row at time 0."""
        if self.reuse is None:
            return {}
        copy, setup = self.registry.tab(self.reuse), self.registry.tab(self.practice)
        sets = {item.set_id: item for item in copy.initial_condition_sets}
        runs = {}
        for record in self.registry.document.datasets:
            link = copy.dataset_link(record.dataset_id)
            series = {item.column_name: item for item in record.observations}
            total, made, hydrolysed = (series.get(name) for name in ("total_PB", "PB", "PH"))
            if (
                record.source_id is not None
                or link is None
                or setup.dataset_link(record.dataset_id) is not None
                or link.set_id not in sets
                or None in (total, made, hydrolysed)
                or dict(link.observation_mapping).get(total.raw_observation_id) != "total_PB"
                or not total.t == made.t == hydrolysed.t
                or not all(
                    isclose(value, first + second, rel_tol=1e-9, abs_tol=1e-15)
                    for value, first, second in zip(total.y, made.y, hydrolysed.y)
                )
            ):
                continue
            values = dict(sets[link.set_id].values)
            run = next((run for run in self._bases if self._starts_from(values, run)), None)
            if run is not None:
                runs[run] = record.dataset_id
        return runs if len(runs) == len(self._bases) else {}

    def _datasets(self) -> set[str]:
        return set(self._processed().values())

    def _bounded(self) -> bool:
        """a ranges from 1e-4 to 1 on a log scale."""
        cells = self._roster().get("a")
        try:
            minimum, maximum = float(cells["Minimum"].text()), float(cells["Maximum"].text())
        except (TypeError, ValueError):
            return False
        return (
            isclose(minimum, 1e-4)
            and isclose(maximum, 1.0)
            and cells["Log Scale"].checkState() == QtCore.Qt.CheckState.Checked
        )

    def _fit_applied(self, _new_tab: bool) -> bool:
        """Apply Fit to Project writes the result into Reactions of the copy, and the parameter table then
        shows each fitted value as its Mechanism Tab Value."""
        roster = self._roster()

        def written(name: str) -> bool:
            cells = roster.get(name)
            try:
                value, result = float(cells["Mechanism Tab Value"].text()), float(cells["Last Fit Result"].text())
            except (TypeError, ValueError):
                return False
            return isclose(value, result, rel_tol=1e-5)

        return (
            self._included_only()
            and self._targets_chosen()
            and self._varied()
            and self._bounded()
            and all(written(name) for name in self._fitted)
        )

    def _create_target(self) -> tuple[Highlight, ...]:
        """The Data page while Create from Data is not open."""
        selection = _spreadsheet_dialogs()[0]
        return () if self._dialog(selection) is not None or self._editor_open() else (self._data_page(),)

    def _bounds_target(self) -> tuple[Highlight, ...]:
        """Set Bounds for Parameters…, or the Parameters tab while another page is shown."""
        window = self._fit_window()
        button = None if window is None else self._shown(
            window, QtWidgets.QPushButton, lambda button: button.text() == "Set Bounds for Parameters…"
        )
        if button is not None:
            return (Highlight(button),)
        return self._parameters_tab()

    def _steps(self) -> tuple[Step, ...]:
        selection = _spreadsheet_dialogs()[0]
        return (
            *self._setup(),
            Step(
                "create_reuse_tab",
                "Copy the Setup Tab",
                "<p>This lesson adds two lines to the M11 model, so it works on a copy of the tab where you set up "
                "the tutorial.</p>"
                "<p>Right-click the current tab, above the plot, and choose <b>Create Copy</b>. The copy is named "
                "after the tab with <b>— Copy</b> added, has the same datasets, mappings, and sets, and becomes "
                "the selected tab.</p>",
                self._tab_target,
                self._copy_made,
                "The copy is ready.",
                waiting="Right-click the current tab and choose Create Copy.",
            ),
            Step(
                "add_constraint",
                "Write k2 from kf1",
                "<p>In a kinetic model, a constraint defines a value, such as a rate constant, from other values. "
                "If the absolute rate constants are unknown but a relative rate constant is known from experiments, "
                "for example from a competition reaction, only one of the two rate constants needs to be fitted. It "
                "is good practice to constrain a model experimentally as thoroughly as possible, because a less "
                "flexible model is less likely to overfit the data. In this lesson, k2 is defined as the fraction "
                "<b>a</b> of kf1.</p>"
                "<p>At the end of <b>Reactions</b>, add two lines:</p>"
                "<p><b>param a = 0.01</b><br><b>param k2 = a * kf1</b></p>"
                "<p>Int + C → PB keeps <b>kf=0.01</b> as written, but <b>param k2</b> now supplies its value: a "
                "times kf1, 0.01 × 1.0 = 0.01. The <b>Help</b> tab beside Reactions describes this.</p>"
                "<p>Click <b>Run</b> in the <b>Mechanism</b> panel. Run checks the new lines and keeps them in "
                "this tab.</p>",
                self._source_target,
                self._constrained,
                "k2 is a times kf1.",
                waiting="Add param a = 0.01 and param k2 = a * kf1 at the end of Reactions, and click Run.",
            ),
            Step(
                "create_data",
                "Add total_PB with Create from Data",
                "<p>base1 to base3 record PB and PH separately. The let total_PB of M11 is their sum, so build a "
                "total_PB column from them for the fit to compare.</p>"
                "<p>In the <b>Data</b> panel, click <b>Create from Data…</b>. Check <b>base1</b>, <b>base2</b>, "
                "and <b>base3</b> only, and click <b>Open Editor</b>. The editor has a worksheet for each "
                "dataset, each with its own tab.</p>"
                "<p>On the base1 worksheet, choose <b>Columns → Add Column</b>. The new column, <b>I</b>, is "
                "named <b>New column 1</b>. Double-click that name, type <b>total_PB</b>, and press <b>Enter</b>. "
                "Click its first cell, type <b>=G1+H1</b>, and press <b>Enter</b>. G is PB and H is PH, so the "
                "whole column becomes PB + PH, in M.</p>"
                "<p>Do the same on the base2 and base3 worksheets.</p>"
                "<p>Click <b>Review and Create…</b>. <b>Review New Datasets</b> maps total_PB to the let total_PB "
                "and gives each new dataset its own set from its row at time 0. Click <b>Create 3 Datasets</b>. "
                "Kindred adds the three datasets, named after base1 to base3 with <b>— processed</b> added, mapped "
                "in this tab only.</p>",
                self._create_target,
                lambda: bool(self._processed()),
                "The three new datasets are mapped.",
                waiting="Click Create from Data…, check base1, base2, and base3 only, add total_PB as =G1+H1 on each "
                "worksheet, click Review and Create…, and click Create 3 Datasets. To redo a run that was left out or "
                "made without total_PB, delete its — processed dataset, if it has one, with Delete… in the Data "
                "panel. Then click Create from Data… and check only that run.",
                dialog=(selection, DatasetMappingBatchDialog),
            ),
            self._open_fit_step(
                "<p>The fit compares the three new datasets with this tab, varying kf1, kr1, and a.</p>"
            ),
            Step(
                "choose_datasets",
                "Choose the Datasets",
                "<p>On <b>Data and Targets</b>, check base1 — processed, base2 — processed, and base3 — "
                "processed in <b>Datasets</b>, and leave the other datasets unchecked, base1 to base3 among "
                "them.</p>",
                lambda: (),
                lambda: self._applied or self._included_only(),
                "The three new datasets are included.",
                waiting="Check base1 — processed, base2 — processed, and base3 — processed only.",
                dialog=FittingWindow,
            ),
            Step(
                "choose_targets",
                "Choose the Targets",
                "<p>The <b>Species Table</b> shows the dataset selected in <b>Datasets</b>. When <b>Include</b> "
                "is checked in a row, the fit compares that species with its mapped column. <b>Include</b> starts "
                "checked for every species with a mapped column.</p>"
                "<p>Water holds a value only at time 0, its starting concentration, so it has no measurements to "
                "fit. PB and PH are measured over time, but this fit compares their sum, total_PB, instead.</p>"
                "<p>Select <b>base1 — processed</b>, uncheck <b>Include</b> for Water, PB, and PH, and click "
                "<b>Copy Targets to Compatible</b>. It copies these choices to every other dataset that has all "
                "the checked species. Then select <b>base2 — processed</b> and <b>base3 — processed</b> to confirm "
                "that only A, B, C, Int, and total_PB are included.</p>",
                lambda: (),
                lambda: self._applied or self._targets_chosen(),
                "Only A, B, C, Int, and total_PB are compared in the new datasets.",
                waiting="Uncheck Include for Water, PB, and PH in base1 — processed, click Copy Targets to "
                "Compatible, and select base2 — processed and base3 — processed to confirm their targets.",
                dialog=FittingWindow,
            ),
            Step(
                "vary_rates",
                "Vary kf1, kr1, and a",
                "<p>Click <b>Parameters</b>. k2 has no row: its value comes from a and kf1.</p>"
                "<p>Check <b>Vary</b> for <b>kf1</b>, <b>kr1</b>, and <b>a</b>. Leave <b>k3</b> and "
                "<b>concentration_floor</b> unchecked.</p>"
                "<p>A known relative rate constant would normally stay fixed: enter its value on the <b>param a</b> "
                "line and leave <b>Vary</b> unchecked for a. This lesson fits a anyway, to show that a calculated "
                "value can be fitted and that its bounds need care.</p>",
                lambda: (),
                lambda: self._applied or self._varied(),
                "kf1, kr1, and a vary.",
                waiting="Check Vary for kf1, kr1, and a only.",
                dialog=FittingWindow,
            ),
            Step(
                "bound_a",
                "Keep a Positive",
                "<p><b>Minimum</b> and <b>Maximum</b> of a start at −0.99 and 1.01, so the fit could try a negative "
                "a. That would make k2 negative, which no rate constant can be, so Kindred rejects every such trial. "
                "Bounds from 10<sup>−4</sup> to 1 keep the search to values Kindred can simulate.</p>"
                "<p>Click <b>Set Bounds for Parameters…</b>. Check a only, enter <b>1e-4</b> in <b>Minimum</b> "
                "and <b>1</b> in <b>Maximum</b>, check <b>Log Scale</b>, and click <b>OK</b>.</p>",
                self._bounds_target,
                lambda: self._applied or self._bounded(),
                "a ranges from 1e-4 to 1 on a log scale.",
                waiting="Set Minimum 1e-4, Maximum 1, and Log Scale for a.",
                dialog=FittingWindow,
            ),
            Step(
                "run_fit",
                "Run the Fit",
                "<p>Click <b>Run Fit</b> and wait for the fit to finish.</p>",
                lambda: (),
                lambda: self._applied or (self._bounded() and self._lesson_fit()),
                "The fit is complete.",
                waiting="Include only the three new datasets. Compare A, B, C, Int, and total_PB. Vary kf1, kr1, and "
                "a, with a from 1e-4 to 1. Then click Run Fit and wait for the fit to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "apply_project",
                "Apply the Fit to This Tab",
                "<p>Click <b>Apply Fit</b> and choose <b>Apply Fit to Project</b>. Kindred writes the fitted "
                "kf1, kr1, and a into Reactions of this tab, which is the copy. <b>Mechanism Tab Value</b> then "
                "shows them.</p>"
                "<p>The setup tab keeps the rate constants of M11, so the other Advanced Fitting lessons can "
                "start from it.</p>",
                lambda: (),
                lambda: self._applied,
                "The fit is applied to the copy.",
                waiting="Click Apply Fit and choose Apply Fit to Project.",
                dialog=FittingWindow,
            ),
            self._finish(
                "<p>base1 to base3 were simulated with <b>kf1 = 0.5</b>, <b>kr1 = 0.01</b>, and "
                "<b>k2 = 0.02</b> M<sup>−1</sup> s<sup>−1</sup>, so <b>a = 0.04</b>. Close the <b>Fitting "
                "Window</b> and compare them with the fitted values in Reactions of the copy.</p>"
            ),
        )


class VaryingInitialConditions(AdvancedFittingLesson):
    title = "Advanced Fitting 3: Varying Initial Conditions"
    description = (
        "Fit a starting concentration that differs from the one recorded, within a range around it, and share one "
        "value between runs made from the same stock solution."
    )
    _sheets = ("stock1", "stock2", "stock3")
    _named = "stock1, stock2, and stock3"
    _first_step = "open_fit"
    _fit_steps = frozenset({
        "reset_setup", "choose_datasets", "choose_targets", "vary_c", "share_c", "run_fit", "read_results",
    })
    _mapped_steps = _fit_steps | {"open_fit"}

    def _c_rows(self) -> dict[str, _Row | None]:
        """The C row last seen in each sheet."""
        found = self._workbook()
        return {sheet: self._rows_seen.get(found[sheet], {}).get("C") for sheet in self._sheets}

    def _c_varied(self) -> bool:
        """stock1 and stock3 were last seen varying C by 20% either side of its Starting IC."""
        rows = self._c_rows()

        def varied(row: _Row | None) -> bool:
            try:
                return row.vary == "symmetric_percent" and isclose(float(row.lower), 20.0)
            except (AttributeError, ValueError):
                return False

        return varied(rows["stock1"]) and varied(rows["stock3"])

    def _c_shared(self) -> bool:
        """stock1 and stock2 were last seen varying C together, and stock3 on its own."""
        rows = self._c_rows()
        return all(row is not None for row in rows.values()) and (
            rows["stock1"].shared and rows["stock2"].shared and not rows["stock3"].shared
        )

    def _steps(self) -> tuple[Step, ...]:
        return (
            *self._setup(),
            self._open_fit_step(
                "<p>This lesson fits the starting concentration of C in <b>stock1</b>, <b>stock2</b>, and "
                "<b>stock3</b>. Each sheet records 0.2 M of C at time 0, the intended concentration, but analysis of "
                "the experiment showed that the real concentration of the stock solutions is uncertain. stock1 and "
                "stock2 were made from the same stock solution, so their starting C must be the same.</p>"
            ),
            self._reset_setup_step(),
            self._choose_datasets_step(),
            self._choose_targets_step(),
            Step(
                "vary_c",
                "Vary the Starting C",
                "<p><b>Vary IC</b> lets the fit move a starting concentration within a range around its "
                "<b>Starting IC</b>.</p>"
                "<p>Select <b>stock1</b>. In the C row, choose <b>± X% (Symmetric)</b> in Vary IC. With Symmetric, "
                "<b>Lower</b> sets the same percentage below and above <b>Starting IC</b>, so <b>Upper</b> shows "
                "the same percentage and cannot be edited. Kindred fills Lower with 5. Enter <b>20</b> in "
                "<b>Lower</b> and press <b>Enter</b>. C can then range 20% either side of 0.2 M, from 0.16 to "
                "0.24 M. Do the same for <b>stock3</b>.</p>",
                lambda: (),
                self._c_varied,
                "C varies by 20% in stock1 and stock3.",
                waiting="In stock1 and stock3, choose ± X% (Symmetric) in Vary IC for C, enter 20 in Lower, and press "
                "Enter. Then select stock1 and stock3 again to confirm them.",
                dialog=FittingWindow,
            ),
            Step(
                "share_c",
                "Share One C",
                "<p>stock1 and stock2 were made from the same stock solution, so they share one starting C.</p>"
                "<p><b>Vary Global</b> fits one starting concentration for several datasets together: the fit varies "
                "a single value of C and uses it in each of them. It can be checked only in a row whose Vary IC is "
                "set.</p>"
                "<p>Select <b>stock1</b> and check <b>Vary Global</b> for C. In <b>Vary Global — C</b>, check "
                "stock2 and click <b>OK</b>. stock2 then shows the range of stock1 and cannot be edited. Then select "
                "<b>stock1</b>, <b>stock2</b>, and <b>stock3</b> to confirm "
                "that Vary Global is checked in stock1 and stock2 and not in stock3.</p>",
                lambda: (),
                lambda: self._c_varied() and self._c_shared(),
                "stock1 and stock2 share one C.",
                waiting="Check Vary Global for C in stock1, check stock2 in Vary Global — C, click OK, and select "
                "stock1, stock2, and stock3.",
                dialog=FittingWindow,
            ),
            Step(
                "run_fit",
                "Run the Fit",
                "<p>Click <b>Run Fit</b> and wait for the fit to finish.</p>",
                lambda: (),
                lambda: self._c_varied() and self._c_shared() and self._lesson_fit(),
                "The fit is complete.",
                waiting="Include only stock1, stock2, and stock3. Compare A, B, C, Int, and total_PB. Vary C as set "
                "above and no rate constant. Then click Run Fit and wait for the fit to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "read_results",
                "Read the Results",
                "<p>Results lists the fitted starting C: one value shared by stock1 and stock2, and one for "
                "stock3.</p>",
                lambda: (),
                dialog=FittingWindow,
            ),
            self._finish(
                "<p>stock1 and stock2 were simulated with <b>C = 0.18 M</b> and stock3 with "
                "<b>C = 0.21 M</b>, with the rate constants of M11. Compare them with the fitted values in "
                "Results.</p>"
            ),
        )


class PerDatasetRateConstants(AdvancedFittingLesson):
    title = "Advanced Fitting 4: Per-Dataset Rate Constants"
    description = (
        "Fit a rate constant that differs between runs: give each dataset its own value, and let runs made under "
        "the same conditions share one."
    )
    _fitted = ("kf1", "kr1", "k2")
    _sheets = ("k2runs1", "k2runs2", "k2runs3")
    _named = "k2runs1, k2runs2, and k2runs3"
    _first_step = "open_fit"
    _fit_steps = frozenset({
        "reset_setup", "choose_datasets", "choose_targets", "vary_rates", "per_dataset", "share_k2", "run_fit",
        "read_results",
    })
    _mapped_steps = _fit_steps | {"open_fit"}

    def _k2_rows(self) -> tuple[QtWidgets.QComboBox | None, dict[str, QtWidgets.QTableWidgetItem]]:
        """The Across Datasets choice of k2, and the Across Datasets cell of each dataset row of k2."""
        roster = self._roster_table()
        headers = [roster.horizontalHeaderItem(column).text() for column in range(roster.columnCount())]
        parameter, scope = headers.index("Parameter"), headers.index("Across Datasets")
        choice, cells = None, {}
        for row in range(roster.rowCount()):
            name = roster.item(row, parameter)
            if name.data(QtCore.Qt.ItemDataRole.UserRole) != "k2":
                continue
            dataset = name.data(QtCore.Qt.ItemDataRole.UserRole + 1)
            if dataset is None:
                choice = roster.cellWidget(row, scope)
            else:
                cells[dataset] = roster.item(row, scope)
        return choice, cells

    def _per_dataset(self) -> bool:
        choice, _cells = self._k2_rows()
        return choice.currentData() == "per_dataset"

    def _k2_shared(self) -> bool:
        """k2runs1 and k2runs2 share one group of k2, and k2runs3 has its own k2."""
        _choice, cells = self._k2_rows()
        found = self._workbook()
        first, second, third = (cells.get(found[sheet]) for sheet in self._sheets)
        if None in (first, second, third):
            return False

        def grouped(cell) -> bool:
            return cell.checkState() == QtCore.Qt.CheckState.Checked

        return grouped(first) and grouped(second) and first.text() == second.text() and not grouped(third)

    def _steps(self) -> tuple[Step, ...]:
        return (
            *self._setup(),
            self._open_fit_step(
                "<p>This lesson fits kf1, kr1, and k2 to <b>k2runs1</b>, <b>k2runs2</b>, and <b>k2runs3</b>.</p>"
                "<p>Within one mechanism, a rate constant has the same value in every run. A run can still need its "
                "own value when something the model does not include differs between runs, such as the temperature, "
                "the concentration of a catalyst, or a reagent in large excess that makes a step pseudo-first order. "
                "Kindred can fit such a rate constant separately for each dataset, or once for a group of "
                "datasets.</p>"
                "<p>Outside the Fitting Window, a mechanism tab holds one value for each rate constant. <b>Apply "
                "Fit</b> writes the mean of the per-dataset values, which <b>Summary</b> also lists, into Reactions "
                "and adds a comment line with each value. A result that is not applied is not saved with the project, "
                "so copy what you need from <b>Summary</b> and keep it outside Kindred. You can also paste it into "
                "the <b>Notes</b> tab beside Reactions once the Fitting Window is closed.</p>"
                "<p>In this lesson, the three runs share kf1 and kr1, but k2 differs: k2runs1 and k2runs2 were made "
                "under the same conditions, and k2runs3 under different ones.</p>"
            ),
            self._reset_setup_step(),
            self._choose_datasets_step(),
            self._choose_targets_step(),
            Step(
                "vary_rates",
                "Vary the Rate Constants",
                "<p>Click <b>Parameters</b> and check <b>Vary</b> for <b>kf1</b>, <b>kr1</b>, and <b>k2</b>. "
                "Leave <b>k3</b> and <b>concentration_floor</b> unchecked.</p>",
                lambda: (),
                self._varied,
                "kf1, kr1, and k2 vary.",
                waiting="Check Vary for kf1, kr1, and k2 only.",
                dialog=FittingWindow,
            ),
            Step(
                "per_dataset",
                "Give Each Dataset Its Own k2",
                "<p><b>Across Datasets</b> reads <b>Shared</b>: one value for every dataset. In the k2 row, choose "
                "<b>Per-dataset</b>. k2 then has a row for each dataset, and each dataset in the fit gets its own "
                "k2.</p>",
                lambda: (),
                lambda: self._varied() and self._per_dataset(),
                "Each dataset has its own k2.",
                waiting="Choose Per-dataset in Across Datasets for k2.",
                dialog=FittingWindow,
            ),
            Step(
                "share_k2",
                "Share k2 Between Two Runs",
                "<p>k2runs1 and k2runs2 were made under the same conditions, so they share one k2.</p>"
                "<p>In the k2runs1 row under k2, check the box in Across Datasets, beside <b>Group —</b>. "
                "<b>Datasets Sharing k2 — C + Int → PB</b> opens. It lists every dataset mapped to this tab, and "
                "datasets not included in the fit are marked <b>(not in fit)</b>. Check k2runs2 and click "
                "<b>OK</b>. The k2runs1 and k2runs2 rows then read <b>Group 1</b>, and k2runs3 keeps its own "
                "k2.</p>",
                lambda: (),
                lambda: self._varied() and self._per_dataset() and self._k2_shared(),
                "k2runs1 and k2runs2 share one k2.",
                waiting="Check Across Datasets in the k2runs1 row of k2, check k2runs2, and click OK.",
                dialog=FittingWindow,
            ),
            Step(
                "run_fit",
                "Run the Fit",
                "<p>Click <b>Run Fit</b> and wait for the fit to finish.</p>",
                lambda: (),
                lambda: self._per_dataset() and self._k2_shared() and self._lesson_fit(),
                "The fit is complete.",
                waiting="Include only k2runs1, k2runs2, and k2runs3. Compare A, B, C, Int, and total_PB. Vary kf1, "
                "kr1, and k2, with k2 shared by k2runs1 and k2runs2. Then click Run Fit and wait for the fit to "
                "finish.",
                dialog=FittingWindow,
            ),
            Step(
                "read_results",
                "Read the Results",
                "<p>Results lists kf1 and kr1 once, shared by every dataset, and k2 twice: once for k2runs1 and "
                "k2runs2 and once for k2runs3.</p>"
                "<p>Click <b>Summary</b>. Under <b>k2 Per-Dataset Values</b>, Kindred lists each fitted k2 with its "
                "<b>Difference from Mean</b>, followed by <b>Count</b>, <b>Mean</b>, <b>Median</b>, <b>Sample "
                "SD</b>, <b>Standard error</b>, and <b>IQR</b>. Group 1 counts once.</p>",
                lambda: (),
                dialog=FittingWindow,
            ),
            self._finish(
                "<p>k2runs1 to k2runs3 were simulated with the kf1 and kr1 of M11, and with "
                "<b>k2 = 0.01</b> M<sup>−1</sup> s<sup>−1</sup> in k2runs1 and k2runs2 and <b>k2 = 0.03</b> "
                "M<sup>−1</sup> s<sup>−1</sup> in k2runs3. Compare them with the fitted values in Results.</p>"
            ),
        )


class ProfileLikelihood(AdvancedFittingLesson):
    title = "Advanced Fitting 5: Profile Likelihood"
    description = (
        "Find how well the data determine each fitted value with profile likelihood, recognize a value the data do "
        "not determine, and hold it fixed."
    )
    _fitted = ("kf1", "kr1", "k2", "k3")
    # The values fitted once k3 is held.
    _held = ("kf1", "kr1", "k2")
    _sheets = tuple(f"profile{number}" for number in range(1, 8))
    _named = "profile1 to profile7"
    _first_step = "open_fit"
    _fit_steps = frozenset({
        "reset_setup", "choose_datasets", "choose_targets", "vary_rates", "profile", "run_fit", "read_profile",
        "hold_k3", "read_held",
    })
    _mapped_steps = _fit_steps | {"open_fit"}

    def _profiled(self, names: tuple[str, ...]) -> bool:
        """Profile Likelihood is on and profiles these values."""
        check = next(
            check for check in self._fit_window().findChildren(QtWidgets.QCheckBox)
            if check.text() == "Profile Likelihood"
        )
        roster = self._roster()
        return check.isChecked() and all(
            roster[name]["Profile"].checkState() == QtCore.Qt.CheckState.Checked
            for name in names
        )

    def _steps(self) -> tuple[Step, ...]:
        return (
            *self._setup(),
            self._open_fit_step(
                "<p>A fit gives one value for each fitted constant. Profile likelihood also gives each a 95% "
                "interval: how far the value can move before the fit becomes significantly worse. This lesson "
                "fits the four rate constants of M11 to <b>profile1</b> to <b>profile7</b>, seven runs with "
                "different starting concentrations.</p>"
            ),
            self._reset_setup_step(),
            self._choose_datasets_step(),
            self._choose_targets_step(),
            Step(
                "vary_rates",
                "Vary the Rate Constants",
                "<p>Click <b>Parameters</b> and check <b>Vary</b> for <b>kf1</b>, <b>kr1</b>, <b>k2</b>, and "
                "<b>k3</b>. Leave <b>concentration_floor</b> unchecked.</p>",
                lambda: (),
                self._varied,
                "kf1, kr1, k2, and k3 vary.",
                waiting="Check Vary for kf1, kr1, k2, and k3 only.",
                dialog=FittingWindow,
            ),
            Step(
                "profile",
                "Turn On Profile Likelihood",
                "<p>When a constant is <b>Shared</b> across datasets, the fit finds one value for it, the one that "
                "best fits all of them together. That number alone does not show how far the value could move "
                "before the fit becomes worse. The data pin some values closely and leave others free to move a long "
                "way. Some stop mattering once they are large enough, so any larger value fits equally well. k3 in "
                "this lesson behaves like that.</p>"
                "<p>In kinetic modeling, an exact value for every elementary step is rarely the useful result, "
                "because each value is only the best fit to the data provided. It is often more useful to report "
                "what the data actually support, for example that one step only needs to be much faster than "
                "another, or that a rate constant is at least a certain size.</p>"
                "<p>Profile likelihood tests this. For each profiled value, Kindred holds it at a series of values "
                "moving away from the fitted one, toward Minimum and toward Maximum. At each one it refits every "
                "other varied value and records how much worse the fit becomes. The interval ends where the fit "
                "becomes significantly worse at the chosen confidence level. If the fit never becomes significantly "
                "worse before the value reaches its Minimum or Maximum, <b>Summary</b> reports that end as <b>not "
                "limited within the search range</b>.</p>"
                "<p>Under <b>Fit Settings</b>, check <b>Profile Likelihood</b>. <b>Profile</b> is then checked for "
                "kf1, kr1, k2, and k3.</p>"
                "<p><b>Configure…</b> opens its options. This lesson leaves them at their defaults:</p>"
                "<ul><li><b>Confidence level</b>: 0.95, the level of each interval.</li>"
                "<li><b>Maximum outward search steps per side</b>: 8, the largest number of evenly spaced values "
                "Kindred tries between the fitted value and each bound, stopping at the first one where the fit is "
                "significantly worse. The last possible one is the bound itself.</li>"
                "<li><b>Maximum evaluations per refit</b>: 300, the evaluations allowed for each refit of the other "
                "values.</li>"
                "<li><b>Optimization method</b>: the method for those refits, Trust Region Reflective by "
                "default.</li></ul>",
                lambda: (),
                lambda: self._varied() and self._profiled(self._fitted),
                "kf1, kr1, k2, and k3 are profiled.",
                waiting="Check Profile Likelihood, with Profile checked for kf1, kr1, k2, and k3.",
                dialog=FittingWindow,
            ),
            Step(
                "run_fit",
                "Run the Fit",
                "<p>Click <b>Run Fit</b> and wait for it to finish. After the fit, Kindred profiles each value in "
                "turn, so this takes longer than a fit alone.</p>",
                lambda: (),
                lambda: self._profiled(self._fitted) and self._lesson_fit(),
                "The fit and its profiles are complete.",
                waiting="Include only profile1 to profile7. Compare A, B, C, Int, and total_PB. Vary and profile kf1, "
                "kr1, k2, and k3. Then click Run Fit and wait for it to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "read_profile",
                "Read the Profiles",
                "<p>Results reads <b>Fit converged · Profile partial</b> and warns that these data do not "
                "determine k3 within its bounds.</p>"
                "<p>Click <b>Summary</b>. k3 ended at its Maximum: <b>Bound Position</b> reads <b>at maximum</b>. "
                "Under <b>Profile Likelihood</b>, its interval is <b>not limited within the search range</b>. k3 "
                "turns PB into PH, and the sheets record only their sum, total_PB. The interval of kf1 has a "
                "lower end only: its upper end could not be located while k3 is free.</p>",
                lambda: (),
                dialog=FittingWindow,
            ),
            Step(
                "hold_k3",
                "Hold k3",
                "<p>It may happen that a rate constant cannot be reliably determined from the provided data, such "
                "as k3 in this case. The sheets record only total_PB, the sum of PB and PH, which shows how fast PB "
                "forms but not how fast it turns into PH. When it is clear that an elementary step is much faster "
                "than the others, it is a valid strategy to hold its rate constant at an arbitrarily high value "
                "rather than fitting it.</p>"
                "<p>On <b>Parameters</b>, uncheck <b>Vary</b> for k3. k3 then keeps its <b>Initial Guess</b>, the "
                "previous result, at its Maximum.</p>"
                "<p>Click <b>Run Fit</b> and wait for it to finish.</p>",
                lambda: (),
                lambda: self._profiled(self._held) and self._lesson_fit(self._held),
                "The fit with k3 held is complete.",
                waiting="Uncheck Vary for k3, click Run Fit, and wait for it to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "read_held",
                "Read the Profiles Again",
                "<p>Results reads <b>Fit converged · Profile complete</b>. Click <b>Summary</b>. Under <b>Profile "
                "Likelihood</b>, the intervals of kf1, kr1, and k2 now each have two ends.</p>",
                lambda: (),
                dialog=FittingWindow,
            ),
            self._finish(
                "<p>Each of profile1 to profile7 was simulated with its own rate constants, each within 5% of "
                "<b>kf1 = 0.5</b>, <b>kr1 = 0.01</b>, <b>k2 = 0.02</b>, and <b>k3 = 100</b> "
                "M<sup>−1</sup> s<sup>−1</sup>.</p>"
            ),
        )


class FittingAgainstMeasuredQuantity(AdvancedFittingLesson):
    title = "Advanced Fitting 6: Fitting Against a Measured Quantity"
    description = (
        "Fit against a measured quantity instead of time: recognize that such data leave the overall rate scale "
        "undetermined, and fix it by fitting one run against time."
    )
    _fitted = ("kf1", "kr1", "k2", "k3")
    # The values fitted once k3 is held.
    _held = ("kf1", "kr1", "k2")
    _sheets = ("base1", "base2", "base3")
    _named = "base1, base2, and base3"
    # The species each sheet compares while Int is its X.
    _targets = frozenset({"A", "B", "C", "PH"})
    # The Fit against choice of each sheet in the first fit and in the second, where base1 is fitted
    # against time and compares Int too.
    _against_int = dict.fromkeys(_sheets, "Int")
    _against_mixed = {"base1": "t", "base2": "Int", "base3": "Int"}
    _mixed_targets = {"base1": _targets | {"Int"}, "base2": _targets, "base3": _targets}
    _first_step = "open_fit"
    _fit_steps = frozenset({
        "reset_setup", "choose_datasets", "fit_against", "choose_targets", "vary_rates", "run_fit", "read_warning",
        "times_ten", "read_scale", "time_base1", "hold_k3", "run_mixed", "read_results",
    })
    _mapped_steps = _fit_steps | {"open_fit"}

    def __init__(self, **owners):
        super().__init__(**owners)
        # The Fit against choice last seen for each dataset the open Fitting Window showed.
        self._x_seen: dict[str, str] = {}
        # The rate constants the first fit against Int ended with.
        self._first_fit: dict[str, float] = {}

    def _observe_fit(self) -> None:
        """Also record the Fit against choice of the dataset the open Fitting Window shows, and the
        result of the first fit against Int."""
        seen = self._seen_fit
        super()._observe_fit()
        window = self._fit_window()
        if window is None or window is not self._seen_fit:
            return
        if window is not seen:
            self._x_seen, self._first_fit = {}, {}
        datasets = window.findChild(QtWidgets.QListWidget, "global_fit_unified_dataset_list")
        x = window.findChild(QtWidgets.QComboBox, "global_fit_parametric_x")
        current = datasets.currentItem()
        if current is not None:
            self._x_seen[current.data(QtCore.Qt.ItemDataRole.UserRole)] = x.currentData()
        if self.step.name in ("run_fit", "read_warning") and self._against(self._against_int) and self._lesson_fit():
            self._first_fit = self._results()

    def _results(self) -> dict[str, float]:
        """The Last Fit Result of each fitted rate constant."""
        roster = self._roster()
        try:
            return {name: float(roster[name]["Last Fit Result"].text()) for name in self._fitted}
        except ValueError:
            return {}

    def _scaled_fit(self) -> bool:
        """The open Fitting Window has a current result against Int whose rate constants are each about
        ten times those of the first fit."""
        first, last = self._first_fit, self._results()
        return (
            bool(first)
            and bool(last)
            and self._against(self._against_int)
            and self._lesson_fit()
            and all(isclose(last[name], 10 * value, rel_tol=0.05) for name, value in first.items())
        )

    def _against(self, choices: dict[str, str]) -> bool:
        """Each sheet was last seen fitted against its choice, "t" for Time."""
        found = self._workbook()
        return bool(found) and all(self._x_seen.get(found[sheet]) == x for sheet, x in choices.items())

    def _compared(self, targets: dict[str, frozenset[str]]) -> bool:
        """Each sheet was last seen comparing exactly its species."""
        found = self._workbook()
        return all(
            frozenset(species for species, row in self._rows_seen.get(found[sheet], {}).items() if row.included)
            == wanted
            for sheet, wanted in targets.items()
        )

    def _base1_timed(self) -> bool:
        return self._against(self._against_mixed) and self._compared(self._mixed_targets)

    def _steps(self) -> tuple[Step, ...]:
        return (
            *self._setup(),
            self._open_fit_step(
                "<p>Some experiments record one measured quantity against another instead of against time, or "
                "record the time unreliably. Kindred can then fit against a measured quantity: it places each row "
                "of the data at the point of the simulated curve that best matches that quantity together with the "
                "other values the fit compares in that row.</p>"
                "<p>This lesson fits the rate constants of M11 to <b>base1</b>, <b>base2</b>, and <b>base3</b>, "
                "first against their measured Int and then with base1 against time.</p>"
            ),
            self._reset_setup_step(),
            self._choose_datasets_step(),
            Step(
                "fit_against",
                "Fit Against Int",
                "<p>Under <b>Fit Data</b>, <b>Fit against</b> reads <b>Time</b>: each row is compared with the "
                "simulated curve at its recorded time. Fit against also lists every measured column of the "
                "dataset.</p>"
                "<p>Select <b>base1</b> and choose <b>Int</b> in Fit against. Do the same for <b>base2</b> and "
                "<b>base3</b>. Int becomes the X of each dataset instead of a target, so its <b>Include</b> is "
                "unchecked and cannot be checked.</p>"
                "<p>Int rises to a peak and falls again, so most Int values occur twice on a curve. A measured "
                "quantity does not need to rise steadily to serve as X.</p>",
                lambda: (),
                lambda: self._against(self._against_int),
                "base1, base2, and base3 are fitted against Int.",
                waiting="Select base1, base2, and base3 in turn and choose Int in Fit against.",
                dialog=FittingWindow,
            ),
            Step(
                "choose_targets",
                "Choose the Targets",
                "<p>The <b>Species Table</b> shows the dataset selected in <b>Datasets</b>. When <b>Include</b> is "
                "checked in a row, the fit compares that species with its mapped column.</p>"
                "<p>Water holds a value only at time 0, its starting concentration, so it has no measurements to fit. "
                "PB never exceeds 3 × 10<sup>−5</sup> M in these sheets, far too little to measure reliably next "
                "to 0.1 to 0.4 M of the other species, so do not compare it either.</p>"
                "<p>Select <b>base1</b>, uncheck <b>Include</b> for Water and PB, and click <b>Copy Targets to "
                "Compatible</b>. It copies these choices to every other dataset that has all the checked species. "
                "Then select <b>base2</b> and <b>base3</b> to confirm that only A, B, C, and PH are included.</p>",
                lambda: (),
                self._targets_chosen,
                "Only A, B, C, and PH are compared in base1, base2, and base3.",
                waiting="Uncheck Include for Water and PB in base1, click Copy Targets to Compatible, and select base2 "
                "and base3 to confirm their targets.",
                dialog=FittingWindow,
            ),
            Step(
                "vary_rates",
                "Vary the Rate Constants",
                "<p>Click <b>Parameters</b> and check <b>Vary</b> for <b>kf1</b>, <b>kr1</b>, <b>k2</b>, and "
                "<b>k3</b>. Leave <b>concentration_floor</b> unchecked.</p>",
                lambda: (),
                self._varied,
                "kf1, kr1, k2, and k3 vary.",
                waiting="Check Vary for kf1, kr1, k2, and k3 only.",
                dialog=FittingWindow,
            ),
            Step(
                "run_fit",
                "Run the Fit",
                "<p>Click <b>Run Fit</b> and wait for the fit to finish.</p>",
                lambda: (),
                lambda: self._against(self._against_int) and self._lesson_fit(),
                "The fit is complete.",
                waiting="Include only base1, base2, and base3, each fitted against Int. Compare A, B, C, and PH. Vary "
                "kf1, kr1, k2, and k3. Then click Run Fit and wait for the fit to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "read_warning",
                "Read the Warning",
                "<p>Results reads <b>Fit converged</b> and warns: <b>These data do not determine the overall rate "
                "scale: multiplying every rate constant by one common factor traces the same curve against the "
                "measured X, only faster or slower.</b></p>"
                "<p>Against Int, the data show which concentrations occur together, but not how fast the reaction "
                "reaches them. Multiplying kf1, kr1, k2, and k3 by 10 gives the same curves of A, B, C, and PH "
                "against Int, reached ten times faster. The next step tries it.</p>",
                lambda: (),
                dialog=FittingWindow,
            ),
            Step(
                "times_ten",
                "Multiply the Guesses by 10",
                "<p>On <b>Parameters</b>, <b>Initial Guess</b> now holds the result of the fit: about 0.516 for kf1, "
                "0.0106 for kr1, 0.0204 for k2, and 28.0 for k3. Enter ten times these as Initial Guess: <b>5.16</b> "
                "for kf1, <b>0.106</b> for kr1, <b>0.204</b> for k2, and <b>280</b> for k3. Then click <b>Run "
                "Fit</b> and wait for the fit to finish.</p>",
                lambda: (),
                self._scaled_fit,
                "The fit is complete.",
                waiting="Enter 5.16, 0.106, 0.204, and 280 as the Initial Guess of kf1, kr1, k2, and k3, then click "
                "Run Fit and wait for the fit to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "read_scale",
                "Compare the Two Fits",
                "<p>Results again reads <b>Fit converged</b> with the same warning, and every rate constant ended "
                "about ten times larger than in the first fit. Both fits match the data equally well: a fit against "
                "a measured quantity alone determines the ratios of the rate constants, not their values.</p>",
                lambda: (),
                dialog=FittingWindow,
            ),
            Step(
                "time_base1",
                "Fit base1 Against Time",
                "<p>One dataset fitted against time sets the rate scale for all of them, while the others stay fitted "
                "against Int.</p>"
                "<p>On <b>Data and Targets</b>, select <b>base1</b> and choose <b>Time</b> in <b>Fit against</b>. "
                "Int can be compared again, but its <b>Include</b> stays unchecked: check it. base1 then compares A, "
                "B, C, Int, and PH over time.</p>",
                lambda: (),
                self._base1_timed,
                "base1 is fitted against time and compares Int.",
                waiting="On Data and Targets, select base1, choose Time in Fit against, and check Include for Int. "
                "Keep base2 and base3 fitted against Int.",
                dialog=FittingWindow,
            ),
            Step(
                "hold_k3",
                "Hold k3",
                "<p>k3 turns PB into PH almost as soon as PB forms, so PH forms as fast as PB does, and only the "
                "small level of PB shows k3. These fits do not compare PB, so they cannot determine k3: fits from "
                "different starting points end with very different k3 and fit the data equally well.</p>"
                "<p>On <b>Parameters</b>, uncheck <b>Vary</b> for k3. k3 then keeps its <b>Initial Guess</b>, the "
                "previous result.</p>",
                lambda: (),
                lambda: self._varied(self._held),
                "kf1, kr1, and k2 vary, and k3 is held.",
                waiting="Uncheck Vary for k3.",
                dialog=FittingWindow,
            ),
            Step(
                "run_mixed",
                "Run the Fit Again",
                "<p>Click <b>Run Fit</b> and wait for the fit to finish.</p>",
                lambda: (),
                lambda: self._included_only()
                and self._base1_timed()
                and self._varied(self._held)
                and self._fit_result(),
                "The fit is complete.",
                waiting="Include only base1, base2, and base3. Fit base1 against Time with Int included, and base2 and "
                "base3 against Int. Vary kf1, kr1, and k2. Then click Run Fit and wait for the fit to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "read_results",
                "Read the Results",
                "<p>Results reads <b>Fit converged</b> without the warning: base1, fitted against time, sets the "
                "rate scale, and base2 and base3 add their measurements against Int.</p>",
                lambda: (),
                dialog=FittingWindow,
            ),
            self._finish(
                "<p>base1 to base3 were simulated with <b>kf1 = 0.5</b>, <b>kr1 = 0.01</b>, <b>k2 = 0.02</b>, and "
                "<b>k3 = 100</b> M<sup>−1</sup> s<sup>−1</sup>. These fits cannot determine k3, so compare kf1, "
                "kr1, and k2 with the fitted values in Results.</p>"
            ),
        )
