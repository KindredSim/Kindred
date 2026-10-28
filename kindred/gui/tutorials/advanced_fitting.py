"""The Advanced Fitting lessons: M11 fitted to the sheets of one workbook, set up once for the series."""

from dataclasses import replace
from functools import cache
from math import isclose
from pathlib import Path
import shutil

from PySide6 import QtCore, QtWidgets

from kindred.core.datasets.foundation import thaw_json_value
from kindred.core.simulator.dsl_parse import prepare_dsl_request_set
from kindred.gui.fitting.constants import DIFFERENTIAL_EVOLUTION_METHOD
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.tutorials.controller import Step
from kindred.gui.tutorials.overlay import Highlight
from kindred.gui.tutorials.practice_lesson import PracticeLesson
from kindred.gui.widgets.dataset_mapping_batch_dialog import DatasetMappingBatchDialog
from kindred.gui.widgets.import_config_dialog import ImportConfigDialog
from kindred.io.paths import file_error_text, resolve_export_start_dir
from kindred.io.resources import get_preset_mechanism_source, get_resource_path

_FOLDER = "AdvancedFitting"
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
# Every sheet is recorded in seconds and M, without a units row; base1-3 also record an absorbance.
_COLUMNS = ("A", "B", "C", "Water", "total_PB")
_UNITS = {
    sheet: ("s", dict.fromkeys(_COLUMNS + (("absorbance",) if sheet.startswith("base") else ()), "M"))
    for sheet in _STARTS
}
_IMPORT_DIALOGS = (ImportConfigDialog, DatasetMappingBatchDialog)
_DE_EVALUATIONS = 3000


@cache
def _preset_equations(preset: str):
    """The preset's equations, which change with its reactions but not with its values."""
    return prepare_dsl_request_set(get_preset_mechanism_source(preset).reactions_text).symbolic_equations_text()


def _shown(parent, kind, predicate=lambda _widget: True):
    """The first matching widget the parent shows, or None while it shows none."""
    return next(
        (widget for widget in parent.findChildren(kind) if widget.isVisibleTo(parent) and predicate(widget)),
        None,
    )


class AdvancedFittingLesson(PracticeLesson):
    """A lesson of the series. The first one started runs the shared setup; the others find
    its tab and start after it. Every lesson applies its fit only to a tab copy, so the setup
    tab keeps the rate constants of M11."""

    _preset_name = "M11"
    _preset_rates = {"kf1": 1.0, "kr1": 0.01, "k2": 0.01, "k3": 100.0}
    _needs_preset_set = False
    # The rate constants the lesson fits.
    _fitted: tuple[str, ...] = ()
    # The first step after the shared setup.
    _first_step = ""
    # Steps taught inside the Fitting Window, and every step that needs the workbook mapped.
    _fit_steps: frozenset[str] = frozenset()
    _mapped_steps: frozenset[str] = frozenset()

    def __init__(self, **owners):
        super().__init__(**owners)
        self._folder: Path | None = None
        self._chosen_parent: str | None = None
        # The Fitting Window last seen, and the tabs present when it opened.
        self._seen_fit: FittingWindow | None = None
        self._tabs_at_open: set[str] = set()
        # Whether the Fitting Window applied a fit to a tab copy.
        self._copied = False
        # The species last seen included for each dataset the open Fitting Window showed.
        self._targets_seen: dict[str, frozenset[str]] = {}
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
            if identity not in self.surfaces:
                continue
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
        if name in self._fit_steps and self._fit_window() is None and not self._copied:
            message, label, _ = self._missing("This exercise needs the Fitting Window open.", "open_fit")
            return message, label, self._reopen_fit
        return None

    def _reopen_fit(self) -> None:
        """Closing the Fitting Window discards its datasets, choices, and result, so its steps
        wait to be done again from Open the Fitting Window."""
        self.revisitRequested.emit("open_fit")

    def _fit_closed(self, *_args) -> None:
        self._seen_fit = None
        if self.step.name in self._fit_steps and not self._copied:
            self._reopen_fit()

    # -- the tutorial file -----------------------------------------------------
    def _save_file(self) -> None:
        window = self.workspace.window()
        parent = QtWidgets.QFileDialog.getExistingDirectory(
            window, "Save Tutorial File", resolve_export_start_dir(self._chosen_parent)
        )
        if not parent:
            return
        self._chosen_parent = parent
        target = Path(parent) / _FOLDER
        if (target / _FILE).exists():
            answer = QtWidgets.QMessageBox.question(
                window,
                "Save Tutorial File",
                f"{target} already contains {_FILE}. Replace it?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                self._folder = target
                return
        try:
            target.mkdir(exist_ok=True)
            shutil.copyfile(get_resource_path(f"tutorial_files/{_FOLDER}/{_FILE}"), target / _FILE)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                window, "Save Tutorial File", file_error_text(exc, exc.filename or target)
            )
            return
        self._folder = target

    def _saved(self) -> bool:
        return self._folder is not None and (self._folder / _FILE).is_file()

    # -- observations ----------------------------------------------------------
    def _workbook(self) -> dict[str, str]:
        """Sheet name -> dataset ID of the newest sheets of advanced_fitting.xlsx imported in this
        series with every column in seconds and M."""
        document = self.registry.document
        runs = {}
        for record in document.datasets:
            if (
                record.dataset_id not in self._datasets_before
                and record.source_id is not None
                and document.imported_source(record.source_id).original_filename == _FILE
            ):
                # The worksheet identity survives renaming the dataset.
                metadata = thaw_json_value(record.source_metadata)
                run = metadata.get("sheet_name")
                if run in _UNITS and (
                    metadata.get("original_time_unit"), metadata.get("original_concentration_units")
                ) == _UNITS[run]:
                    runs[run] = record.dataset_id
        return runs if len(runs) == len(_UNITS) else {}

    def _import_dialog_open(self) -> bool:
        dialog = self._dialog(ImportConfigDialog)
        return dialog is not None and dialog.windowTitle() == f"Import: {_FILE}"

    def _expected_mapping(self, dataset_id: str) -> dict[str, str]:
        # M11 has no absorbance; Advanced Fitting 2 defines it and maps the column.
        return {
            series.raw_observation_id: series.column_name
            for series in self.registry.document.dataset(dataset_id).observations
            if series.column_name != "absorbance"
        }

    @staticmethod
    def _starts(run: str) -> dict[str, float]:
        a, b, c = _STARTS[run]
        return {"A": a, "B": b, "C": c, "Int": 0.0, "Water": _WATER, "PB": 0.0, "PH": 0.0, "BOH": 0.0}

    def _mapped(self) -> bool:
        """Each sheet maps its columns by name to its own set, which starts from its row at time 0."""
        found = self._workbook()
        tab = self.tab
        sets = {item.set_id: item for item in tab.initial_condition_sets}
        used = set()
        for run, dataset_id in found.items():
            link = tab.dataset_link(dataset_id)
            if (
                link is None
                or link.set_id not in sets
                or link.set_id in used
                or dict(link.observation_mapping) != self._expected_mapping(dataset_id)
                or not all(
                    isclose(dict(sets[link.set_id].values).get(species, -1.0), value, abs_tol=1e-12)
                    for species, value in self._starts(run).items()
                )
            ):
                return False
            used.add(link.set_id)
        return bool(found)

    def _fit_window(self) -> FittingWindow | None:
        """The open Fitting Window, also while a dialog opened from it is in front."""
        widget = QtWidgets.QApplication.activeModalWidget()
        while widget is not None and not isinstance(widget, FittingWindow):
            widget = widget.parentWidget()
        return widget

    def _observe_fit(self) -> None:
        """Record whether an open Fitting Window of the lesson tab applied the lesson's final fit to a
        tab copy. The window blocks every other edit, so while it is open a new tab can only come from
        Apply Fit to Tab Copy, which needs a current result."""
        window = self._fit_window()
        if window is None or self.practice is None or self.registry.active_tab_id != self.practice:
            return
        tabs = {tab.tab_id for tab in self.registry.document.tabs}
        if window is not self._seen_fit:
            self._seen_fit, self._tabs_at_open, self._targets_seen = window, tabs, {}
            window.destroyed.connect(self._fit_closed)
        shown = self._shown_targets(window)
        if shown is not None:
            self._targets_seen[shown[0]] = shown[1]
        # A copy of an earlier result does not count, now or later.
        new, self._tabs_at_open = tabs - self._tabs_at_open, tabs
        self._copied = self._copied or bool(new and self._final_setup())

    def _included_only(self, sheets: tuple[str, ...]) -> bool:
        """The open Fitting Window includes exactly the datasets of these sheets."""
        window = self._fit_window()
        datasets = None if window is None else window.findChild(QtWidgets.QListWidget, "global_fit_unified_dataset_list")
        found = self._workbook()
        if datasets is None or not found:
            return False
        checked = {
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in (datasets.item(row) for row in range(datasets.count()))
            if item.checkState() == QtCore.Qt.CheckState.Checked
        }
        return checked == {found[sheet] for sheet in sheets}

    @staticmethod
    def _shown_targets(window: FittingWindow) -> tuple[str, frozenset[str]] | None:
        """The dataset the Species Table shows and the species it includes. The table shows one
        dataset at a time, and only that one can be changed or copied to the others."""
        datasets = window.findChild(QtWidgets.QListWidget, "global_fit_unified_dataset_list")
        table = window.findChild(QtWidgets.QTableWidget, "global_fit_unified_species_table")
        current = None if datasets is None else datasets.currentItem()
        if current is None or table is None:
            return None
        headers = [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
        include, species = headers.index("Include"), headers.index("Species")
        included = frozenset(
            table.item(row, species).text()
            for row in range(table.rowCount())
            if (item := table.item(row, include)) is not None
            and item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable
            and item.checkState() == QtCore.Qt.CheckState.Checked
        )
        return current.data(QtCore.Qt.ItemDataRole.UserRole), included

    def _roster(self) -> dict[str, dict[str, QtWidgets.QTableWidgetItem]]:
        """Parameter -> its row of the open Fitting Window's parameter table, by column title."""
        window = self._fit_window()
        roster = None if window is None else window.findChild(QtWidgets.QTableWidget, "fit_parameter_roster")
        if roster is None:
            return {}
        headers = [roster.horizontalHeaderItem(column).text() for column in range(roster.columnCount())]
        rows = {}
        for row in range(roster.rowCount()):
            cells = {header: roster.item(row, column) for column, header in enumerate(headers)}
            name = cells["Parameter"]
            # Rows of single datasets follow their parameter row.
            if name is not None and name.data(QtCore.Qt.ItemDataRole.UserRole + 1) is None:
                rows[name.data(QtCore.Qt.ItemDataRole.UserRole)] = cells
        return rows

    def _varied(self) -> bool:
        """The open Fitting Window varies the lesson's rate constants and nothing else."""
        roster = self._roster()
        varied = {
            name for name, cells in roster.items()
            if cells["Vary"] is not None and cells["Vary"].checkState() == QtCore.Qt.CheckState.Checked
        }
        return bool(roster) and varied == set(self._fitted)

    def _fit_result(self) -> bool:
        """The open Fitting Window has a current result to apply. Changing the fit setup makes the
        shown result stale until the fit runs again."""
        window = self._fit_window()
        apply = None if window is None else _shown(
            window, QtWidgets.QPushButton, lambda button: button.text().replace("&", "") == "Apply Fit"
        )
        return apply is not None and apply.isEnabled()

    def _final_setup(self) -> bool:
        """The open Fitting Window holds the setup whose result the lesson applies to a tab copy."""
        return False

    # -- targets ---------------------------------------------------------------
    def _page_tab(self, page) -> Highlight:
        """The tab of a Data panel page, for a page that is not shown."""
        tabs = self.bundle.right_panel.findChild(QtWidgets.QTabWidget)
        bar = tabs.tabBar()
        return Highlight(bar, lambda: bar.tabRect(tabs.indexOf(page)), lambda: self.workspace.reveal_dock("data"))

    def _data_page(self) -> Highlight:
        """The Data page of the Data panel, or its tab while another page is shown."""
        panel = self.bundle.data_panel
        return self._panel(panel, "data") if panel.isVisible() else self._page_tab(panel)

    def _open_targets(self) -> tuple[Highlight, ...]:
        return () if self._dialog(_IMPORT_DIALOGS) is not None else (self._data_page(),)

    def _review_targets(self) -> tuple[Highlight, ...]:
        """Nothing in the mapping review; the Data page once the review is closed."""
        return () if self._dialog(DatasetMappingBatchDialog) is not None else (self._data_page(),)

    def _fit_page(self, title: str, widget_type, name: str) -> tuple[Highlight, ...]:
        """A control of the open Fitting Window, or the tab of its page while another page is shown."""
        window = self._fit_window()
        if window is None:
            return ()
        widget = window.findChild(widget_type, name)
        if widget is not None and widget.isVisible():
            return (Highlight(widget),)
        bar = window.findChild(QtWidgets.QTabBar, "global_fit_top_tabs")
        index = next(index for index in range(bar.count()) if bar.tabText(index) == title)
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
                "<li><b>base1</b> to <b>base3</b>: Calculated Quantities and Constraints</li>"
                "<li><b>stock1</b> to <b>stock3</b>: Varying Initial Conditions</li>"
                "<li><b>k2runs1</b> to <b>k2runs3</b>: Per-Dataset Rate Constants</li>"
                "<li><b>profile1</b> to <b>profile7</b>: Profile Likelihood</li></ul>"
                "<p>Each sheet records A and total_PB from 0 to 1800 s, and base1 to base3 also record an "
                "<b>absorbance</b>. The row at time 0 gives the starting concentrations of A, B, C, and Water.</p>"
                "<p>The data were simulated from M11 with rate constants or starting concentrations that differ "
                "from those in Reactions, plus about 1% noise. Each lesson ends with the values used.</p>"
                "<p>Every sheet is recorded in seconds and M, the default units, and names its columns exactly as "
                "M11 does. Every column except absorbance therefore maps automatically.</p>",
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
                dialog=_IMPORT_DIALOGS,
            ),
            Step(
                "import_sheets",
                "Import Every Sheet",
                "<p>All 19 sheets are checked under <b>Sheets</b>. Each is read in seconds and M, as recorded.</p>"
                "<p>Click <b>Import</b>.</p>",
                lambda: (),
                lambda: bool(self._workbook()),
                "The workbook is imported.",
                dialog=_IMPORT_DIALOGS,
            ),
            Step(
                "accept_mapping",
                "Accept the Mapping",
                "<p>The mapping review maps the columns of each dataset to their species and to total_PB. M11 has "
                "no absorbance, so in base1 to base3 that column stays under <b>Unused columns</b>. "
                "Advanced Fitting 2 maps it.</p>"
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
        )


class BoundsAndSolvers(AdvancedFittingLesson):
    title = "Advanced Fitting 1: Bounds and Solvers"
    description = (
        "Fit rate constants whose true values lie far from the guesses: recognize a fit stopped by its bounds, "
        "widen the bounds on a log scale, and search the whole range with Differential Evolution."
    )
    _fitted = ("kf1", "kr1", "k2")
    _sheets = ("bounds1", "bounds2", "bounds3")
    _first_step = "open_fit"
    _fit_steps = frozenset({
        "choose_datasets", "choose_targets", "vary_rates", "run_fit", "read_results", "set_bounds", "run_wide",
        "use_de", "apply_copy",
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
            except (AttributeError, ValueError):
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
        method = None if window is None else window.findChild(QtWidgets.QComboBox, "global_fit_method")
        # The label stays found while another page of the window is shown.
        label = None if window is None else next(
            (label for label in window.findChildren(QtWidgets.QLabel) if label.text() == "Maximum evaluations:"), None
        )
        try:
            evaluations = int(label.buddy().text())
        except (AttributeError, ValueError):
            return False
        # A typed number applies on Enter or focus loss; until then the box is modified.
        return (
            method.currentData() == DIFFERENTIAL_EVOLUTION_METHOD
            and not label.buddy().isModified()
            and evaluations >= _DE_EVALUATIONS
        )

    def _final_setup(self) -> bool:
        return self._bounds_widened() and self._de_chosen()

    def _targets_chosen(self) -> bool:
        """bounds1 to bounds3 were each last seen comparing only A and total_PB."""
        found = self._workbook()
        return all(self._targets_seen.get(found.get(sheet)) == self._targets for sheet in self._sheets)

    def _lesson_fit(self) -> bool:
        """The open Fitting Window has a current result for bounds1 to bounds3 comparing A and total_PB,
        with kf1, kr1, and k2 varied."""
        return (
            self._included_only(self._sheets) and self._targets_chosen() and self._varied() and self._fit_result()
        )

    def _bounds_target(self) -> tuple[Highlight, ...]:
        """Set Bounds for Parameters…, or the Parameters tab while another page is shown."""
        window = self._fit_window()
        button = None if window is None else _shown(
            window, QtWidgets.QPushButton, lambda button: button.text() == "Set Bounds for Parameters…"
        )
        if button is not None and button.isVisible():
            return (Highlight(button),)
        return self._fit_page("Parameters", QtWidgets.QTableWidget, "fit_parameter_roster")

    def _steps(self) -> tuple[Step, ...]:
        return (
            *self._setup(),
            Step(
                "open_fit",
                "Open the Fitting Window",
                "<p>This lesson fits kf1, kr1, and k2 of M11 to <b>bounds1</b>, <b>bounds2</b>, and "
                "<b>bounds3</b>. Their true values lie far from those in Reactions, as when a mechanism meets a "
                "new system for the first time.</p>"
                "<p>Choose <b>Fitting → Global Fit…</b>.</p>",
                lambda: (),
                lambda: self._fit_window() is not None or self._copied,
                "The Fitting Window is open.",
                dialog=FittingWindow,
            ),
            Step(
                "choose_datasets",
                "Choose the Datasets",
                "<p>In <b>Datasets</b>, check bounds1, bounds2, and bounds3, and leave the other datasets "
                "unchecked. They belong to the other lessons.</p>",
                lambda: self._fit_page("Data and Targets", QtWidgets.QListWidget, "global_fit_unified_dataset_list"),
                lambda: self._copied or self._included_only(self._sheets),
                "bounds1, bounds2, and bounds3 are included.",
                waiting="Check bounds1, bounds2, and bounds3 only.",
                dialog=FittingWindow,
            ),
            Step(
                "choose_targets",
                "Choose the Targets",
                "<p>The <b>Species Table</b> shows the dataset selected in <b>Datasets</b>. A checked "
                "<b>Include</b> compares that species with its column in the fit, and every mapped column starts "
                "checked.</p>"
                "<p>B, C, and Water hold a value only at time 0, the starting concentration, so they have no "
                "measurements to fit. Compare only species measured over time.</p>"
                "<p>Select <b>bounds1</b>, uncheck <b>Include</b> for B, C, and Water, and click <b>Copy Targets "
                "to Compatible</b>. It copies these choices to every other dataset that has all the checked "
                "species. Then select <b>bounds2</b> and <b>bounds3</b> to check that only A and total_PB are "
                "included.</p>",
                lambda: (),
                lambda: self._copied or self._targets_chosen(),
                "Only A and total_PB are compared in bounds1, bounds2, and bounds3.",
                waiting="Uncheck Include for B, C, and Water in bounds1, click Copy Targets to Compatible, and "
                "select bounds2 and bounds3 to check them.",
                dialog=FittingWindow,
            ),
            Step(
                "vary_rates",
                "Vary the Rate Constants",
                "<p>Click <b>Parameters</b> and check <b>Vary</b> for <b>kf1</b>, <b>kr1</b>, and <b>k2</b>. "
                "Leave <b>k3</b> and <b>concentration_floor</b> unchecked.</p>"
                "<p><b>Minimum</b> and <b>Maximum</b> start at 1/100 and 100 times the value in Reactions. "
                "kr1 and k2, at 0.01 in Reactions, can therefore range only from 0.0001 to 1.</p>",
                lambda: self._fit_page("Parameters", QtWidgets.QTableWidget, "fit_parameter_roster"),
                lambda: self._copied or self._varied(),
                "kf1, kr1, and k2 vary.",
                waiting="Check Vary for kf1, kr1, and k2 only.",
                dialog=FittingWindow,
            ),
            Step(
                "run_fit",
                "Run the Fit",
                "<p>Click <b>Run Fit</b> and wait for the fit to finish.</p>",
                lambda: (),
                lambda: self._copied or self._lesson_fit(),
                "The fit is complete.",
                waiting="With bounds1, bounds2, and bounds3 included, comparing only A and total_PB, and kf1, kr1, "
                "and k2 varied, click Run Fit and wait for the fit to finish.",
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
                lambda: self._copied or self._bounds_widened(),
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
                lambda: self._copied or (self._bounds_widened() and self._lesson_fit()),
                "The fit with the wider bounds is complete.",
                waiting="Click Run Fit with the wider bounds and wait for the fit to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "use_de",
                "Search the Whole Range",
                "<p><b>Trust Region Reflective</b>, the default <b>Optimization method</b>, is a local optimization "
                "method. This means that the quality of the found solution is heavily dependent on the initial "
                "guess, and any solutions found are very likely to be local minima. An alternative Optimization "
                "method, albeit significantly more expensive, is <b>Differential Evolution</b>, a global search "
                'algorithm. Differential Evolution attempts to "escape" from local minima in order to find a better '
                "solution across the provided parameter space. It searches the whole range between Minimum and "
                "Maximum, so it is not necessarily guess dependent, and the quality of the results depends on the "
                "provided parameter space (Minimum and Maximum values), as well as on the number of evaluations "
                "(1000 by default).</p>"
                "<p>In the <b>Parameters</b> tab, under <b>Fit Settings</b>, choose <b>Differential Evolution</b> in Optimization method, set "
                f"<b>Maximum evaluations</b> to <b>{_DE_EVALUATIONS}</b>, and click <b>Run Fit</b>. With the "
                "default 1000, it stops early, and Results reads <b>Optimizer did not converge</b>.</p>"
                "<p>It finds the same values as the previous fit. In this case, the local minimum found by Trust "
                "Region Reflective is also the global minimum, a consequence of the simple model and the few values "
                "fitted. More complex models, more species, or more datasets are highly unlikely to give a local "
                "minimum that is also the global minimum. It is therefore advisable to use Differential Evolution "
                "when no reliable estimates are known, whereas local methods such as Trust Region Reflective are "
                "best for faster fits when a local minimum is sufficient, and for refining global solutions, for "
                "example after adding another dataset or value to the fit.</p>",
                lambda: (),
                lambda: self._copied or (self._final_setup() and self._lesson_fit()),
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
                lambda: self._copied,
                "The fitted copy is in the project.",
                waiting="Click Apply Fit and choose Apply Fit to Tab Copy.",
                dialog=FittingWindow,
            ),
            self._finish(
                "<p>advanced_fitting.xlsx simulated bounds1 to bounds3 with <b>kf1 = 0.05</b>, <b>kr1 = 3</b>, and "
                "<b>k2 = 2</b> M<sup>−1</sup> s<sup>−1</sup>. Close the <b>Fitting Window</b> and compare them with "
                "the fitted values in Reactions of the copy.</p>"
            ),
        )
