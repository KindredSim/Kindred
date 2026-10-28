"""Fit the rate constants of M5 to the three runs of an Excel workbook."""

from dataclasses import replace
from pathlib import Path
import shutil

from PySide6 import QtCore, QtWidgets

from kindred.core.datasets.foundation import thaw_json_value
from kindred.core.project_presentation_state import SimulationSeriesAxisSelection
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.parameter_enumeration import enumerate_parameter_coordinates
from kindred.gui.tutorials.controller import Step
from kindred.gui.tutorials.overlay import Highlight
from kindred.gui.tutorials.practice_lesson import PracticeLesson
from kindred.gui.widgets.dataset_mapping_batch_dialog import DatasetMappingBatchDialog
from kindred.gui.widgets.import_config_dialog import ImportConfigDialog
from kindred.io.paths import file_error_text, resolve_export_start_dir
from kindred.io.resources import get_resource_path

_FOLDER = "FitBasics"
_FILE = "fit_runs.xlsx"
_RUNS = ("run1", "run2", "run3")
_A0 = {"run1": 0.5, "run2": 1.0, "run3": 2.0}
# Every sheet has no units row, so it is read in the default units.
_UNITS = ("s", {"A": "M", "B": "M", "C": "M"})
# The rate constants the lesson fits.
_RATES = ("kf1", "kr1", "k2")
_PLOTTED = (SimulationSeriesAxisSelection("B"),)
_IMPORT_DIALOGS = (ImportConfigDialog, DatasetMappingBatchDialog)


def _shown(parent, kind, predicate=lambda _widget: True):
    """The first matching widget the parent shows, or None while it shows none."""
    return next(
        (widget for widget in parent.findChildren(kind) if widget.isVisibleTo(parent) and predicate(widget)),
        None,
    )


class FittingBasics(PracticeLesson):
    title = "Fitting Basics"
    description = (
        "Fit the rate constants of M5 to three runs in an Excel workbook: choose the datasets "
        "and species to compare, choose the rate constants to vary, run the fit, read the results, and apply them."
    )
    _preset_name = "M5"
    _preset_rates = {"kf1": 1.5, "kr1": 0.25, "k2": 0.1}
    _run_steps = frozenset({"run_data", "run_again"})
    _needs_preset_set = False
    # Steps taught inside the Fitting Window.
    _fit_steps = frozenset({
        "choose_datasets", "choose_species", "vary_rates", "run_fit", "read_results", "apply_copy", "apply_fit",
    })
    _mapped_steps = _fit_steps | {"show_data", "choose_y", "run_data", "open_fit", "run_again"}

    def __init__(self, **owners):
        super().__init__(**owners)
        self._folder: Path | None = None
        self._chosen_parent: str | None = None
        # Datasets present when the lesson started.
        self._datasets_before = {record.dataset_id for record in self.registry.document.datasets}
        # The Fitting Window last seen, with the rate constants and tabs present when it opened.
        self._seen_fit: FittingWindow | None = None
        self._rates_at_open: dict[str, float] = {}
        self._tabs_at_open: set[str] = set()
        # What the Fitting Window applied: a tab copy, and the rate constants written into Reactions.
        self._copied = False
        self._applied_rates: dict[str, float] | None = None

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
                "This exercise needs run1, run2, and run3 of fit_runs.xlsx imported with all their columns in seconds and M.",
                "open_workbook",
            )
        if name in self._mapped_steps and not self._mapped():
            return self._missing(
                "This exercise needs each workbook dataset mapped to its own set from its data.", "accept_mapping"
            )
        if name in self._fit_steps and self._fit_window() is None and not self._fit_applied():
            message, label, _ = self._missing("This exercise needs the Fitting Window open.", "open_fit")
            return message, label, self._reopen_fit
        return None

    def _reopen_fit(self) -> None:
        """Closing the Fitting Window discards its datasets, Vary choices, and result, so their
        steps wait to be done again."""
        for name in ("run_fit", "vary_rates", "choose_datasets", "open_fit"):
            self.revisitRequested.emit(name)

    def _fit_closed(self, *_args) -> None:
        self._seen_fit = None
        if self.step.name in self._fit_steps and not self._fit_applied():
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
    def _rates(self) -> dict[str, float]:
        values = {item.name: float(item.value) for item in enumerate_parameter_coordinates(self.tab.prepared_dsl)}
        return {name: values[name] for name in _RATES if name in values}

    def _observe_fit(self) -> None:
        """Record what an open Fitting Window of the lesson tab applies. The window blocks every other
        edit, so while it is open a new tab can only come from Apply Fit to Tab Copy, and new rate
        constants only from Apply Fit to Project."""
        window = self._dialog(FittingWindow)
        if window is None or self.practice is None or self.registry.active_tab_id != self.practice:
            return
        tabs = {tab.tab_id for tab in self.registry.document.tabs}
        if window is not self._seen_fit:
            self._seen_fit, self._rates_at_open, self._tabs_at_open = window, self._rates(), tabs
            window.destroyed.connect(self._fit_closed)
        self._copied = self._copied or bool(tabs - self._tabs_at_open)
        if (rates := self._rates()) != self._rates_at_open:
            self._applied_rates = rates

    def _fit_applied(self) -> bool:
        """Reactions holds the rate constants that Apply Fit to Project wrote. Applying the fit
        also proves the Fitting Window steps that lead to it."""
        return self._applied_rates is not None and self._rates() == self._applied_rates

    def _mechanism_loaded(self) -> bool:
        # Applying the fit replaces the preset rates with the fitted ones.
        return self.tab.prepared_dsl is not None and (self._preset_rates_loaded() or self._fit_applied())

    def _workbook(self) -> dict[str, str]:
        """Run name -> dataset ID of the newest sheets of fit_runs.xlsx imported in this lesson
        with every column in seconds and M."""
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
                if run in _RUNS and (
                    metadata.get("original_time_unit"), metadata.get("original_concentration_units")
                ) == _UNITS:
                    runs[run] = record.dataset_id
        return runs if len(runs) == len(_RUNS) else {}

    def _import_dialog_open(self) -> bool:
        dialog = self._dialog(ImportConfigDialog)
        return dialog is not None and dialog.windowTitle() == f"Import: {_FILE}"

    def _expected_mapping(self, dataset_id: str) -> dict[str, str]:
        return {
            series.raw_observation_id: series.column_name
            for series in self.registry.document.dataset(dataset_id).observations
        }

    def _mapped(self) -> bool:
        """Each run maps A, B, and C by name to its own set, which starts from its row at time 0."""
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
                or dict(sets[link.set_id].values) != {"A": _A0[run], "B": 0.0, "C": 0.0}
            ):
                return False
            used.add(link.set_id)
        return bool(found)

    def _overlays_shown(self) -> bool:
        """Plot Overlays checks every column of each workbook dataset."""
        overlays = dict(self.registry.presentation(self.tab_id).overlay_raw_observation_ids_by_dataset)
        return all(
            set(self._expected_mapping(dataset_id)) <= set(overlays.get(dataset_id, ()))
            for dataset_id in self._workbook().values()
        )

    def _y_chosen(self) -> bool:
        return self.registry.presentation(self.tab_id).selected_y_series == _PLOTTED

    def _workbook_sets(self) -> tuple[str, ...]:
        """The set of each workbook run; the Run steps are blocked until each is mapped."""
        found = self._workbook()
        return tuple(self.tab.dataset_link(found[run]).set_id for run in _RUNS)

    def _fit_window(self) -> FittingWindow | None:
        return self._dialog(FittingWindow)

    def _included(self) -> bool:
        """The open Fitting Window includes the three workbook datasets."""
        window = self._fit_window()
        datasets = None if window is None else window.findChild(QtWidgets.QListWidget, "global_fit_unified_dataset_list")
        if datasets is None:
            return False
        checked = {
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in (datasets.item(row) for row in range(datasets.count()))
            if item.checkState() == QtCore.Qt.CheckState.Checked
        }
        return set(self._workbook().values()) <= checked

    def _varied(self) -> bool:
        """The open Fitting Window varies kf1, kr1, and k2, and leaves concentration_floor fixed."""
        window = self._fit_window()
        roster = None if window is None else window.findChild(QtWidgets.QTableWidget, "fit_parameter_roster")
        if roster is None:
            return False
        headers = [roster.horizontalHeaderItem(column).text() for column in range(roster.columnCount())]
        name_column, vary_column = headers.index("Parameter"), headers.index("Vary")
        varied = {}
        for row in range(roster.rowCount()):
            name, vary = roster.item(row, name_column), roster.item(row, vary_column)
            # Rows of single datasets follow their parameter row.
            if name is not None and vary is not None and name.data(QtCore.Qt.ItemDataRole.UserRole + 1) is None:
                varied[name.data(QtCore.Qt.ItemDataRole.UserRole)] = vary.checkState() == QtCore.Qt.CheckState.Checked
        return all(varied.get(name) for name in _RATES) and not varied.get("concentration_floor", False)

    def _fit_result(self) -> bool:
        """The open Fitting Window has a current result to apply."""
        window = self._fit_window()
        apply = None if window is None else _shown(
            window, QtWidgets.QPushButton, lambda button: button.text().replace("&", "") == "Apply Fit"
        )
        return apply is not None and apply.isEnabled()

    def _ran(self, rates: dict[str, float]) -> bool:
        identities = self._workbook_sets()
        return all(
            (settings := self.tab.resolved_execution_settings(identity)).t_start == 0 and settings.t_end == 600
            for identity in identities
        ) and self._run_ok(rates, identities)

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

    def _overlay_target(self) -> Highlight:
        """The Plot Overlays list, which stays in view while its datasets are expanded."""
        panel = self.bundle.overlay_panel
        if not panel.isVisible():
            return self._page_tab(panel)
        return self._panel(panel.findChild(QtWidgets.QTreeWidget), "data")

    def _y_target(self) -> tuple[Highlight, ...]:
        """The Y selector above the plot."""
        plot = self.workspace.plot_for_tab(self.tab_id)
        button = _shown(plot, QtWidgets.QToolButton, lambda button: button.text().startswith("Y:"))
        return () if button is None else (Highlight(button),)

    def _run_targets_now(self) -> tuple[Highlight, ...]:
        """Initial Conditions and both Run buttons, once the Fitting Window is closed."""
        return () if self._fit_window() is not None else (self._initial_target(), *self._run_targets())

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

    def _steps(self) -> tuple[Step, ...]:
        create_tab, load_preset = self._setup_steps()
        return (
            create_tab,
            replace(
                load_preset,
                title="Load M5",
                instruction="<p>Open <b>Examples → Preset Mechanisms → M5: A &lt;-&gt; B, B -&gt; C</b>.</p>"
                "<p>Follow the highlighted menu entries. There is no need to click <b>Next</b> between menus.</p>"
                "<p>M5 has three species, <b>A</b>, <b>B</b>, and <b>C</b>, and three rate constants: "
                "<b>kf1</b> and <b>kr1</b> for the reversible step A ⇌ B, and <b>k2</b> for B → C. "
                "In this lesson, the sets come from the data.</p>",
                complete=lambda: self.tab.prepared_dsl is not None and self._preset_rates_loaded(),
                success="M5 is loaded.",
            ),
            Step(
                "save_file",
                "Save the Tutorial File",
                "<p>This lesson uses <b>fit_runs.xlsx</b>, an Excel workbook supplied with Kindred.</p>"
                "<p>Click <b>Save Tutorial File…</b> and choose a folder. Kindred creates a "
                "<b>FitBasics</b> folder there and copies the workbook into it.</p>",
                lambda: None,
                self._saved,
                "fit_runs.xlsx is in the FitBasics folder.",
                waiting="Save the tutorial file to continue.",
                action=("Save Tutorial File…", self._save_file),
            ),
            Step(
                "about_workbook",
                "What Is in the Workbook",
                "<p>fit_runs.xlsx holds three runs of M5, one per sheet: <b>run1</b>, <b>run2</b>, and <b>run3</b>. "
                "They start from 0.5, 1, and 2 M of A, with no B or C, and record A, B, and C from 0 to 600 s.</p>"
                "<p>The data were simulated from M5 with rate constants that differ from those in Reactions, "
                "plus about 1% noise. The end of the lesson gives the values used.</p>"
                "<p>Every sheet is recorded in seconds and M, the default units, and names its columns exactly as the "
                "species in Reactions. Every column therefore maps automatically, and the row at time 0 supplies the "
                "starting concentrations. <b>Import and Map Data</b> covers other units and names.</p>",
                lambda: None,
            ),
            Step(
                "open_workbook",
                "Open the Workbook",
                "<p>Choose <b>File → Import Data…</b>, or click <b>Import Data…</b> in the <b>Data</b> panel. "
                "Open <b>fit_runs.xlsx</b> from the FitBasics folder.</p>",
                self._open_targets,
                lambda: self._import_dialog_open() or bool(self._workbook()),
                "fit_runs.xlsx is open in the import dialog or already imported.",
                dialog=_IMPORT_DIALOGS,
            ),
            Step(
                "import_sheets",
                "Import Every Sheet",
                "<p>run1, run2, and run3 are checked under <b>Sheets</b>. Each is read in seconds and M, "
                "as recorded.</p>"
                "<p>Click <b>Import</b>.</p>",
                lambda: (),
                lambda: bool(self._workbook()),
                "The workbook is imported.",
                dialog=_IMPORT_DIALOGS,
            ),
            Step(
                "accept_mapping",
                "Accept the Mapping",
                "<p>The mapping review maps A, B, and C of each dataset to their species. "
                "<b>Initial-Condition Sets</b> previews a new set for each dataset, starting from its row at "
                "time 0: A at 0.5, 1, and 2 M, and B and C at 0.</p>"
                "<p>Click <b>OK</b>. Each dataset gets a new set named after it.</p>",
                self._review_targets,
                self._mapped,
                "Each sheet is mapped to its own new set.",
                waiting="Click OK in the mapping review. If it was closed, choose each workbook dataset in "
                "Dataset in the Data panel, click Change IC Set…, then Create IC Set from Data…, and click OK.",
                dialog=DatasetMappingBatchDialog,
            ),
            Step(
                "show_data",
                "Show the Data on the Plot",
                "<p>Imported data are drawn only for the columns checked in <b>Plot Overlays</b>, beside "
                "<b>Data</b> in the Data panel. Click <b>Plot Overlays</b>.</p>"
                "<p>In the highlighted list, check the three workbook datasets. Checking a dataset checks all its "
                "columns.</p>",
                self._overlay_target,
                self._overlays_shown,
                "The workbook data are shown on the plot.",
                waiting="Check the three workbook datasets.",
            ),
            Step(
                "choose_y",
                "Choose What to Plot",
                "<p>Open the <b>Y</b> selector above the plot and select only <b>B</b>, then close the selector. "
                "The intermediate shows most clearly how well the rate constants match the data.</p>"
                "<p>Markers follow the same choice: only the B columns stay on the plot.</p>",
                self._y_target,
                self._y_chosen,
                "Only B is selected for Y.",
            ),
            Step(
                "run_data",
                "Run the Sets from the Workbook",
                "<p>The workbook runs from 0 to 600 s. In <b>Initial Conditions</b>, check that "
                "<b><i>t</i><sub>start</sub></b> is 0 and <b><i>t</i><sub>end</sub></b> is 600 seconds, and press "
                "<b>Enter</b> after each edit.</p>"
                "<p>Select only the three sets mapped to the workbook datasets, "
                "such as <b>fit_runs.xlsx::run1</b>, with Ctrl-click, or Cmd-click on macOS. Then click <b>Run</b>.</p>"
                "<p>Compare the B curve of each run with its markers.</p>",
                lambda: (self._initial_target(), *self._run_targets()),
                lambda: self._ran(self._preset_rates),
                "The sets from the workbook have run.",
                waiting="Set t_start to 0 and t_end to 600 seconds, select only the three workbook sets, and click Run.",
            ),
            Step(
                "open_fit",
                "Open the Fitting Window",
                "<p>The simulated curves are far from the data. M5 predicts that B forms and is consumed much "
                "faster than measured, so the rate constants of M5 do not describe these runs well. Fitting the "
                "model to the data, with the rate constants as fit variables, finds values that describe them "
                "better.</p>"
                "<p>Choose <b>Fitting → Global Fit…</b>.</p>"
                "<p>The <b>Fitting Window</b> works on the datasets mapped to this tab. Nothing in the project "
                "changes until a fit is applied.</p>",
                lambda: (),
                lambda: self._fit_window() is not None or self._fit_applied(),
                "The Fitting Window is open, or the fit is applied.",
                dialog=FittingWindow,
            ),
            Step(
                "choose_datasets",
                "Choose the Datasets",
                "<p>On <b>Data and Targets</b>, <b>Datasets</b> lists every dataset mapped to this tab. The "
                "checkbox beside each name includes that dataset in the fit. Clicking a name shows the settings "
                "of that dataset beside the list and its species in the <b>Species Table</b>.</p>"
                "<p>Check the three workbook datasets. One value of each rate constant then has to explain all "
                "three runs.</p>",
                lambda: self._fit_page("Data and Targets", QtWidgets.QListWidget, "global_fit_unified_dataset_list"),
                lambda: self._fit_applied() or self._included(),
                "The three workbook datasets are included.",
                waiting="Check the three workbook datasets in Datasets.",
                dialog=FittingWindow,
            ),
            Step(
                "choose_species",
                "Choose the Species",
                "<p>The <b>Species Table</b> shows the dataset selected in <b>Datasets</b>. A checked "
                "<b>Include</b> compares that species with its column in the fit.</p>"
                "<p>Every mapped column starts checked, so A, B, and C are all compared here. Uncheck a species "
                "to leave its measurements out of a fit.</p>",
                lambda: (),
                dialog=FittingWindow,
            ),
            Step(
                "vary_rates",
                "Vary the Rate Constants",
                "<p>Click <b>Parameters</b>. <b>Vary</b> starts unchecked for every value, so a fit changes "
                "nothing until values are chosen.</p>"
                "<p>Check <b>Vary</b> for <b>kf1</b>, <b>kr1</b>, and <b>k2</b>.</p>"
                "<p>Leave <b>concentration_floor</b> unchecked. M5 includes it so that the let quantity "
                "<b>yield_C = [C] / max([A]_0, concentration_floor)</b> never divides by zero when [A]_0 is 0.</p>"
                "<p>A fit starts from <b>Initial Guess</b>, the value in Reactions, and searches between "
                "<b>Minimum</b> and <b>Maximum</b>.</p>",
                lambda: self._fit_page("Parameters", QtWidgets.QTableWidget, "fit_parameter_roster"),
                lambda: self._fit_applied() or self._varied(),
                "kf1, kr1, and k2 vary.",
                waiting="Check Vary for kf1, kr1, and k2 only.",
                dialog=FittingWindow,
            ),
            Step(
                "run_fit",
                "Run the Fit",
                "<p>Click <b>Run Fit</b>. <b>Results</b> opens on <b>All Datasets</b>, and the fit starts after a "
                "short setup. Results shows the best curves found so far while the fit runs.</p>"
                "<p><b>Pause</b> interrupts the fit until <b>Resume</b> is clicked. <b>Stop</b> ends the fit early "
                "and keeps the best values found. After a fit, Initial Guess holds the best values, so the next "
                "fit starts from them.</p>",
                lambda: (),
                lambda: self._fit_applied() or self._fit_result(),
                "The fit is complete.",
                waiting="Click Run Fit and wait for the fit to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "read_results",
                "Read the Results",
                "<p>Each dataset has its own plot, and <b>All Datasets</b> shows them together. The fitted values "
                "of kf1, kr1, and k2 appear above the plots. <b>Shared</b> means one value describes all three "
                "datasets. <b>Per-dataset</b>, the other choice in <b>Across Datasets</b> on Parameters, fits a "
                "separate value for each dataset.</p>"
                "<p><b>Summary</b> opens the full report. It lists the <b>Estimate</b>, <b>Initial Guess</b>, "
                "<b>Bounds</b>, and <b>Bound Position</b> of each value, and <b>R²</b> for each species of each "
                "dataset. Bound Position shows whether a value ended at its Minimum or Maximum.</p>",
                lambda: (),
                dialog=FittingWindow,
            ),
            Step(
                "apply_copy",
                "Apply the Fit to a Tab Copy",
                "<p>Click <b>Apply Fit</b> and choose <b>Apply Fit to Tab Copy</b>. Kindred adds a copy of this "
                "tab, named after it with <b>— Fit</b> added, and writes the fitted values into Reactions of the "
                "copy, rounded to six significant digits. This tab stays unchanged.</p>",
                lambda: (),
                lambda: self._copied,
                "The fitted copy is in the project.",
                waiting="Click Apply Fit and choose Apply Fit to Tab Copy. After Apply Fit to Project, a copy needs "
                "a new fit: click Run Fit again, or use Skip Step.",
                dialog=FittingWindow,
            ),
            Step(
                "apply_fit",
                "Apply the Fit to the Project",
                "<p>Click <b>Apply Fit</b> again and choose <b>Apply Fit to Project</b>. Kindred writes the fitted "
                "values into Reactions of this tab, rounded to six significant digits.</p>"
                "<p>After Apply Fit to Project, Apply Fit to Tab Copy is no longer available for the same fit, so "
                "a copy comes first when both are wanted.</p>"
                "<p>Applying does not save the project file. Use <b>File → Save Project</b> to save your work.</p>",
                lambda: (),
                self._fit_applied,
                "The fitted values are in Reactions.",
                waiting="Click Apply Fit and choose Apply Fit to Project. If applying leaves the values of M5 in "
                "Reactions, run the fit again and apply the new result.",
                dialog=FittingWindow,
            ),
            Step(
                "run_again",
                "Run the Sets Again",
                "<p>Close the <b>Fitting Window</b>. In <b>Initial Conditions</b>, select the three workbook sets "
                "again and click <b>Run</b>.</p>"
                "<p>The curves now use the fitted rate constants and pass through the markers.</p>",
                self._run_targets_now,
                lambda: self._fit_applied() and self._ran(self._rates()),
                "The sets have run with the fitted rate constants.",
                waiting="Close the Fitting Window, select only the three workbook sets, and click Run. "
                "Each runs from 0 to 600 seconds.",
                dialog=FittingWindow,
            ),
            replace(
                self._finish_step(),
                instruction="<p>fit_runs.xlsx was simulated from M5 with <b>kf1 = 0.02</b>, <b>kr1 = 0.05</b>, and "
                "<b>k2 = 0.03</b> s<sup>−1</sup>. Compare them with the fitted values in Reactions.</p>"
                + self._finish_step().instruction,
            ),
        )
