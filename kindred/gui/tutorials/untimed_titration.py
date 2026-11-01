"""Advanced Fitting 7: fit binding constants to NMR titrations that record no time, against a measured total."""

from dataclasses import dataclass, replace
from functools import cache
from math import isclose

from PySide6 import QtCore, QtWidgets

from kindred.core.datasets.foundation import thaw_json_value
from kindred.core.simulator.dsl_parse import prepare_dsl_request_set
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.parameter_enumeration import enumerate_parameter_coordinates
from kindred.gui.tutorials.controller import Step
from kindred.gui.tutorials.practice_lesson import IMPORT_DIALOGS, PracticeLesson
from kindred.gui.widgets.dataset_mapping_batch_dialog import DatasetMappingBatchDialog
from kindred.gui.widgets.import_config_dialog import ImportConfigDialog

_FILE = "binding_titration.xlsx"
# The mechanism the lesson asks for, line by line as the card shows it.
_MECHANISM = (
    "RA -> A + RA ; k=1e-4",
    "RC -> C + RC ; k=1e-4",
    "2*A <-> AA ; kf=1e7 ; Keq=100",
    "2*C <-> CC ; kf=1e7 ; Keq=100",
    "A + C <-> AC ; kf=1e7 ; Keq=1000",
    "param concentration_floor = 1e-18; unit = M",
    "param dA = 7.0",
    "param dAA = 7.5",
    "param dAC_A = 8.0",
    "param dC = 3.0",
    "param dCC = 3.4",
    "param dAC_C = 4.0",
    "let A_total = [A] + 2*[AA] + [AC]",
    "let C_total = [C] + 2*[CC] + [AC]",
    "let shift_A = (dA*[A] + 2*dAA*[AA] + dAC_A*[AC]) / max(A_total, concentration_floor)",
    "let shift_C = (dC*[C] + 2*dCC*[CC] + dAC_C*[AC]) / max(C_total, concentration_floor)",
)
_VALUES = {
    "k1": 1e-4, "k2": 1e-4, "kf3": 1e7, "Keq3": 100.0, "kf4": 1e7, "Keq4": 100.0, "kf5": 1e7, "Keq5": 1000.0,
    "concentration_floor": 1e-18, "dA": 7.0, "dAA": 7.5, "dAC_A": 8.0, "dC": 3.0, "dCC": 3.4, "dAC_C": 4.0,
}
_LETS = frozenset({"A_total", "C_total", "shift_A", "shift_C"})
# The starting concentrations, in M, that the first row of each sheet gives.
_STARTS = {
    "dilute_A": {"RA": 1.0},
    "dilute_C": {"RC": 1.0},
    "titrate_A": {"RA": 1.0, "C": 0.002},
    "titrate_C": {"RC": 1.0, "A": 0.002},
}
# Every sheet has no time column and no units row: concentrations in mM, shifts dimensionless.
_UNITS = {
    "dilute_A": {"RA": "mM", "A_total": "mM", "shift_A": "1"},
    "dilute_C": {"RC": "mM", "C_total": "mM", "shift_C": "1"},
    "titrate_A": {"RA": "mM", "C": "mM", "A_total": "mM", "shift_A": "1", "shift_C": "1"},
    "titrate_C": {"RC": "mM", "A": "mM", "C_total": "mM", "shift_A": "1", "shift_C": "1"},
}
_DILUTIONS = ("dilute_A", "dilute_C")
_TITRATIONS = ("titrate_A", "titrate_C")
# The measured total each sheet is fitted against, and the shifts it compares.
_AGAINST = {"dilute_A": "A_total", "dilute_C": "C_total", "titrate_A": "A_total", "titrate_C": "C_total"}
_COMPARED = {
    "dilute_A": frozenset({"shift_A"}),
    "dilute_C": frozenset({"shift_C"}),
    "titrate_A": frozenset({"shift_A", "shift_C"}),
    "titrate_C": frozenset({"shift_A", "shift_C"}),
}
_SPAN = (0.0, 1000.0)
_TOTAL_WEIGHT = 1e3
# The values each fit varies.
_DILUTION_VALUES = ("Keq3", "Keq4", "dA", "dAA", "dC", "dCC")
_TITRATION_VALUES = ("Keq5", "dAC_A", "dAC_C")


@cache
def _expected_equations():
    """The equations of the mechanism, which change with its reactions but not with its values."""
    return prepare_dsl_request_set("\n".join(_MECHANISM)).symbolic_equations_text()


@dataclass(frozen=True)
class _FitData:
    """The Fit Data settings of a dataset: its Fit against choice, its Simulation time start and end, and
    whether they are applied, with no typed edit waiting and no error shown; and from the Species Table,
    the species it compares and the Weight of its X."""

    against: str
    start: str
    end: str
    settled: bool
    compared: frozenset[str]
    x_weight: str


class TitrationsWithoutTime(PracticeLesson):
    title = "Advanced Fitting 7: Titrations Without Time"
    description = (
        "Fit binding constants to NMR titrations that record no time: write the binding mechanism, import data "
        "without a time column that give their starting concentrations in the first row, and fit the dilutions "
        "and then the titrations against a measured total."
    )
    _needs_preset_set = False
    _sample_folder = "BindingTitration"
    _sample_file = _FILE
    _preset_rates = {}
    # Steps taught inside the Fitting Window, and every step that needs the workbook mapped.
    _fit_steps = frozenset({
        "choose_dilutions", "dilutions_against", "dilutions_targets", "vary_dilutions", "run_dilutions",
        "read_dilutions", "choose_titrations", "titrations_against", "titrations_targets", "vary_titrations",
        "run_titrations", "read_titrations",
    })
    _mapped_steps = _fit_steps | {"open_fit"}
    # Steps that need the mechanism of the lesson.
    _typed_steps = _mapped_steps | {"map_sets"}

    def __init__(self, **owners):
        super().__init__(**owners)
        # Datasets present when the lesson started.
        self._datasets_before = {record.dataset_id for record in self.registry.document.datasets}
        # The Fitting Window last seen, and the Fit Data settings last seen for each dataset it showed.
        self._seen_fit: FittingWindow | None = None
        self._data_seen: dict[str, _FitData] = {}

    def _mechanism_loaded(self) -> bool:
        """The lesson checks its own mechanism once the practice tab is in place."""
        return True

    def blocked(self):
        self._observe_fit()
        missing = super().blocked()
        if missing is not None:
            return missing
        name = self.step.name
        if name in self._typed_steps and not self._typed():
            return self._missing(
                "This exercise needs the binding mechanism of this lesson in Reactions, accepted with Run.",
                "enter_mechanism",
            )
        if not self._workbook() and (
            name in self._typed_steps or (name == "import_sheets" and self._dialog(ImportConfigDialog) is None)
        ):
            return self._missing(
                "This exercise needs every sheet of binding_titration.xlsx imported without a time column, with "
                "each concentration in mM and each shift dimensionless.",
                "open_workbook",
            )
        if name in self._mapped_steps and not self._mapped():
            return self._missing(
                "This exercise needs each workbook dataset mapped to a set from its first data row.", "map_sets"
            )
        if name in self._fit_steps and self._fit_window() is None:
            message, label, _ = self._missing("This exercise needs the Fitting Window open.", "open_fit")
            return message, label, self._reopen_fit
        return None

    def _reopen_fit(self) -> None:
        """Closing the Fitting Window discards its datasets, choices, and result, so its steps wait to be
        done again from Open the Fitting Window."""
        self.revisitRequested.emit("open_fit")

    def _fit_closed(self, *_args) -> None:
        self._seen_fit = None
        if self.step.name in self._fit_steps:
            self._reopen_fit()

    # -- observations ----------------------------------------------------------
    def _typed(self) -> bool:
        """Reactions holds the reactions of the lesson with the values and lets the card gives."""
        prepared = self.tab.prepared_dsl
        if prepared is None or prepared.symbolic_equations_text() != _expected_equations():
            return False
        values = {item.name: item.value for item in enumerate_parameter_coordinates(prepared)}
        return all(name in values and isclose(values[name], value) for name, value in _VALUES.items()) and (
            _LETS <= set(prepared.analysis.observable_names)
        )

    def _workbook(self) -> dict[str, str]:
        """Sheet name -> dataset ID of the newest sheets of binding_titration.xlsx imported in this lesson
        without a time column, with each concentration in mM and each shift dimensionless."""
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
                if (
                    run in _UNITS
                    and metadata.get("time_column") is None
                    and metadata.get("original_concentration_units") == _UNITS[run]
                ):
                    runs[run] = record.dataset_id
        return runs if len(runs) == len(_UNITS) else {}

    def _mapped(self) -> bool:
        """Each sheet maps its columns by name and uses a set of its own that starts from its first row,
        every species it does not give at 0."""
        found = self._workbook()
        tab = self.tab
        sets = {item.set_id: dict(item.values) for item in tab.initial_condition_sets}
        for run, dataset_id in found.items():
            link = tab.dataset_link(dataset_id)
            if (
                link is None
                or link.set_id not in sets
                or dict(link.observation_mapping) != {
                    series.raw_observation_id: series.column_name
                    for series in self.registry.document.dataset(dataset_id).observations
                }
                or not all(
                    isclose(value, _STARTS[run].get(species, 0.0), abs_tol=1e-12)
                    for species, value in sets[link.set_id].items()
                )
            ):
                return False
        return bool(found)

    def _fit_window(self) -> FittingWindow | None:
        """The open Fitting Window, also while a dialog opened from it is in front."""
        widget = QtWidgets.QApplication.activeModalWidget()
        while widget is not None and not isinstance(widget, FittingWindow):
            widget = widget.parentWidget()
        return widget

    def _observe_fit(self) -> None:
        """Record the Fit Data settings and the Species Table of the dataset that an open Fitting Window of
        the lesson tab shows. The window shows one dataset at a time."""
        window = self._fit_window()
        if window is None or self.practice is None or self.registry.active_tab_id != self.practice:
            return
        if window is not self._seen_fit:
            self._seen_fit, self._data_seen = window, {}
            window.destroyed.connect(self._fit_closed)
        datasets = window.findChild(QtWidgets.QListWidget, "global_fit_unified_dataset_list")
        against = window.findChild(QtWidgets.QComboBox, "global_fit_parametric_x")
        start = window.findChild(QtWidgets.QLineEdit, "global_fit_t_start")
        end = window.findChild(QtWidgets.QLineEdit, "global_fit_simulation_end")
        error = window.findChild(QtWidgets.QLabel, "global_fit_data_range_error")
        table = window.findChild(QtWidgets.QTableWidget, "global_fit_unified_species_table")
        # The Fitting Window always builds these widgets.
        current = datasets.currentItem()
        if current is None:
            return
        headers = [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
        include, weight, species = (headers.index(title) for title in ("Include", "Weight", "Species"))
        compared, x_weight = set(), ""
        for row in range(table.rowCount()):
            name, check, cell = table.item(row, species), table.item(row, include), table.item(row, weight)
            if (
                bool(check.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                and check.checkState() == QtCore.Qt.CheckState.Checked
            ):
                compared.add(name.text())
            if name.text() == against.currentData():
                x_weight = cell.text()
        self._data_seen[current.data(QtCore.Qt.ItemDataRole.UserRole)] = _FitData(
            against.currentData(),
            start.text(),
            end.text(),
            # A typed number applies on Enter or focus loss; until then the field is modified.
            not start.isModified() and not end.isModified() and not error.text(),
            frozenset(compared),
            x_weight,
        )

    def _seen(self, sheets: tuple[str, ...]) -> dict[str, _FitData | None]:
        """The Fit Data settings last seen for each of these sheets."""
        found = self._workbook()
        # Fit steps require the workbook.
        return {sheet: self._data_seen.get(found[sheet]) for sheet in sheets}

    def _set_up(self, sheets: tuple[str, ...]) -> bool:
        """Each sheet was last seen fitted against its total and simulated from 0 to 1000 s."""

        def ready(sheet: str, data: _FitData | None) -> bool:
            try:
                return (
                    data.against == _AGAINST[sheet]
                    and data.settled
                    and all(isclose(float(text), value) for text, value in zip((data.start, data.end), _SPAN))
                )
            except (AttributeError, ValueError):
                return False

        return all(ready(sheet, data) for sheet, data in self._seen(sheets).items())

    def _targets_set(self, sheets: tuple[str, ...]) -> bool:
        """Each sheet was last seen comparing only its shifts, with its total weighted 1e3."""

        def weighted(data: _FitData) -> bool:
            try:
                return isclose(float(data.x_weight), _TOTAL_WEIGHT)
            except ValueError:
                return False

        return all(
            data is not None and data.compared == _COMPARED[sheet] and weighted(data)
            for sheet, data in self._seen(sheets).items()
        )

    def _included_only(self, sheets: tuple[str, ...]) -> bool:
        """The open Fitting Window includes exactly these workbook datasets."""
        # Fit steps run their checks only while the window is open and the workbook is imported.
        datasets = self._fit_window().findChild(QtWidgets.QListWidget, "global_fit_unified_dataset_list")
        found = self._workbook()
        checked = {
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in (datasets.item(row) for row in range(datasets.count()))
            if item.checkState() == QtCore.Qt.CheckState.Checked
        }
        return checked == {found[sheet] for sheet in sheets}

    def _varied(self, names: tuple[str, ...]) -> bool:
        """The open Fitting Window varies these values, and nothing else."""
        roster = self._fit_window().findChild(QtWidgets.QTableWidget, "fit_parameter_roster")
        headers = [roster.horizontalHeaderItem(column).text() for column in range(roster.columnCount())]
        name_column, vary_column = headers.index("Parameter"), headers.index("Vary")
        varied = set()
        for row in range(roster.rowCount()):
            name, vary = roster.item(row, name_column), roster.item(row, vary_column)
            # Rows of single datasets follow their parameter row.
            if (
                name.data(QtCore.Qt.ItemDataRole.UserRole + 1) is None
                and vary.checkState() == QtCore.Qt.CheckState.Checked
            ):
                varied.add(name.data(QtCore.Qt.ItemDataRole.UserRole))
        return varied == set(names)

    def _fit_result(self) -> bool:
        """The open Fitting Window has a current result to apply. Changing the fit setup makes the shown
        result stale until the fit runs again."""
        apply = self._shown(
            self._fit_window(), QtWidgets.QPushButton, lambda button: button.text().replace("&", "") == "Apply Fit"
        )
        return apply.isEnabled()

    def _fitted(self, sheets: tuple[str, ...], names: tuple[str, ...]) -> bool:
        """The open Fitting Window has a current result for these sheets, set up as the lesson asks, with
        these values varied."""
        return (
            self._included_only(sheets)
            and self._set_up(sheets)
            and self._targets_set(sheets)
            and self._varied(names)
            and self._fit_result()
        )

    def _steps(self) -> tuple[Step, ...]:
        create_tab, _load_preset = self._setup_steps()
        return (
            replace(
                create_tab,
                instruction=create_tab.instruction
                + "<p>Unlike Advanced Fitting 1 to 6, this lesson does not use their shared setup. It has its own tab, "
                "mechanism, and workbook.</p>",
            ),
            Step(
                "enter_mechanism",
                "Enter the Mechanism",
                "<p>This lesson fits NMR titrations of two species, <b>A</b> and <b>C</b>. Each dimerizes, as "
                "<b>AA</b> and <b>CC</b>, and the two bind each other, as <b>AC</b>.</p>"
                "<p>In <b>Reactions</b>, replace the text with these lines. You can select them on this card and "
                "copy them.</p>"
                "<p>" + "<br>".join(line.replace("<", "&lt;").replace(">", "&gt;") for line in _MECHANISM) + "</p>"
                "<p>Click <b>Run</b> in the <b>Mechanism</b> panel. Run checks the lines and keeps them in this tab. "
                "The next step explains them.</p>",
                self._source_target,
                self._typed,
                "The binding mechanism is accepted.",
                waiting="Replace the text in Reactions with the lines above and click Run.",
            ),
            Step(
                "about_mechanism",
                "The Binding Mechanism",
                "<p>Binding is fast on the NMR timescale, so each species shows one averaged signal: its chemical "
                "shift is the average of the shifts of its free and bound forms, weighted by the share of it in "
                "each form. The experiments record these shifts against the total concentration, with no time.</p>"
                "<p>Kindred simulates over time, so the reservoirs <b>RA</b> and <b>RC</b> add A and C at a "
                "constant rate of 0.1 mM s<sup>−1</sup>. The binding steps are made arbitrarily fast, with kf = "
                "10<sup>7</sup> M<sup>−1</sup> s<sup>−1</sup>, so every species stays at equilibrium while the "
                "totals rise through every recorded total. Only <b>Keq</b>, the equilibrium constant of each "
                "binding step, matters.</p>"
                "<p>The params <b>dA</b>, <b>dAA</b>, and <b>dAC_A</b> are the shifts of A, in ppm, when free, in "
                "AA, and in AC, and <b>dC</b>, <b>dCC</b>, and <b>dAC_C</b> those of C. The lets <b>A_total</b> "
                "and <b>C_total</b> are the total concentrations, and <b>shift_A</b> and <b>shift_C</b> the "
                "averaged shifts. As in M11, dividing by the larger of the total and <b>concentration_floor</b> "
                "keeps each shift defined at the start, while its total is still 0.</p>",
                lambda: None,
            ),
            Step(
                "save_file",
                "Save the Tutorial File",
                "<p>This lesson uses <b>binding_titration.xlsx</b>, an Excel workbook supplied with Kindred.</p>"
                "<p>Click <b>Save Tutorial File…</b> and choose a folder. Kindred creates a "
                "<b>BindingTitration</b> folder there and copies the workbook into it.</p>",
                lambda: None,
                self._saved,
                "binding_titration.xlsx is in the BindingTitration folder.",
                waiting="Save the tutorial file to continue.",
                action=("Save Tutorial File…", self._save_file),
            ),
            Step(
                "about_workbook",
                "What Is in the Workbook",
                "<p>binding_titration.xlsx holds four experiments, one per sheet, with no time column and no units "
                "row. The first row of each sheet also gives the starting values the simulation needs: the "
                "reservoir that adds A or C, and in the titrations the 2 mM of the other species.</p>"
                "<ul><li><b>dilute_A</b>: A alone at 12 totals from 0.5 to 100 mM, recording A_total and "
                "shift_A.</li>"
                "<li><b>dilute_C</b>: C alone at the same totals, recording C_total and shift_C.</li>"
                "<li><b>titrate_A</b>: A added to 2 mM of C at 16 totals from 0.25 to 10 mM, recording A_total, "
                "shift_A, and shift_C.</li>"
                "<li><b>titrate_C</b>: C added to 2 mM of A at the same totals, recording C_total, shift_A, and "
                "shift_C.</li></ul>"
                "<p>The dilutions show how A and C dimerize, and the titrations how they bind each other. "
                "Concentrations are recorded in mM and shifts in ppm.</p>"
                "<p>The data were calculated from this mechanism at equilibrium with other values of Keq and of four "
                "of the shifts, plus noise. The end of the lesson gives the values used.</p>",
                lambda: None,
            ),
            Step(
                "open_workbook",
                "Open the Workbook",
                "<p>Choose <b>File → Import Data…</b>, or click <b>Import Data…</b> in the <b>Data</b> panel. "
                "Open <b>binding_titration.xlsx</b> from the BindingTitration folder.</p>",
                self._open_targets,
                lambda: self._import_dialog_open() or bool(self._workbook()),
                "binding_titration.xlsx is open in the import dialog or already imported.",
                dialog=IMPORT_DIALOGS,
            ),
            Step(
                "import_sheets",
                "Set the Units and Import",
                "<p>Each sheet has its own settings, and dilute_A is shown first. <b>Time column</b> reads <b>No "
                "experimental time</b>, because no column holds a time.</p>"
                "<p>Without a units row, Kindred reads every column in <b>Default unit</b>, M. The concentrations "
                "are in mM, so choose <b>mM</b> in Default unit. The shifts are in ppm, a ratio: in <b>Species "
                "Columns</b>, choose <b>Dimensionless</b> in the <b>Unit</b> of shift_A. <b>Conversion</b> then "
                "shows mM → M for RA and A_total and 1 → 1 for shift_A.</p>"
                "<p>Select <b>dilute_C</b>, <b>titrate_A</b>, and <b>titrate_C</b> in turn and do the same, with "
                "Dimensionless for every shift. Then click <b>Import</b>.</p>",
                lambda: (),
                lambda: bool(self._workbook()),
                "The workbook is imported.",
                waiting="In each sheet, choose mM in Default unit and Dimensionless in the Unit of each shift, then "
                "click Import.",
                dialog=IMPORT_DIALOGS,
            ),
            Step(
                "map_sets",
                "Take the Starting Concentrations from the Data",
                "<p>The mapping review maps each column to the species or let of the same name. <b>Action</b> reads "
                "<b>Create Set</b>, which makes a new set from the data, and <b>Initial concentrations from</b> "
                "reads <b>Choose…</b>: Recorded data at t = 0 is unavailable, since the sheets record no time.</p>"
                "<p>Choose <b>First data row</b> in Initial concentrations from, so that each sheet starts from the "
                "values in its first row. Click <b>Copy Settings to Other Datasets</b> to make the same choice for "
                "the other three sheets, and click <b>OK</b>.</p>",
                self._review_targets,
                self._mapped,
                "Each dataset starts from its first data row.",
                waiting="Choose First data row in Initial concentrations from for every dataset, keep Create Set, and "
                "click OK.",
                dialog=DatasetMappingBatchDialog,
            ),
            Step(
                "open_fit",
                "Open the Fitting Window",
                "<p>Choose <b>Fitting → Global Fit…</b>.</p>"
                "<p>The lesson fits in two stages: first the dilutions, which determine how A "
                "and C dimerize, and then the titrations, with those values held, which determine how they bind "
                "each other.</p>",
                lambda: (),
                lambda: self._fit_window() is not None,
                "The Fitting Window is open.",
                dialog=FittingWindow,
            ),
            Step(
                "choose_dilutions",
                "Choose the Dilutions",
                "<p>In <b>Datasets</b>, check dilute_A and dilute_C. Each is marked <b>Data error</b> until the next "
                "step sets its Fit against and Simulation time.</p>",
                lambda: (),
                lambda: self._included_only(_DILUTIONS),
                "dilute_A and dilute_C are included.",
                waiting="Check dilute_A and dilute_C only.",
                dialog=FittingWindow,
            ),
            Step(
                "dilutions_against",
                "Fit Against the Totals",
                "<p>Select <b>dilute_A</b>. Under <b>Fit Data</b>, <b>Fit against</b> reads <b>Time</b>, but these "
                "sheets record no time. Fit against a measured total instead: choose <b>A_total</b> in Fit "
                "against. Kindred places each row at the point of the simulated curve that best matches its total "
                "together with its shift.</p>"
                "<p><b>Simulation time (s)</b> is blank: with no recorded time, Kindred cannot propose one. The "
                "simulation has to run until the total reaches every recorded value, and 1000 s at 0.1 mM "
                "s<sup>−1</sup> reaches 100 mM. Enter <b>0</b> and <b>1000</b>, and press <b>Enter</b> after "
                "each.</p>"
                "<p>Select <b>dilute_C</b> and do the same, with <b>C_total</b> in Fit against.</p>",
                lambda: (),
                lambda: self._set_up(_DILUTIONS),
                "dilute_A and dilute_C are fitted against their totals from 0 to 1000 s.",
                waiting="For dilute_A and dilute_C in turn, choose its total in Fit against, enter 0 and 1000 in "
                "Simulation time (s), and press Enter after each.",
                dialog=FittingWindow,
            ),
            Step(
                "dilutions_targets",
                "Choose the Targets and Weight the Totals",
                "<p>Select <b>dilute_A</b>. In the <b>Species Table</b>, RA, the reservoir, holds a value only in "
                "the first row, so it has no measurements to fit: uncheck its <b>Include</b>.</p>"
                "<p>A_total is now the X of dilute_A, so its Include is unchecked, but its <b>Weight</b> still "
                "counts: a row can match the curve at a different total, and the Weight sets how much that moved "
                "total counts against a missed shift. The totals come from how each sample was made, so weight "
                "them more than the shifts: enter <b>1e3</b> in the Weight of A_total.</p>"
                "<p>Select <b>dilute_C</b> and do the same, with RC and C_total.</p>",
                lambda: (),
                lambda: self._targets_set(_DILUTIONS),
                "dilute_A and dilute_C compare only their shifts, with each total weighted 1e3.",
                waiting="For dilute_A and dilute_C in turn, uncheck Include for RA or RC and enter 1e3 in the Weight "
                "of its total.",
                dialog=FittingWindow,
            ),
            Step(
                "vary_dilutions",
                "Vary the Dimerization Values",
                "<p>Click <b>Parameters</b> and check <b>Vary</b> for <b>Keq3</b>, <b>dA</b>, and <b>dAA</b>, the "
                "dimerization of A, and <b>Keq4</b>, <b>dC</b>, and <b>dCC</b>, that of C. AC cannot form in the "
                "dilutions, so they cannot determine Keq5, dAC_A, or dAC_C. Leave every other value unchecked.</p>",
                lambda: (),
                lambda: self._varied(_DILUTION_VALUES),
                "Keq3, Keq4, dA, dAA, dC, and dCC vary.",
                waiting="Check Vary for Keq3, Keq4, dA, dAA, dC, and dCC only.",
                dialog=FittingWindow,
            ),
            Step(
                "run_dilutions",
                "Fit the Dilutions",
                "<p>Click <b>Run Fit</b> and wait for the fit to finish.</p>",
                lambda: (),
                lambda: self._fitted(_DILUTIONS, _DILUTION_VALUES),
                "The fit is complete.",
                waiting="Include only dilute_A and dilute_C, each fitted against its total from 0 to 1000 s, "
                "comparing only its shift, with its total weighted 1e3. Vary Keq3, Keq4, dA, dAA, dC, and dCC. Then "
                "click Run Fit and wait for the fit to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "read_dilutions",
                "Read the Dimerization Results",
                "<p>Results reads <b>Fit converged</b>. <b>Keq3</b> is the constant of AA and <b>Keq4</b> that of "
                "CC, and dA, dAA, dC, and dCC are the shifts of the free and dimerized forms.</p>",
                lambda: (),
                dialog=FittingWindow,
            ),
            Step(
                "choose_titrations",
                "Choose the Titrations",
                "<p>Click <b>Data and Targets</b>. In <b>Datasets</b>, uncheck dilute_A and dilute_C, and check "
                "titrate_A and titrate_C.</p>",
                lambda: (),
                lambda: self._included_only(_TITRATIONS),
                "titrate_A and titrate_C are included.",
                waiting="On Data and Targets, check titrate_A and titrate_C only.",
                dialog=FittingWindow,
            ),
            Step(
                "titrations_against",
                "Fit the Titrations Against Their Totals",
                "<p>Select <b>titrate_A</b>, choose <b>A_total</b> in Fit against, enter <b>0</b> and <b>1000</b> "
                "in Simulation time (s), and press Enter after each. Select <b>titrate_C</b> and do the same, with "
                "<b>C_total</b>. The titrations reach their largest total, 10 mM, after 100 s, so 1000 s covers "
                "them too.</p>",
                lambda: (),
                lambda: self._set_up(_TITRATIONS),
                "titrate_A and titrate_C are fitted against their totals from 0 to 1000 s.",
                waiting="For titrate_A and titrate_C in turn, choose its total in Fit against, enter 0 and 1000 in "
                "Simulation time (s), and press Enter after each.",
                dialog=FittingWindow,
            ),
            Step(
                "titrations_targets",
                "Choose the Targets and Weight the Totals",
                "<p>Select <b>titrate_A</b>, uncheck <b>Include</b> for RA and C, and enter <b>1e3</b> in the "
                "Weight of A_total. Select <b>titrate_C</b>, uncheck Include for RC and A, and enter <b>1e3</b> in "
                "the Weight of C_total. Each titration then compares only shift_A and shift_C.</p>",
                lambda: (),
                lambda: self._targets_set(_TITRATIONS),
                "titrate_A and titrate_C compare only their shifts, with each total weighted 1e3.",
                waiting="In titrate_A, uncheck Include for RA and C and enter 1e3 in the Weight of A_total. In "
                "titrate_C, uncheck Include for RC and A and enter 1e3 in the Weight of C_total.",
                dialog=FittingWindow,
            ),
            Step(
                "vary_titrations",
                "Hold the Dimerization Values",
                "<p>Click <b>Parameters</b>. <b>Initial Guess</b> of Keq3, Keq4, dA, dAA, dC, and dCC now holds the "
                "result of the dilution fit. Uncheck <b>Vary</b> for these six, so that the titration fit keeps "
                "them, and check Vary for <b>Keq5</b>, <b>dAC_A</b>, and <b>dAC_C</b>.</p>",
                lambda: (),
                lambda: self._varied(_TITRATION_VALUES),
                "Keq5, dAC_A, and dAC_C vary, and the dimerization values are held.",
                waiting="Uncheck Vary for Keq3, Keq4, dA, dAA, dC, and dCC, and check it for Keq5, dAC_A, and dAC_C "
                "only.",
                dialog=FittingWindow,
            ),
            Step(
                "run_titrations",
                "Fit the Titrations",
                "<p>Click <b>Run Fit</b> and wait for the fit to finish.</p>",
                lambda: (),
                lambda: self._fitted(_TITRATIONS, _TITRATION_VALUES),
                "The fit is complete.",
                waiting="Include only titrate_A and titrate_C, each fitted against its total from 0 to 1000 s, "
                "comparing only its shifts, with its total weighted 1e3. Vary Keq5, dAC_A, and dAC_C. Then click Run "
                "Fit and wait for the fit to finish.",
                dialog=FittingWindow,
            ),
            Step(
                "read_titrations",
                "Read the Binding Results",
                "<p>Results reads <b>Fit converged</b>. <b>Keq5</b> is the constant of AC, and dAC_A and dAC_C are the "
                "shifts of A and C in it. Compared with Keq3 and Keq4, Keq5 shows how much more strongly A and C "
                "bind each other than themselves.</p>",
                lambda: (),
                dialog=FittingWindow,
            ),
            replace(
                self._finish_step(),
                instruction="<p>binding_titration.xlsx was calculated with <b>Keq3 = 50</b>, <b>Keq4 = 20</b>, and "
                "<b>Keq5 = 2000</b> M<sup>−1</sup>, with <b>dA = 7.0</b>, <b>dAA = 7.6</b>, <b>dAC_A = 8.2</b>, "
                "<b>dC = 3.0</b>, <b>dCC = 3.5</b>, and <b>dAC_C = 4.1</b> ppm, and with noise of 0.01 ppm. Compare "
                "them with your fitted values, Keq5, dAC_A, and dAC_C in Results and the dilution values in Initial "
                "Guess on Parameters: the noise in the shifts, and the totals that a Weight of 1e3 still lets move, "
                "put each fitted value a little off.</p>"
                + self._finish_step().instruction,
                dialog=FittingWindow,
            ),
        )
