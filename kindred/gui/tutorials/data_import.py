"""Import an Excel workbook and map its columns to M11."""

from dataclasses import replace
from math import isclose

from PySide6 import QtWidgets

from kindred.core.datasets.foundation import thaw_json_value
from kindred.core.project_presentation_state import SimulationSeriesAxisSelection
from kindred.gui.tutorials.controller import Step
from kindred.gui.tutorials.overlay import Highlight
from kindred.gui.tutorials.practice_lesson import IMPORT_DIALOGS, PracticeLesson
from kindred.gui.widgets.dataset_mapping_batch_dialog import DatasetMappingBatchDialog
from kindred.gui.widgets.import_config_dialog import ImportConfigDialog
from kindred.gui.widgets.observation_mapping_widget import ObservationMappingWidget

_FILE = "runs.xlsx"
_RUNS = ("run1", "run2", "run3")
_WATER = {"run1": 0.05, "run2": 0.1, "run3": 0.2}
# The time unit and column units of each sheet: run1 in min and mM as its
# units row declares, run2 in s and mM and run3 in s and M without one.
_UNITS = {
    "run1": ("min", dict.fromkeys(("A", "B", "C", "Water", "PB", "PH", "BOH", "total_PB"), "mM")),
    "run2": ("s", dict.fromkeys(("A", "B", "C", "Water", "PB", "PH", "BOH", "PB_total"), "mM")),
    "run3": ("s", dict.fromkeys(("PB", "PH", "BOH", "total_PB"), "M")),
}
# run2 names the let total_PB differently from M11.
_RENAMED = {"PB_total": "total_PB"}


class ImportAndMapData(PracticeLesson):
    title = "Import and Map Data"
    description = (
        "Import an Excel workbook whose sheets differ in units and column names, set units and map "
        "a column by hand, type missing starting concentrations, and run the sets created from the data."
    )
    _preset_name = "M11"
    _preset_rates = {"kf1": 1.0, "kr1": 0.01, "k2": 0.01, "k3": 100.0}
    _run_steps = frozenset({"run_data"})
    _needs_preset_set = False
    _sample_folder = "DataImport"
    _sample_file = _FILE
    _import_steps = frozenset({"read_units", "set_units", "import_sheets"})
    _review_steps = frozenset({"map_by_hand", "type_starts", "accept_mapping"})
    _data_steps = _review_steps | {"check_dataset", "show_data", "choose_y", "run_data", "export_original"}

    def __init__(self, **owners):
        super().__init__(**owners)
        # Datasets present when the lesson started.
        self._datasets_before = {record.dataset_id for record in self.registry.document.datasets}

    def blocked(self):
        missing = super().blocked()
        if missing is not None:
            return missing
        name = self.step.name
        if not self._workbook() and (
            name in self._data_steps
            or (name in self._import_steps and self._dialog(ImportConfigDialog) is None)
        ):
            return self._missing(
                "This exercise needs run1, run2, and run3 of runs.xlsx imported with all their columns "
                "and their own units: run1 in min and mM, run2 in s and mM, and run3 in s and M.",
                "open_workbook",
            )
        if name in self._review_steps and self._dialog(DatasetMappingBatchDialog) is None and not self._mapped():
            # The mapping is finished at a later step, so this recovery moves forward.
            return (
                "The mapping review was closed before the mapping was finished. Finish it in the Data panel, "
                "or use Skip Step to continue without this exercise.",
                "Finish in the Data Panel",
                lambda: self.revisitRequested.emit("check_dataset"),
            )
        if name == "run_data" and not self._mapped():
            return self._missing(
                "This exercise needs each workbook dataset mapped to its own set from its data.", "check_dataset"
            )
        return None

    # -- observations ----------------------------------------------------------
    def _run_time_set(self) -> bool:
        defaults = self.tab.run_defaults
        return (
            not self.bundle.simulation_panel.has_run_defaults_draft
            and defaults.t_start == 0
            and defaults.t_end == 150
        )

    def _workbook(self) -> dict[str, str]:
        """Run name -> dataset ID of the newest sheets of runs.xlsx imported in this lesson
        with every column and its own units."""
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
                ) == _UNITS[run]:
                    runs[run] = record.dataset_id
        return runs if len(runs) == len(_RUNS) else {}

    def _default_unit(self) -> QtWidgets.QComboBox | None:
        """The Default unit control of the open import dialog."""
        dialog = self._dialog(ImportConfigDialog)
        label = None if dialog is None else self._shown(dialog, QtWidgets.QLabel, lambda label: label.text() == "Default unit")
        return None if label is None else label.buddy()

    def _run2_units_set(self) -> bool:
        """The import dialog shows run2 with the default unit mM."""
        dialog = self._dialog(ImportConfigDialog)
        sheets = None if dialog is None else self._shown(dialog, QtWidgets.QListWidget)
        control = self._default_unit()
        return (
            sheets is not None
            and sheets.currentItem() is not None
            and sheets.currentItem().text() == "run2"
            and control is not None
            and control.currentData() == "mM"
        )

    def _expected_mapping(self, dataset_id: str) -> dict[str, str]:
        return {
            series.raw_observation_id: _RENAMED.get(series.column_name, series.column_name)
            for series in self.registry.document.dataset(dataset_id).observations
        }

    @staticmethod
    def _starts_match(values: dict[str, float | None], run: str) -> bool:
        expected = {
            "A": 0.2, "B": 0.2, "C": 0.2, "Int": 0.0, "Water": _WATER[run],
            "PB": 0.0, "PH": 0.0, "BOH": 0.0,
        }
        return values.keys() == expected.keys() and all(
            isclose(values[name] or 0.0, value, abs_tol=1e-12) for name, value in expected.items()
        )

    def _mapped(self) -> bool:
        """Each run is mapped by name, and PB_total by hand, to its own set from its data."""
        found = self._workbook()
        tab = self.tab
        sets = {item.set_id: item for item in tab.initial_condition_sets}
        for run, dataset_id in found.items():
            link = tab.dataset_link(dataset_id)
            if (
                link is None
                or link.set_id not in sets
                or dict(link.observation_mapping) != self._expected_mapping(dataset_id)
                or not self._starts_match(dict(sets[link.set_id].values), run)
            ):
                return False
        return bool(found)

    def _editors(self) -> dict[str, ObservationMappingWidget]:
        """The open review's mapping of each run, identified by its mapped columns."""
        review = self._dialog(DatasetMappingBatchDialog)
        if review is None:
            return {}
        owners = {
            raw_id: run for run, dataset_id in self._workbook().items() for raw_id in self._expected_mapping(dataset_id)
        }
        editors = {}
        for editor in review.findChildren(ObservationMappingWidget):
            runs = {owners.get(raw_id) for raw_id, _target in editor.confirmed_mapping()}
            if len(runs) == 1 and None not in runs:
                editors[runs.pop()] = editor
        return editors

    def _unmapped_runs(self) -> tuple[str, ...]:
        """Runs whose page in the open review does not yet map PB_total by hand."""
        editors = self._editors()
        found = self._workbook()
        return tuple(
            run for run in _RUNS
            if run not in editors or dict(editors[run].confirmed_mapping()) != self._expected_mapping(found[run])
        )

    @staticmethod
    def _starts(editor: ObservationMappingWidget) -> dict[str, float | None]:
        return {start.species: start.value for start in editor.starts()}

    def _typed(self) -> bool:
        """The run3 page of the open review shows the typed starting concentrations."""
        editor = self._editors().get("run3")
        return editor is not None and self._starts_match(self._starts(editor), "run3")

    def _overlays_shown(self) -> bool:
        """Plot Overlays checks the column of each run that maps to total_PB, which later steps compare."""
        overlays = dict(self.registry.presentation(self.tab_id).overlay_raw_observation_ids_by_dataset)
        return all(
            any(
                target == "total_PB" and raw_id in overlays.get(dataset_id, ())
                for raw_id, target in self._expected_mapping(dataset_id).items()
            )
            for dataset_id in self._workbook().values()
        )

    def _workbook_sets(self) -> tuple[str, ...]:
        """The set of each workbook run; the Run step is blocked until each is mapped."""
        found = self._workbook()
        return tuple(self.tab.dataset_link(found[run]).set_id for run in _RUNS)

    def _ran(self) -> bool:
        identities = self._workbook_sets()
        return (
            all(
                (settings := self.tab.resolved_execution_settings(identity)).t_start == 0 and settings.t_end == 150
                for identity in identities
            )
            and self._run_ok(self._preset_rates, identities)
        )

    # -- targets ---------------------------------------------------------------
    def _y_target(self) -> tuple[Highlight, ...]:
        """The Y selector above the plot."""
        plot = self.workspace.plot_for_tab(self.tab_id)
        button = self._shown(plot, QtWidgets.QToolButton, lambda button: button.text().startswith("Y:"))
        return () if button is None else (Highlight(button),)

    def _y_chosen(self) -> bool:
        return self.registry.presentation(self.tab_id).selected_y_series == (
            SimulationSeriesAxisSelection("total_PB"),
        )

    def _overlay_target(self) -> Highlight:
        """The Plot Overlays list, which stays in view while its datasets are expanded."""
        panel = self.bundle.overlay_panel
        if not panel.isVisible():
            return self._page_tab(panel)
        return self._panel(panel.findChild(QtWidgets.QTreeWidget), "data")

    def _steps(self) -> tuple[Step, ...]:
        create_tab, load_preset = self._setup_steps()
        in_review = "Complete this in the mapping review."
        return (
            create_tab,
            replace(
                load_preset,
                title="Load M11",
                instruction="<p>Open <b>Examples → Preset Mechanisms → M11: PB cascade</b>.</p>"
                "<p>M11 has eight species: <b>A</b>, <b>B</b>, <b>C</b>, <b>Int</b>, <b>Water</b>, "
                "<b>PB</b>, <b>PH</b>, and <b>BOH</b>. It also defines two let quantities, "
                "<b>total_PB = [PB] + [PH]</b> and <b>selectivity</b>. "
                "A measured column can map to a let quantity as well as to a species.</p>"
                "<p>M11 adds <b>preset_set1</b>. In this lesson, further sets come from the data.</p>",
                complete=self._mechanism_loaded,
                success="M11 is loaded.",
            ),
            Step(
                "save_file",
                "Save the Tutorial File",
                "<p>This lesson uses <b>runs.xlsx</b>, an Excel workbook supplied with Kindred.</p>"
                "<p>Click <b>Save Tutorial File…</b> and choose a folder. Kindred creates a "
                "<b>DataImport</b> folder there and copies the workbook into it.</p>",
                lambda: None,
                self._saved,
                "runs.xlsx is in the DataImport folder.",
                waiting="Save the tutorial file to continue.",
                action=("Save Tutorial File…", self._save_file),
            ),
            Step(
                "set_run_time",
                "Set the Run Time First",
                "<p>The measurements run from 0 to 150 s. In <b>Initial Conditions</b>, set "
                "<b><i>t</i><sub>start</sub></b> to <b>0</b> and <b><i>t</i><sub>end</sub></b> to <b>150</b> seconds. "
                "Press <b>Enter</b> after each edit.</p>"
                "<p>Set the time before importing. A set created from data starts at the time of the row its "
                "starting concentrations come from. When this time differs from <i>t</i><sub>start</sub>, the set gets its own "
                "Run settings, shown checked under <b>Custom</b>, and later edits here no longer change it.</p>",
                lambda: (
                    self._panel(self.bundle.simulation_panel.t_start_edit, "batch_initial_conditions"),
                    self._panel(self.bundle.simulation_panel.t_end_edit, "batch_initial_conditions"),
                ),
                self._run_time_set,
                "The Run time is 0 to 150 seconds.",
            ),
            Step(
                "about_workbook",
                "What Is in the Workbook",
                "<p>runs.xlsx holds three runs of M11, one per sheet: <b>run1</b>, <b>run2</b>, and <b>run3</b>. "
                "Each run starts with 0.2 M of A, B, and C and a different amount of Water: 0.05, 0.1, and "
                "0.2 M. The sheets differ in three ways:</p>"
                "<ul><li><b>Units:</b> run1 is recorded in min and mM, and a units row below its headers says so. "
                "run2 is recorded in s and mM, and run3 in s and M. Neither has a units row.</li>"
                "<li><b>Names:</b> Kindred maps a column automatically when its header matches the name of a "
                "species or a let quantity in Reactions exactly, including capitals. run1 and run3 name the "
                "measured sum of PB and PH <b>total_PB</b>, as M11 does. run2 names it <b>PB_total</b>, so that "
                "column stays unmapped until it is mapped by hand.</li>"
                "<li><b>Starting concentrations:</b> run1 and run2 give A, B, C, and Water in the row at time 0. "
                "run3 has no columns for them.</li></ul>"
                "<p>Single CSV files, one run per file, can also be imported. A workbook is recommended, "
                "because it keeps all runs and their units in one file.</p>",
                lambda: None,
            ),
            Step(
                "open_workbook",
                "Open the Workbook",
                "<p>Choose <b>File → Import Data…</b>, or click <b>Import Data…</b> in the <b>Data</b> panel. "
                "Open <b>runs.xlsx</b> from the DataImport folder.</p>",
                self._open_targets,
                lambda: self._import_dialog_open() or bool(self._workbook()),
                "runs.xlsx is open in the import dialog or already imported.",
                dialog=IMPORT_DIALOGS,
            ),
            Step(
                "read_units",
                "Kindred Reads the Units",
                "<p><b>Sheets</b> lists every sheet in the workbook. <b>Notes</b> has no data rows, so Kindred "
                "leaves it unchecked. Each sheet has its own settings, and run1 is shown first.</p>"
                "<p>Kindred works internally in seconds for time and M for concentrations, and converts each "
                "column to these on import. Kindred recognized the units row of run1, so <b>First Data Row Contains Units</b> is checked "
                "and the preview labels that row <b>Units</b>. <b>Time unit</b> shows <b>From file (min)</b>, and "
                "in <b>Species Columns</b> each column shows <b>From file (mM)</b>. <b>Conversion</b> shows "
                "mM → M, so 50 mM of Water becomes 0.05 M.</p>",
                lambda: (),
                dialog=IMPORT_DIALOGS,
            ),
            Step(
                "set_units",
                "Set the Units of run2",
                "<p>Select <b>run2</b>. It is recorded in seconds and mM and has no units row. Without a row "
                "specifying units, Kindred assumes its default units: seconds for time and M for concentrations. "
                "The time of run2 already matches, but its concentrations do not, so choose <b>mM</b> in "
                "<b>Default unit</b>.</p>"
                "<p>run3 has no units row either. It is recorded in s and M, which are the defaults, so it needs "
                "no change.</p>",
                lambda: (),
                lambda: self._run2_units_set() or bool(self._workbook()),
                "run2 is read in s and mM.",
                waiting="Select run2, then choose mM in Default unit.",
                dialog=IMPORT_DIALOGS,
            ),
            Step(
                "import_sheets",
                "Import Every Sheet",
                "<p>run1, run2, and run3 are checked under <b>Sheets</b>, so all three are imported. Leave "
                "<b>Use for other sheets</b> unchecked. It gives every other sheet the settings of the shown "
                "sheet, but these sheets differ in columns and units, so each keeps its own. Keeping every "
                "sheet of a workbook in the same columns and units lets one set of settings cover them all, "
                "instead of setting up each sheet separately.</p>"
                "<p>Click <b>Import</b>.</p>",
                lambda: (),
                lambda: bool(self._workbook()),
                "The workbook is imported.",
                dialog=IMPORT_DIALOGS,
            ),
            Step(
                "map_by_hand",
                "Map a Column by Hand",
                "<p>The mapping review lists the three new datasets by sheet. In the project, each is named "
                "after the workbook and its sheet, such as <b>runs.xlsx::run1</b>.</p>"
                "<p>Every column of run1 and run3 is mapped automatically. In run2, <b>PB_total</b> is listed "
                "under <b>Unused columns</b>. Unmapped columns take no part in fits or starting concentrations.</p>"
                "<p>Select <b>run2</b> in <b>Datasets</b>, or click its row in <b>Initial-Condition Sets</b> "
                "below, and choose <b>PB_total</b> in the <b>Column</b> of "
                "<b>total_PB</b>, which is marked <b>(let)</b>.</p>",
                lambda: (),
                lambda: not self._unmapped_runs() or self._mapped(),
                "PB_total is mapped in run2.",
                waiting=in_review,
                dialog=DatasetMappingBatchDialog,
            ),
            Step(
                "type_starts",
                "Type the Missing Starting Concentrations",
                "<p>With <b>Recorded data at t = 0</b> in <b>Initial concentrations from</b>, run1 and run2 start "
                "from the values in their row at time 0, converted to M. run3 has no columns for A, B, C, and "
                "Water. A species without a column and without a typed value starts at 0, so nothing would react.</p>"
                "<p>Select <b>run3</b> in <b>Datasets</b>, or click its row in <b>Initial-Condition Sets</b> "
                "below, and type <b>0.2</b> for A, B, C, and Water in "
                "<b>Initial Concentration (M)</b>. Int starts at 0, as intended.</p>",
                lambda: (),
                lambda: self._typed() or self._mapped(),
                "The starting concentrations of run3 are ready.",
                waiting=in_review,
                dialog=DatasetMappingBatchDialog,
            ),
            Step(
                "accept_mapping",
                "Accept the Mapping",
                "<p><b>Initial-Condition Sets</b> previews the new set of each dataset: A, B, and C at 0.2 M "
                "and Water at 0.05, 0.1, and 0.2 M.</p>"
                "<p>Click <b>OK</b>. Each dataset gets a new set named after it.</p>",
                lambda: (),
                self._mapped,
                "Each sheet is mapped to its own new set.",
                waiting=in_review,
                dialog=DatasetMappingBatchDialog,
            ),
            Step(
                "check_dataset",
                "Check a Dataset in the Data Panel",
                "<p>In the <b>Data</b> panel, choose a workbook dataset in <b>Dataset</b>. "
                "<b>Initial-condition set</b> shows the set it is mapped to, and <b>Columns → Model</b> shows "
                "how many of its columns are mapped.</p>"
                "<p>Mappings belong to the mechanism tab. To change one later, use <b>Map Columns…</b> to "
                "choose columns, or <b>Change IC Set…</b> to choose another set or create one from the data.</p>"
                "<p>If a workbook dataset is not mapped as described, choose it in <b>Dataset</b> and click "
                "<b>Change IC Set…</b>, then <b>Create IC Set from Data…</b>. In that dialog, choose "
                "<b>PB_total</b> for <b>total_PB</b> where it is unmapped, and type <b>0.2</b> for each of A, B, C, "
                "and Water that has no value from the data, before clicking <b>OK</b>.</p>",
                self._data_page,
                self._mapped,
                "Each workbook dataset is mapped to its own set.",
                waiting="A workbook dataset is not mapped as described.",
            ),
            Step(
                "show_data",
                "Show the Data on the Plot",
                "<p>Imported data are drawn only for the columns checked in <b>Plot Overlays</b>, beside "
                "<b>Data</b> in the Data panel. Click <b>Plot Overlays</b>.</p>"
                "<p>In the highlighted list, check the three workbook datasets. Checking a dataset checks all its "
                "columns. Expand it to choose single columns. The comparison later needs the total_PB column "
                "of each dataset, named PB_total in run2.</p>",
                self._overlay_target,
                self._overlays_shown,
                "The workbook data are shown on the plot.",
                waiting="Check at least the total_PB column of each workbook dataset, PB_total in run2.",
            ),
            Step(
                "choose_y",
                "Choose What to Plot",
                "<p>Three runs with every species and let quantity crowd the plot. Open the <b>Y</b> selector "
                "above the plot and select only <b>total_PB</b>, then close the selector.</p>"
                "<p>Markers follow the same choice: only the columns mapped to total_PB stay on the plot.</p>",
                self._y_target,
                self._y_chosen,
                "Only total_PB is selected for Y.",
            ),
            Step(
                "run_data",
                "Run the Sets from the Workbook",
                "<p>In <b>Initial Conditions</b>, select only the three sets mapped to the workbook datasets, "
                "such as <b>runs.xlsx::run1</b>, with Ctrl-click, or Cmd-click on macOS. Then click <b>Run</b>. "
                "<b>Initial-condition set</b>, on the <b>Data</b> tab of the <b>Data</b> panel, shows the set of "
                "each dataset.</p>"
                "<p>Compare the total_PB curve of each run with its markers.</p>",
                lambda: (self._initial_target(), *self._run_targets()),
                self._ran,
                "The sets from the workbook have run.",
                waiting="Select only the three workbook sets and click Run. Each runs from 0 to 150 seconds.",
            ),
            Step(
                "export_original",
                "Export the Original File",
                "<p>The project keeps an exact copy of every imported file and saves it in the project file, so "
                "the original data stay with the project even if the file is moved or deleted.</p>"
                "<p>In the <b>Data</b> panel, click the <b>Data</b> tab, choose a workbook dataset in <b>Dataset</b>, "
                "and click <b>Export Original…</b>. Save the copy under a new name, such as <b>runs_copy.xlsx</b>. It is the "
                "whole workbook as imported, with every sheet and its original units, whichever of the three "
                "datasets is chosen.</p>"
                "<p>Check that saving completed. If you canceled the dialog, try again before clicking "
                "<b>Continue</b>.</p>",
                lambda: None,
                manual=True,
            ),
            self._finish_step(),
        )
