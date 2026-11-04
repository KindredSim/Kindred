"""Add interventions to one mechanism, one at a time, then fit when a light went on and off."""

from dataclasses import replace
from html import escape
from math import isclose

from PySide6 import QtCore, QtWidgets

from kindred.core.intervention_schedule import (
    InterventionFlux,
    InterventionHold,
    InterventionJump,
    InterventionProtocol,
    InterventionSet,
    InterventionStopAction,
    InterventionTrigger,
    ObservableTriggerCondition,
    SpeciesTriggerCondition,
)
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.parameter_enumeration import enumerate_parameter_coordinates
from kindred.gui.tutorials.controller import Step
from kindred.gui.tutorials.practice_lesson import IMPORT_DIALOGS, PracticeLesson
from kindred.gui.widgets.solver_settings import SimulationSettingsDialog

# The reactions of I8, without its stopping intervention.
_PRACTICE_RATES = {"kf1": 0.02, "kr1": 0.004, "k2": 0.006, "k3": 0.002}
_PRACTICE_SET = "practice_set"
_EXAMPLE_RATES = {"k1": 0.01, "k2": 0.003}
_EXAMPLE_SET = "intervention_set1"
_FILE = "light_runs.xlsx"
_RUNS = ("Run 1", "Run 2", "Run 3")
# Every sheet has a units row of seconds and M.
_UNITS = ("s", {"A": "M", "B": "M"})
# The starting concentrations the row at time 0 of each sheet gives its set: the prepared ones.
_STARTS = {
    "Run 1": {"A": 1.0, "B": 0.0, "Light": 0.0},
    "Run 2": {"A": 0.6, "B": 0.0, "Light": 0.0},
    "Run 3": {"A": 1.4, "B": 0.0, "Light": 0.0},
}
# The light periods of I11, each written as its own hold and set with times from parameters.
_PERIODS = (("on1", "off1"), ("on2", "off2"), ("on3", "off3"))
_TIMES = {"on1": 0.0, "off1": 120.0, "on2": 180.0, "off2": 300.0, "on3": 360.0, "off3": 480.0}
_FITTED = ("k1", "k2", *_TIMES)

# The protocol that holds B and then sets it to 0 in each repetition, without its start.
_HOLD_THEN_SET = {
    "every": 120.0,
    "count": 3,
    "duration": 60.0,
    "before": (),
    "during": ({"op": "hold", "species": "B", "value": 0.3},),
    "after": ({"op": "set", "species": "B", "value": 0.0},),
}


def _lines(*lines: str) -> str:
    """Mechanism text in bold, one line per row, as the user types it."""
    return "<p><b>" + "<br>".join(escape(line, quote=False) for line in lines) + "</b></p>"


def _period(on: str, off: str) -> tuple[str, ...]:
    return (
        f"param {on} = {_TIMES[on]:g} ; unit = s",
        f"param {off} = {_TIMES[off]:g} ; unit = s",
        f"intervention: op=hold; species=Light; start_time_param={on}; end_time_param={off}; value=1.0",
        f"intervention: op=set; species=Light; time_param={off}; value=0.0",
    )


def _matches(item, expected: dict) -> bool:
    """item has every field that expected names, with that value. A tuple lists the actions of a group
    in order, and a mapping describes a nested action."""
    for name, wanted in expected.items():
        actual = getattr(item, name)
        if isinstance(wanted, float):
            if actual is None or not isclose(actual, wanted, abs_tol=1e-12):
                return False
        elif isinstance(wanted, tuple):
            if len(actual) != len(wanted) or not all(map(_matches, actual, wanted)):
                return False
        elif isinstance(wanted, dict):
            if not _matches(actual, wanted):
                return False
        elif actual != wanted:
            return False
    return True


def _one(kind, **fields):
    """Accepts a schedule of exactly one operation of this kind with these fields."""
    return lambda operations: (
        len(operations) == 1 and isinstance(operations[0], kind) and _matches(operations[0], fields)
    )


class Interventions(PracticeLesson):
    title = "Interventions"
    description = (
        "Add interventions to a mechanism one at a time: set, jump, flux, hold, times from parameters, "
        "repeating protocols, triggers, and a stop. Then fit the times at which a light went on and off in "
        "three simulated runs of I11."
    )
    _practice_species = ("A", "B", "C", "D")
    _preset_rates = _PRACTICE_RATES
    _needs_preset_set = False
    _sample_folder = "Interventions"
    _sample_file = _FILE
    _sheet_units = dict.fromkeys(_RUNS, _UNITS)
    _run_steps = frozenset({
        "load_preset", "set", "jump", "flux", "hold", "time_param", "protocol_before", "protocol_during",
        "protocol_after", "protocol_groups", "protocol_param", "trigger", "stop", "ignore_stop", "load_example",
        "run_data", "own_times",
    })
    # Steps on the practice mechanism after it is pasted, and steps on I11 after it is loaded.
    _practice_steps = frozenset({
        "set", "jump", "flux", "hold", "time_param", "time_slider", "protocol_before", "protocol_during",
        "protocol_after", "protocol_groups", "protocol_collide", "protocol_param", "protocol_slider", "trigger",
        "stop", "ignore_stop",
    })
    _fit_steps = frozenset({"choose_datasets", "vary", "set_bounds", "run_fit"})
    _mapped_steps = _fit_steps | {"run_data", "own_times", "open_fit"}
    _example_steps = _mapped_steps | {"save_file", "import_data"}

    def __init__(self, **owners):
        super().__init__(**owners)
        # Slider steps whose Preview has been seen; they complete once it is reset.
        self._previews_seen: set[str] = set()
        self._seen_fit: FittingWindow | None = None

    def blocked(self):
        self._observe_fit()
        missing = super().blocked()
        if missing is not None:
            return missing
        name = self.step.name
        if name in self._practice_steps and not self._practice_ready():
            return self._missing(
                "This exercise needs the practice mechanism in Reactions and practice_set with A = 1 and B, C, "
                "and D = 0.",
                "load_preset",
            )
        if name in self._example_steps and not self._example_loaded():
            return self._missing("This exercise needs I11 in Reactions.", "load_example")
        if name in self._mapped_steps and not self._workbook():
            return self._missing(
                "This exercise needs Run 1, Run 2, and Run 3 of light_runs.xlsx imported in seconds and M.",
                "import_data",
            )
        if name in self._mapped_steps and not self._mapped():
            return self._missing(
                "This exercise needs each light_runs.xlsx dataset mapped to its own set from its data.",
                "import_data",
            )
        if name in self._fit_steps | {"open_fit"} and not self._own_times():
            return self._missing(
                "This exercise needs each light period written as its own hold and set, with times from on1 "
                "to off3.",
                "own_times",
            )
        if name in self._fit_steps and self._fit_window() is None:
            message, label, _ = self._missing("This exercise needs the Fitting Window open.", "open_fit")
            return message, label, self._reopen_fit
        return None

    # -- the practice mechanism ------------------------------------------------
    def _mechanism_loaded(self) -> bool:
        # blocked() checks the mechanism that each part of the lesson needs.
        return True

    def _set_named(self, name: str) -> str | None:
        return next((item.set_id for item in self.tab.initial_condition_sets if item.name == name), None)

    def _practice_ready(self) -> bool:
        """The practice reactions and rates, and practice_set with A = 1 and the products at 0."""
        if self.tab.prepared_dsl is None or not self._preset_rates_loaded():
            return False
        self.preset_set = self._set_named(_PRACTICE_SET)
        return self.preset_set is not None and self._values(self.preset_set, 1.0)

    def _practice_result(self):
        return self.registry.tab_runtime(self.tab_id).result_repository.current_result(self.preset_set)

    @staticmethod
    def _parameter(result, name: str, value: float) -> bool:
        parameters = {item: number for item, number, _unit in result.parameters}
        return name in parameters and isclose(parameters[name], value)

    def _practice_ran(self, operations, parameters: dict[str, float] | None = None) -> bool:
        """practice_set ran from 0 to 600 s in this step, with the practice rates, these parameter
        values, and an intervention schedule that operations accepts."""
        if not self._practice_ready() or not self._run_ok(_PRACTICE_RATES, (self.preset_set,)):
            return False
        settings = self.tab.resolved_execution_settings(self.preset_set)
        result = self._practice_result()
        return (
            settings.t_start == 0
            and settings.t_end == 600
            and operations(result.intervention_execution.authored.operations)
            and all(self._parameter(result, name, value) for name, value in (parameters or {}).items())
        )

    def _previewed(self, parameter: str, value: float, operations) -> bool:
        """A completed Preview of practice_set moved this parameter away from its value in Reactions,
        with a schedule that operations accepts, and Reset has since discarded it."""
        name = self.step.name
        if self.jobs.busy or not self._practice_ready():
            return False
        repo = self.registry.tab_runtime(self.tab_id).result_repository
        if self.preset_set in repo.preview_set_ids():
            result = repo.current_result(self.preset_set)
            if not self._parameter(result, parameter, value) and operations(
                result.intervention_execution.authored.operations
            ):
                self._previews_seen.add(name)
            return False
        return name in self._previews_seen

    def _triggered(self) -> bool:
        """The trigger ran with the fields the step asks for, and acted during the Run."""
        return self._practice_ran(_one(
            InterventionTrigger,
            condition=SpeciesTriggerCondition("B"),
            threshold=0.25,
            direction="rising",
            max_count=1,
            min_interval=0.0,
            then={"op": "jump", "species": "A", "delta": 0.5},
        )) and bool(self._practice_result().intervention_execution.trigger_firings)

    _stop_trigger = staticmethod(_one(
        InterventionTrigger,
        condition=ObservableTriggerCondition("conversion"),
        threshold=0.8,
        direction="rising",
        then=InterventionStopAction(),
    ))

    def _stopped(self) -> bool:
        return self._practice_ran(self._stop_trigger) and self._practice_result().termination.cause == "stop"

    def _stop_ignored(self) -> bool:
        """With Use Authored Stopping Interventions off, practice_set ran to 600 s past the stop that
        Reactions keeps. A Run without stops leaves them out of its result, so the stop is read from the
        accepted model, and the result has no intervention left."""
        return (
            not self.tab.scientific_settings.use_authored_stops
            and self._practice_ran(lambda operations: not operations)
            # Only a schedule with times from parameters has no compiled contract. The line above admits
            # only a model whose interventions are all stops, so this one has a contract.
            and self._stop_trigger(self.tab.prepared_dsl.intervention_execution_contract().authored.operations)
            and self._practice_result().termination.cause == "endpoint"
        )

    # -- I11 and the simulated runs --------------------------------------------
    def _example_loaded(self) -> bool:
        if self.tab.prepared_dsl is None:
            return False
        values = {item.name: item.value for item in enumerate_parameter_coordinates(self.tab.prepared_dsl)}
        return all(
            name in values and isclose(values[name], rate, rel_tol=1e-8) for name, rate in _EXAMPLE_RATES.items()
        )

    def _example_ran(self) -> bool:
        identity = self._set_named(_EXAMPLE_SET)
        return (
            self._example_loaded()
            and identity is not None
            and dict(self.tab.initial_condition_set(identity).values) == {"A": 1.0, "Light": 0.0, "B": 0.0}
            and self._ran_from_zero((identity,), _one(InterventionProtocol))
        )

    def _ran_from_zero(self, identities: tuple, operations) -> bool:
        """These sets ran from 0 to 600 s in this step with the rates of I11, each with a schedule that
        operations accepts."""
        if not self._run_ok(_EXAMPLE_RATES, identities):
            return False
        repo = self.registry.tab_runtime(self.tab_id).result_repository
        return all(
            (settings := self.tab.resolved_execution_settings(identity)).t_start == 0
            and settings.t_end == 600
            and operations(repo.current_result(identity).intervention_execution.authored.operations)
            for identity in identities
        )

    def _mapped(self) -> bool:
        """Each run maps A and B by name to its own set, which starts from its row at time 0."""
        found = self._workbook()
        tab = self.tab
        sets = {item.set_id: item for item in tab.initial_condition_sets}
        for run, dataset_id in found.items():
            link = tab.dataset_link(dataset_id)
            if (
                link is None
                or link.set_id not in sets
                or dict(link.observation_mapping) != self._expected_mapping(dataset_id)
            ):
                return False
            values = dict(sets[link.set_id].values)
            if values.keys() != _STARTS[run].keys() or not all(
                isclose(values[species], value, abs_tol=1e-12) for species, value in _STARTS[run].items()
            ):
                return False
        return bool(found)

    def _overlays_shown(self) -> bool:
        """Plot Overlays checks every column of each workbook dataset."""
        overlays = dict(self.registry.presentation(self.tab_id).overlay_raw_observation_ids_by_dataset)
        return all(
            set(self._expected_mapping(dataset_id)) <= set(overlays.get(dataset_id, ()))
            for dataset_id in self._workbook().values()
        )

    @staticmethod
    def _periods(operations) -> bool:
        """A hold of Light at 1 and a set of Light to 0 for each period, with the times from its
        parameters, and nothing else."""
        holds = {
            (operation.start_time_param, operation.end_time_param)
            for operation in operations
            if isinstance(operation, InterventionHold)
            and operation.species == "Light"
            and operation.value == 1.0
        }
        sets = {
            operation.time_param
            for operation in operations
            if isinstance(operation, InterventionSet) and operation.species == "Light" and operation.value == 0.0
        }
        return (
            len(operations) == 2 * len(_PERIODS)
            and holds == set(_PERIODS)
            and sets == {off for _on, off in _PERIODS}
        )

    def _times_loaded(self) -> bool:
        values = {item.name: item.value for item in enumerate_parameter_coordinates(self.tab.prepared_dsl)}
        return all(name in values and isclose(values[name], time) for name, time in _TIMES.items())

    def _own_times(self) -> bool:
        """Reactions writes each light period as its own hold and set with times from on1 to off3."""
        dsl = self.tab.prepared_dsl
        # Times from parameters keep the accepted schedule parameterized until a Run resolves them.
        schedule = None if dsl is None else dsl.parameterized_intervention_schedule_for_binding()
        return schedule is not None and self._periods(schedule.operations) and self._times_loaded()

    def _ran_own_times(self) -> bool:
        return (
            self._example_loaded()
            and self._times_loaded()
            and self._ran_from_zero(self._workbook_sets(), self._periods)
        )

    # -- the Fitting Window ----------------------------------------------------
    def _observe_fit(self) -> None:
        window = self._fit_window()
        if window is not None and window is not self._seen_fit:
            self._seen_fit = window
            window.destroyed.connect(self._fit_closed)

    def _fit_closed(self, *_args) -> None:
        self._seen_fit = None
        if self.step.name in self._fit_steps:
            self._reopen_fit()

    def _included_only(self) -> bool:
        """The open Fitting Window includes exactly the three workbook datasets."""
        datasets = self._fit_window().findChild(QtWidgets.QListWidget, "global_fit_unified_dataset_list")
        checked = {
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in (datasets.item(row) for row in range(datasets.count()))
            if item.checkState() == QtCore.Qt.CheckState.Checked
        }
        return checked == set(self._workbook().values())

    def _roster(self) -> dict[str, tuple[dict[str, QtWidgets.QTableWidgetItem], QtWidgets.QComboBox]]:
        """Parameter -> its row of the parameter table by column title, and its Across Datasets choice."""
        roster = self._fit_window().findChild(QtWidgets.QTableWidget, "fit_parameter_roster")
        headers = [roster.horizontalHeaderItem(column).text() for column in range(roster.columnCount())]
        scope = headers.index("Across Datasets")
        rows = {}
        for row in range(roster.rowCount()):
            cells = {header: roster.item(row, column) for column, header in enumerate(headers)}
            name = cells["Parameter"]
            # Rows of single datasets follow their parameter row.
            if name.data(QtCore.Qt.ItemDataRole.UserRole + 1) is None:
                rows[name.data(QtCore.Qt.ItemDataRole.UserRole)] = (cells, roster.cellWidget(row, scope))
        return rows

    def _varied(self) -> bool:
        """The open Fitting Window varies k1, k2, and the six times, each time for every dataset."""
        roster = self._roster()
        varied = {
            name for name, (cells, _choice) in roster.items()
            if cells["Vary"].checkState() == QtCore.Qt.CheckState.Checked
        }
        return varied == set(_FITTED) and all(roster[name][1].currentData() == "per_dataset" for name in _TIMES)

    def _bounded(self) -> bool:
        """Every time ranges from 0 or later, and each range ends before the range of the next time
        begins, so every candidate keeps the periods in order."""
        roster = self._roster()
        try:
            ranges = [
                (float(roster[name][0]["Minimum"].text()), float(roster[name][0]["Maximum"].text()))
                for name in _TIMES
            ]
        except (KeyError, ValueError):
            return False
        return ranges[0][0] >= 0 and all(
            earlier[1] < later[0] for earlier, later in zip(ranges, ranges[1:])
        )

    # -- steps -----------------------------------------------------------------
    def _steps(self) -> tuple[Step, ...]:
        create_tab, _load_preset = self._setup_steps()
        practice = self._practice_ran
        # This lesson highlights nothing. Each card says in words where to look.
        def nothing():
            return ()

        return (
            replace(create_tab, target=nothing),
            Step(
                "load_preset",
                "Paste the Model",
                "<p>This lesson adds interventions to one model, one at a time. Each step replaces the "
                "intervention of the step before.</p>"
                "<p>Paste the model below into the <b>Reactions</b> box. This is the same model as <b>I8</b> in "
                "<b>Examples → Intervention Examples</b>, without its intervention.</p>"
                + _lines(
                    "A <-> B ; kf=0.02 ; kr=0.004",
                    "B -> C ; k=0.006",
                    "B -> D ; k=0.002",
                    "",
                    "practice_set = {",
                    "[A] = 1.0",
                    "[B] = 0.0",
                    "[C] = 0.0",
                    "[D] = 0.0",
                    "}",
                )
                + "<p>In <b>Initial Conditions</b>, check that <b><i>t</i><sub>start</sub></b> is 0 and "
                "<b><i>t</i><sub>end</sub></b> is 600 seconds. Press <b>Enter</b> after each edit. Then click "
                "<b>Run</b>.</p>"
                "<p>In this model as written, A is monotonically consumed, B accumulates to approximately 0.5 M "
                "at 70 s, and C and D reach approximately 0.75 and 0.25 M by 600 s.</p>",
                nothing,
                lambda: practice(lambda operations: not operations),
                "practice_set has run without interventions.",
                waiting="Paste the model into Reactions, set t_start to 0 and t_end to 600 seconds, and click Run.",
            ),
            Step(
                "set",
                "Set a Concentration",
                "<p>Interventions are operations that change simulation values once a given condition is "
                "fulfilled, such as a time <i>t</i> or a user-defined quantity, such as the conversion of a "
                "species. An intervention is input as one line in the model beginning with "
                "<b>intervention: op=</b>, followed by fields specific to that operation, separated by "
                "semicolons. <b>op=</b> names the operation:</p>"
                "<ul>"
                "<li><b>set</b> gives a species a new concentration at one time.</li>"
                "<li><b>jump</b> adds an amount to the concentration of a species at one time.</li>"
                "<li><b>flux</b> adds or removes a species at a constant rate over an interval.</li>"
                "<li><b>hold</b> keeps the concentration of a species constant over an interval.</li>"
                "<li><b>protocol</b> repeats a group of operations on a regular schedule.</li>"
                "<li><b>trigger</b> acts when a watched concentration or calculated quantity crosses a "
                "threshold.</li>"
                "</ul>"
                "<p><b>scale</b> and <b>withdraw</b> act in the context of volume and are covered in the "
                "<b>Models with Explicit Volume</b> tutorial.</p>"
                "<p>By default, times are given in seconds and concentrations in M for intervention fields, "
                "unless defined as a parameter (as shown in step 8). Multiple interventions can be defined per "
                "model. When several act at the same time, Kindred applies them in a fixed order, and it refuses "
                "combinations that contradict each other, such as two different <b>set</b> values for one "
                "species at the same time. The <b>Help</b> tab beside Reactions lists these rules under "
                "<b>Actions at the Same Time</b>.</p>"
                "<p>Add this line to the model and click <b>Run</b>:</p>"
                + _lines("intervention: op=set; species=A; time=200; value=1.0")
                + "<p><b>op=set</b> sets the concentration of a species. At <b>time</b> = 200 s, [A] becomes "
                "<b>value</b> = 1.0 M, independent of its previous value. The simulation continues from this "
                "point.</p>"
                "<p>At 200 s, A goes from about 0.09 M to 1 M and then continues reacting.</p>",
                nothing,
                lambda: practice(_one(InterventionSet, species="A", time=200.0, value=1.0)),
                "A is set to 1 M at 200 s.",
                waiting="Add the set of A at 200 s to 1.0 M and click Run.",
            ),
            Step(
                "marks",
                "Read the Intervention Marks",
                "<p>Interventions are labeled in the plot at their respective times with a vertical line and a "
                "label.</p>"
                "<p>Labels can be dragged with the left mouse button, and indicators can be disabled.</p>"
                "<p>Right-click the plot and choose <b>Appearance → Show Intervention Indicators</b> or <b>Show "
                "Intervention Labels</b> to hide or show them.</p>",
                nothing,
            ),
            Step(
                "jump",
                "Add an Amount",
                "<p>Replace the <b>set</b> line with this one and click <b>Run</b>:</p>"
                + _lines("intervention: op=jump; species=A; time=200; delta=0.5")
                + "<p><b>op=jump</b> changes the concentration of one species by a set amount. At 200 s, "
                "<b>delta</b> = 0.5 M is added to whatever [A] is then. Unlike <b>set</b>, the result depends on "
                "[A] at that moment. A negative <b>delta</b> removes an amount up to 0, for example, if [A] = "
                "0.5 M, <b>delta</b> = -0.5 M or <b>delta</b> = -5.0 M would both result in [A] = 0 M, as no "
                "negative concentrations are allowed.</p>"
                "<p>At 200 s, A rises by 0.5 M, from about 0.1 to 0.6 M. C ends near 1.05 M.</p>",
                nothing,
                lambda: practice(_one(InterventionJump, species="A", time=200.0, delta=0.5)),
                "0.5 M of A is added at 200 s.",
                waiting="Replace the set with the jump of A at 200 s by 0.5 M and click Run.",
            ),
            Step(
                "flux",
                "Feed Continuously",
                "<p>Replace the <b>jump</b> with this line and click <b>Run</b>:</p>"
                + _lines("intervention: op=flux; species=A; start_time=0; end_time=120; flux_rate=0.005")
                + "<p><b>op=flux</b> adds a species at a continuous rate over an interval. From "
                "<b>start_time</b> to <b>end_time</b>, A is added at <b>flux_rate</b>, in M/s. A negative "
                "<b>flux_rate</b> removes it at that rate.</p>"
                "<p>During the first 120 s, A is replenished by flux at the same time as it equilibrates with B, "
                "whereas after 120 s no further A is added.</p>",
                nothing,
                lambda: practice(_one(InterventionFlux, species="A", start_time=0.0, end_time=120.0, flux_rate=0.005)),
                "A is fed from 0 to 120 s.",
                waiting="Replace the jump with the flux of A from 0 to 120 s at 0.005 M/s and click Run.",
            ),
            Step(
                "hold",
                "Hold a Concentration",
                "<p>Replace the <b>flux</b> with this line and click <b>Run</b>:</p>"
                + _lines("intervention: op=hold; species=B; start_time=60; end_time=240; value=0.3")
                + "<p><b>op=hold</b> fixes the concentration of one species over an interval. From 60 to 240 s, "
                "[B] stays at <b>value</b> = 0.3 M, and while B continues to participate in reactions normally, "
                "it is neither consumed nor generated. When the <b>hold</b> ends, B continues on in the reaction "
                "from 0.3 M.</p>"
                "<p>At 60 s, B drops from about 0.5 to 0.3 M and stays constant until 240 s, then continues "
                "reacting normally from 0.3 M.</p>",
                nothing,
                lambda: practice(_one(InterventionHold, species="B", start_time=60.0, end_time=240.0, value=0.3)),
                "B is held at 0.3 M from 60 to 240 s.",
                waiting="Replace the flux with the hold of B from 60 to 240 s at 0.3 M and click Run.",
            ),
            Step(
                "time_param",
                "Take a Time from a Parameter",
                "<p>A time or value can be defined from a <b>param</b> line instead. This is useful to explore "
                "variations with sliders or fits. Add <b>_param</b> to a field name, such as <b>time_param</b> "
                "instead of plain <b>time</b>, and give the name of the parameter. Such parameters can also carry "
                "units for time, such as <b>s</b>, <b>min</b>, or <b>h</b>, and for concentration, such as "
                "<b>M</b>, <b>mM</b>, or <b>uM</b>. Kindred converts them to seconds and M.</p>"
                "<p>Replace the <b>hold</b> with these two lines and click <b>Run</b>:</p>"
                + _lines(
                    "param t_add = 200 ; unit = s",
                    "intervention: op=jump; species=A; time_param=t_add; delta=0.5",
                ),
                nothing,
                lambda: practice(
                    _one(InterventionJump, species="A", time_param="t_add", delta=0.5), {"t_add": 200.0}
                ),
                "The jump takes its time from t_add.",
                waiting="Replace the hold with t_add = 200 s and the jump of A by 0.5 M at time_param=t_add, and "
                "click Run.",
            ),
            Step(
                "time_slider",
                "Move the Time with Its Slider",
                "<p>This is identical to step 5, because t_add is 200 s. But it is possible to vary t_add, for "
                "example through sliders or fitting, without changing the accepted model, the text the last Run "
                "accepted.</p>"
                "<p>In the <b>Interactive Sliders</b> panel, open <b>Visible Sliders</b> and check <b>t_add</b>. "
                "Move the slider.</p>"
                "<p>The Preview shows the <b>jump</b> at the new time.</p>"
                "<p>Subsequently, click <b>Reset</b> in the Interactive Sliders panel. The slider returns to "
                "200 s, and the Preview is discarded.</p>",
                nothing,
                lambda: self._previewed("t_add", 200.0, _one(InterventionJump, time_param="t_add")),
                "The Preview moved the jump, and Reset discarded it.",
                waiting="Move the t_add slider, then click Reset in the Interactive Sliders panel.",
            ),
            Step(
                "protocol_before",
                "What a Protocol Is",
                "<p>A protocol repeats groups of actions on a schedule. It starts at <b>start_time</b> and "
                "repeats <b>count</b> times, every <b>every</b> seconds. One pass is a repetition.</p>"
                "<p>Each repetition can have three groups:</p>"
                "<ul>"
                "<li><b>before</b> acts once, at the start of the repetition.</li>"
                "<li><b>during</b> acts throughout the first <b>duration</b> seconds of the repetition.</li>"
                "<li><b>after</b> acts once, when <b>duration</b> ends.</li>"
                "</ul>"
                "<p>A protocol is one <b>intervention:</b> line with <b>op=protocol</b> and <b>kind=repeat</b>. "
                "Each group is enclosed in braces, with the fields of one action separated by commas.</p>"
                "<p>Replace both lines with this one and click <b>Run</b>:</p>"
                + _lines(
                    "intervention: op=protocol; kind=repeat; start_time=60; every=120; count=3; "
                    "before={op=jump,species=A,delta=0.3}"
                )
                + "<p>A protocol that only contains a <b>before</b> group happens instantaneously and has no "
                "<b>duration</b>.</p>"
                "<p>With this input, A rises by 0.3 M at 60, 180, and 300 s.</p>"
                "<p>For any model, a detailed log listing each scheduled action from Interventions is found at "
                "<b>Simulation → Inspect Mechanism…</b>.</p>",
                nothing,
                lambda: practice(_one(
                    InterventionProtocol,
                    start_time=60.0,
                    every=120.0,
                    count=3,
                    duration=None,
                    before=({"op": "jump", "species": "A", "delta": 0.3},),
                    during=(),
                    after=(),
                )),
                "A is added three times.",
                waiting="Replace both lines with the protocol and click Run.",
            ),
            Step(
                "protocol_during",
                "Act Throughout a Repetition",
                "<p>Replace the <b>protocol</b> with the following input and click <b>Run</b>:</p>"
                + _lines(
                    "intervention: op=protocol; kind=repeat; start_time=60; every=120; count=3; duration=60; "
                    "during={op=flux,species=A,flux_rate=0.005}"
                )
                + "<p><b>during</b> intervals accept <b>op=flux</b> or <b>op=hold</b> as valid operations. "
                "<b>before</b> and <b>after</b> inputs take <b>op=set</b>, <b>op=jump</b>, as well as "
                "<b>op=scale</b>, or <b>op=withdraw</b> in the context of volume (see the <b>Models with Explicit "
                "Volume</b> tutorial).</p>"
                "<p>A is continuously fed into the reaction for the first 60 s of each repetition, from 60 to "
                "120 s, 180 to 240 s, and 300 to 360 s.</p>",
                nothing,
                lambda: practice(_one(
                    InterventionProtocol,
                    start_time=60.0,
                    every=120.0,
                    count=3,
                    duration=60.0,
                    before=(),
                    during=({"op": "flux", "species": "A", "flux_rate": 0.005},),
                    after=(),
                )),
                "A is fed three times.",
                waiting="Replace the protocol with the one that feeds A during each repetition and click Run.",
            ),
            Step(
                "protocol_after",
                "Act at the End of a Repetition",
                "<p>Replace the <b>protocol</b> with this one and click <b>Run</b>:</p>"
                + _lines(
                    "intervention: op=protocol; kind=repeat; start_time=60; every=120; count=3; duration=60; "
                    "during={op=hold,species=B,value=0.3}; after={op=set,species=B,value=0.0}"
                )
                + "<p>A <b>hold</b> that ends does not reset its species concentration. Without <b>after</b>, B "
                "would go on from 0.3 M after each <b>hold</b>. The <b>after</b> operation ensures B is set to 0 "
                "when each <b>hold</b> ends.</p>"
                "<p>In this simulation, B is flat at 0.3 M from 60 to 120 s, drops to 0 at 120 s, and forms "
                "again from A until the next <b>hold</b> at 180 s.</p>",
                nothing,
                lambda: practice(_one(InterventionProtocol, start_time=60.0, **_HOLD_THEN_SET)),
                "B is held and then set to 0 three times.",
                waiting="Replace the protocol with the one that holds B and then sets it to 0, and click Run.",
            ),
            Step(
                "protocol_groups",
                "Several Actions in One Group",
                "<p>A group can contain several actions, separated by semicolons. The fields within each action "
                "stay separated by commas.</p>"
                "<p>Replace the <b>protocol</b> with this one and click <b>Run</b>:</p>"
                + _lines(
                    "intervention: op=protocol; kind=repeat; start_time=60; every=120; count=3; duration=60; "
                    "during={op=hold,species=B,value=0.3}; after={op=set,species=B,value=0.0; "
                    "op=jump,species=A,delta=0.2}"
                )
                + "<p>Actions in a group act in the order written. Here, when each <b>hold</b> ends, <b>after</b> "
                "first sets B to 0 and then adds 0.2 M to A.</p>"
                "<p>At 120, 240, and 360 s, B drops to 0 and A rises by 0.2 M.</p>",
                nothing,
                lambda: practice(_one(
                    InterventionProtocol,
                    start_time=60.0,
                    **{
                        **_HOLD_THEN_SET,
                        "after": (
                            {"op": "set", "species": "B", "value": 0.0},
                            {"op": "jump", "species": "A", "delta": 0.2},
                        ),
                    },
                )),
                "Each repetition ends with two actions.",
                waiting="Add the jump of A by 0.2 M after the set of B in after, and click Run.",
            ),
            Step(
                "protocol_collide",
                "When Repetitions Collide",
                "<p>Repetitions must not act on the same species in contradicting ways. Change the "
                "<b>protocol</b> as below, one change at a time, and click <b>Run</b> after each. Kindred refuses "
                "each and names the conflict.</p>"
                "<ul>"
                "<li><b>every=60</b>: each <b>after</b> sets B to 0 at the same time as the next <b>hold</b> "
                "starts holding it at 0.3 M.</li>"
                "<li><b>every=50</b>: each <b>hold</b> of B overlaps the next one, and only one <b>hold</b> per "
                "species may be active at a time.</li>"
                "</ul>"
                "<p>Then click <b>Revert</b> beside Reactions to go back to the <b>protocol</b> of step 13.</p>",
                nothing,
                manual=True,
            ),
            Step(
                "protocol_param",
                "Start a Protocol from a Parameter",
                "<p>Of the <b>protocol</b> fields, only <b>start_time</b> can come from a parameter, as "
                "<b>start_time_param</b>. <b>every</b>, <b>count</b>, and <b>duration</b> are always plain "
                "numbers.</p>"
                "<p>Replace the <b>protocol</b> with these two lines and click <b>Run</b>:</p>"
                + _lines(
                    "param t_first = 60 ; unit = s",
                    "intervention: op=protocol; kind=repeat; start_time_param=t_first; every=120; count=3; "
                    "duration=60; during={op=hold,species=B,value=0.3}; after={op=set,species=B,value=0.0}",
                ),
                nothing,
                lambda: practice(
                    _one(InterventionProtocol, start_time_param="t_first", **_HOLD_THEN_SET), {"t_first": 60.0}
                ),
                "The protocol takes its start from t_first.",
                waiting="Replace the protocol with t_first = 60 s and the protocol with start_time_param=t_first, "
                "and click Run.",
            ),
            Step(
                "protocol_slider",
                "Move the Protocol with Its Slider",
                "<p>This is identical to step 12, because t_first is 60 s.</p>"
                "<p>In the <b>Interactive Sliders</b> panel, open <b>Visible Sliders</b> and check "
                "<b>t_first</b>. Move the slider.</p>"
                "<p>The Preview shows every repetition move with the start. The spacing and the <b>duration</b> "
                "stay the same.</p>"
                "<p>Subsequently, click <b>Reset</b> in the Interactive Sliders panel. The slider returns to "
                "60 s, and the Preview is discarded.</p>",
                nothing,
                lambda: self._previewed("t_first", 60.0, _one(InterventionProtocol, start_time_param="t_first")),
                "The Preview moved the protocol, and Reset discarded it.",
                waiting="Move the t_first slider, then click Reset in the Interactive Sliders panel.",
            ),
            Step(
                "trigger",
                "Act When a Concentration Crosses a Threshold",
                "<p><b>op=trigger</b> acts when a specified concentration crosses a user-defined threshold, at "
                "whatever time that happens, instead of at a time given in advance.</p>"
                "<p>Replace both lines with this one and click <b>Run</b>:</p>"
                + _lines(
                    "intervention: op=trigger; watch_species=B; threshold=0.25; direction=rising; max_count=1; "
                    "min_interval=0; then={op=jump,species=A,delta=0.5}"
                )
                + "<ul>"
                "<li><b>watch_species</b> names the species to watch, and <b>threshold</b> the concentration at "
                "which the operation is executed, in M.</li>"
                "<li><b>direction=rising</b> acts when [B] reaches or crosses 0.25 M from a lower concentration. "
                "<b>falling</b> acts when it crosses from a higher concentration, and <b>either</b> is direction "
                "agnostic. Without <b>direction</b> specified, <b>either</b> applies.</li>"
                "<li><b>max_count</b> is the most times the trigger can act if conditions are fulfilled multiple "
                "times. Here it is limited to trigger once.</li>"
                "<li><b>min_interval</b> is the minimum interval, in seconds, between two actions. A crossing "
                "within that time is ignored. 0 adds no waiting time.</li>"
                "<li><b>then</b> contains the action, enclosed in braces: one <b>set</b>, <b>jump</b>, "
                "<b>scale</b>, or <b>withdraw</b>, with its fields separated by commas and without a time, "
                "because the action occurs at the crossing.</li>"
                "</ul>"
                "<p>A species that starts at or beyond the threshold has not crossed it. The crossing must happen "
                "after <i>t</i><sub>start</sub>.</p>"
                "<p>B reaches 0.25 M at about 16 s, and A rises by 0.5 M at that moment.</p>",
                nothing,
                self._triggered,
                "A is added when B first rises through 0.25 M.",
                waiting="Replace both lines with the trigger and click Run.",
            ),
            Step(
                "stop",
                "Stop at a Conversion",
                "<p>A <b>trigger</b> can also be employed as a condition to end the simulation, with "
                "<b>then={op=stop}</b>. A stopping trigger can watch a species, or a calculated quantity defined "
                "with <b>let</b>, such as conversion.</p>"
                "<p>Replace the <b>trigger</b> with these two lines and click <b>Run</b>:</p>"
                + _lines(
                    "let conversion = 1 - [A] / [A]_0",
                    "intervention: op=trigger; watch_observable=conversion; threshold=0.8; direction=rising; "
                    "then={op=stop}",
                )
                + "<p><b>let</b> defines a calculated quantity, here the fraction of A consumed. <b>[A]_0</b> is "
                "the starting concentration of A. <b>watch_observable</b> names the <b>let</b> quantity to watch, "
                "in place of <b>watch_species</b>. A stopping trigger acts only once, so it takes no "
                "<b>max_count</b> or <b>min_interval</b>. The result keeps everything up to the stop.</p>"
                "<p>The simulation ends at approximately 105 s, when the conversion reaches 0.8.</p>",
                nothing,
                self._stopped,
                "The Run stopped when the conversion reached 0.8.",
                waiting="Replace the trigger with the conversion and the stopping trigger, and click Run.",
            ),
            Step(
                "ignore_stop",
                "Run Past the Stop",
                "<p>Stopping triggers can be disabled in general without having to remove them from the model. "
                "In <b>Simulation → Simulation Settings…</b>, uncheck <b>Use Authored Stopping "
                "Interventions</b>, and click <b>OK</b>. Click <b>Run</b> again.</p>"
                "<p>The curves reach 600 s again. Note that this setting applies only to this tab.</p>",
                nothing,
                self._stop_ignored,
                "The Run ignored the stop and reached 600 s.",
                waiting="Uncheck Use Authored Stopping Interventions in Simulation Settings and click Run.",
                dialog=SimulationSettingsDialog,
            ),
            Step(
                "load_example",
                "Load I11",
                "<p>Open <b>Examples → Intervention Examples → I11: Light–dark photoswitching</b>. I11 "
                "replaces the model in this tab.</p>"
                "<p>In I11, a photochemical reaction, Light is explicitly included as a species. A + Light → B "
                "forms B only while the light source is on, and B → A turns B back into A independently of light. "
                "A <b>protocol</b> holds [Light] = 1 M for the first 120 s of every 180 s, three times, and an "
                "<b>after</b> sets it to 0 at the end of each period. As in step 7, Light is not consumed while "
                "it is held.</p>"
                "<p>Select only <b>intervention_set1</b> in <b>Initial Conditions</b> and click <b>Run</b>.</p>",
                nothing,
                self._example_ran,
                "I11 has run.",
                waiting="Load I11, select only intervention_set1, and click Run.",
            ),
            Step(
                "save_file",
                "Save the Tutorial File",
                "<p><b>light_runs.xlsx</b> contains three simulated runs of I11, one per sheet: <b>Run 1</b>, "
                "<b>Run 2</b>, and <b>Run 3</b>. Each records A and B from 0 to 600 s, with noise added. Each run "
                "was simulated with the light switched on and off at slightly different times than in the "
                "<b>protocol</b> of I11, and with slightly different values of k1 and k2.</p>"
                "<p>Click <b>Save Tutorial File…</b> and choose a folder. Kindred creates an "
                "<b>Interventions</b> folder there and copies the workbook into it.</p>",
                nothing,
                self._saved,
                "light_runs.xlsx is in the Interventions folder.",
                waiting="Save the tutorial file to continue.",
                action=("Save Tutorial File…", self._save_file),
            ),
            Step(
                "import_data",
                "Import the Runs",
                "<p>Choose <b>File → Import Data…</b>. Open <b>light_runs.xlsx</b> from the Interventions folder, "
                "keep the three sheets checked, and click <b>Import</b>.</p>"
                "<p>In the mapping review, click <b>OK</b>. Each run gets its own set, which starts from its row "
                "at time 0.</p>",
                nothing,
                self._mapped,
                "Each run is mapped to its own new set.",
                waiting="Import the three sheets of light_runs.xlsx and click OK in the mapping review. If the "
                "review was closed, choose each run in Dataset in the Data panel, click Change IC Set…, then "
                "Create IC Set from Data…, and click OK.",
                dialog=IMPORT_DIALOGS,
            ),
            Step(
                "run_data",
                "Compare I11 with the Runs",
                "<p>In <b>Plot Overlays</b>, beside <b>Data</b> in the Data panel, check the three "
                "light_runs.xlsx datasets.</p>"
                "<p>In <b>Initial Conditions</b>, select only the three sets of the runs, such as "
                "<b>light_runs.xlsx::Run 1</b>, with Ctrl-click, or Cmd-click on macOS. Click <b>Run</b>.</p>"
                "<p>The curves of I11 follow the regular periods of its <b>protocol</b>, while the markers of "
                "each run follow the periods with which that run was simulated. Where the light of a run was "
                "switched on or off earlier or later, its markers rise or fall earlier or later than its "
                "curves.</p>",
                nothing,
                lambda: self._overlays_shown()
                and self._ran_from_zero(self._workbook_sets(), _one(InterventionProtocol)),
                "The runs are shown and simulated with the protocol.",
                waiting="Check the three datasets in Plot Overlays, select only their three sets, and click Run.",
            ),
            Step(
                "own_times",
                "Give Each Period Its Own Times",
                "<p>A fit can vary a time only when the time comes from a parameter. In a <b>protocol</b>, only "
                "<b>start_time</b> can come from a parameter, and every repetition has the same spacing and "
                "<b>duration</b>, so a fit cannot find when each period began and ended. Instead, each period "
                "can be written as its own <b>hold</b> and <b>set</b>, with their times from parameters.</p>"
                "<p>Delete the <b>protocol</b> line and type the first period:</p>"
                + _lines(*_period(*_PERIODS[0]))
                + "<p>As in step 12, the <b>set</b> at the end of the <b>hold</b> turns the light off.</p>"
                "<p>Then paste the other two:</p>"
                + _lines(*_period(*_PERIODS[1]), *_period(*_PERIODS[2]))
                + "<p>Select the three run sets and click <b>Run</b>.</p>",
                nothing,
                self._ran_own_times,
                "Each period has its own times.",
                waiting="Write the three periods with times from on1 to off3, select the three run sets, and click "
                "Run.",
            ),
            Step(
                "open_fit",
                "Open the Fitting Window",
                "<p>This is identical to step 23, because the times are those of the <b>protocol</b>.</p>"
                "<p>Choose <b>Fitting → Global Fit…</b>.</p>",
                nothing,
                lambda: self._fit_window() is not None,
                "The Fitting Window is open.",
                dialog=FittingWindow,
            ),
            Step(
                "choose_datasets",
                "Choose the Datasets",
                "<p>On <b>Data and Targets</b>, check the three light_runs.xlsx datasets in <b>Datasets</b>, "
                "and leave any others unchecked. The fit compares the simulated A and B of each set with the data "
                "of its run.</p>",
                nothing,
                self._included_only,
                "The three runs are included.",
                waiting="Check the three light_runs.xlsx datasets only.",
                dialog=FittingWindow,
            ),
            Step(
                "vary",
                "Vary the Rates and the Times",
                "<p>Click <b>Parameters</b>. Check <b>Vary</b> for <b>k1</b>, <b>k2</b>, <b>on1</b>, "
                "<b>off1</b>, <b>on2</b>, <b>off2</b>, <b>on3</b>, and <b>off3</b>.</p>"
                "<p>In <b>Across Datasets</b>, choose <b>Per-dataset</b> for each of the six times, so that the "
                "fit finds separate times for each run. k1 and k2 stay shared, so the fit finds one value of each "
                "for all three runs.</p>",
                nothing,
                lambda: self._included_only() and self._varied(),
                "k1 and k2 vary, and each run has its own times.",
                waiting="Vary k1, k2, and the six times, with Per-dataset for each time.",
                dialog=FittingWindow,
            ),
            Step(
                "set_bounds",
                "Keep the Periods in Order",
                "<p>The default <b>Minimum</b> and <b>Maximum</b> of each time overlap with those of the "
                "neighboring times, and the range of on1 goes below 0. Kindred refuses times that are negative, "
                "that end a period before it starts, or that make two periods overlap, so the fit cannot use "
                "those values.</p>"
                "<p>Click <b>Set Bounds for Parameters…</b> and give each time a range that starts at 0 or later "
                "and ends before the range of the next time begins. For example: on1 0 to 30, off1 90 to 145, "
                "on2 155 to 210, off2 270 to 325, on3 335 to 390, and off3 450 to 510.</p>",
                nothing,
                lambda: self._included_only() and self._varied() and self._bounded(),
                "Every range keeps the periods in order.",
                waiting="Give each time a range that starts at 0 or later and ends before the range of the next "
                "time begins.",
                dialog=FittingWindow,
            ),
            Step(
                "run_fit",
                "Run the Fit",
                "<p>Click <b>Run Fit</b> and wait for the fit to finish. It varies 20 values: k1, k2, and six "
                "times for each of the three runs.</p>",
                nothing,
                lambda: self._included_only() and self._varied() and self._bounded() and self._fit_result(),
                "The fit is complete.",
                waiting="Include only the three runs, vary k1, k2, and the six times per dataset, keep the periods "
                "in order, and click Run Fit.",
                dialog=FittingWindow,
            ),
            Step(
                "finish",
                "Read the Results",
                "<p>Results lists k1 and k2 once, and each time once for each run. <b>Summary</b> lists every "
                "fitted value.</p>"
                "<p>The three runs were simulated with <b>k1 = 0.0135</b> M<sup>−1</sup> s<sup>−1</sup> and "
                "<b>k2 = 0.0024</b> s<sup>−1</sup>, and with the light switched on during these periods:</p>"
                "<ul>"
                "<li>Run 1: 0 to 112, 176 to 301, and 365 to 479 s</li>"
                "<li>Run 2: 4 to 127, 183 to 296, and 358 to 485 s</li>"
                "<li>Run 3: 0 to 117, 189 to 305, and 361 to 470 s</li>"
                "</ul>"
                "<p>Compare them with the fitted values.</p>"
                "<p>Closing the Fitting Window without <b>Apply Fit</b> discards the result.</p>"
                "<p>Click <b>Finish</b> to close this lesson. Open <b>Help → Interactive Tutorials…</b> to choose "
                "another.</p>",
                nothing,
                dialog=FittingWindow,
            ),
        )
