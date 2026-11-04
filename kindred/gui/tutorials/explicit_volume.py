"""Add, dilute, withdraw, scale and feed solutions in one model that tracks its vessel volume, one step at a time."""

from dataclasses import replace
from html import escape
from math import isclose

from kindred.core.intervention_schedule import InterventionPhysical, InterventionProtocol
from kindred.core.project_presentation_state import VesselVolumeAxisSelection
from kindred.gui.tutorials.controller import Step
from kindred.gui.tutorials.practice_lesson import PracticeLesson

# The reactions of I13, with B present from the start.
_RATES = {"kf1": 0.02, "kr1": 0.004, "kf2": 0.01, "kr2": 0.002}
_SET = "practice_set"
_STARTS = {"A": 1.0, "B": 1.0, "C": 0.0, "D": 0.0}
# Kindred keeps volumes in m³, flow rates in m³/s, and stock concentrations in M.
_ML = 1e-6


def _lines(*lines: str) -> str:
    """Mechanism text in bold, one line per row, as the user types it."""
    return "<p><b>" + "<br>".join(escape(line, quote=False) for line in lines) + "</b></p>"


def _quantity(quantity, wanted) -> bool:
    """The quantity comes from the parameter that wanted names, or has the value wanted. None means absent."""
    if quantity is None or wanted is None:
        return quantity is wanted
    if isinstance(wanted, str):
        return quantity.parameter == wanted
    return quantity.parameter is None and isclose(quantity.value, wanted)


def _action(physical, op: str, amount, stock: dict | None = None, initial_volume=None) -> bool:
    """A solution addition or inflow, scale or withdrawal with this amount, exactly this stock, and this
    start volume."""
    stock = stock or {}
    composition = dict(physical.stock)
    return (
        physical.op == op
        and _quantity(physical.amount, amount)
        and composition.keys() == stock.keys()
        and all(_quantity(composition[name], value) for name, value in stock.items())
        and _quantity(physical.initial_volume, initial_volume)
    )


def _at(time: float, op: str, amount, stock: dict | None = None, initial_volume=None):
    """Accepts a solution addition, scale or withdrawal at this time."""
    return lambda operation: (
        isinstance(operation, InterventionPhysical)
        and operation.time is not None
        and isclose(operation.time, time)
        and _action(operation.physical, op, amount, stock, initial_volume)
    )


def _between(start: float, end: float, amount, stock: dict, initial_volume):
    """Accepts a solution inflow from start to end."""
    return lambda operation: (
        isinstance(operation, InterventionPhysical)
        and operation.start_time is not None
        and operation.end_time is not None
        and isclose(operation.start_time, start)
        and isclose(operation.end_time, end)
        and _action(operation.physical, "flux", amount, stock, initial_volume)
    )


def _only(*accepted):
    """Accepts a schedule of exactly these operations, in any order."""
    return lambda operations: len(operations) == len(accepted) and all(
        any(accept(operation) for operation in operations) for accept in accepted
    )


def _dose_and_sample(operations) -> bool:
    """One protocol from 60 s, every 180 s, three times, for 120 s, that adds 2 mL of 2 M B before and
    withdraws 4 mL after, with the start volume of 10 mL on one of those two actions."""
    if len(operations) != 1 or not isinstance(operations[0], InterventionProtocol):
        return False
    protocol = operations[0]
    if len(protocol.before) != 1 or len(protocol.after) != 1 or protocol.during:
        return False
    addition, sample = protocol.before[0].physical, protocol.after[0].physical
    if addition is None or sample is None or (addition.initial_volume is None) == (sample.initial_volume is None):
        return False
    return (
        protocol.start_time is not None
        and isclose(protocol.start_time, 60.0)
        and isclose(protocol.every, 180.0)
        and protocol.count == 3
        and protocol.duration is not None
        and isclose(protocol.duration, 120.0)
        and _quantity(addition.initial_volume or sample.initial_volume, 10 * _ML)
        and _action(replace(addition, initial_volume=None), "jump", 2 * _ML, {"B": 2.0})
        and _action(replace(sample, initial_volume=None), "withdraw", 4 * _ML)
    )


class ModelsWithExplicitVolume(PracticeLesson):
    title = "Models with Explicit Volume"
    description = (
        "Add solutions, solvent, and inflows to a model that tracks its vessel volume, withdraw aliquots, scale "
        "concentrations, take volumes and concentrations from parameters, and dose and sample in a repeating "
        "protocol."
    )
    _preset_rates = _RATES
    _needs_preset_set = False
    _run_steps = frozenset({
        "load_preset", "stock", "two_species", "solvent", "withdraw", "scale", "inflow", "volume_params",
        "protocol",
    })
    # Steps on the practice mechanism after it is pasted.
    _practice_steps = _run_steps - {"load_preset"}

    def blocked(self):
        missing = super().blocked()
        if missing is not None:
            return missing
        if self.step.name in self._practice_steps and not self._practice_ready():
            return self._missing(
                "This exercise needs the practice mechanism in Reactions and practice_set with A and B = 1 and "
                "C and D = 0.",
                "load_preset",
            )
        return None

    # -- the practice mechanism ------------------------------------------------
    def _mechanism_loaded(self) -> bool:
        # blocked() checks the practice mechanism for the steps that need it.
        return True

    def _practice_ready(self) -> bool:
        """The practice reactions and rates, and practice_set with A and B at 1 and C and D at 0."""
        if self.tab.prepared_dsl is None or not self._preset_rates_loaded():
            return False
        self.preset_set = next(
            (item.set_id for item in self.tab.initial_condition_sets if item.name == _SET), None
        )
        return (
            self.preset_set is not None
            and not self.registry.tab_runtime(self.tab_id).initial_text_by_set_id.get(self.preset_set)
            and dict(self.tab.initial_condition_set(self.preset_set).values) == _STARTS
        )

    @staticmethod
    def _parameter(result, name: str, value: float) -> bool:
        parameters = {item: number for item, number, _unit in result.parameters}
        return name in parameters and isclose(parameters[name], value)

    def _practice_ran(self, operations, parameters: dict[str, float] | None = None) -> bool:
        """practice_set ran from 0 to 600 s in this step, with the practice rates, these parameter values,
        and an intervention schedule that operations accepts."""
        if not self._practice_ready() or not self._run_ok(_RATES, (self.preset_set,)):
            return False
        settings = self.tab.resolved_execution_settings(self.preset_set)
        result = self.registry.tab_runtime(self.tab_id).result_repository.current_result(self.preset_set)
        return (
            settings.t_start == 0
            and settings.t_end == 600
            and operations(result.intervention_execution.authored.operations)
            and all(self._parameter(result, name, value) for name, value in (parameters or {}).items())
        )

    def _volume_shown(self) -> bool:
        """The Y selection of the practice tab includes Vessel volume."""
        state = self.registry.presentation(self.tab_id)
        return state.selected_y_uses_default_all or VesselVolumeAxisSelection() in state.selected_y_series

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
                "<p>This lesson adds volume-aware interventions to one model, one at a time. Each step replaces "
                "the intervention of the step before.</p>"
                "<p>Paste the model below into the <b>Reactions</b> box. This is the same model as <b>I13</b> in "
                "<b>Examples → Intervention Examples</b>, without its interventions, and with B present from the start.</p>"
                + _lines(
                    "A + B <-> C ; kf=0.02 ; kr=0.004",
                    "C + B <-> D ; kf=0.01 ; kr=0.002",
                    "",
                    "practice_set = {",
                    "[A] = 1.0",
                    "[B] = 1.0",
                    "[C] = 0.0",
                    "[D] = 0.0",
                    "}",
                )
                + "<p>In <b>Initial Conditions</b>, check that <b><i>t</i><sub>start</sub></b> is 0 and "
                "<b><i>t</i><sub>end</sub></b> is 600 seconds. Press <b>Enter</b> after each edit. Then click "
                "<b>Run</b>.</p>",
                nothing,
                lambda: practice(lambda operations: not operations),
                "practice_set has run without interventions.",
                waiting="Paste the model into Reactions, set t_start to 0 and t_end to 600 seconds, and click Run.",
            ),
            Step(
                "stock",
                "Add a Stock Solution",
                "<p>The Interventions lesson teaches <b>set</b>, <b>jump</b>, <b>flux</b>, <b>hold</b>, "
                "<b>protocol</b>, and <b>trigger</b> in detail. Those change concentrations only. The interventions "
                "in this lesson change reaction volumes.</p>"
                "<p>Add this line to the model and click <b>Run</b>:</p>"
                + _lines(
                    "intervention: op=jump; time=240; initial_volume=10 mL; "
                    "solution={volume=2 mL,stock={species=B,value=5 M}}"
                )
                + "<p><b>solution={...}</b> turns a <b>jump</b> operation into the addition of a solution, and "
                "replaces the <b>species</b> and <b>delta</b> of an ordinary <b>jump</b>. Inside the braces, the "
                "fields are separated by commas. <b>volume</b> is the volume added. <b>stock={...}</b> gives the "
                "concentration of each species in the added solution, here 5 M of B. At 240 s, Kindred adds 2 mL "
                "of the B stock solution to the main reaction, and every species already present in the main "
                "reaction changes concentration due to the dilution from 10 to 12 mL.</p>"
                "<p>To calculate a dilution, Kindred needs the volume in the vessel at simulation start. "
                "<b>initial_volume</b> gives it, here 10 mL. A model that adds or removes liquid needs it exactly "
                "once. Write it on any one intervention that adds a solution, feeds one in, or withdraws an "
                "aliquot, outside <b>solution={...}</b>. It always means the volume at simulation start, not at "
                "the time of that intervention. Volumes need an explicit unit, such as <b>mL</b> or <b>uL</b>. "
                "Stock concentrations are in M unless given another unit, such as <b>mM</b>.</p>"
                "<p>Kindred plots the volume as <b>Vessel volume</b>, on the same axis as the concentrations and "
                "in mL, the unit of <b>initial_volume</b>. It rises from 10 to 12 mL at 240 s, which compresses "
                "the concentrations at the bottom of the plot. Open the <b>Y</b> selector above the plot and "
                "uncheck <b>Vessel volume</b>. The concentrations fill the axis again. Vessel volume stays "
                "unchecked for later Runs until you check it again.</p>",
                nothing,
                lambda: practice(_only(_at(240.0, "jump", 2 * _ML, {"B": 5.0}, 10 * _ML)))
                and not self._volume_shown(),
                "2 mL of 5 M B is added at 240 s.",
                waiting="Add 2 mL of 5 M B at 240 s with initial_volume=10 mL, click Run, and uncheck Vessel volume "
                "in Y.",
            ),
            Step(
                "two_species",
                "Two Species in One Stock",
                "<p>Replace the line with this one and click <b>Run</b>:</p>"
                + _lines(
                    "intervention: op=jump; time=240; initial_volume=10 mL; "
                    "solution={volume=2 mL,stock={species=B,value=2 M; species=D,value=1 M}}"
                )
                + "<p>A stock can contain several species. Separate its entries with semicolons. Within an entry, "
                "<b>species</b> and <b>value</b> stay separated by a comma. A species the stock does not list has "
                "zero concentration in it, so A and C are only diluted.</p>",
                nothing,
                lambda: practice(_only(_at(240.0, "jump", 2 * _ML, {"B": 2.0, "D": 1.0}, 10 * _ML))),
                "2 mL of a stock of B and D is added at 240 s.",
                waiting="Replace the line with the addition of 2 mL of 2 M B and 1 M D at 240 s, and click Run.",
            ),
            Step(
                "solvent",
                "Add Solvent",
                "<p>Replace the line with this one and click <b>Run</b>:</p>"
                + _lines("intervention: op=jump; time=240; initial_volume=10 mL; solution={volume=10 mL,stock={}}")
                + "<p><b>stock={}</b> is pure solvent. Adding 10 mL to 10 mL doubles the volume and halves every "
                "concentration.</p>",
                nothing,
                lambda: practice(_only(_at(240.0, "jump", 10 * _ML, {}, 10 * _ML))),
                "10 mL of solvent is added at 240 s.",
                waiting="Replace the line with the addition of 10 mL of solvent at 240 s, and click Run.",
            ),
            Step(
                "withdraw",
                "Withdraw an Aliquot",
                "<p>Replace the line with this one and click <b>Run</b>:</p>"
                + _lines("intervention: op=withdraw; time=240; initial_volume=10 mL; volume=2 mL")
                + "<p><b>op=withdraw</b> removes an aliquot of the well-mixed contents at <b>time</b>. "
                "<b>volume</b> is the volume removed.</p>"
                "<p>No concentrations are changed in withdrawal operations, so the concentration curves are the "
                "same as in step 2, whereas the vessel volume changes. Open the <b>Y</b> selector above the plot "
                "and check <b>Vessel volume</b> again. It drops from 10 to 8 mL at 240 s. Leave it checked for the "
                "rest of the lesson. A withdrawal must leave liquid in the vessel. A run withdrawing x mL or more "
                "from a current vessel volume of x mL is rejected.</p>",
                nothing,
                lambda: practice(_only(_at(240.0, "withdraw", 2 * _ML, None, 10 * _ML))) and self._volume_shown(),
                "2 mL is withdrawn at 240 s.",
                waiting="Replace the line with the withdrawal of 2 mL at 240 s, click Run, and check Vessel volume "
                "in Y.",
            ),
            Step(
                "scale",
                "Scale Concentrations",
                "<p><b>op=scale</b> multiplies every concentration not held by a <b>hold</b> by <b>factor</b> at "
                "<b>time</b>, and divides the volume by the same factor. It needs no volume declaration, and it cannot carry "
                "<b>initial_volume</b>.</p>"
                "<p>Replace the withdrawal with these two lines and click <b>Run</b>:</p>"
                + _lines(
                    "intervention: op=scale; time=240; factor=0.5",
                    "intervention: op=withdraw; time=400; initial_volume=10 mL; volume=2 mL",
                )
                + "<p>The withdrawal carries <b>initial_volume</b>, so the volume is tracked. Its declaration sets "
                "the volume at simulation start, although the withdrawal occurs only at 400 s. The volume rises "
                "from 10 to 20 mL at 240 s and drops to 18 mL at 400 s.</p>"
                "<p>Without a volume declaration in the model, <b>scale</b> changes concentrations only. "
                "<b>factor</b> must be positive.</p>",
                nothing,
                lambda: practice(_only(
                    _at(240.0, "scale", 0.5),
                    _at(400.0, "withdraw", 2 * _ML, None, 10 * _ML),
                )),
                "Concentrations are halved at 240 s, and 2 mL is withdrawn at 400 s.",
                waiting="Replace the withdrawal with the scale by 0.5 at 240 s and the withdrawal of 2 mL at 400 s, "
                "and click Run.",
            ),
            Step(
                "inflow",
                "Feed a Solution",
                "<p>Replace both lines with this one and click <b>Run</b>:</p>"
                + _lines(
                    "intervention: op=flux; start_time=60; end_time=360; initial_volume=10 mL; "
                    "solution={volume_rate=1 mL/min,stock={species=B,value=2 M}}"
                )
                + "<p><b>solution={...}</b> on a <b>flux</b> supplies a solution continuously from "
                "<b>start_time</b> to <b>end_time</b>, and replaces the <b>species</b> and <b>flux_rate</b> of an "
                "ordinary <b>flux</b>. <b>volume_rate</b> is the volume that flows in per unit time.</p>"
                "<p>From 60 to 360 s, 1 mL of a stock solution containing 2 M B flows into the reaction each "
                "minute, so Vessel volume rises linearly from 10 to 15 mL.</p>",
                nothing,
                lambda: practice(_only(_between(60.0, 360.0, _ML / 60, {"B": 2.0}, 10 * _ML))),
                "2 M B flows in from 60 to 360 s.",
                waiting="Replace both lines with the inflow of 2 M B at 1 mL/min from 60 to 360 s, and click Run.",
            ),
            Step(
                "volume_params",
                "Volumes and Concentrations from Parameters",
                "<p>A volume, flow rate, stock concentration, or factor can come from a <b>param</b> line. Add "
                "<b>_param</b> to the field name and give the name of the parameter: <b>initial_volume_param</b>, "
                "<b>volume_param</b>, <b>volume_rate_param</b>, <b>factor_param</b>, and <b>value_param</b> for a "
                "stock concentration. Give the parameter its unit with <b>unit =</b>, and Kindred converts it.</p>"
                "<p>Replace the line with these four and click <b>Run</b>:</p>"
                + _lines(
                    "param V0 = 10 ; unit = mL",
                    "param dose = 2000 ; unit = uL",
                    "param c_B = 5000 ; unit = mM",
                    "intervention: op=jump; time=240; initial_volume_param=V0; "
                    "solution={volume_param=dose,stock={species=B,value_param=c_B}}",
                )
                + "<p>Kindred evaluates <b>initial_volume_param</b> once, at simulation start. V0, dose, and c_B "
                "are now in <b>Visible Sliders</b> in the <b>Interactive Sliders</b> panel, so they can be varied "
                "without changing the model.</p>",
                nothing,
                lambda: practice(
                    _only(_at(240.0, "jump", "dose", {"B": "c_B"}, "V0")),
                    {"V0": 10 * _ML, "dose": 2 * _ML, "c_B": 5.0},
                ),
                "The addition takes its volumes and stock from V0, dose, and c_B.",
                waiting="Replace the line with V0 = 10 mL, dose = 2000 uL, c_B = 5000 mM, and the addition that "
                "uses them, and click Run.",
            ),
            Step(
                "protocol",
                "Dose and Sample in a Protocol",
                "<p>Replace the four lines with this one and click <b>Run</b>:</p>"
                + _lines(
                    "intervention: op=protocol; kind=repeat; start_time=60; every=180; count=3; duration=120; "
                    "before={op=jump,initial_volume=10 mL,solution={volume=2 mL,stock={species=B,value=2 M}}}; "
                    "after={op=withdraw,volume=4 mL}"
                )
                + "<p>Each repetition adds 2 mL of 2 M B at its start and withdraws a 4 mL sample 120 s later. "
                "Within a group, the fields of an action are separated by commas, and the solution and stock keep "
                "their own braces. In a repeating action, <b>initial_volume</b> goes inside the action in its "
                "group, not on the protocol line. It still counts once, as the volume at simulation start, not at "
                "each repetition.</p>"
                "<p>A trigger's <b>then={...}</b> also accepts <b>scale</b> and <b>withdraw</b>. A trigger action "
                "that carries <b>initial_volume</b> has it inside its braces, as here.</p>",
                nothing,
                lambda: practice(_dose_and_sample),
                "Each repetition adds B and withdraws a sample.",
                waiting="Replace the four lines with the protocol that adds 2 mL of 2 M B and withdraws 4 mL, and "
                "click Run.",
            ),
            Step(
                "finish",
                "Lesson Complete",
                "<p><b>Examples → Intervention Examples</b> has a ready-made model for each operation of this "
                "lesson:</p>"
                "<ul>"
                "<li><b>I13</b>: Add stock solution and solvent</li>"
                "<li><b>I14</b>: Pump a stock solution</li>"
                "<li><b>I15</b>: Withdraw an aliquot</li>"
                "<li><b>I16</b>: Scale concentrations</li>"
                "<li><b>I17</b>: Repeated dosing and sampling</li>"
                "</ul>"
                "<p>Click <b>Finish</b> to close this lesson. Open <b>Help → Interactive Tutorials…</b> to choose "
                "another.</p>",
                nothing,
            ),
        )
