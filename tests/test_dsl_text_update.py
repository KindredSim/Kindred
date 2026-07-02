import pytest

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.simulator.computational_mode import GENERATED_BLOCK_END, GENERATED_BLOCK_START
from kindred.core.simulator.dsl_text_update import (
    analyze_parameter_updates_to_dsl_text,
    analyze_step_parameter_update,
)
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.gui.main_window import MainWindow
from kindred.gui.main_window_variable_runtime import MainWindowVariableRuntime
from kindred.gui.mixins.fitting_mixin import FittingMixin
from kindred.gui.parameter_enumeration import enumerate_step_parameters_for_gui


def test_updating_derived_kr_on_keq_authority_equilibrium_is_blocked():
    source = "equilibrium: A <-> B; kf=6.0; Keq=3.0"

    outcome = analyze_step_parameter_update(source, "kr1", 4.0)

    assert not outcome.writable
    assert outcome.warning_reason == "target_unwritable"
    assert outcome.updated_text == source


def test_updating_keq_on_keq_authority_equilibrium_is_writable():
    source = "equilibrium: A <-> B; kf=6.0; Keq=3.0"

    outcome = analyze_step_parameter_update(source, "Keq1", 4.0)

    assert outcome.writable
    assert outcome.warning_reason is None
    assert "Keq=4" in outcome.updated_text


def test_exact_keq_token_is_case_insensitive_for_text_update():
    source = "equilibrium: A <-> B; kf=6.0; keq=3.0"

    outcome = analyze_step_parameter_update(source, "Keq1", 4.0)

    assert outcome.writable
    assert outcome.warning_reason is None
    assert "Keq=4" in outcome.updated_text


@pytest.mark.parametrize("key", ["K", "K_eq", "k_eq"])
def test_text_update_does_not_treat_legacy_aliases_as_keq_tokens(key):
    source = f"equilibrium: A <-> B; kf=6.0; {key}=3.0"

    outcome = analyze_step_parameter_update(source, "Keq1", 4.0)

    assert not outcome.writable
    assert outcome.updated_text == source
    assert outcome.warning_reason in {
        "constraint_analysis_failed",
        "missing_target",
        "target_unwritable",
    }


def test_updating_derived_kr_on_dg_authority_equilibrium_is_blocked():
    source = "equilibrium: A <-> B; kf=6.0; dG_eq=-1.0"

    outcome = analyze_step_parameter_update(source, "kr1", 4.0)

    assert not outcome.writable
    assert outcome.warning_reason == "target_unwritable"
    assert outcome.updated_text == source


def test_gui_enumeration_marks_dg_authority_reverse_rate_as_derived():
    mechanism = parse_dsl_to_mechanism("equilibrium: A <-> B; kf=6.0; dG_eq=-1.0", initials={})

    _variables, metadata = enumerate_step_parameters_for_gui(mechanism)

    assert metadata["kr1"]["editable"] is False
    assert metadata["kr1"]["derived"] is True


def test_updating_kf_on_dg_authority_equilibrium_preserves_public_authority():
    source = "equilibrium: A <-> B; kf=6.0; dG_eq=-1.0"

    outcome = analyze_step_parameter_update(source, "kf1", 7.0)

    assert outcome.writable
    assert "kf=7" in outcome.updated_text
    assert "dG_eq=-1.0" in outcome.updated_text
    assert "kr=" not in outcome.updated_text
    parse_dsl_to_mechanism(outcome.updated_text, initials={})


class _PreviewRecorder:
    def __init__(self) -> None:
        self.values = None
        self.schema_text = None

    def sync_committed_slider_values(self, values, *, schema_text=None):
        self.values = dict(values or {})
        self.schema_text = schema_text


class _PlotRecorder:
    def __init__(self) -> None:
        self.parameters = None

    def update_parameters(self, parameters):
        self.parameters = dict(parameters or {})


class _VariableRuntimeWindow:
    def __init__(self, source: MechanismAuthoringSource) -> None:
        self._source = source
        self._simulation_solver_owner = None
        self._mechanism_helpers = None
        self._preview_session = _PreviewRecorder()
        self._plot = _PlotRecorder()
        self._plot_parameter_summary_stale = False
        self._energy_mode_channels = []
        self.slider_values = None
        self.slider_metadata = None
        self.cleared = False

    def canonical_mechanism_source(self) -> MechanismAuthoringSource:
        return self._source

    def temperature_spinbox_value(self) -> float:
        return 298.15

    def wegscheider_cyclicity_enabled(self) -> bool:
        return False

    def set_variable_sliders(
        self,
        variables,
        *,
        metadata=None,
        preserve_visibility=False,
        visibility_scope_signature=None,
    ):
        self.slider_values = dict(variables or {})
        self.slider_metadata = dict(metadata or {})

    def clear_variable_sliders(self) -> None:
        self.cleared = True
        self.slider_values = {}
        self.slider_metadata = {}

    def variable_slider_values(self):
        return dict(self.slider_values or {})

    def main_plot(self):
        return self._plot

    def _record_best_effort_failure(self, *args, **kwargs):
        return None


class _ValueBox:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def value(self) -> float:
        return float(self._value)


class _SolverOwner:
    def wegscheider_cyclicity_enabled(self) -> bool:
        return False


class _VariableRuntimeProbe:
    def dsl_global_temperature_k(self, _dsl_text: str):
        return None


class _GeneratedFastMaterializationHarness:
    _apply_energy_overrides_to_computational_mode_fast_equilibria = (
        MainWindow._apply_energy_overrides_to_computational_mode_fast_equilibria
    )
    _collect_energy_overrides = MainWindow._collect_energy_overrides
    _runtime_parameter_values_for_materialization = MainWindow._runtime_parameter_values_for_materialization

    def __init__(self, metadata=None) -> None:
        self._metadata = dict(metadata or {})
        self._temperature_spinbox = _ValueBox(298.15)
        self._simulation_solver_owner = _SolverOwner()
        self._variable_runtime = _VariableRuntimeProbe()
        self.failures = []

    def variable_metadata(self):
        return dict(self._metadata)

    def _record_best_effort_failure(self, *args, **kwargs):
        self.failures.append((args, kwargs))


class _FitProjectTextHarness(FittingMixin, _GeneratedFastMaterializationHarness):
    def __init__(self, source_text: str) -> None:
        _GeneratedFastMaterializationHarness.__init__(self)
        self._source_text = str(source_text)

    def mechanism_reactions_text_raw(self) -> str:
        return self._source_text

    def _current_fit_authoritative_parameter_values(self):
        return {}

    def _current_fit_step_constraint_context(self):
        return {
            "temperature_K": 298.15,
            "wegscheider_cyclicity_enabled": False,
        }


def test_variable_runtime_does_not_parse_hidden_normalized_source_when_canonical_source_is_invalid():
    source = MechanismAuthoringSource.from_parts(
        reactions_text="equilibrium: A <-> B; k=2; kr=1",
        state_network_dsl="",
    )
    window = _VariableRuntimeWindow(source)
    runtime = MainWindowVariableRuntime(window)

    runtime.extract_and_populate_variables()

    assert window.slider_values in (None, {})
    assert window._preview_session.values is None


def test_energy_mode_fast_equilibrium_variables_come_from_parsed_mechanism_metadata_not_generated_text():
    generated_text = "\n".join(
        [
            "energy=kJ/mol",
            GENERATED_BLOCK_START,
            "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0",
            GENERATED_BLOCK_END,
            "init: A=1.0, B=0.0",
        ]
    )
    mechanism = parse_dsl_to_mechanism(generated_text, initials={})
    window = _VariableRuntimeWindow(
        MechanismAuthoringSource.from_parts(
            reactions_text="reaction: Placeholder -> Other; k=1",
            state_network_dsl="",
        )
    )
    runtime = MainWindowVariableRuntime(window)

    runtime.populate_energy_mode_variables_from_mechanism(
        mechanism,
        refresh_sliders=True,
    )

    assert "dG_eq_fast__feq__A__B" in (window.slider_values or {})
    assert window.slider_metadata["dG_eq_fast__feq__A__B"]["role"] == "dG_eq_fast"


def test_generated_fast_equilibrium_routes_through_normal_variable_refresh():
    generated_text = "\n".join(
        [
            "energy=kJ/mol",
            GENERATED_BLOCK_START,
            "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0",
            GENERATED_BLOCK_END,
            "init: A=1.0, B=0.0",
        ]
    )
    window = _VariableRuntimeWindow(
        MechanismAuthoringSource.from_parts(
            reactions_text=generated_text,
            state_network_dsl="",
        )
    )
    runtime = MainWindowVariableRuntime(window)

    runtime.extract_and_populate_variables()

    assert "dG_eq_fast__feq__A__B" in (window.slider_values or {})
    assert "kf1" not in (window.slider_values or {})
    assert window.slider_metadata["dG_eq_fast__feq__A__B"]["role"] == "dG_eq_fast"


def test_generated_fast_equilibrium_refresh_preserves_public_step_sliders():
    generated_text = "\n".join(
        [
            "energy=kJ/mol",
            GENERATED_BLOCK_START,
            "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0",
            GENERATED_BLOCK_END,
            "equilibrium: C <-> D; kf=4.0; dG_eq=2.0",
            "init: A=1.0, B=0.0, C=1.0, D=0.0",
        ]
    )
    window = _VariableRuntimeWindow(
        MechanismAuthoringSource.from_parts(
            reactions_text=generated_text,
            state_network_dsl="",
        )
    )
    runtime = MainWindowVariableRuntime(window)

    runtime.extract_and_populate_variables()

    assert "dG_eq_fast__feq__A__B" in (window.slider_values or {})
    assert "kf1" not in (window.slider_values or {})
    assert "kr1" not in (window.slider_values or {})
    assert "Keq1" not in (window.slider_values or {})
    assert "kf2" in (window.slider_values or {})
    assert "kr2" in (window.slider_values or {})
    assert "Keq2" in (window.slider_values or {})


def test_energy_mode_does_not_expose_public_equilibrium_as_generated_fast_slider():
    generated_text = "\n".join(
        [
            "energy=kJ/mol",
            GENERATED_BLOCK_START,
            "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0",
            GENERATED_BLOCK_END,
            "equilibrium: C <-> D; kf=4.0; dG_eq=2.0",
            "init: A=1.0, B=0.0, C=1.0, D=0.0",
        ]
    )
    mechanism = parse_dsl_to_mechanism(generated_text, initials={})
    window = _VariableRuntimeWindow(
        MechanismAuthoringSource.from_parts(
            reactions_text=generated_text,
            state_network_dsl="",
        )
    )
    runtime = MainWindowVariableRuntime(window)

    runtime.populate_energy_mode_variables_from_mechanism(
        mechanism,
        refresh_sliders=True,
    )

    fast_sliders = [
        name for name in (window.slider_values or {}) if str(name).startswith("dG_eq_fast__")
    ]
    assert fast_sliders == ["dG_eq_fast__feq__A__B"]


def test_text_update_does_not_rewrite_generated_block_indexed_parameters():
    source = "\n".join(
        [
            "energy=kJ/mol",
            GENERATED_BLOCK_START,
            "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0",
            GENERATED_BLOCK_END,
            "equilibrium: C <-> D; kf=4.0; dG_eq=2.0",
            "init: A=1.0, B=0.0, C=1.0, D=0.0",
        ]
    )

    analysis = analyze_parameter_updates_to_dsl_text(
        source,
        {
            "kf1": 99.0,
            "kf2": 5.0,
        },
    )

    assert "kf1" in analysis.missing
    assert "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0" in analysis.updated_text
    assert "equilibrium: C <-> D" in analysis.updated_text
    assert "kf=5" in analysis.updated_text
    assert "dG_eq=2.0" in analysis.updated_text


def test_generated_fast_energy_override_materializes_by_cm_id_not_indexed_step_map():
    source = "\n".join(
        [
            "energy=kJ/mol",
            GENERATED_BLOCK_START,
            "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0",
            GENERATED_BLOCK_END,
            "equilibrium: C <-> D; kf=4.0; dG_eq=2.0",
            "init: A=1.0, B=0.0, C=1.0, D=0.0",
        ]
    )
    metadata = {
        "dG_eq_fast__feq__A__B": {
            "type": "energy",
            "role": "dG_eq_fast",
            "cm_id": "feq__A__B",
            "unit": "kJ/mol",
            "kf_fixed": 10.0,
            "std_ratio": 1.0,
        }
    }
    harness = _GeneratedFastMaterializationHarness(metadata)

    updated = harness._apply_energy_overrides_to_computational_mode_fast_equilibria(
        source,
        overrides={"dG_eq_fast__feq__A__B": 2.5},
    )

    assert "equilibrium: A <-> B; kf=10.0; dG_eq=2.5; cm_id=feq__A__B; cm_std_ratio=1.0" in updated
    assert "kr=" not in updated
    parse_dsl_to_mechanism(updated, initials={})


def test_fit_project_text_apply_persists_generated_fast_parameter_by_cm_id():
    source = "\n".join(
        [
            "energy=kJ/mol",
            GENERATED_BLOCK_START,
            "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0",
            GENERATED_BLOCK_END,
            "init: A=1.0, B=0.0",
        ]
    )
    harness = _FitProjectTextHarness(source)

    update = harness._prepare_fit_parameter_project_text(
        {"dG_eq_fast__feq__A__B": 2.5},
        source_text=source,
    )

    assert update["missing"] == []
    assert update["update_errors"] == []
    assert "dG_eq=2.5" in update["updated_text"]
    assert "cm_id=feq__A__B" in update["updated_text"]
    parse_dsl_to_mechanism(str(update["updated_text"]), initials={})
