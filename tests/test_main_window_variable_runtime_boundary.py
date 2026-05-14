from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import pytest

from kindred.core.simulation_preparation import prepare_bound_mechanism
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.gui.main_window_variable_runtime import MainWindowVariableRuntime

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeRuntimeHost:
    def __init__(
        self,
        *,
        reactions_text: str,
        state_network_dsl: str = "",
        temperature_k: float = 298.15,
        wegscheider_enabled: bool = False,
    ) -> None:
        self._reactions_text = str(reactions_text)
        self._state_network_dsl = str(state_network_dsl)
        self._temperature_k = float(temperature_k)
        self._wegscheider_enabled = bool(wegscheider_enabled)
        self._wegscheider_cyclicity_enabled = bool(wegscheider_enabled)
        self._temperature_spinbox = SimpleNamespace(value=lambda: float(self._temperature_k))
        self._slider_values: Dict[str, float] = {}
        self._slider_metadata: Dict[str, Dict[str, object]] = {}
        self._slider_set_calls: list[tuple[Dict[str, float], Dict[str, Dict[str, object]], bool]] = []
        self._slider_clear_calls = 0
        self._parameter_summary_updates: list[Dict[str, Tuple[float, str]]] = []
        self._temperature_override_calls: list[tuple[bool, str]] = []
        self._temperature_mode_indicator_texts: list[str] = []
        self._best_effort_failures: list[tuple[str, str]] = []
        self._sim_controller = SimpleNamespace(
            run_state=SimpleNamespace(),
            ensure_parallel_batch_runtime_ready=lambda *, wait=False: None,
        )
        self._slider_runtime_dirty = False
        self._slider_overrides: Dict[str, float] = {}
        self._slider_overrides_by_set: Dict[str, Dict[str, float]] = {}
        self._slider_runtime = object()
        self._step_index_map: list[object] = []
        self._energy_mode_channels: list[object] = []
        self._plot_parameter_summary_stale = False
        self._temperature_tooltip_failed = False
        self._preview_session = SimpleNamespace(
            sync_committed_slider_values=self._sync_committed_slider_values,
        )

    def mechanism_reactions_text_raw(self) -> str:
        return str(self._reactions_text)

    def mechanism_state_network_dsl_raw(self) -> str:
        return str(self._state_network_dsl)

    def slider_overrides(self, set_id: str | None = None) -> Dict[str, float]:
        if set_id is not None and str(set_id) in self._slider_overrides_by_set:
            return dict(self._slider_overrides_by_set[str(set_id)])
        return dict(self._slider_overrides)

    def _parameter_algebra_spec_for_ui(self):
        return None

    def _get_mechanism_text(self) -> str:
        return str(self._reactions_text)

    def set_mechanism_reactions_text_with_optional_undo(
        self,
        new_text: str,
        description: str,
        *,
        record_undo: bool,
    ) -> None:
        _ = (description, record_undo)
        self._reactions_text = str(new_text)

    def finalize_authoritative_mechanism_widget_write(self, *, dispatch_consumers: bool) -> None:
        _ = dispatch_consumers

    def temperature_spinbox_value(self) -> float:
        return float(self._temperature_k)

    def wegscheider_cyclicity_enabled(self) -> bool:
        return bool(self._wegscheider_enabled)

    def set_variable_sliders(
        self,
        variables: Dict[str, float],
        *,
        metadata: Dict[str, Dict[str, object]] | None = None,
        preserve_visibility: bool = False,
        visibility_scope_signature: object | None = None,
    ) -> None:
        self._slider_values = dict(variables)
        self._slider_metadata = dict(metadata or {})
        self._slider_set_calls.append(
            (
                dict(variables),
                dict(metadata or {}),
                bool(preserve_visibility),
                visibility_scope_signature,
            )
        )

    def variable_slider_values(self) -> Dict[str, float]:
        return dict(self._slider_values)

    def clear_variable_sliders(self) -> None:
        self._slider_clear_calls += 1
        self._slider_values = {}
        self._slider_metadata = {}

    def set_slider_runtime_dirty(self, value: bool) -> None:
        self._slider_runtime_dirty = bool(value)

    def update_main_plot_parameter_summary(self, parameters: Dict[str, Tuple[float, str]]) -> None:
        self._parameter_summary_updates.append(dict(parameters))

    def set_temperature_override_state(self, *, enabled: bool, tooltip: str) -> None:
        self._temperature_override_calls.append((bool(enabled), str(tooltip)))

    def set_temperature_mode_indicator_text(self, text: str) -> None:
        self._temperature_mode_indicator_texts.append(str(text))

    def _record_best_effort_failure(
        self,
        code: str,
        *,
        message: str,
        exc: object | None = None,
        max_logs: int | None = None,
    ) -> None:
        _ = (exc, max_logs)
        self._best_effort_failures.append((str(code), str(message)))

    def _sync_committed_slider_values(self, values: Dict[str, float]) -> None:
        self._slider_overrides = dict(values or {})


def _energy_mode_dsl(*, temperature_k: float = 200.0) -> str:
    return "\n".join(
        [
            f"T={temperature_k}",
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "state: B, kind=GS, energy=-5, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
            "edge: TS1,B",
        ]
    )


@pytest.mark.unit
def test_runtime_extract_and_parameter_refresh_use_public_main_window_boundary() -> None:
    host = _FakeRuntimeHost(reactions_text="reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0")
    runtime = MainWindowVariableRuntime(host)

    runtime.extract_and_populate_variables()

    assert host._slider_set_calls
    assert host._slider_set_calls[-1][2] is False
    assert host._slider_set_calls[-1][3] == runtime._slider_visibility_scope_signature(
        host.variable_slider_values(),
        runtime.variable_metadata(),
    )
    assert host.variable_slider_values()["k1"] == pytest.approx(1.0)
    assert host.slider_overrides()["k1"] == pytest.approx(1.0)
    assert "k1" in runtime.variable_metadata()
    assert host._parameter_summary_updates
    assert host._parameter_summary_updates[-1]["k1"][0] == pytest.approx(1.0)


@pytest.mark.unit
def test_runtime_extract_does_not_warm_parallel_pool_during_variable_refresh() -> None:
    class _FakePoolController:
        def __init__(self) -> None:
            self.calls = 0
            self.run_state = SimpleNamespace()

        def ensure_parallel_batch_runtime_ready(self, *, wait: bool = False) -> None:
            _ = wait
            self.calls += 1

    host = _FakeRuntimeHost(reactions_text="reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0")
    host._sim_controller = _FakePoolController()
    runtime = MainWindowVariableRuntime(host)

    runtime.extract_and_populate_variables()
    runtime.extract_and_populate_variables()

    assert host._sim_controller.calls == 0


@pytest.mark.unit
def test_runtime_extract_does_not_prepare_slider_runtime_on_gui_thread(monkeypatch) -> None:
    host = _FakeRuntimeHost(reactions_text="reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0")
    runtime = MainWindowVariableRuntime(host)
    calls: list[dict[str, object]] = []

    def _fake_prepare_bound_mechanism(*, mechanism_text, param_names, temperature_K, initials, **kwargs):
        calls.append(
            {
                "mechanism_text": str(mechanism_text),
                "param_names": list(param_names),
                "temperature_K": float(temperature_K),
                "initials": dict(initials),
                "kwargs": dict(kwargs),
            }
        )
        return SimpleNamespace(param_names=list(param_names), bindings={})

    monkeypatch.setattr(
        "kindred.gui.main_window_variable_runtime.prepare_bound_mechanism",
        _fake_prepare_bound_mechanism,
    )

    runtime.extract_and_populate_variables()

    assert calls == []
    assert runtime.slider_runtime_dirty() is True


@pytest.mark.unit
def test_runtime_extract_ignores_supported_named_inline_initial_set_blocks_without_rewriting_text() -> None:
    host = _FakeRuntimeHost(
        reactions_text="\n".join(
            [
                "reaction: A -> B; k=1.0",
                "",
                "Set B = {",
                "[A] = 1.5",
                "# comment-only import metadata",
                "}",
            ]
        )
    )
    runtime = MainWindowVariableRuntime(host)

    runtime.extract_and_populate_variables()

    assert host._slider_set_calls
    assert host.variable_slider_values()["k1"] == pytest.approx(1.0)
    assert "Set B = {" in host.mechanism_reactions_text_raw()
    assert host._slider_clear_calls == 0


@pytest.mark.unit
def test_runtime_extract_keeps_unsupported_let_named_set_blocks_invalid() -> None:
    host = _FakeRuntimeHost(
        reactions_text="\n".join(
            [
                "reaction: A -> B; k=1.0",
                "",
                "let baseline = {",
                "}",
            ]
        )
    )
    runtime = MainWindowVariableRuntime(host)

    runtime.extract_and_populate_variables()

    assert host._slider_set_calls == []
    assert host.variable_slider_values() == {}


@pytest.mark.unit
def test_runtime_sanitize_mechanism_parameter_conflicts_canonicalizes_keq_to_k() -> None:
    host = _FakeRuntimeHost(reactions_text="equilibrium: A <-> B ; kf=6 ; Keq=3")
    runtime = MainWindowVariableRuntime(host)

    sanitized_text, baseline_variables, baseline_metadata = runtime.sanitize_mechanism_parameter_conflicts(
        host.mechanism_reactions_text_raw()
    )

    assert sanitized_text == "equilibrium: A <-> B ; kf=6, Keq=3"
    assert baseline_variables == {}
    assert baseline_metadata == {}


@pytest.mark.unit
def test_runtime_extract_keq_alias_writeback_preserves_equilibrium_value() -> None:
    original_text = "equilibrium: A <-> B ; kf=6 ; Keq=3"
    host = _FakeRuntimeHost(reactions_text=original_text)
    runtime = MainWindowVariableRuntime(host)

    sanitized_text, _, _ = runtime.sanitize_mechanism_parameter_conflicts(original_text)
    original = parse_dsl_to_mechanism(original_text, initials={})
    sanitized = parse_dsl_to_mechanism(sanitized_text, initials={})

    assert sanitized_text == "equilibrium: A <-> B ; kf=6, Keq=3"
    assert float(original.equilibria[0].kf) == pytest.approx(float(sanitized.equilibria[0].kf))
    assert float(original.equilibria[0].kr) == pytest.approx(float(sanitized.equilibria[0].kr))


@pytest.mark.unit
def test_runtime_sanitize_duplicate_keq_aliases_leaves_text_unchanged() -> None:
    text = "equilibrium: A <-> B ; kf=1 ; Keq=3 ; Keq=5"
    host = _FakeRuntimeHost(reactions_text=text)
    runtime = MainWindowVariableRuntime(host)

    sanitized_text, baseline_variables, baseline_metadata = runtime.sanitize_mechanism_parameter_conflicts(text)

    assert sanitized_text == text
    assert baseline_variables == {}
    assert baseline_metadata == {}


@pytest.mark.unit
def test_runtime_sanitize_case_only_duplicate_keq_aliases_leaves_text_unchanged() -> None:
    text = "equilibrium: A <-> B ; kf=1 ; Keq=3 ; keq=5"
    host = _FakeRuntimeHost(reactions_text=text)
    runtime = MainWindowVariableRuntime(host)

    sanitized_text, baseline_variables, baseline_metadata = runtime.sanitize_mechanism_parameter_conflicts(text)

    assert sanitized_text == text
    assert baseline_variables == {}
    assert baseline_metadata == {}


@pytest.mark.unit
def test_runtime_sanitize_duplicate_reaction_k_tokens_leaves_text_unchanged() -> None:
    text = "reaction: A -> B ; k=3 ; k=5"
    host = _FakeRuntimeHost(reactions_text=text)
    runtime = MainWindowVariableRuntime(host)

    sanitized_text, baseline_variables, baseline_metadata = runtime.sanitize_mechanism_parameter_conflicts(text)

    assert sanitized_text == text
    assert baseline_variables == {}
    assert baseline_metadata == {}


@pytest.mark.unit
def test_runtime_sanitize_duplicate_reversible_kf_tokens_leaves_text_unchanged() -> None:
    text = "reaction: A <-> B ; kf=3 ; kf=5 ; K=2"
    host = _FakeRuntimeHost(reactions_text=text)
    runtime = MainWindowVariableRuntime(host)

    sanitized_text, baseline_variables, baseline_metadata = runtime.sanitize_mechanism_parameter_conflicts(text)

    assert sanitized_text == text
    assert baseline_variables == {}
    assert baseline_metadata == {}


@pytest.mark.unit
def test_runtime_energy_mode_sync_uses_public_main_window_boundary() -> None:
    host = _FakeRuntimeHost(reactions_text="")
    mechanism = SimpleNamespace(metadata={"temperature_K": 200.0})

    MainWindowVariableRuntime(host).sync_energy_mode_temperature_from_mechanism(mechanism)

    assert host._temperature_mode_indicator_texts == ["Temperature: 200.00 K (from DSL)"]
    assert host._temperature_override_calls == [(False, "Overridden by energy-mode DSL (T=...).")]


@pytest.mark.unit
def test_runtime_energy_mode_population_uses_public_slider_and_plot_seams() -> None:
    dsl_text = _energy_mode_dsl()
    mechanism = parse_dsl_to_mechanism(dsl_text, initials={})
    host = _FakeRuntimeHost(reactions_text=dsl_text, temperature_k=298.15)
    runtime = MainWindowVariableRuntime(host)

    runtime.populate_energy_mode_variables_from_mechanism(
        mechanism,
        refresh_sliders=True,
        preserve_visibility=True,
    )

    assert host._slider_set_calls
    assert host._slider_set_calls[-1][2] is True
    assert host._slider_set_calls[-1][3] == runtime._slider_visibility_scope_signature(
        host.variable_slider_values(),
        runtime.variable_metadata(),
    )
    slider_values = host.variable_slider_values()
    assert any(name.startswith("dGact_fwd__") for name in slider_values)
    assert any(name.startswith("dG_eq__") for name in slider_values)
    assert host._parameter_summary_updates
    assert "Keq (A→B via TS1)" in host._parameter_summary_updates[-1]


@pytest.mark.unit
def test_runtime_energy_mode_population_accepts_structured_energy_result_mechanism() -> None:
    from kindred.core.kinetics import K_from_deltaG_eq

    host = _FakeRuntimeHost(
        reactions_text="",
        state_network_dsl="\n".join(
            [
                "energy=kJ/mol",
                "T=298.15",
                "state: A, kind=GS, energy=0, members=A",
                "state: B, kind=GS, energy=5, members=B",
                "state: TS1, kind=TS, energy=25",
                "edge: A,TS1",
                "edge: TS1,B",
            ]
        ),
        temperature_k=298.15,
    )
    runtime = MainWindowVariableRuntime(host)
    bound = prepare_bound_mechanism(
        mechanism_text="# State Network\n" + host.mechanism_state_network_dsl_raw(),
        param_names=["dGact_fwd__TS1__A__B", "dG_eq__TS1__A__B"],
        temperature_K=298.15,
        initials={},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )

    runtime.populate_energy_mode_variables_from_mechanism(
        bound.mechanism,
        refresh_sliders=True,
        preserve_visibility=True,
    )

    assert host._parameter_summary_updates
    assert host._parameter_summary_updates[-1]["Keq (A→B via TS1)"][0] == pytest.approx(
        K_from_deltaG_eq(5000.0, 298.15)
    )
    assert host._best_effort_failures == []


@pytest.mark.unit
def test_runtime_visibility_scope_signature_tracks_slider_universe_not_dsl_values() -> None:
    host = _FakeRuntimeHost(reactions_text="reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0")
    runtime = MainWindowVariableRuntime(host)

    runtime.extract_and_populate_variables()
    first_scope = host._slider_set_calls[-1][3]

    host._reactions_text = "reaction: A -> B; k=4.0\ninitial: A=3.0\ninitial: B=0.0"
    runtime.extract_and_populate_variables(preserve_visibility=True)
    second_scope = host._slider_set_calls[-1][3]

    host._reactions_text = "reaction: C -> D; k=4.0\ninitial: C=3.0\ninitial: D=0.0"
    runtime.extract_and_populate_variables(preserve_visibility=True)
    third_scope = host._slider_set_calls[-1][3]

    assert second_scope == first_scope
    assert third_scope != first_scope


@pytest.mark.unit
def test_runtime_visibility_scope_signature_ignores_same_universe_reorder() -> None:
    runtime = MainWindowVariableRuntime(_FakeRuntimeHost(reactions_text=""))

    first_scope = runtime._slider_visibility_scope_signature(
        {"k1": 1.0, "a": 2.0, "b": 3.0},
        {
            "k1": {"type": "reaction", "index": 1, "role": "k", "label": "Step 1: A -> B", "unit": "1/s"},
            "a": {"type": "scalar", "index": 0, "role": "scalar", "label": "Scalar parameter", "unit": "1", "line": 4},
            "b": {"type": "scalar", "index": 0, "role": "scalar", "label": "Scalar parameter", "unit": "1", "line": 5},
        },
    )
    reordered_scope = runtime._slider_visibility_scope_signature(
        {"k1": 4.0, "b": 30.0, "a": 20.0},
        {
            "k1": {"type": "reaction", "index": 1, "role": "k", "label": "Step 1: A -> B", "unit": "1/s"},
            "b": {"type": "scalar", "index": 0, "role": "scalar", "label": "Scalar parameter", "unit": "1", "line": 4},
            "a": {"type": "scalar", "index": 0, "role": "scalar", "label": "Scalar parameter", "unit": "1", "line": 5},
        },
    )
    different_scope = runtime._slider_visibility_scope_signature(
        {"k1": 1.0, "a": 2.0, "c": 3.0},
        {
            "k1": {"type": "reaction", "index": 1, "role": "k", "label": "Step 1: A -> B", "unit": "1/s"},
            "a": {"type": "scalar", "index": 0, "role": "scalar", "label": "Scalar parameter", "unit": "1", "line": 4},
            "c": {"type": "scalar", "index": 0, "role": "scalar", "label": "Scalar parameter", "unit": "1", "line": 5},
        },
    )

    assert reordered_scope == first_scope
    assert different_scope != first_scope


@pytest.mark.unit
def test_prepare_slider_runtime_rebuilds_after_superset_preview_to_avoid_stale_binding_leak(monkeypatch) -> None:
    class _Binding:
        def __init__(self, value: float) -> None:
            self.value = float(value)

        def set(self, value: float) -> None:
            self.value = float(value)

    class _Runtime:
        def __init__(self, param_names: list[str]) -> None:
            self.param_names = list(param_names)
            self.bindings = {name: _Binding(0.0) for name in param_names}

    created_runtimes: list[_Runtime] = []

    def _fake_prepare_bound_mechanism(
        *,
        mechanism_text: str,
        param_names: list[str],
        temperature_K: float,
        initials: Dict[str, float],
        use_advanced_dsl: bool,
        wegscheider_cyclicity_enabled: bool,
    ) -> _Runtime:
        _ = (
            mechanism_text,
            temperature_K,
            initials,
            use_advanced_dsl,
            wegscheider_cyclicity_enabled,
        )
        runtime = _Runtime(param_names)
        created_runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(
        "kindred.gui.main_window_variable_runtime.prepare_bound_mechanism",
        _fake_prepare_bound_mechanism,
    )
    host = _FakeRuntimeHost(reactions_text="reaction: A -> B; k=1.0")
    host._slider_overrides_by_set = {
        "set-a": {"k1": 1.5, "k2": 2.5},
        "set-b": {"k1": 3.5},
    }
    runtime = MainWindowVariableRuntime(host)

    first_runtime = runtime.prepare_slider_runtime(set_id="set-a")

    assert first_runtime is not None
    assert runtime.apply_slider_overrides_to_bindings(first_runtime, set_id="set-a") is True
    assert first_runtime.bindings["k1"].value == pytest.approx(1.5)
    assert first_runtime.bindings["k2"].value == pytest.approx(2.5)

    second_runtime = runtime.prepare_slider_runtime(set_id="set-b")

    assert second_runtime is not None
    assert second_runtime is not first_runtime
    assert runtime.apply_slider_overrides_to_bindings(second_runtime, set_id="set-b") is True
    assert second_runtime.bindings["k1"].value == pytest.approx(3.5)
    assert "k2" not in second_runtime.bindings

    reused_runtime = runtime.prepare_slider_runtime(set_id="set-b")

    assert reused_runtime is second_runtime
    assert len(created_runtimes) == 2


@pytest.mark.unit
def test_prepare_slider_runtime_builds_structured_energy_bindings_from_raw_state_network() -> None:
    host = _FakeRuntimeHost(
        reactions_text="",
        state_network_dsl="\n".join(
            [
                "energy=kJ/mol",
                "T=298.15",
                "state: A, kind=GS, energy=0, members=A",
                "state: B, kind=GS, energy=5, members=B",
                "state: TS1, kind=TS, energy=25",
                "edge: A,TS1",
                "edge: TS1,B",
            ]
        ),
        temperature_k=298.15,
    )
    host._slider_overrides_by_set = {
        "set-a": {
            "dGact_fwd__TS1__A__B": 32.0,
            "dG_eq__TS1__A__B": 7.0,
        }
    }
    runtime = MainWindowVariableRuntime(host)
    runtime.set_variable_metadata(
        {
            "dGact_fwd__TS1__A__B": {
                "type": "energy",
                "role": "dG_act_fwd",
                "ts": "TS1",
                "reactant": "A",
                "product": "B",
            },
            "dG_eq__TS1__A__B": {
                "type": "energy",
                "role": "dG_eq",
                "ts": "TS1",
                "reactant": "A",
                "product": "B",
            },
        }
    )

    prepared = runtime.prepare_slider_runtime(set_id="set-a")

    assert prepared is not None
    assert set(prepared.bindings) >= {
        "dGact_fwd__TS1__A__B",
        "dG_eq__TS1__A__B",
    }

    equilibrium = prepared.mechanism.equilibria[0]
    initial_kf = float(equilibrium.kf())
    initial_kr = float(equilibrium.kr())

    assert runtime.apply_slider_overrides_to_bindings(prepared, set_id="set-a") is True
    assert float(equilibrium.kf()) != pytest.approx(initial_kf)
    assert float(equilibrium.kr()) != pytest.approx(initial_kr)
    assert float(equilibrium.kf()) > 0.0
    assert float(equilibrium.kr()) > 0.0


@pytest.mark.unit
def test_prepare_slider_runtime_records_wegscheider_cyclicity_block_reason() -> None:
    host = _FakeRuntimeHost(
        reactions_text="\n".join(
            [
                "equilibrium: A <-> B ; kf=1 ; K=2",
                "equilibrium: B <-> C ; kf=1 ; K=3",
                "equilibrium: C <-> A ; kf=1 ; K=7",
                "initial: A=1",
                "initial: B=0",
                "initial: C=0",
            ]
        ),
        wegscheider_enabled=True,
    )
    host._slider_overrides = {"Keq1": 2.0}
    runtime = MainWindowVariableRuntime(host)

    prepared = runtime.prepare_slider_runtime(set_id="set-a")

    assert prepared is None
    assert runtime.slider_runtime_unavailable_reason() == "unresolved Wegscheider cyclicity"


@pytest.mark.unit
def test_main_window_source_no_longer_reaches_through_run_state_for_variable_metadata() -> None:
    source = (REPO_ROOT / "kindred" / "gui" / "main_window.py").read_text(encoding="utf-8")
    runtime_source = (REPO_ROOT / "kindred" / "gui" / "main_window_variable_runtime.py").read_text(encoding="utf-8")

    assert "run_state.variable_metadata" not in source
    assert "run_state.variable_metadata" not in runtime_source


@pytest.mark.unit
def test_simulation_run_state_no_longer_declares_variable_metadata() -> None:
    source = (
        REPO_ROOT / "kindred" / "gui" / "controllers" / "simulation_run_state.py"
    ).read_text(encoding="utf-8")

    assert "variable_metadata" not in source
