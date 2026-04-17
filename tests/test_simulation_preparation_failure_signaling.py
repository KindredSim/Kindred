from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def test_simulation_preparation_error_carries_stage_metadata() -> None:
    from kindred.core.simulation_preparation import SimulationPreparationError

    exc = SimulationPreparationError("parse", "bad dsl")

    assert exc.stage == "parse"
    assert str(exc) == "bad dsl"


def test_prepare_simulation_worker_run_reports_sparse_jacobian_fallback(monkeypatch) -> None:
    from kindred.core.simulation_preparation import prepare_simulation_worker_run

    monkeypatch.setattr(
        "kindred.core.sparse_jacobian.build_sparse_jacobian",
        lambda _mechanism: (_ for _ in ()).throw(RuntimeError("sparsity blew up")),
    )

    prepared = prepare_simulation_worker_run(
        mechanism_text="reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0",
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 1.0),
        solver_config={"solver": "BDF", "use_sparse_jacobian": True, "grid": {"N": 5}},
    )

    assert prepared.jacobian_func is None
    assert prepared.warnings == [
        "Sparse Jacobian unavailable; falling back to dense Jacobian: sparsity blew up"
    ]


def test_prepare_simulation_worker_run_disables_sparse_jacobian_for_temperature_schedule(monkeypatch) -> None:
    from kindred.core.simulation_preparation import prepare_simulation_worker_run

    called = {"n": 0}

    def _build_sparse(_mechanism):
        called["n"] += 1
        raise AssertionError("sparse Jacobian builder should not be called for scheduled temperature")

    monkeypatch.setattr("kindred.core.sparse_jacobian.build_sparse_jacobian", _build_sparse)

    prepared = prepare_simulation_worker_run(
        mechanism_text="\n".join(
            [
                "temp_step: t=[0,0.5,1.0], T=[300,600]",
                "energy=kJ/mol",
                "reaction: A -> B; A=1e3; Ea=50",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 1.0),
        solver_config={"solver": "BDF", "use_sparse_jacobian": True, "grid": {"N": 5}},
    )

    assert called["n"] == 0
    assert prepared.jacobian_func is None
    assert prepared.temperature_schedule is not None
    assert prepared.warnings == [
        "Sparse Jacobian disabled for scheduled-temperature run; falling back to dense Jacobian."
    ]


def test_prepare_simulation_worker_run_rejects_unknown_prepared_payload_version(monkeypatch) -> None:
    from kindred.core.simulation_preparation import (
        SimulationPreparationError,
        prepare_simulation_worker_run,
    )

    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_args, **_kwargs: None,
    )

    mechanism = SimpleNamespace(
        metadata={},
        species={"A": SimpleNamespace(initial_conc=1.0)},
        species_names=lambda: ["A"],
    )

    with pytest.raises(SimulationPreparationError, match="Unsupported prepared payload version"):
        prepare_simulation_worker_run(
            mechanism_text="reaction: A -> A; k=1.0",
            initials={"A": 1.0},
            t_span=(0.0, 1.0),
            solver_config={"solver": "BDF", "grid": {"N": 5}},
            prepared_payload={
                "version": 99,
                "mechanism": mechanism,
                "rhs": lambda _t, y: y,
                "y0": np.asarray([1.0], dtype=float),
                "species_names": ["A"],
            },
        )


def test_prepare_bound_mechanism_stops_before_binding_on_namespace_prepass_failure(monkeypatch) -> None:
    from kindred.core.simulation_preparation import (
        prepare_bound_mechanism,
    )
    from kindred.core.exceptions import FitSimulationError

    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.mechanism_parameter_namespace",
        lambda _mechanism: (_ for _ in ()).throw(ValueError("namespace sentinel")),
    )
    monkeypatch.setattr(
        "kindred.core.simulation_preparation._bind_parameters_to_mechanism",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("bind reached")),
    )

    with pytest.raises(FitSimulationError, match="namespace sentinel") as exc:
        prepare_bound_mechanism(
            "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0\n",
            ["k1"],
            initials={"A": 1.0, "B": 0.0},
        )

    assert "parameter_algebra" in str(exc.value)


def test_prepare_simulation_worker_run_accepts_structured_execution_request_without_text(monkeypatch) -> None:
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        prepare_simulation_worker_run,
    )

    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "kindred.core.ode_builder.build_ode_rhs_from_mechanism",
        lambda _mechanism: (lambda _t, y: y),
    )

    mechanism = SimpleNamespace(
        metadata={},
        species={"A": SimpleNamespace(initial_conc=1.0)},
        species_names=lambda: ["A"],
    )

    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload={
                "version": 2,
                "mechanism": mechanism,
                "species_names": ["A"],
                "y0": np.asarray([1.0], dtype=float),
                "mechanism_text": "",
                "temperature_schedule": None,
                "jacobian_func": None,
            },
            initials={"A": 3.0},
            t_span=(0.0, 1.0),
            solver_config={"solver": "BDF", "grid": {"N": 5}},
            mechanism_text="",
            simulation_identity={"schema_id": "schema", "param_fingerprint": "fingerprint"},
        )
    )

    assert prepared.request.y0.tolist() == [3.0]
    assert prepared.species_names == ["A"]


def test_prepare_simulation_worker_run_structured_prepared_request_ignores_stale_mechanism_text(
    monkeypatch,
) -> None:
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        prepare_simulation_worker_run,
    )

    seen: dict[str, object] = {}

    def _record_spec(spec, *, mechanism, require_mutable):
        seen["spec"] = spec
        seen["require_mutable"] = require_mutable
        seen["mechanism"] = mechanism
        return {}

    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("stale text path should not run")),
    )
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.parameter_algebra_spec_from_mechanism",
        lambda _mechanism: "structured-spec",
    )
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_spec_to_mechanism",
        _record_spec,
    )
    monkeypatch.setattr(
        "kindred.core.ode_builder.build_ode_rhs_from_mechanism",
        lambda _mechanism: (lambda _t, y: y),
    )

    mechanism = SimpleNamespace(
        metadata={},
        species={"A": SimpleNamespace(initial_conc=1.0)},
        species_names=lambda: ["A"],
    )

    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload={
                "version": 2,
                "mechanism": mechanism,
                "species_names": ["A"],
                "y0": np.asarray([1.0], dtype=float),
                "mechanism_text": "",
                "temperature_schedule": None,
                "jacobian_func": None,
            },
            initials={"A": 3.0},
            t_span=(0.0, 1.0),
            solver_config={"solver": "BDF", "grid": {"N": 5}},
            mechanism_text="reaction: A -> B; k=PRIMARY",
            simulation_identity={"schema_id": "schema", "param_fingerprint": "fingerprint"},
        )
    )

    assert prepared.request.y0.tolist() == [3.0]
    assert seen["spec"] == "structured-spec"
    assert seen["require_mutable"] is True
    assert seen["mechanism"] is mechanism


def test_prepare_simulation_worker_run_keeps_energy_prepared_request_authoritative(
    monkeypatch,
) -> None:
    from kindred.core.kinetics import K_from_deltaG_eq
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        prepare_bound_mechanism,
        prepare_simulation_worker_run,
    )

    bound = prepare_bound_mechanism(
        "\n".join(
            [
                "energy=kJ/mol",
                "T=298.15",
                "equilibrium: A <-> B; kf=10.0; kr=2.0; dG_eq=4.0",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        ["dG_eq_fast__feq__A__B"],
        temperature_K=298.15,
        initials={"A": 1.0, "B": 0.0},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )
    bound.bindings["dG_eq_fast__feq__A__B"].set(-5.0)

    monkeypatch.setattr(
        "kindred.core.simulator.dsl.parse_dsl_to_mechanism",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("stale text path should not run")),
    )

    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=bound.as_serializable_execution_payload(),
            initials={"A": 3.0},
            t_span=(0.0, 1.0),
            solver_config={"solver": "BDF", "grid": {"N": 5}},
            mechanism_text="reaction: A -> B; k=PRIMARY",
            simulation_identity={"schema_id": "schema", "param_fingerprint": "fingerprint"},
        )
    )

    equilibrium = prepared.mechanism.equilibria[0]
    expected_K = K_from_deltaG_eq(-5000.0, 298.15)

    assert prepared.request.y0.tolist() == [3.0, 0.0]
    assert float(equilibrium.kf) == pytest.approx(10.0)
    assert float(equilibrium.kr()) == pytest.approx(10.0 / expected_K)


def test_fitting_preparation_uses_same_failure_stage_as_simulation_preparation() -> None:
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.fitting_objective import build_fitting_objective
    from kindred.core.simulation_preparation import (
        SimulationPreparationError,
        prepare_simulation_worker_run,
    )

    with pytest.raises(SimulationPreparationError) as sim_excinfo:
        prepare_simulation_worker_run(
            mechanism_text="reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0",
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 1.0),
            solver_config={"solver": "BDF", "rtol": 0.0, "grid": {"N": 5}},
        )

    objective = build_fitting_objective(
        mechanism_text="reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0",
        param_names=["k1"],
        t_exp=np.asarray([0.0, 1.0], dtype=float),
        y_exp=np.asarray([0.0, 1.0], dtype=float),
        target_species="B",
        solver="BDF",
        rtol=0.0,
    )

    with pytest.raises(FitSimulationError) as fit_excinfo:
        objective(np.asarray([0.2], dtype=float))

    failure = fit_excinfo.value.details["failure"]
    assert failure["kind"] == "preparation_error"
    assert failure["details"]["stage"] == sim_excinfo.value.stage


def test_simulation_failure_user_message_formats_prepared_payload_stage() -> None:
    from kindred.core.simulation_failure import (
        build_simulation_failure,
        simulation_failure_user_message,
    )

    payload = build_simulation_failure(
        "preparation_error",
        "worker payload is invalid",
        details={"stage": "prepared_payload"},
    )

    assert simulation_failure_user_message(payload) == (
        "Prepared simulation payload invalid:\n\nworker payload is invalid"
    )


def test_simulation_failure_user_message_ignores_stack_trace_in_main_text() -> None:
    from kindred.core.simulation_failure import (
        build_simulation_failure,
        simulation_failure_user_message,
    )

    payload = build_simulation_failure(
        "preparation_error",
        "worker payload is invalid",
        details={"stage": "prepared_payload"},
        context={"stack_trace": "Traceback line 1\nTraceback line 2"},
    )

    assert simulation_failure_user_message(payload) == (
        "Prepared simulation payload invalid:\n\nworker payload is invalid"
    )


def test_heaviside_rejects_non_numeric_argument_with_explicit_error() -> None:
    from kindred.core.algebra.symbols import heaviside

    with pytest.raises(TypeError, match="heaviside expects a numeric argument"):
        heaviside("bad-input")
