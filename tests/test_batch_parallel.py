from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import copy
import gc
from multiprocessing.reduction import ForkingPickler

import numpy as np
import pytest

from kindred.core import batch_parallel
from kindred.core.mechanism import Mechanism


pytestmark = pytest.mark.unit


def test_mechanism_clone_copies_mutable_containers_and_shares_rate_objects():
    rate_obj = object()

    mech = Mechanism()
    mech.add_species("A", 0.0)
    mech.add_species("B", 0.0)
    mech.add_reaction({"A": -1.0, "B": 1.0}, rate=rate_obj, overrides={"kappa": 1.0})
    mech.add_equilibrium(
        {"A": -1.0},
        {"A": 1.0},
        Keq=1.0,
        metadata={"tag": "eq1"},
    )

    cloned = mech.clone()

    assert cloned is not mech
    assert cloned.species is not mech.species
    assert cloned.reactions is not mech.reactions
    assert cloned.equilibria is not mech.equilibria
    assert cloned.metadata is not mech.metadata
    assert cloned.metadata["declaration_order"] is not mech.metadata["declaration_order"]

    assert cloned.reactions[0] is not mech.reactions[0]
    assert cloned.reactions[0].rate is rate_obj
    assert cloned.reactions[0].stoich is not mech.reactions[0].stoich
    assert cloned.reactions[0].overrides is not mech.reactions[0].overrides
    assert cloned.equilibria[0].metadata is not mech.equilibria[0].metadata

    cloned.set_initial("A", 2.0)
    cloned.metadata["touched"] = True
    cloned.reactions[0].overrides["kappa"] = 2.0

    assert mech.species["A"].initial_conc == 0.0
    assert mech.metadata.get("touched") is None
    assert mech.reactions[0].overrides["kappa"] == 1.0


def test_ensure_worker_cache_bound_does_not_call_gc_collect_on_eviction(monkeypatch):
    original_cache = OrderedDict(batch_parallel._WORKER_PREPARED_CACHE)
    try:
        calls = {"n": 0}

        def fake_collect() -> int:
            calls["n"] += 1
            return 0

        monkeypatch.setattr(gc, "collect", fake_collect)
        monkeypatch.setattr(batch_parallel, "_WORKER_CACHE_MAXSIZE", 1)

        batch_parallel._WORKER_PREPARED_CACHE.clear()
        for i in range(5):
            batch_parallel._WORKER_PREPARED_CACHE[str(i)] = {"blob": bytearray(1024)}

        batch_parallel._ensure_worker_cache_bound()

        assert len(batch_parallel._WORKER_PREPARED_CACHE) == 1
        assert calls["n"] == 0
    finally:
        batch_parallel._WORKER_PREPARED_CACHE.clear()
        batch_parallel._WORKER_PREPARED_CACHE.update(original_cache)


def test_run_batch_simulation_task_sanitizes_unpicklable_solve_ode_exceptions(monkeypatch):
    class _UnpicklableError(Exception):
        pass

    class _FakeSpecies:
        def __init__(self) -> None:
            self.initial_conc = 0.0

    class _FakeMechanism:
        def __init__(self) -> None:
            self.species = {"A": _FakeSpecies()}
            self.metadata = {}

        def clone(self):
            cloned = copy.copy(self)
            cloned.species = {k: copy.copy(v) for k, v in self.species.items()}
            cloned.metadata = dict(self.metadata)
            return cloned

        def set_initial(self, name: str, initial_conc: float) -> None:
            self.species[str(name)].initial_conc = float(initial_conc)

    class _FakeBound:
        def __init__(self) -> None:
            self.species_names = ["A"]
            self.y0 = [1.0]
            self.mechanism = _FakeMechanism()

        def rhs(self, *_args, **_kwargs):
            raise AssertionError("rhs should not be called when solve_ode raises")

    monkeypatch.setattr(batch_parallel, "_prepared_entry", lambda **_kwargs: {"bound": _FakeBound()})
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_a, **_k: None,
    )

    def _boom(_req):
        raise _UnpicklableError(lambda x: x)

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _boom)

    payload = batch_parallel.run_batch_simulation_task(
        {
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "LSODA"},
            "t_end": 1.0,
            "set_id": "id1",
            "set_name": "set1",
        }
    )

    assert payload["success"] is False
    ForkingPickler.dumps(payload["error"])


def test_run_batch_simulation_task_returns_structured_error_payload_on_solver_failure(monkeypatch):
    class _FakeSpecies:
        def __init__(self) -> None:
            self.initial_conc = 0.0

    class _FakeMechanism:
        def __init__(self) -> None:
            self.species = {"A": _FakeSpecies()}
            self.metadata = {}

        def clone(self):
            cloned = copy.copy(self)
            cloned.species = {k: copy.copy(v) for k, v in self.species.items()}
            cloned.metadata = dict(self.metadata)
            return cloned

        def set_initial(self, name: str, initial_conc: float) -> None:
            self.species[str(name)].initial_conc = float(initial_conc)

    class _FakeBound:
        def __init__(self) -> None:
            self.species_names = ["A"]
            self.y0 = [1.0]
            self.mechanism = _FakeMechanism()

        def rhs(self, *_args, **_kwargs):
            raise AssertionError("rhs should not be called when solve_ode raises")

    monkeypatch.setattr(batch_parallel, "_prepared_entry", lambda **_kwargs: {"bound": _FakeBound()})
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", lambda _req: (_ for _ in ()).throw(RuntimeError("solver blew up")))

    payload = batch_parallel.run_batch_simulation_task(
        {
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "LSODA"},
            "t_end": 1.0,
            "set_id": "id1",
            "set_name": "set1",
        }
    )

    assert payload["success"] is False
    assert isinstance(payload.get("error"), dict)
    assert payload["error"]["kind"] == "simulation_error"
    assert payload["error"]["message"] == "solver blew up"


def test_run_batch_simulation_task_uses_shared_preparation_failure_payload_for_invalid_solver_config():
    payload = batch_parallel.run_batch_simulation_task(
        {
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "LSODA", "rtol": "bad"},
            "t_end": 1.0,
            "set_id": "id1",
            "set_name": "set1",
        }
    )

    assert payload["success"] is False
    assert payload["error"]["kind"] == "preparation_error"
    assert payload["error"]["details"]["stage"] == "solver_config"


def test_run_batch_simulation_task_reports_algebra_errors_with_shared_schema(monkeypatch):
    import numpy as np

    class _FakeSpecies:
        def __init__(self) -> None:
            self.initial_conc = 0.0

    class _FakeMechanism:
        def __init__(self) -> None:
            self.species = {"A": _FakeSpecies()}
            self.metadata = {"algebra_text": "let bad = nope + 1"}

        def clone(self):
            cloned = copy.copy(self)
            cloned.species = {k: copy.copy(v) for k, v in self.species.items()}
            cloned.metadata = dict(self.metadata)
            return cloned

        def set_initial(self, name: str, initial_conc: float) -> None:
            self.species[str(name)].initial_conc = float(initial_conc)

    class _FakeBound:
        def __init__(self) -> None:
            self.species_names = ["A"]
            self.y0 = np.asarray([1.0], dtype=float)
            self.mechanism = _FakeMechanism()

        def rhs(self, *_args, **_kwargs):
            return None

    class _FakeResult:
        def __init__(self) -> None:
            self.t = np.asarray([0.0, 1.0], dtype=float)
            self.Y = np.asarray([[1.0, 0.5]], dtype=float)
            self.provenance = {}
            self.fallback_occurred = False
            self.fallback_message = None

    monkeypatch.setattr(batch_parallel, "_prepared_entry", lambda **_kwargs: {"bound": _FakeBound()})
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", lambda _req: _FakeResult())

    def _boom(*_args, **_kwargs):
        raise RuntimeError("algebra exploded")

    monkeypatch.setattr("kindred.core.algebra.simulation_series.evaluate_algebra_series_for_simulation", _boom)

    payload = batch_parallel.run_batch_simulation_task(
        {
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "LSODA"},
            "t_end": 1.0,
            "set_id": "id1",
            "set_name": "set1",
        }
    )

    errors = payload.get("algebra_errors")
    assert isinstance(errors, list)
    assert errors
    assert errors[0]["kind"] == "algebra_error"
    assert errors[0]["message"] == "algebra exploded"


def test_run_batch_simulation_task_emits_worker_style_success_payload_fields(monkeypatch):
    import numpy as np

    class _FakeSpecies:
        def __init__(self) -> None:
            self.initial_conc = 0.0

    class _FakeMechanism:
        def __init__(self) -> None:
            self.species = {"A": _FakeSpecies()}
            self.metadata = {}

        def clone(self):
            cloned = copy.copy(self)
            cloned.species = {k: copy.copy(v) for k, v in self.species.items()}
            cloned.metadata = dict(self.metadata)
            return cloned

        def set_initial(self, name: str, initial_conc: float) -> None:
            self.species[str(name)].initial_conc = float(initial_conc)

    class _FakeBound:
        def __init__(self) -> None:
            self.species_names = ["A"]
            self.y0 = np.asarray([1.0], dtype=float)
            self.mechanism = _FakeMechanism()

        def rhs(self, *_args, **_kwargs):
            return None

    class _FakeResult:
        def __init__(self) -> None:
            self.t = np.asarray([0.0, 1.0], dtype=float)
            self.Y = np.asarray([[1.0, 0.5]], dtype=float)
            self.nfev = 17
            self.provenance = {"path": "batch"}
            self.fallback_occurred = False
            self.fallback_message = None

    monkeypatch.setattr(batch_parallel, "_prepared_entry", lambda **_kwargs: {"bound": _FakeBound()})
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "kindred.core.algebra.simulation_series.evaluate_algebra_series_for_simulation",
        lambda *_a, **_k: ({}, {}),
    )
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", lambda _req: _FakeResult())

    payload = batch_parallel.run_batch_simulation_task(
        {
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "LSODA"},
            "t_end": 1.0,
            "set_id": "id1",
            "set_name": "set1",
        }
    )

    assert payload["success"] is True
    assert payload["warnings"] == []
    assert payload["message"] == "Simulation completed successfully"
    assert payload["solver"] == "LSODA"
    assert payload["nfev"] == 17
    assert payload["provenance"] == {"path": "batch"}


def test_run_batch_simulation_task_secondary_payload_reports_base_species_count(monkeypatch):
    import numpy as np

    class _FakeSpecies:
        def __init__(self) -> None:
            self.initial_conc = 0.0

    class _FakeMechanism:
        def __init__(self) -> None:
            self.species = {"A": _FakeSpecies()}
            self.metadata = {}

        def clone(self):
            cloned = copy.copy(self)
            cloned.species = {k: copy.copy(v) for k, v in self.species.items()}
            cloned.metadata = dict(self.metadata)
            return cloned

        def set_initial(self, name: str, initial_conc: float) -> None:
            self.species[str(name)].initial_conc = float(initial_conc)

    class _FakeBound:
        def __init__(self) -> None:
            self.species_names = ["A"]
            self.y0 = np.asarray([1.0], dtype=float)
            self.mechanism = _FakeMechanism()

        def rhs(self, *_args, **_kwargs):
            return None

    class _FakeResult:
        def __init__(self) -> None:
            self.t = np.asarray([0.0, 1.0], dtype=float)
            self.Y = np.asarray([[1.0, 0.5]], dtype=float)
            self.provenance = {}
            self.fallback_occurred = False
            self.fallback_message = None

    monkeypatch.setattr(batch_parallel, "_prepared_entry", lambda **_kwargs: {"bound": _FakeBound()})
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "kindred.core.algebra.simulation_series.evaluate_algebra_series_for_simulation",
        lambda *_a, **_k: ({"Alg": np.asarray([0.2, 0.3], dtype=float)}, {}),
    )
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", lambda _req: _FakeResult())

    payload = batch_parallel.run_batch_simulation_task(
        {
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "LSODA"},
            "t_end": 1.0,
            "set_id": "id1",
            "set_name": "set1",
        }
    )

    assert payload["success"] is True
    assert "mechanism" not in payload
    assert payload["base_species_count"] == 1


def test_run_batch_simulation_task_preserves_structured_execution_request_provenance_text(monkeypatch):
    import numpy as np

    class _FakeSpecies:
        def __init__(self) -> None:
            self.initial_conc = 0.0

    class _FakeMechanism:
        def __init__(self) -> None:
            self.species = {"A": _FakeSpecies()}
            self.metadata = {}

        def clone(self):
            cloned = copy.copy(self)
            cloned.species = {k: copy.copy(v) for k, v in self.species.items()}
            cloned.metadata = dict(self.metadata)
            return cloned

    class _FakeResult:
        def __init__(self) -> None:
            self.t = np.asarray([0.0, 1.0], dtype=float)
            self.Y = np.asarray([[3.0, 2.0]], dtype=float)
            self.provenance = {}
            self.fallback_occurred = False
            self.fallback_message = None

    seen: dict[str, object] = {}

    def _fake_build_rhs(mechanism):
        seen["mechanism"] = mechanism
        return lambda _t, y: y

    def _fake_solve(request):
        seen["request"] = request
        return _FakeResult()

    monkeypatch.setattr(
        "kindred.core.ode_builder.build_ode_rhs_from_mechanism",
        _fake_build_rhs,
    )
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("stale text path should not run")),
    )
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve)

    payload = batch_parallel.run_batch_simulation_task(
        {
            "run_id": 1,
            "set_id": "id1",
            "set_name": "set1",
            "include_mechanism_in_result_payload": False,
            "mechanism_text": "reaction: A -> B; k=PRIMARY",
            "execution_request": {
                "prepared_payload": {
                    "version": 2,
                    "mechanism": _FakeMechanism(),
                    "species_names": ["A"],
                    "y0": np.asarray([1.0], dtype=float),
                    "mechanism_text": "",
                    "temperature_schedule": None,
                    "jacobian_func": None,
                },
                "initials": {"A": 3.0},
                "t_span": (0.0, 1.0),
                "solver_config": {"solver": "LSODA", "grid": {"N": 2}},
                "mechanism_text": "reaction: A -> B; k=SET1",
                "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
            },
        }
    )

    assert payload["success"] is True
    assert "mechanism" not in payload
    assert payload["mechanism_text"] == "reaction: A -> B; k=SET1"
    assert seen["mechanism"] is not None
    assert np.asarray(seen["request"].y0).tolist() == [3.0]  # type: ignore[union-attr]


def test_run_batch_simulation_task_can_include_mechanism_for_primary_completion_payloads(monkeypatch):
    import numpy as np

    class _FakeSpecies:
        def __init__(self) -> None:
            self.initial_conc = 0.0

    class _FakeMechanism:
        def __init__(self) -> None:
            self.species = {"A": _FakeSpecies()}
            self.metadata = {}

        def clone(self):
            cloned = copy.copy(self)
            cloned.species = {k: copy.copy(v) for k, v in self.species.items()}
            cloned.metadata = dict(self.metadata)
            return cloned

    class _FakeResult:
        def __init__(self) -> None:
            self.t = np.asarray([0.0, 1.0], dtype=float)
            self.Y = np.asarray([[3.0, 2.0]], dtype=float)
            self.provenance = {}
            self.fallback_occurred = False
            self.fallback_message = None

    seen: dict[str, object] = {}

    def _fake_build_rhs(mechanism):
        seen["mechanism"] = mechanism
        return lambda _t, y: y

    def _fake_solve(request):
        seen["request"] = request
        return _FakeResult()

    monkeypatch.setattr(
        "kindred.core.ode_builder.build_ode_rhs_from_mechanism",
        _fake_build_rhs,
    )
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("stale text path should not run")),
    )
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve)

    payload = batch_parallel.run_batch_simulation_task(
        {
            "run_id": 1,
            "set_id": "id1",
            "set_name": "set1",
            "include_mechanism_in_result_payload": True,
            "mechanism_text": "reaction: A -> B; k=PRIMARY",
            "execution_request": {
                "prepared_payload": {
                    "version": 2,
                    "mechanism": _FakeMechanism(),
                    "species_names": ["A"],
                    "y0": np.asarray([1.0], dtype=float),
                    "mechanism_text": "",
                    "temperature_schedule": None,
                    "jacobian_func": None,
                },
                "initials": {"A": 3.0},
                "t_span": (0.0, 1.0),
                "solver_config": {"solver": "LSODA", "grid": {"N": 2}},
                "mechanism_text": "reaction: A -> B; k=SET1",
                "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
            },
        }
    )

    assert payload["success"] is True
    assert payload["mechanism"] is seen["mechanism"]
    assert payload["mechanism_text"] == "reaction: A -> B; k=SET1"
    assert seen["mechanism"] is not None
    assert np.asarray(seen["request"].y0).tolist() == [3.0]  # type: ignore[union-attr]


def test_run_batch_simulation_task_logs_warning_on_invalid_t_span_and_falls_back(monkeypatch, caplog):
    import numpy as np

    class _FakeSpecies:
        def __init__(self) -> None:
            self.initial_conc = 0.0

    class _FakeMechanism:
        def __init__(self) -> None:
            self.species = {"A": _FakeSpecies()}
            self.metadata = {}

        def clone(self):
            cloned = copy.copy(self)
            cloned.species = {k: copy.copy(v) for k, v in self.species.items()}
            cloned.metadata = dict(self.metadata)
            return cloned

        def set_initial(self, name: str, initial_conc: float) -> None:
            self.species[str(name)].initial_conc = float(initial_conc)

    class _FakeBound:
        def __init__(self) -> None:
            self.species_names = ["A"]
            self.y0 = [1.0]
            self.mechanism = _FakeMechanism()

        def rhs(self, *_args, **_kwargs):
            return None

    class _FakeResult:
        def __init__(self) -> None:
            self.t = np.asarray([0.0, 1.0], dtype=float)
            self.Y = np.asarray([[1.0, 1.0]], dtype=float)
            self.provenance = {}
            self.fallback_occurred = False
            self.fallback_message = None

    monkeypatch.setattr(batch_parallel, "_prepared_entry", lambda **_kwargs: {"bound": _FakeBound()})
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "kindred.core.algebra.simulation_series.evaluate_algebra_series_for_simulation",
        lambda *_a, **_k: ({}, {}),
    )

    def _solve(req):
        assert req.t_span == (0.0, 1.0)
        return _FakeResult()

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _solve)

    caplog.set_level("WARNING")
    payload = batch_parallel.run_batch_simulation_task(
        {
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "LSODA"},
            "t_span": "not-a-seq",
            "t_end": 1.0,
            "set_id": "id1",
            "set_name": "set1",
        }
    )
    assert payload["t"][0] == 0.0
    assert any("Invalid t_span" in rec.getMessage() for rec in caplog.records)


def test_run_batch_simulation_task_normalizes_solver_config_defaults(monkeypatch):
    import numpy as np

    class _FakeSpecies:
        def __init__(self) -> None:
            self.initial_conc = 0.0

    class _FakeMechanism:
        def __init__(self) -> None:
            self.species = {"A": _FakeSpecies()}
            self.metadata = {}

        def clone(self):
            cloned = copy.copy(self)
            cloned.species = {k: copy.copy(v) for k, v in self.species.items()}
            cloned.metadata = dict(self.metadata)
            return cloned

        def set_initial(self, name: str, initial_conc: float) -> None:
            self.species[str(name)].initial_conc = float(initial_conc)

    class _FakeBound:
        def __init__(self) -> None:
            self.species_names = ["A"]
            self.y0 = np.asarray([1.0], dtype=float)
            self.mechanism = _FakeMechanism()

        def rhs(self, *_args, **_kwargs):
            return None

    class _FakeResult:
        def __init__(self) -> None:
            self.t = np.asarray([0.0, 1.0], dtype=float)
            self.Y = np.asarray([[1.0, 1.0]], dtype=float)
            self.provenance = {}
            self.fallback_occurred = False
            self.fallback_message = None

    monkeypatch.setattr(batch_parallel, "_prepared_entry", lambda **_kwargs: {"bound": _FakeBound()})
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "kindred.core.algebra.simulation_series.evaluate_algebra_series_for_simulation",
        lambda *_a, **_k: ({}, {}),
    )
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", lambda _req: _FakeResult())

    payload = batch_parallel.run_batch_simulation_task(
        {
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "LSODA"},
            "t_end": 1.0,
            "set_id": "id1",
            "set_name": "set1",
        }
    )

    solver_config = payload["solver_config"]
    assert solver_config["solver"] == "LSODA"
    assert solver_config["rtol"] == 1e-6
    assert solver_config["atol"] == 1e-12
    assert solver_config["grid"] == {"N": 100}
    assert solver_config["temperature_K"] == 298.15
    assert solver_config["wegscheider_cyclicity_enabled"] is False
    assert solver_config["use_sparse_jacobian"] is False


def test_run_batch_simulation_task_does_not_mutate_prepared_mechanism_and_avoids_frozen_instance_error(monkeypatch):
    import numpy as np

    @dataclass(frozen=True)
    class _FrozenSpecies:
        initial_conc: float = 0.0

    class _FakeMechanism:
        def __init__(self) -> None:
            self.species = {"A": _FrozenSpecies(0.0)}
            self.metadata = {}

        def clone(self):
            cloned = copy.copy(self)
            cloned.species = dict(self.species)
            cloned.metadata = dict(self.metadata)
            return cloned

        def set_initial(self, name: str, initial_conc: float) -> None:
            key = str(name)
            self.species[key] = replace(self.species[key], initial_conc=float(initial_conc))

    class _FakeBound:
        def __init__(self, mech: _FakeMechanism) -> None:
            self.species_names = ["A"]
            self.y0 = np.asarray([0.0], dtype=float)
            self.mechanism = mech

        def rhs(self, *_args, **_kwargs):
            return None

    class _FakeResult:
        def __init__(self) -> None:
            self.t = np.asarray([0.0, 1.0], dtype=float)
            self.Y = np.asarray([[1.0, 1.0]], dtype=float)
            self.provenance = {}
            self.fallback_occurred = False
            self.fallback_message = None

    prototype = _FakeMechanism()
    bound = _FakeBound(prototype)
    monkeypatch.setattr(batch_parallel, "_prepared_entry", lambda **_kwargs: {"bound": bound})

    seen_apply: list[object] = []
    seen_eval: list[object] = []

    def _apply(_text, *, mechanism, require_mutable: bool):
        assert require_mutable is True
        seen_apply.append(mechanism)
        if hasattr(mechanism, "metadata") and isinstance(mechanism.metadata, dict):
            mechanism.metadata["touched"] = True
        return None

    def _eval(mechanism, *, t, species_series, initials):
        seen_eval.append(mechanism)
        return ({}, {})

    monkeypatch.setattr("kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism", _apply)
    monkeypatch.setattr("kindred.core.algebra.simulation_series.evaluate_algebra_series_for_simulation", _eval)
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", lambda _req: _FakeResult())

    batch_parallel.run_batch_simulation_task(
        {
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "LSODA", "use_sparse_jacobian": False, "wegscheider_cyclicity_enabled": True},
            "t_end": 1.0,
            "initials": {"A": 1.0},
            "set_id": "id1",
            "set_name": "set1",
        }
    )
    batch_parallel.run_batch_simulation_task(
        {
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "LSODA", "use_sparse_jacobian": False, "wegscheider_cyclicity_enabled": False},
            "t_end": 1.0,
            "initials": {"A": 2.0},
            "set_id": "id1",
            "set_name": "set1",
        }
    )

    assert len(seen_apply) == 2
    assert len(seen_eval) == 2
    assert seen_apply[0] is not prototype
    assert seen_apply[1] is not prototype
    assert seen_apply[0] is not seen_apply[1]
    assert seen_eval[0] is seen_apply[0]
    assert seen_eval[1] is seen_apply[1]
    assert prototype.metadata == {}
    assert prototype.species["A"].initial_conc == 0.0


def test_batch_mechanism_signature_stable_across_param_fingerprint_for_prepared_runtime_reuse():
    """Prepared-runtime reuse key must NOT vary with param_fingerprint.

    The prepared-runtime cache stores a compiled mechanism that is cloned and
    bound per-set.  Only structural factors (schema, temperature, sparse
    jacobian, wegscheider) should affect the reuse key.  Param overrides are
    applied after cloning, so different param_fingerprints must share the same
    prepared-runtime signature.
    """
    shared_identity = {
        "version": 1,
        "schema_id": "schema-1",
        "solver": {
            "temperature_K": 298.15,
            "use_sparse_jacobian": False,
            "wegscheider_cyclicity_enabled": False,
        },
        "t_end": 10.0,
        "preview_batch_cache_token": "",
        "execution_flags": (),
    }

    sig_a = batch_parallel.batch_mechanism_signature(
        simulation_identity={**shared_identity, "param_fingerprint": "params-a"},
    )
    sig_b = batch_parallel.batch_mechanism_signature(
        simulation_identity={**shared_identity, "param_fingerprint": "params-b"},
    )

    # Same structural identity → same prepared-runtime reuse key
    assert sig_a == sig_b


def test_batch_mechanism_signature_varies_with_structural_identity():
    """Prepared-runtime key must vary when structural factors change."""
    base = {
        "version": 1,
        "schema_id": "schema-1",
        "param_fingerprint": "params-a",
        "solver": {
            "temperature_K": 298.15,
            "use_sparse_jacobian": False,
            "wegscheider_cyclicity_enabled": False,
        },
        "t_end": 10.0,
        "preview_batch_cache_token": "",
        "execution_flags": (),
    }
    sig_base = batch_parallel.batch_mechanism_signature(simulation_identity=base)

    # Different schema_id
    different_schema = dict(base, schema_id="schema-2")
    sig_schema = batch_parallel.batch_mechanism_signature(simulation_identity=different_schema)
    assert sig_base != sig_schema

    # Different temperature
    different_temp = dict(base)
    different_temp["solver"] = dict(base["solver"], temperature_K=310.0)
    sig_temp = batch_parallel.batch_mechanism_signature(simulation_identity=different_temp)
    assert sig_base != sig_temp


def test_run_batch_simulation_task_non_structured_text_path_invalidates_worker_cache_by_mechanism_text(
    monkeypatch,
):
    original_cache = OrderedDict(batch_parallel._WORKER_PREPARED_CACHE)

    class _FakeResult:
        def __init__(self, y0: np.ndarray) -> None:
            self.t = np.asarray([0.0, 1.0], dtype=float)
            self.Y = np.column_stack([y0, y0])
            self.provenance = {}
            self.fallback_occurred = False
            self.fallback_message = None

    seen_rates: list[float] = []

    def _capture_request(req):
        y0 = np.asarray(req.y0, dtype=float).reshape(-1)
        dydt = np.asarray(req.rhs(0.0, y0), dtype=float).reshape(-1)
        seen_rates.append(float(-dydt[0]))
        return _FakeResult(y0)

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _capture_request)

    base_task = {
        "solver_config": {"solver": "LSODA", "temperature_K": 298.15},
        "t_span": (0.0, 1.0),
        "set_id": "id1",
        "set_name": "set1",
        # Deliberately stale structural signature: pre-fix this caused the
        # second task to reuse the first prepared runtime and ignore the new DSL.
        "mechanism_signature": "structural-signature",
    }

    try:
        batch_parallel._WORKER_PREPARED_CACHE.clear()

        first = batch_parallel.run_batch_simulation_task(
            {
                **base_task,
                "mechanism_text": "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0\n",
            }
        )
        second = batch_parallel.run_batch_simulation_task(
            {
                **base_task,
                "mechanism_text": "reaction: A -> B; k=10.0\ninitial: A=1.0\ninitial: B=0.0\n",
            }
        )

        assert first["success"] is True
        assert second["success"] is True
        assert seen_rates == pytest.approx([1.0, 10.0])
    finally:
        batch_parallel._WORKER_PREPARED_CACHE.clear()
        batch_parallel._WORKER_PREPARED_CACHE.update(original_cache)


def test_run_batch_simulation_task_honors_energy_prepared_payload_over_stale_text(monkeypatch):
    from kindred.core.kinetics import K_from_deltaG_eq
    from kindred.core.simulation_preparation import prepare_bound_mechanism

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

    class _FakeResult:
        def __init__(self) -> None:
            self.t = np.asarray([0.0, 1.0], dtype=float)
            self.Y = np.asarray([[1.0, 0.5], [0.0, 0.5]], dtype=float)
            self.provenance = {}
            self.fallback_occurred = False
            self.fallback_message = None

    monkeypatch.setattr(
        "kindred.core.algebra.simulation_series.evaluate_algebra_series_for_simulation",
        lambda *_args, **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        "kindred.core.simulator.solvers.solve_ode",
        lambda _request: _FakeResult(),
    )

    payload = batch_parallel.run_batch_simulation_task(
        {
            "run_id": 1,
            "set_id": "id1",
            "set_name": "set1",
            "include_mechanism_in_result_payload": True,
            "mechanism_text": "reaction: A -> B; k=PRIMARY",
            "execution_request": {
                "prepared_payload": bound.as_serializable_execution_payload(),
                "initials": {"A": 3.0, "B": 0.0},
                "t_span": (0.0, 1.0),
                "solver_config": {"solver": "LSODA", "grid": {"N": 2}},
                "mechanism_text": "reaction: A -> B; k=PRIMARY",
                "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
            },
        }
    )

    equilibrium = payload["mechanism"].equilibria[0]
    expected_K = K_from_deltaG_eq(-5000.0, 298.15)

    assert payload["success"] is True
    assert float(equilibrium.kf) == pytest.approx(10.0)
    assert float(equilibrium.kr()) == pytest.approx(10.0 / expected_K)
