import types

import numpy as np
import pytest

from kindred.core.exceptions import InitialConditionError, TimeGridError
from kindred.core.simulator import solvers
from kindred.core.simulator.jacobian import JacobianConfig


pytestmark = pytest.mark.unit


def test_solve_ode_validates_t_eval_and_y0():
    req_base = dict(rhs=lambda t, y: -y, t_span=(0.0, 1.0), y0=np.array([1.0]))

    with pytest.raises(InitialConditionError, match="y0 must be a 1D finite array"):
        solvers.solve_ode(solvers.SimulationRequest(**{**req_base, "y0": np.array([[1.0]])}))

    with pytest.raises(InitialConditionError, match="y0 must be a 1D finite array"):
        solvers.solve_ode(solvers.SimulationRequest(**{**req_base, "y0": np.array([np.inf])}))

    with pytest.raises(TimeGridError, match="t_eval must contain at least one time point"):
        solvers.solve_ode(solvers.SimulationRequest(**{**req_base, "t_eval": np.array([])}))

    with pytest.raises(TimeGridError, match="t_eval must contain only finite values"):
        solvers.solve_ode(solvers.SimulationRequest(**{**req_base, "t_eval": np.array([0.0, np.nan])}))

    with pytest.raises(TimeGridError, match="t_eval must be strictly increasing"):
        solvers.solve_ode(solvers.SimulationRequest(**{**req_base, "t_eval": np.array([0.0, 0.0, 1.0])}))

    with pytest.raises(TimeGridError, match="t_eval must lie within t_span"):
        solvers.solve_ode(solvers.SimulationRequest(**{**req_base, "t_eval": np.array([-1.0, 0.0, 1.0])}))


def test_solve_ode_mapping_adapter_rejects_unknown_keys():
    req = dict(
        rhs=lambda t, y: -y,
        t_span=(0.0, 1.0),
        y0=np.array([1.0]),
        t_eval=np.array([0.0, 1.0]),
        unknown_field="boom",
    )
    with pytest.raises(TypeError, match="Unknown SimulationRequest field"):
        solvers.solve_ode(req)


def test_solve_ode_mapping_adapter_allow_unknown_keys_executes(monkeypatch):
    def fake_solve_ivp(*, fun, t_span, y0, **kwargs):
        t_eval = np.asarray(kwargs["t_eval"], float)
        y = np.vstack([np.ones_like(t_eval)])
        return types.SimpleNamespace(success=True, message="ok", t=t_eval, y=y, t_events=[])

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    req = dict(
        rhs=lambda t, y: -y,
        t_span=(0.0, 1.0),
        y0=np.array([1.0]),
        t_eval=np.array([0.0, 1.0]),
        unknown_field="boom",
    )
    out = solvers.solve_ode(req, allow_unknown_keys=True)
    assert out.t.shape == (2,)


def test_solve_ode_scipy_path_injects_temperature_calls_progress_sets_note_and_clamps(monkeypatch):
    calls = {"progress": 0, "rhs_T": []}

    class TempSchedule:
        def __call__(self, t: float) -> float:
            return 300.0 + float(t)

        def to_dict(self):
            raise RuntimeError("nope")

        def __str__(self) -> str:
            return "TempSchedule(str)"

    def rhs(t: float, y: np.ndarray, *, T: float):
        calls["rhs_T"].append(float(T))
        return -y

    def progress_cb(_t, _t0, _t1):
        calls["progress"] += 1

    def fake_solve_ivp(*, fun, t_span, y0, **kwargs):
        span = float(t_span[1]) - float(t_span[0])
        for bucket in range(11):
            fun(float(t_span[0]) + span * (bucket / 10.0), np.asarray(y0, float))

        t_eval = np.asarray(kwargs["t_eval"], float)
        y = np.vstack([-np.ones_like(t_eval)])
        return types.SimpleNamespace(success=True, message="ok", t=t_eval, y=y, t_events=[np.array([0.5])])

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    req = solvers.SimulationRequest(
        rhs=rhs,  # type: ignore[arg-type]
        t_span=(0.0, 1.0),
        y0=np.array([1.0]),
        solver="unknown_solver_name",
        t_eval=np.array([0.0, 0.5, 1.0]),
        positivity="clamp",
        temperature_schedule=TempSchedule(),
        progress_callback=progress_cb,
    )

    out = solvers.solve_ode(req)

    assert calls["progress"] == 11
    assert all(t >= 300.0 for t in calls["rhs_T"])
    assert out.provenance["emulation_note"] == "Unknown solver name; using BDF"
    assert out.provenance["temperature_schedule"] == "TempSchedule(str)"
    assert out.provenance["events"] == [[0.5]]
    assert float(out.Y[0, 0]) == 0.0


def test_solve_ode_scipy_bdf_uses_solver_default_when_no_symbolic_jacobian(monkeypatch):
    seen = {}

    def fake_solve_ivp(*, fun, t_span, y0, **kwargs):
        seen.update(kwargs)
        t_eval = np.asarray(kwargs["t_eval"], float)
        y = np.vstack([np.array([-1.0, -2.0, -3.0])[: t_eval.size]])
        return types.SimpleNamespace(success=True, message="ok", t=t_eval, y=y, t_events=[])

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    cfg = JacobianConfig(mode="banded", ml=1, mu=1)
    req = solvers.SimulationRequest(
        rhs=lambda t, y: -y,
        t_span=(0.0, 1.0),
        y0=np.array([1.0]),
        solver="BDF",
        t_eval=np.array([0.0, 0.5, 1.0]),
        rosenbrock_jacobian=cfg,
        positivity="clamp",
        pos_indices=[0],
    )

    out = solvers.solve_ode(req)

    assert seen["method"] == "BDF"
    assert "jac" not in seen
    assert "jac_sparsity" not in seen
    assert float(out.Y[0, 0]) == 0.0


def test_solve_ode_scipy_bdf_accepts_sparsity_hint_without_supplied_jacobian(monkeypatch):
    seen = {}
    sparsity = np.asarray([[True, False], [True, True]], dtype=bool)

    def fake_solve_ivp(*, fun, t_span, y0, **kwargs):
        seen.update(kwargs)
        t_eval = np.asarray(kwargs["t_eval"], float)
        y = np.vstack([np.ones_like(t_eval), np.zeros_like(t_eval)])
        return types.SimpleNamespace(success=True, message="ok", t=t_eval, y=y, t_events=[])

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    solvers.solve_ode(
        solvers.SimulationRequest(
            rhs=lambda _t, y: np.asarray([-y[0], y[0] - y[1]], dtype=float),
            t_span=(0.0, 1.0),
            y0=np.array([1.0, 0.0]),
            solver="BDF",
            t_eval=np.array([0.0, 0.5, 1.0]),
            jac_sparsity=sparsity,
        )
    )

    assert "jac" not in seen
    assert seen["jac_sparsity"] is sparsity


def test_solve_ode_builds_time_grid_and_passes_solver_kwargs(monkeypatch):
    seen = {}

    def fake_solve_ivp(*, fun, t_span, y0, **kwargs):
        seen.update(kwargs)
        fun(t_span[0], np.asarray(y0, float))
        t_eval = np.asarray(kwargs["t_eval"], float)
        y = np.vstack([np.ones_like(t_eval)])
        return types.SimpleNamespace(success=True, message="ok", t=t_eval, y=y, t_events=[])

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    out = solvers.solve_ode(
        solvers.SimulationRequest(
            rhs=lambda t, y: -y,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            solver="Radau",
            grid={"N": 3},
            max_step=0.1,
            first_step=0.01,
            events=[lambda t, y: y[0] - 0.5],
            progress_callback=lambda *_a: None,
        )
    )

    assert np.all(np.isfinite(out.Y))
    assert seen["max_step"] == 0.1
    assert seen["first_step"] == 0.01
    assert "events" in seen
    assert "mass" not in seen
    assert np.asarray(seen["t_eval"]).shape == (3,)


def test_solve_ode_progress_callback_cancellation_propagates(monkeypatch):
    def progress_cb(*_a):
        raise RuntimeError("cancel")

    def fake_solve_ivp(*, fun, t_span, y0, **kwargs):
        for i in range(10):
            fun(t_span[0] + 1e-6 * i, np.asarray(y0, float))
        return types.SimpleNamespace(success=True, message="ok", t=np.array([0.0]), y=np.array([[1.0]]), t_events=[])

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    with pytest.raises(RuntimeError, match="cancel"):
        solvers.solve_ode(
            solvers.SimulationRequest(
                rhs=lambda t, y: -y,
                t_span=(0.0, 1.0),
                y0=np.array([1.0]),
                solver="Radau",
                grid={"N": 2},
                progress_callback=progress_cb,
            )
        )


def test_scipy_method_for_unknown_names_fall_back_to_bdf_with_note():
    method, note = solvers._scipy_method_for("UNKNOWN_SOLVER_A")
    assert method == "BDF"
    assert note == "Unknown solver name; using BDF"

    method2, note2 = solvers._scipy_method_for("UNKNOWN_SOLVER_B")
    assert method2 == "BDF"
    assert note2 == "Unknown solver name; using BDF"


@pytest.mark.parametrize(
    ("solver_name", "expected"),
    [
        ("", ("BDF", "Unknown solver name; using BDF")),
        (None, ("BDF", "Unknown solver name; using BDF")),
        ("   ", ("BDF", "Unknown solver name; using BDF")),
        ("legacy_solver", ("BDF", "Unknown solver name; using BDF")),
        ("LEGACY_SOLVER", ("BDF", "Unknown solver name; using BDF")),
        ("Legacy_Solver", ("BDF", "Unknown solver name; using BDF")),
        (123, ("BDF", "Unknown solver name; using BDF")),
        ("x" * 10000, ("BDF", "Unknown solver name; using BDF")),
    ],
)
def test_normalize_solver_name_handles_unknown_inputs(solver_name, expected):
    assert solvers.normalize_solver_name(solver_name) == expected


def test_solve_ode_temperature_schedule_is_used_for_solver_rhs(monkeypatch):
    seen_T = []

    class TempSchedule:
        def __call__(self, t: float) -> float:
            return 123.0 + float(t)

        def __str__(self) -> str:
            return "TempSchedule"

    def rhs(_t: float, y: np.ndarray, *, T: float):
        seen_T.append(float(T))
        return -y

    def fake_solve_ivp(*, fun, t_span, y0, **kwargs):
        assert "jac" not in kwargs
        assert "jac_sparsity" not in kwargs
        fun(t_span[0], np.asarray(y0, float))
        t_eval = np.asarray(kwargs["t_eval"], float)
        y = np.vstack([np.ones_like(t_eval)])
        return types.SimpleNamespace(success=True, message="ok", t=t_eval, y=y, t_events=[])

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    solvers.solve_ode(
        solvers.SimulationRequest(
            rhs=rhs,  # type: ignore[arg-type]
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            solver="BDF",
            t_eval=np.array([0.0, 1.0]),
            temperature_schedule=TempSchedule(),
        )
    )

    assert seen_T


def test_solve_ode_intervention_interval_symbolic_jacobian_uses_scheduled_rhs(monkeypatch):
    symbolic_jacobian_calls = []

    def forbidden_symbolic_jacobian(_t: float, _y: np.ndarray):
        symbolic_jacobian_calls.append(True)
        raise AssertionError("base symbolic jacobian does not include intervention interval semantics")

    setattr(
        forbidden_symbolic_jacobian,
        "_kindred_symbolic_jacobian_identity",
        {"kind": "jacobian", "artifact_fingerprint": "test"},
    )

    def fake_solve_ivp(*, fun, t_span, y0, **kwargs):
        assert "jac" not in kwargs
        fun(t_span[0], np.asarray(y0, float))
        t_eval = np.asarray(kwargs["t_eval"], float)
        y = np.vstack([np.ones_like(t_eval)])
        return types.SimpleNamespace(success=True, message="ok", t=t_eval, y=y, t_events=[])

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    out = solvers.solve_ode(
        solvers.SimulationRequest(
            rhs=lambda _t, y: np.asarray([0.0 * float(y[0])]),
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            solver="BDF",
            grid={"N": 2},
            jacobian_func=forbidden_symbolic_jacobian,
            species_names=("A",),
            intervention_schedule={
                "intervals": [{"kind": "source", "species": "A", "start": 0.0, "end": 1.0, "rate": 2.0}]
            },
        )
    )

    assert symbolic_jacobian_calls == []
    assert out.provenance["has_intervention_schedule"] is True
    assert out.provenance["intervention_symbolic_jacobian_disabled"] is True


def test_solve_ode_empty_intervention_schedule_object_is_noop_without_species_names():
    from kindred.core.intervention_schedule import InterventionSchedule

    request = solvers.SimulationRequest(
        rhs=lambda _t, y: np.asarray([0.0 * float(y[0])]),
        t_span=(0.0, 1.0),
        y0=np.array([1.0]),
        solver="BDF",
        grid={"N": 2},
        intervention_schedule=InterventionSchedule(),
    )

    out = solvers.solve_ode(request)

    assert out.provenance["has_intervention_schedule"] is False
    np.testing.assert_allclose(out.Y[0], np.array([1.0, 1.0]))


def test_solve_ode_scheduled_terminal_event_stops_later_segments_and_preserves_events():
    def terminal_event(t: float, _y: np.ndarray) -> float:
        return float(t) - 0.5

    terminal_event.terminal = True  # type: ignore[attr-defined]
    terminal_event.direction = 0  # type: ignore[attr-defined]

    request = solvers.SimulationRequest(
        rhs=lambda _t, y: np.asarray([0.0 * float(y[0])]),
        t_span=(0.0, 2.0),
        y0=np.array([1.0]),
        solver="BDF",
        t_eval=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        events=[terminal_event],
        species_names=("A",),
        intervention_schedule={
            "instant_events": [
                {"op": "set", "species": "A", "time": 1.0, "value": 3.0}
            ]
        },
    )

    out = solvers.solve_ode(request)

    np.testing.assert_allclose(out.t, np.array([0.0, 0.5]))
    assert out.provenance["events"] == [[0.5]]
    assert out.provenance["intervention_segments"] == 1


def test_solve_ode_scheduled_event_terminal_flags_stop_inside_segment():
    def terminal_event(t: float, _y: np.ndarray) -> float:
        return float(t) - 0.5

    request = solvers.SimulationRequest(
        rhs=lambda _t, y: np.asarray([0.0 * float(y[0])]),
        t_span=(0.0, 1.0),
        y0=np.array([1.0]),
        solver="BDF",
        t_eval=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        events=[terminal_event],
        event_terminal=[True],
        species_names=("A",),
        intervention_schedule={
            "instant_events": [
                {"op": "set", "species": "A", "time": 0.8, "value": 3.0}
            ]
        },
    )

    out = solvers.solve_ode(request)

    np.testing.assert_allclose(out.t, np.array([0.0, 0.25, 0.5]))
    assert out.provenance["events"] == [[0.5]]
    assert out.provenance["intervention_segments"] == 1


def test_solve_ode_scheduled_event_terminal_flags_accept_numpy_arrays():
    def terminal_event(t: float, _y: np.ndarray) -> float:
        return float(t) - 0.5

    request = solvers.SimulationRequest(
        rhs=lambda _t, y: np.asarray([0.0 * float(y[0])]),
        t_span=(0.0, 1.0),
        y0=np.array([1.0]),
        solver="BDF",
        t_eval=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        events=[terminal_event],
        event_terminal=np.array([True, False]),
        species_names=("A",),
        intervention_schedule={
            "instant_events": [
                {"op": "set", "species": "A", "time": 0.8, "value": 3.0}
            ]
        },
    )

    out = solvers.solve_ode(request)

    np.testing.assert_allclose(out.t, np.array([0.0, 0.25, 0.5]))
    assert out.provenance["events"] == [[0.5]]
    assert out.provenance["intervention_segments"] == 1


def test_solve_ode_scheduled_terminal_before_first_requested_sample_returns_empty_truncated_output():
    def terminal_event(t: float, _y: np.ndarray) -> float:
        return float(t) - 0.5

    request = solvers.SimulationRequest(
        rhs=lambda _t, y: np.asarray([0.0 * float(y[0])]),
        t_span=(0.0, 1.0),
        y0=np.array([1.0]),
        solver="BDF",
        t_eval=np.array([0.75, 1.0]),
        events=[terminal_event],
        event_terminal=[True],
        species_names=("A",),
        intervention_schedule={
            "instant_events": [
                {"op": "set", "species": "A", "time": 0.8, "value": 3.0}
            ]
        },
    )

    out = solvers.solve_ode(request)

    np.testing.assert_allclose(out.t, np.array([], dtype=float))
    assert out.Y.shape == (1, 0)
    assert out.provenance["events"] == [[0.5]]
    assert out.provenance["intervention_segments"] == 1


def test_solve_ode_state_trigger_resumes_from_event_state_between_requested_samples():
    request = solvers.SimulationRequest(
        rhs=lambda _t, _y: np.array([1.0]),
        t_span=(0.0, 1.0),
        y0=np.array([0.0]),
        solver="BDF",
        t_eval=np.array([0.0, 1.0]),
        species_names=("A",),
        intervention_schedule={
            "trigger_events": [
                {
                    "op": "trigger",
                    "trigger_species": "A",
                    "threshold": 0.5,
                    "direction": "rising",
                    "species": "A",
                    "action": "add",
                    "amount": 1.0,
                    "max_count": 1,
                    "min_interval": 0.0,
                }
            ]
        },
    )

    out = solvers.solve_ode(request)

    np.testing.assert_allclose(out.t, np.array([0.0, 1.0]))
    assert out.provenance["intervention_trigger_events"] == [
        {
            "time": pytest.approx(0.5, abs=1e-6),
            "trigger_species": "A",
            "species": "A",
            "action": "add",
        }
    ]
    assert float(out.Y[0, -1]) == pytest.approx(2.0, abs=1e-6)


def test_solve_ode_state_trigger_rearms_after_min_interval_inside_same_segment():
    request = solvers.SimulationRequest(
        rhs=lambda _t, _y: np.array([1.0]),
        t_span=(0.0, 2.0),
        y0=np.array([0.0]),
        solver="BDF",
        t_eval=np.array([0.0, 2.0]),
        species_names=("A",),
        intervention_schedule={
            "trigger_events": [
                {
                    "op": "trigger",
                    "trigger_species": "A",
                    "threshold": 0.5,
                    "direction": "rising",
                    "species": "A",
                    "action": "set",
                    "value": 0.0,
                    "max_count": 2,
                    "min_interval": 0.25,
                }
            ]
        },
    )

    out = solvers.solve_ode(request)

    assert out.provenance["intervention_trigger_events"] == [
        {
            "time": pytest.approx(0.5, abs=1e-6),
            "trigger_species": "A",
            "species": "A",
            "action": "set",
        },
        {
            "time": pytest.approx(1.0, abs=1e-6),
            "trigger_species": "A",
            "species": "A",
            "action": "set",
        },
    ]
    np.testing.assert_allclose(out.t, np.array([0.0, 2.0]))
    assert float(out.Y[0, -1]) == pytest.approx(1.0, abs=1e-6)


def test_solve_ode_state_trigger_rearm_slice_drops_out_of_bounds_first_step():
    request = solvers.SimulationRequest(
        rhs=lambda _t, _y: np.array([1.0]),
        t_span=(0.0, 2.0),
        y0=np.array([0.0]),
        solver="BDF",
        first_step=0.1,
        t_eval=np.array([0.0, 2.0]),
        species_names=("A",),
        intervention_schedule={
            "trigger_events": [
                {
                    "op": "trigger",
                    "trigger_species": "A",
                    "threshold": 0.5,
                    "direction": "rising",
                    "species": "A",
                    "action": "set",
                    "value": 0.0,
                    "max_count": 2,
                    "min_interval": 0.0,
                }
            ]
        },
    )

    out = solvers.solve_ode(request)

    assert out.provenance["intervention_trigger_events"] == [
        {
            "time": pytest.approx(0.5, abs=1e-6),
            "trigger_species": "A",
            "species": "A",
            "action": "set",
        },
        {
            "time": pytest.approx(1.0, abs=1e-6),
            "trigger_species": "A",
            "species": "A",
            "action": "set",
        },
    ]
    np.testing.assert_allclose(out.t, np.array([0.0, 2.0]))
    assert float(out.Y[0, -1]) == pytest.approx(1.0, abs=1e-6)


def test_solve_ode_state_trigger_at_requested_sample_keeps_t_eval_grid_stable():
    request = solvers.SimulationRequest(
        rhs=lambda _t, _y: np.array([1.0]),
        t_span=(0.0, 1.0),
        y0=np.array([0.0]),
        solver="BDF",
        t_eval=np.array([0.0, 0.5, 1.0]),
        species_names=("A",),
        intervention_schedule={
            "trigger_events": [
                {
                    "op": "trigger",
                    "trigger_species": "A",
                    "threshold": 0.5,
                    "direction": "rising",
                    "species": "A",
                    "action": "add",
                    "amount": 1.0,
                    "max_count": 1,
                    "min_interval": 0.0,
                }
            ]
        },
    )

    out = solvers.solve_ode(request)

    np.testing.assert_allclose(out.t, np.array([0.0, 0.5, 1.0]))
    assert float(out.Y[0, 1]) == pytest.approx(1.5, abs=1e-6)
    assert float(out.Y[0, -1]) == pytest.approx(2.0, abs=1e-6)


def test_solve_ode_state_trigger_ignores_surplus_user_event_terminal_flags():
    request = solvers.SimulationRequest(
        rhs=lambda _t, _y: np.array([1.0]),
        t_span=(0.0, 1.0),
        y0=np.array([0.0]),
        solver="BDF",
        t_eval=np.array([0.0, 0.75, 1.0]),
        event_terminal=[False],
        species_names=("A",),
        intervention_schedule={
            "trigger_events": [
                {
                    "op": "trigger",
                    "trigger_species": "A",
                    "threshold": 0.5,
                    "direction": "rising",
                    "species": "A",
                    "action": "add",
                    "amount": 1.0,
                    "max_count": 1,
                    "min_interval": 0.0,
                }
            ]
        },
    )

    out = solvers.solve_ode(request)

    np.testing.assert_allclose(out.t, np.array([0.0, 0.75, 1.0]))
    assert float(out.Y[0, 1]) == pytest.approx(1.75, abs=1e-6)
    assert float(out.Y[0, -1]) == pytest.approx(2.0, abs=1e-6)


def test_solve_ode_state_trigger_respects_active_clamp_interval_at_event_sample():
    request = solvers.SimulationRequest(
        rhs=lambda _t, _y: np.array([0.0, 1.0]),
        t_span=(0.0, 1.0),
        y0=np.array([0.0, 0.0]),
        solver="BDF",
        t_eval=np.array([0.0, 0.5, 1.0]),
        species_names=("A", "B"),
        intervention_schedule={
            "intervals": [
                {"kind": "clamp", "species": "A", "start": 0.0, "end": 1.0, "value": 0.0}
            ],
            "trigger_events": [
                {
                    "op": "trigger",
                    "trigger_species": "B",
                    "threshold": 0.5,
                    "direction": "rising",
                    "species": "A",
                    "action": "add",
                    "amount": 1.0,
                    "max_count": 1,
                    "min_interval": 0.0,
                }
            ],
        },
    )

    out = solvers.solve_ode(request)

    np.testing.assert_allclose(out.t, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(out.Y[0], np.array([0.0, 0.0, 0.0]), atol=1e-7)
    assert out.provenance["intervention_trigger_events"][0]["species"] == "A"


def test_solve_ode_state_trigger_at_clamp_interval_end_is_not_reclamped():
    request = solvers.SimulationRequest(
        rhs=lambda _t, _y: np.array([0.0, 1.0]),
        t_span=(0.0, 1.0),
        y0=np.array([0.0, 0.0]),
        solver="BDF",
        t_eval=np.array([0.0, 0.5, 1.0]),
        species_names=("A", "B"),
        intervention_schedule={
            "intervals": [
                {"kind": "clamp", "species": "A", "start": 0.0, "end": 0.5, "value": 0.0}
            ],
            "trigger_events": [
                {
                    "op": "trigger",
                    "trigger_species": "B",
                    "threshold": 0.5,
                    "direction": "rising",
                    "species": "A",
                    "action": "add",
                    "amount": 1.0,
                    "max_count": 1,
                    "min_interval": 0.0,
                }
            ],
        },
    )

    out = solvers.solve_ode(request)

    np.testing.assert_allclose(out.t, np.array([0.0, 0.5, 1.0]))
    assert float(out.Y[0, 1]) == pytest.approx(1.0, abs=1e-7)
    assert float(out.Y[0, -1]) == pytest.approx(1.0, abs=1e-7)


def test_solve_ode_state_trigger_requires_event_state_when_event_between_samples(monkeypatch):
    def fake_solve_ivp(*, t_span, y0, **kwargs):
        t_eval = np.asarray(kwargs["t_eval"], float)
        return types.SimpleNamespace(
            success=True,
            message="ok",
            status=1,
            t=t_eval[:1],
            y=np.asarray(y0, float).reshape(-1, 1),
            t_events=[np.array([0.5])],
        )

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    request = solvers.SimulationRequest(
        rhs=lambda _t, _y: np.array([1.0]),
        t_span=(0.0, 1.0),
        y0=np.array([0.0]),
        solver="BDF",
        t_eval=np.array([0.0, 1.0]),
        species_names=("A",),
        intervention_schedule={
            "trigger_events": [
                {
                    "op": "trigger",
                    "trigger_species": "A",
                    "threshold": 0.5,
                    "direction": "rising",
                    "species": "A",
                    "action": "add",
                    "amount": 1.0,
                    "max_count": 1,
                    "min_interval": 0.0,
                }
            ]
        },
    )

    with pytest.raises(Exception, match="did not provide an event-time state"):
        solvers.solve_ode(request)


def test_solve_ode_exercises_implicit_alternatives_and_raises_after_exhaustion(monkeypatch):
    class DummySolution:
        def __init__(self):
            self.success = False
            self.message = "fail"
            self.t = np.array([0.0, 0.1])

    calls = []

    def fake_solve_ivp(*_a, **kwargs):
        calls.append(kwargs.get("method"))
        return DummySolution()

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    with pytest.raises(Exception) as exc_info:
        solvers.solve_ode(
            solvers.SimulationRequest(
                rhs=lambda t, y: -y,
                t_span=(0.0, 1.0),
                y0=np.array([1.0]),
                solver="BDF",
                grid={"N": 2},
            )
        )

    assert calls == ["BDF", "Radau"]
    assert "attempted methods: BDF, Radau" in str(exc_info.value)


def test_solve_ode_records_successful_fallback_provenance(monkeypatch):
    class DummySolution:
        def __init__(self, *, success: bool, message: str, t_eval: np.ndarray):
            self.success = success
            self.message = message
            self.t = np.asarray(t_eval, float)
            self.y = np.vstack([np.ones_like(self.t)])
            self.t_events = []

    calls = []

    def fake_solve_ivp(*_a, **kwargs):
        method = kwargs.get("method")
        calls.append(method)
        t_eval = np.asarray(kwargs["t_eval"], float)
        if method == "BDF":
            return DummySolution(success=False, message="BDF failure", t_eval=t_eval[:1])
        return DummySolution(success=True, message="ok", t_eval=t_eval)

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    out = solvers.solve_ode(
        solvers.SimulationRequest(
            rhs=lambda t, y: -y,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            solver="BDF",
            grid={"N": 2},
        )
    )

    assert calls == ["BDF", "Radau"]
    assert out.fallback_occurred is True
    assert out.provenance["solver_alternative_used"] == "Radau"
    assert out.provenance["solver_requested"] == "BDF"
    assert out.provenance["solver_used"] == "Radau"
    assert "solver" not in out.provenance


def test_solve_ode_records_segmented_intervention_fallback_solver_used(monkeypatch):
    class DummySolution:
        def __init__(self, *, success: bool, message: str, t_eval: np.ndarray):
            self.success = success
            self.message = message
            self.t = np.asarray(t_eval, float)
            self.y = np.vstack([np.ones_like(self.t)])
            self.t_events = []

    calls = []

    def fake_solve_ivp(*_a, **kwargs):
        method = kwargs.get("method")
        calls.append(method)
        t_eval = np.asarray(kwargs["t_eval"], float)
        if method == "BDF":
            return DummySolution(success=False, message="BDF failure", t_eval=t_eval[:1])
        return DummySolution(success=True, message="ok", t_eval=t_eval)

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    out = solvers.solve_ode(
        solvers.SimulationRequest(
            rhs=lambda _t, y: -y,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            solver="BDF",
            grid={"N": 2},
            species_names=("A",),
            intervention_schedule={
                "intervals": [{"kind": "source", "species": "A", "start": 0.0, "end": 1.0, "rate": 0.0}]
            },
        )
    )

    assert calls == ["BDF", "Radau"]
    assert out.fallback_occurred is True
    assert out.provenance["solver_alternative_used"] == "Radau"
    assert out.provenance["solver_requested"] == "BDF"
    assert out.provenance["solver_used"] == "Radau"
    assert out.provenance["intervention_segment_solvers"] == ["Radau"]


def test_simulation_request_rejects_deprecated_native_solver_params():
    with pytest.raises(TypeError):
        solvers.SimulationRequest(
            rhs=lambda t, y: -y,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            adaptive=True,  # type: ignore[call-arg]
        )


def test_progress_rhs_reports_by_time_not_call_count():
    calls = {"n": 0}

    def cb(_t: float, _t0: float, _t1: float) -> None:
        calls["n"] += 1

    pr = solvers._ProgressRhs(
        lambda _t, y: y,
        callback=cb,
        t0=0.0,
        t1=1.0,
        every=5,
    )

    for _ in range(20):
        pr(0.0, np.array([1.0]))

    assert calls["n"] <= 1
