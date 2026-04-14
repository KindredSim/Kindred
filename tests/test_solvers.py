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


def test_solve_ode_scipy_bdf_omits_jac_sparsity_for_banded_and_clamps_selected_indices(monkeypatch):
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
    assert "jac" in seen
    assert "jac_sparsity" not in seen
    jac_dense = seen["jac"](0.0, np.array([1.0]))
    assert jac_dense.shape == (1, 1)
    assert float(out.Y[0, 0]) == 0.0


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
    method, note = solvers._scipy_method_for("ROS3")
    assert method == "BDF"
    assert note == "Unknown solver name; using BDF"

    method2, note2 = solvers._scipy_method_for("ROS4")
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


def test_make_scipy_jac_converts_real_banded_storage_to_dense():
    cfg = JacobianConfig(mode="banded", ml=1, mu=1)
    jac = solvers._make_scipy_jac(
        lambda _t, y: np.array([-2.0 * y[0] + y[1], y[0] - 3.0 * y[1]]),
        cfg,
    )
    Jd = jac(0.0, np.array([1.0, 2.0]))
    assert Jd.shape == (2, 2)
    assert np.allclose(Jd, np.array([[-2.0, 1.0], [1.0, -3.0]]))


def test_solve_ode_temperature_schedule_is_used_for_jacobian_rhs(monkeypatch):
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
        kwargs["jac"](t_span[0], np.asarray(y0, float))
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
