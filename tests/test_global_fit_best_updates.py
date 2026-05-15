import numpy as np
import pytest

from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult
from kindred.core.fitting_completion import GlobalFitCompletion
from kindred.gui.fitting.worker import GlobalFitWorker

pytestmark = pytest.mark.gui



def test_global_fit_worker_emits_best_updated_only_on_improvement(qt_app):
    emitted = []

    def fake_fit_global(*args, **kwargs):
        fit_evaluator = args[0]
        assert callable(fit_evaluator)
        assert hasattr(fit_evaluator, "evaluate_series")
        progress = kwargs.get("progress_callback")
        assert progress is not None
        progress(1, 10.0, {"k": 1.0})
        progress(2, 12.0, {"k": 1.1})
        progress(3, 9.0, {"k": 0.9})
        progress(4, 9.0, {"k": 0.9})
        progress(5, 8.0, {"k": 0.8})
        return GlobalFitResult(
            shared_params={"k": 0.8},
            dataset_params={"ds": {}},
            uncertainties=None,
            global_chi_squared=1.0,
            global_r_squared=0.0,
            dataset_info=[
                DatasetFitInfo(
                    dataset_id="ds",
                    r_squared=0.0,
                    chi_squared=1.0,
                    rmse=1.0,
                    mae=1.0,
                    residuals=np.array([0.0]),
                    n_points=1,
                    weight=1.0,
                )
            ],
            nfev=5,
            message="ok",
            completion=GlobalFitCompletion(
                status="ok",
                optimizer_converged=True,
                nonfinite_metrics=False,
            ),
            covariance=None,
            objective_residuals=np.array([0.0]),
            model_series={"ds": {}},
            residual_series={"ds": {}},
        )

    datasets = [{"id": "ds", "t": np.array([0.0]), "y": np.array([0.0]), "species": "A"}]

    def simulation(_params):
        return {"t": np.array([0.0]), "species": {"A": np.array([0.0])}}

    worker = GlobalFitWorker(
        datasets,
        {"k1": 1.0},
        fit_evaluator=simulation,
        fit_func=fake_fit_global,
        best_update_interval_s=0.0,
        plot_update_interval_s=0.0,
    )
    worker.bestUpdated.connect(lambda payload: emitted.append(payload))

    worker._execute()

    assert [p["cost"] for p in emitted] == [10.0, 9.0, 8.0]


def test_global_fit_worker_best_payload_stats_remain_raw_under_target_weighting(qt_app):
    from kindred.core.fitting_evaluation import CallableFittingEvaluator

    t_obs = np.linspace(0.0, 1.0, 5, dtype=float)

    def simulation(_params):
        return {
            "t": t_obs.copy(),
            "species": {
                "A": np.zeros_like(t_obs),
                "B": np.zeros_like(t_obs),
            },
        }

    worker = GlobalFitWorker(
        [
            {
                "id": "ds1",
                "t": t_obs.copy(),
                "y": np.vstack([np.ones_like(t_obs), 2.0 * np.ones_like(t_obs)]),
                "species": ["A", "B"],
                "target_weights": {"A": 5.0, "B": 1.0},
            }
        ],
        {"k": 0.2},
        fit_evaluator=CallableFittingEvaluator(simulation),
        best_update_interval_s=0.0,
    )

    model_series, residual_series, plot_model_series, plot_model_x, dataset_stats = worker._build_best_payload_series(
        shared_params={"k": 0.2},
        dataset_params={},
    )

    np.testing.assert_allclose(model_series["ds1"]["A"], np.zeros_like(t_obs))
    np.testing.assert_allclose(plot_model_series["ds1"]["A"], np.zeros_like(t_obs))
    np.testing.assert_allclose(plot_model_x["ds1"], t_obs)
    np.testing.assert_allclose(
        residual_series["ds1"]["A"],
        -np.ones_like(t_obs),
    )
    np.testing.assert_allclose(
        residual_series["ds1"]["B"],
        np.full_like(t_obs, -2.0),
    )
    assert dataset_stats["ds1"]["chi_squared"] == pytest.approx(2.5)


def test_global_fit_worker_best_payload_accepts_raw_callable_evaluator(qt_app):
    t_obs = np.linspace(0.0, 1.0, 5, dtype=float)
    emitted = []

    def simulation(params):
        return {
            "t": t_obs.copy(),
            "species": {
                "A": np.exp(-float(params["k"]) * t_obs),
            },
        }

    worker = GlobalFitWorker(
        [
            {
                "id": "ds1",
                "t": t_obs.copy(),
                "y": np.exp(-0.2 * t_obs),
                "species": "A",
            }
        ],
        {"k": 0.2},
        fit_evaluator=simulation,
        best_update_interval_s=0.0,
        plot_update_interval_s=0.0,
    )
    worker.bestUpdated.connect(lambda payload: emitted.append(payload))

    worker._maybe_emit_best(1, 0.1, {"k": 0.2})

    assert len(emitted) == 1
    payload = emitted[0]
    assert payload["model_series"]["ds1"]["A"].shape == t_obs.shape
    assert payload["plot_model_series"]["ds1"]["A"].shape == t_obs.shape
    np.testing.assert_allclose(payload["plot_model_x"]["ds1"], t_obs)
    assert payload["dataset_stats"]["ds1"]["chi_squared"] == pytest.approx(0.0)


def test_global_fit_worker_best_payload_accepts_evaluate_series_only_evaluator(qt_app):
    t_obs = np.linspace(0.0, 1.0, 5, dtype=float)
    emitted = []

    class _EvaluateOnly:
        def evaluate_series(self, params):
            return {
                "t": t_obs.copy(),
                "species": {
                    "A": np.exp(-float(params["k"]) * t_obs),
                },
            }

    worker = GlobalFitWorker(
        [
            {
                "id": "ds1",
                "t": t_obs.copy(),
                "y": np.exp(-0.2 * t_obs),
                "species": "A",
            }
        ],
        {"k": 0.2},
        fit_evaluator=_EvaluateOnly(),
        best_update_interval_s=0.0,
        plot_update_interval_s=0.0,
    )
    worker.bestUpdated.connect(lambda payload: emitted.append(payload))

    worker._maybe_emit_best(1, 0.1, {"k": 0.2})

    assert len(emitted) == 1
    payload = emitted[0]
    assert payload["model_series"]["ds1"]["A"].shape == t_obs.shape
    assert payload["plot_model_series"]["ds1"]["A"].shape == t_obs.shape
    np.testing.assert_allclose(payload["plot_model_x"]["ds1"], t_obs)
    assert payload["dataset_stats"]["ds1"]["chi_squared"] == pytest.approx(0.0)


def test_global_fit_worker_passes_runtime_session_to_fit_boundary(qt_app):
    seen: dict[str, object] = {}
    events: list[str] = []

    class _RuntimeSession:
        @property
        def ledger(self):
            return None

        def warm(self, *, cancellation_check=None, lane_count=None):
            raise AssertionError("runtime readiness must be established before worker execution")

    runtime_session = _RuntimeSession()

    def fake_fit_global(*_args, **kwargs):
        events.append("fit")
        seen["runtime_session"] = kwargs.get("runtime_session")
        seen["max_runtime_lanes"] = kwargs.get("max_runtime_lanes")
        return GlobalFitResult(
            shared_params={"k": 1.0},
            dataset_params={"ds": {}},
            uncertainties=None,
            global_chi_squared=0.0,
            global_r_squared=1.0,
            dataset_info=[],
            nfev=1,
            message="ok",
            completion=GlobalFitCompletion(
                status="ok",
                optimizer_converged=True,
                nonfinite_metrics=False,
            ),
        )

    t_obs = np.asarray([0.0, 1.0], dtype=float)
    worker = GlobalFitWorker(
        [{"id": "ds", "t": t_obs, "y": np.zeros_like(t_obs), "species": "A"}],
        {"k": 1.0},
        fit_evaluator=lambda _params: {"t": t_obs, "species": {"A": np.zeros_like(t_obs)}},
        fit_runtime_session=runtime_session,
        fit_runtime_max_lanes=3,
        fit_func=fake_fit_global,
    )

    worker._execute()

    assert events == ["fit"]
    assert seen == {"runtime_session": runtime_session, "max_runtime_lanes": 3}


def test_global_fit_worker_rejects_exact_serial_evaluator_without_runtime_session(qt_app):
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    context = prepare_fitting_execution_context(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        param_names=["k1"],
        t_end=1.0,
        num_points=2,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="initial:",
    )
    evaluator = SerialFittingEvaluator(context)
    called: list[str] = []

    def fake_fit_global(*_args, **_kwargs):
        called.append("fit")
        raise AssertionError("GUI worker must not fall through to core runtime-session creation")

    worker = GlobalFitWorker(
        [{"id": "ds", "t": np.asarray([0.0]), "y": np.asarray([0.0]), "species": "A"}],
        {"k1": 1.0},
        fit_evaluator=evaluator,
        fit_runtime_session=None,
        fit_func=fake_fit_global,
    )

    with pytest.raises(RuntimeError, match="required fitting runtime session is not ready"):
        worker._execute()

    assert called == []


def test_global_fit_worker_rejects_exact_serial_evaluator_with_unready_runtime_session(qt_app):
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    class _UnreadySession:
        def is_ready(self, *, lane_count=None) -> bool:
            return False

    context = prepare_fitting_execution_context(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        param_names=["k1"],
        t_end=1.0,
        num_points=2,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="initial:",
    )
    evaluator = SerialFittingEvaluator(context)
    called: list[str] = []

    def fake_fit_global(*_args, **_kwargs):
        called.append("fit")
        raise AssertionError("GUI worker must not warm an unready runtime session during fit execution")

    worker = GlobalFitWorker(
        [{"id": "ds", "t": np.asarray([0.0]), "y": np.asarray([0.0]), "species": "A"}],
        {"k1": 1.0},
        fit_evaluator=evaluator,
        fit_runtime_session=_UnreadySession(),
        fit_func=fake_fit_global,
    )

    with pytest.raises(RuntimeError, match="required fitting runtime session is not ready"):
        worker._execute()

    assert called == []


def test_global_fit_worker_accepts_ready_runtime_session_with_default_lane_count(qt_app):
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    class _ReadySession:
        def __init__(self) -> None:
            self.ready_lane_counts: list[object] = []

        def is_ready(self, *, lane_count=None) -> bool:
            self.ready_lane_counts.append(lane_count)
            return lane_count is None

    context = prepare_fitting_execution_context(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        param_names=["k1"],
        t_end=1.0,
        num_points=2,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="initial:",
    )
    evaluator = SerialFittingEvaluator(context)
    runtime_session = _ReadySession()
    captured: dict[str, object] = {}

    def fake_fit_global(*_args, **kwargs):
        captured["runtime_session"] = kwargs.get("runtime_session")
        captured["max_runtime_lanes"] = kwargs.get("max_runtime_lanes")
        return GlobalFitResult(
            shared_params={"k1": 1.0},
            dataset_params={"ds": {}},
            uncertainties=None,
            global_chi_squared=0.0,
            global_r_squared=1.0,
            dataset_info=[],
            nfev=1,
            message="ok",
            completion=GlobalFitCompletion(
                status="ok",
                optimizer_converged=True,
                nonfinite_metrics=False,
            ),
            covariance=None,
            objective_residuals=np.zeros(1, dtype=float),
            model_series={"ds": {}},
            residual_series={"ds": {}},
        )

    worker = GlobalFitWorker(
        [{"id": "ds", "t": np.asarray([0.0]), "y": np.asarray([0.0]), "species": "A"}],
        {"k": 1.0},
        fit_evaluator=evaluator,
        fit_runtime_session=runtime_session,
        fit_func=fake_fit_global,
    )

    payload = worker._execute()

    assert payload is not None
    assert runtime_session.ready_lane_counts == [None]
    assert captured == {"runtime_session": runtime_session, "max_runtime_lanes": None}


def test_global_fit_worker_cancel_notifies_runtime_session(qt_app):
    events: list[str] = []

    class _RuntimeSession:
        @property
        def ledger(self):
            return None

        def cancel_run(self):
            events.append("runtime:cancel")

    worker = GlobalFitWorker(
        [{"id": "ds", "t": np.asarray([0.0]), "y": np.asarray([0.0]), "species": "A"}],
        {"k": 1.0},
        fit_evaluator=lambda _params: {"t": np.asarray([0.0]), "species": {"A": np.asarray([0.0])}},
        fit_runtime_session=_RuntimeSession(),
        fit_func=lambda *_args, **_kwargs: None,
    )

    worker.cancel()

    assert worker._cancelled is True
    assert events == ["runtime:cancel"]


def test_global_fit_worker_best_payload_uses_runtime_session_evaluator_when_available(qt_app):
    calls: list[dict[str, float]] = []

    class _RuntimeEvaluator:
        def evaluate_series(self, params):
            calls.append(dict(params))
            t = np.asarray([0.0, 1.0], dtype=float)
            return {"t": t, "species": {"A": np.asarray([2.0, 3.0], dtype=float)}}

    class _RuntimeSession:
        @property
        def ledger(self):
            return None

        def evaluator(self, *, cancellation_check=None):
            assert cancellation_check is not None
            return _RuntimeEvaluator()

    t_obs = np.asarray([0.0, 1.0], dtype=float)

    def direct_evaluator(_params):
        raise AssertionError("best payload must use the runtime-session evaluator")

    worker = GlobalFitWorker(
        [{"id": "ds", "t": t_obs, "y": np.asarray([2.0, 3.0], dtype=float), "species": "A"}],
        {"k": 1.0},
        fit_evaluator=direct_evaluator,
        fit_runtime_session=_RuntimeSession(),
        fit_func=lambda *_args, **_kwargs: None,
    )

    model_series, residual_series, _plot_model_series, _plot_model_x, dataset_stats = worker._build_best_payload_series(
        shared_params={"k": 1.0},
        dataset_params={},
    )

    assert calls == [{"k": 1.0}]
    np.testing.assert_allclose(model_series["ds"]["A"], np.asarray([2.0, 3.0], dtype=float))
    np.testing.assert_allclose(residual_series["ds"]["A"], np.asarray([0.0, 0.0], dtype=float))
    assert dataset_stats["ds"]["chi_squared"] == pytest.approx(0.0)


def test_parametric_x_time_guided_best_payload_uses_penalized_alignment_like_final_replay(qt_app):
    from kindred.core.fitting_evaluation import CallableFittingEvaluator

    t_obs = np.linspace(0.0, 1.0, 41, dtype=float)
    x_obs = t_obs * (1.0 - t_obs)
    y_obs = t_obs.copy()
    t_sim = np.linspace(0.0, 1.0, 401, dtype=float)

    def simulation(params):
        a = float(params.get("a", 0.5))
        return {
            "t": t_sim.copy(),
            "species": {
                "X": a * t_sim * (1.0 - t_sim),
                "Y": t_sim.copy(),
            },
        }

    worker = GlobalFitWorker(
        [
            {
                "id": "ds1",
                "t": t_obs.copy(),
                "y": y_obs.copy(),
                "species": "Y",
                "x_name": "X",
                "x_obs": x_obs.copy(),
                "x_mapping_mode": "time_guided",
            }
        ],
        {"a": 0.5},
        fit_evaluator=CallableFittingEvaluator(simulation),
        best_update_interval_s=0.0,
        plot_update_interval_s=0.0,
    )

    model_series, residual_series, plot_model_series, plot_model_x, dataset_stats = worker._build_best_payload_series(
        shared_params={"a": 0.5},
        dataset_params={},
    )

    assert set(model_series) == {"ds1"}
    assert set(residual_series) == {"ds1"}
    assert set(plot_model_series) == {"ds1"}
    assert set(plot_model_x) == {"ds1"}
    assert set(dataset_stats) == {"ds1"}
    assert model_series["ds1"]["Y"].shape == t_obs.shape
    assert residual_series["ds1"]["Y"].shape == t_obs.shape


def test_parametric_x_auto_best_payload_falls_back_for_out_of_range_x_like_final_replay(qt_app):
    from kindred.core.fitting_evaluation import CallableFittingEvaluator

    t_obs = np.asarray([0.0, 0.5, 1.0], dtype=float)
    x_obs = np.asarray([-0.25, 0.5, 1.25], dtype=float)
    y_obs = np.asarray([0.0, 0.5, 1.0], dtype=float)
    t_sim = np.linspace(0.0, 1.0, 101, dtype=float)

    def simulation(_params):
        return {
            "t": t_sim.copy(),
            "species": {
                "X": t_sim.copy(),
                "Y": t_sim.copy(),
            },
        }

    worker = GlobalFitWorker(
        [
            {
                "id": "ds1",
                "t": t_obs.copy(),
                "y": y_obs.copy(),
                "species": "Y",
                "x_name": "X",
                "x_obs": x_obs.copy(),
                "x_mapping_mode": "auto",
            }
        ],
        {},
        fit_evaluator=CallableFittingEvaluator(simulation),
        best_update_interval_s=0.0,
        plot_update_interval_s=0.0,
    )

    model_series, residual_series, plot_model_series, plot_model_x, dataset_stats = worker._build_best_payload_series(
        shared_params={},
        dataset_params={},
    )

    assert set(model_series) == {"ds1"}
    assert set(residual_series) == {"ds1"}
    assert set(plot_model_series) == {"ds1"}
    assert set(plot_model_x) == {"ds1"}
    assert set(dataset_stats) == {"ds1"}
    assert model_series["ds1"]["Y"].shape == t_obs.shape


def test_parametric_x_best_payload_and_final_replay_use_ready_runtime_evaluator(qt_app):
    from kindred.core.analysis.global_fitting import fit_global

    t_obs = np.linspace(0.0, 1.0, 21, dtype=float)
    x_obs = t_obs * (1.0 - t_obs)
    y_obs = t_obs.copy()
    t_sim = np.linspace(0.0, 1.0, 201, dtype=float)
    calls: list[dict[str, float]] = []

    class _RuntimeEvaluator:
        def evaluate_series(self, params):
            calls.append(dict(params))
            a = float(params.get("a", 1.0))
            return {
                "t": t_sim.copy(),
                "species": {
                    "X": t_sim * (1.0 - t_sim),
                    "Y": a * t_sim,
                },
            }

    class _RuntimeSession:
        @property
        def ledger(self):
            return None

        def evaluator(self, *, cancellation_check=None):
            assert cancellation_check is not None
            return _RuntimeEvaluator()

    def direct_evaluator(_params):
        raise AssertionError("Parametric-X fitting must use the ready runtime evaluator")

    dataset = {
        "id": "ds1",
        "t": t_obs.copy(),
        "y": y_obs.copy(),
        "species": "Y",
        "x_name": "X",
        "x_obs": x_obs.copy(),
        "x_mapping_mode": "time_guided",
    }
    runtime_session = _RuntimeSession()
    worker = GlobalFitWorker(
        [dict(dataset)],
        {"a": 1.0},
        fit_evaluator=direct_evaluator,
        fit_runtime_session=runtime_session,
        best_update_interval_s=0.0,
        plot_update_interval_s=0.0,
    )

    model_series, residual_series, plot_model_series, plot_model_x, dataset_stats = worker._build_best_payload_series(
        shared_params={"a": 1.0},
        dataset_params={},
    )

    result = fit_global(
        direct_evaluator,
        [dict(dataset)],
        {"a": 1.0},
        bounds={"a": (0.95, 1.05)},
        method="trf",
        max_nfev=1,
        runtime_session=runtime_session,
        max_runtime_lanes=1,
        cancellation_check=lambda: False,
    )

    assert set(model_series) == {"ds1"}
    assert set(residual_series) == {"ds1"}
    assert set(plot_model_series) == {"ds1"}
    assert set(plot_model_x) == {"ds1"}
    assert set(dataset_stats) == {"ds1"}
    assert set(result.model_series) == {"ds1"}
    assert set(result.plot_model_x) == {"ds1"}
    assert sum(1 for call in calls if "a" in call) >= 2
