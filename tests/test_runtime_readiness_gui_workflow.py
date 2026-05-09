from __future__ import annotations

import threading

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets


pytestmark = [pytest.mark.gui, pytest.mark.slow, pytest.mark.real_runtime_readiness]


def _select_batch_rows(main_window, rows: list[int]) -> None:
    table = main_window._batch_table
    assert table is not None
    selection = table.selectionModel()
    assert selection is not None
    selection.clearSelection()
    table.setCurrentIndex(main_window._batch_model.index(int(rows[0]), 0))
    for row in rows:
        selection.select(
            main_window._batch_model.index(int(row), 0),
            QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows,
        )
    main_window._refresh_batch_display_from_focus_and_shown()


def _slider_handle_center(slider: QtWidgets.QSlider) -> QtCore.QPoint:
    option = QtWidgets.QStyleOptionSlider()
    slider.initStyleOption(option)
    handle = slider.style().subControlRect(
        QtWidgets.QStyle.CC_Slider,
        option,
        QtWidgets.QStyle.SC_SliderHandle,
        slider,
    )
    return handle.center()


def test_initial_condition_add_select_reselect_is_passive_runtime_work(
    main_window,
    qtbot,
    qt_app,
    monkeypatch,
):
    runtime_ensures: list[bool] = []
    batch_warms: list[bool] = []
    scheduled: list[tuple[int, object]] = []
    recomputes: list[dict[str, object]] = []

    main_window.show()
    qtbot.waitUntil(lambda: main_window.isVisible(), timeout=1000)
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: runtime_ensures.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: batch_warms.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda **kwargs: recomputes.append(dict(kwargs)),
    )
    def _record_runtime_timer(delay_ms, fn):
        qualname = str(getattr(fn, "__qualname__", ""))
        if "_FitDialogWorkerRegistry.schedule_cleanup" not in qualname:
            scheduled.append((int(delay_ms), fn))

    monkeypatch.setattr(QtCore.QTimer, "singleShot", _record_runtime_timer)

    main_window._set_runtime_backed_controls_ready(True)
    runtime_ensures.clear()
    batch_warms.clear()
    scheduled.clear()
    recomputes.clear()

    main_window._add_batch_set()
    qt_app.processEvents()
    _select_batch_rows(main_window, [0])
    qt_app.processEvents()
    _select_batch_rows(main_window, [1])
    qt_app.processEvents()
    _select_batch_rows(main_window, [0])
    qt_app.processEvents()
    main_window._on_batch_current_changed()
    main_window._on_batch_selection_changed()
    main_window._on_slider_edit_targets_changed()

    assert runtime_ensures == []
    assert batch_warms == []
    assert scheduled == []
    assert recomputes == []


class _RecordingOwner:
    def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
        self.payload = dict(payload)
        self.fast_mode = bool(fast_mode)
        self.ready = False
        self.start_calls: list[dict[str, object]] = []
        self.close_calls: list[bool] = []

    @property
    def simulation_plan_payload(self) -> dict[str, object]:
        return dict(self.payload)

    @property
    def is_ready(self) -> bool:
        return bool(self.ready)

    def start(self, *, wait: bool = True) -> None:
        self.start_calls.append(
            {
                "wait": bool(wait),
                "thread_id": threading.get_ident(),
            }
        )
        self.ready = True

    def close(self, *, kill: bool = False) -> None:
        self.close_calls.append(bool(kill))
        self.ready = False


def test_schedule_bearing_runtime_readiness_reuses_and_invalidates_by_schedule_identity(
    main_window,
    qtbot,
    qt_app,
    monkeypatch,
):
    from kindred.core.simulation_containment import contained_owner_identity_payload

    owners: list[_RecordingOwner] = []

    def _owner_factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _RecordingOwner:
        owner = _RecordingOwner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        owners.append(owner)
        return owner

    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: None,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "parallel_batch_runtime_ready",
        lambda: True,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: None,
    )
    main_window.simulation_controller._contained_simulation_owner_factory = _owner_factory
    main_window.show()
    qtbot.waitUntil(lambda: main_window.isVisible(), timeout=1000)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=set; species=A; time=0.0; value=2.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    qt_app.processEvents()
    _select_batch_rows(main_window, [0])
    qt_app.processEvents()

    main_window.simulation_controller._ensure_interactive_simulation_runtime_available_for_mode(
        fast_mode=False,
        wait=True,
    )

    assert len(owners) == 1
    first_owner = owners[0]
    first_identity = contained_owner_identity_payload(first_owner.simulation_plan_payload)
    first_identity_key = str(first_identity.get("simulation_identity_key") or "")
    assert first_identity_key
    assert first_owner.is_ready is True

    main_window.simulation_controller._ensure_interactive_simulation_runtime_available_for_mode(
        fast_mode=False,
        wait=True,
    )
    assert owners == [first_owner]
    assert first_owner.is_ready is True

    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=set; species=A; time=0.0; value=3.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    qt_app.processEvents()

    main_window.simulation_controller._ensure_interactive_simulation_runtime_available_for_mode(
        fast_mode=False,
        wait=True,
    )

    assert len(owners) == 2
    second_owner = owners[1]
    second_identity = contained_owner_identity_payload(second_owner.simulation_plan_payload)
    assert str(second_identity.get("simulation_identity_key") or "") != first_identity_key
    assert first_owner.close_calls
    assert second_owner.is_ready is True


@pytest.mark.parametrize("preset_name", ["M1", "M9"])
def test_fresh_load_run_selected_and_slider_reuse_ready_exact_runtime_owners(
    main_window,
    qtbot,
    qt_app,
    monkeypatch,
    preset_name: str,
):
    from kindred.core.simulation_plan import SimulationPlan

    owners_by_mode: dict[bool, list[_RecordingOwner]] = {False: [], True: []}
    workers: list[object] = []
    prepare_calls: list[dict[str, object]] = []
    main_thread_id = threading.main_thread().ident

    def _owner_factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _RecordingOwner:
        owner = _RecordingOwner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        owners_by_mode[bool(fast_mode)].append(owner)
        return owner

    class _RecordingContainedWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        result_ready = QtCore.Signal(dict)
        error = QtCore.Signal(object)

        def __init__(
            self,
            *,
            owner,
            simulation_plan_payload,
            include_mechanism_in_result_payload=True,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.owner = owner
            self._owner = owner
            self.simulation_plan_payload = dict(simulation_plan_payload)
            self.include_mechanism_in_result_payload = bool(include_mechanism_in_result_payload)
            self.running = False
            plan = SimulationPlan.from_payload(self.simulation_plan_payload)
            request = plan.to_execution_request().to_payload()
            self.fast_mode = plan.execution_mode == "preview"
            self.mechanism_text = str(request.get("mechanism_text") or "")
            self.parameter_overrides = {
                str(name): float(value)
                for name, value in dict(request.get("parameter_overrides") or {}).items()
            }
            self.initials = {
                str(name): float(value)
                for name, value in dict(request.get("initials") or {}).items()
            }
            workers.append(self)

        def start(self) -> None:
            self.running = True

            def _finish() -> None:
                point_count = 12 if self.fast_mode else 8
                t = np.linspace(0.0, 1.0, point_count)
                initial_a = max(0.0, float(self.initials.get("A", 1.0)))
                initial_b = max(0.0, float(self.initials.get("B", 0.0)))
                amplitude = 0.7
                if self.fast_mode:
                    override_value = next(iter(self.parameter_overrides.values()), 1.0)
                    amplitude = max(0.05, min(0.95, 1.0 / (1.0 + float(override_value))))
                final_a = initial_a * amplitude
                final_b = initial_b + initial_a * (1.0 - amplitude)
                y = np.vstack(
                    [
                        np.linspace(initial_a, final_a, point_count),
                        np.linspace(initial_b, final_b, point_count),
                    ]
                )
                self.running = False
                self.progress.emit(100, "done")
                self.result_ready.emit(
                    {
                        "t": t,
                        "Y": y,
                        "species_names": ["A", "B"],
                        "mechanism": None,
                        "mechanism_text": self.mechanism_text,
                        "solver_config": {"solver": "BDF"},
                        "algebra_scalars": {},
                        "algebra_errors": [],
                    }
                )

            QtCore.QTimer.singleShot(0, _finish)

        def cancel(self) -> None:
            self.running = False

        def isRunning(self) -> bool:
            return bool(self.running)

        def wait(self, *_args, **_kwargs) -> bool:
            self.running = False
            return True

        def terminate(self) -> None:
            self.running = False

    real_prepare = main_window._variable_runtime.prepare_slider_runtime

    def _record_prepare(*args, **kwargs):
        prepare_calls.append({"args": args, "kwargs": dict(kwargs)})
        return real_prepare(*args, **kwargs)

    main_window.simulation_controller._contained_simulation_owner_factory = _owner_factory

    def _ensure_interactive(*, wait=False):
        for fast_mode in (False, True):
            main_window.simulation_controller._ensure_interactive_simulation_runtime_available_for_mode(
                fast_mode=fast_mode,
                wait=bool(wait),
            )

    monkeypatch.setattr(main_window.simulation_controller, "ensure_interactive_simulation_runtimes_available", _ensure_interactive)
    monkeypatch.setattr("kindred.gui.simulation_worker.ContainedSimulationWorker", _RecordingContainedWorker)
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: None,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "parallel_batch_runtime_ready",
        lambda: True,
    )
    monkeypatch.setattr(main_window._variable_runtime, "prepare_slider_runtime", _record_prepare)

    main_window.show()
    qtbot.waitUntil(lambda: main_window.isVisible(), timeout=1000)
    main_window._load_preset_mechanism(preset_name)
    qt_app.processEvents()
    _select_batch_rows(main_window, [0])
    qt_app.processEvents()

    qtbot.waitUntil(
        lambda: len(owners_by_mode[False]) == 1
        and len(owners_by_mode[True]) == 1
        and owners_by_mode[False][0].is_ready
        and owners_by_mode[True][0].is_ready,
        timeout=1000,
    )
    assert prepare_calls == []
    assert all(call["wait"] is True and call["thread_id"] != main_thread_id for call in owners_by_mode[False][0].start_calls)
    assert all(call["wait"] is True and call["thread_id"] != main_thread_id for call in owners_by_mode[True][0].start_calls)

    ordinary_owner = owners_by_mode[False][0]
    preview_owner = owners_by_mode[True][0]
    ordinary_start_calls = list(ordinary_owner.start_calls)
    preview_start_calls = list(preview_owner.start_calls)
    ready_checks: list[dict[str, object]] = []
    real_acquire_owner = main_window.simulation_controller._runtime_application.acquire_ready_owner

    def _record_ready_check(**kwargs):
        owner = real_acquire_owner(**kwargs)
        ready_checks.append(
            {
                "fast_mode": kwargs.get("mode") == "preview",
                "owner": owner,
                "payload": dict(kwargs.get("payload") or {}),
            }
        )
        return owner

    monkeypatch.setattr(
        main_window.simulation_controller._runtime_application,
        "acquire_ready_owner",
        _record_ready_check,
    )

    main_window.simulation_controller.run_simulation()
    qtbot.waitUntil(
        lambda: any(not worker.fast_mode for worker in workers) or bool(ready_checks),
        timeout=1000,
    )
    assert ready_checks and ready_checks[-1]["owner"] is ordinary_owner

    assert owners_by_mode[False] == [ordinary_owner]
    assert ordinary_owner.start_calls == ordinary_start_calls
    explicit_workers = [worker for worker in workers if not worker.fast_mode]
    assert explicit_workers[-1].owner is ordinary_owner
    qtbot.waitUntil(
        lambda: main_window._run_btn.isEnabled()
        and bool(getattr(main_window._plot_tabs._main_plot, "_series", {}) or {}),
        timeout=1000,
    )
    assert main_window._run_btn.isEnabled()
    explicit_series = {
        str(name): np.asarray(values, dtype=float).copy()
        for name, values in dict(getattr(main_window._plot_tabs._main_plot, "_series", {}) or {}).items()
    }
    assert explicit_series

    sliders = main_window._mechanism_editor._variable_sliders
    editable_names = [
        str(name)
        for name in getattr(sliders, "_sliders", {}).keys()
        if sliders.has_variable(str(name))
    ]
    assert editable_names
    slider_name = editable_names[0]
    slider_widget = sliders._sliders[slider_name]
    prior_fast_worker_count = len([worker for worker in workers if worker.fast_mode])
    prior_ready_check_count = len(ready_checks)
    slider_widget.setValue(sliders._value_to_slider_pos(slider_name, 2.0))
    from kindred.core.simulation_containment import contained_owner_identity_payload

    qtbot.waitUntil(
        lambda: len([worker for worker in workers if worker.fast_mode]) > prior_fast_worker_count
        or len(ready_checks) > prior_ready_check_count,
        timeout=1000,
    )
    assert any(worker.fast_mode for worker in workers), (
        f"ready_checks={[(item['fast_mode'], item['owner'] is preview_owner) for item in ready_checks]!r} "
        f"startup_identity={contained_owner_identity_payload(preview_owner.simulation_plan_payload)!r} "
        f"action_identity={contained_owner_identity_payload(ready_checks[-1]['payload'])!r} "
        f"worker_count={len(workers)} pending={main_window.simulation_controller._pending_slider_simulation!r} "
        f"running={main_window.simulation_controller._simulation_running!r} "
        f"status={main_window._status_label.text()!r}"
    )
    qtbot.waitUntil(
        lambda: any(
            not np.array_equal(
                np.asarray(values, dtype=float),
                explicit_series.get(str(name), np.asarray([], dtype=float)),
            )
            for name, values in dict(getattr(main_window._plot_tabs._main_plot, "_series", {}) or {}).items()
        ),
        timeout=1000,
    )
    first_preview_series = {
        str(name): np.asarray(values, dtype=float).copy()
        for name, values in dict(getattr(main_window._plot_tabs._main_plot, "_series", {}) or {}).items()
    }
    assert first_preview_series

    first_preview_count = len([worker for worker in workers if worker.fast_mode])
    slider_widget.setValue(sliders._value_to_slider_pos(slider_name, 1.5))
    qtbot.waitUntil(
        lambda: len([worker for worker in workers if worker.fast_mode]) > first_preview_count,
        timeout=1000,
    )
    qtbot.waitUntil(
        lambda: any(
            not np.array_equal(
                np.asarray(values, dtype=float),
                first_preview_series.get(str(name), np.asarray([], dtype=float)),
            )
            for name, values in dict(getattr(main_window._plot_tabs._main_plot, "_series", {}) or {}).items()
        ),
        timeout=1000,
    )
    second_preview_series = {
        str(name): np.asarray(values, dtype=float).copy()
        for name, values in dict(getattr(main_window._plot_tabs._main_plot, "_series", {}) or {}).items()
    }
    assert second_preview_series

    preview_workers = [worker for worker in workers if worker.fast_mode]
    assert preview_workers[-2].owner is preview_owner, (
        f"startup_identity={contained_owner_identity_payload(preview_owner.simulation_plan_payload)!r} "
        f"first_identity={contained_owner_identity_payload(preview_workers[-2].simulation_plan_payload)!r} "
        f"second_identity={contained_owner_identity_payload(preview_workers[-1].simulation_plan_payload)!r} "
        f"owners={len(owners_by_mode[True])}"
    )
    assert preview_workers[-1].owner is preview_owner
    assert owners_by_mode[True] == [preview_owner]
    assert preview_owner.start_calls == preview_start_calls
    assert prepare_calls == []

    reset_preview_count = len([worker for worker in workers if worker.fast_mode])
    main_window._on_reset_slider_overrides_clicked()
    qt_app.processEvents()

    slider_widget.setValue(sliders._value_to_slider_pos(slider_name, 2.25))
    qtbot.waitUntil(
        lambda: len([worker for worker in workers if worker.fast_mode]) > reset_preview_count,
        timeout=1000,
    )
    qtbot.waitUntil(
        lambda: any(
            not np.array_equal(
                np.asarray(values, dtype=float),
                second_preview_series.get(str(name), np.asarray([], dtype=float)),
            )
            for name, values in dict(getattr(main_window._plot_tabs._main_plot, "_series", {}) or {}).items()
        ),
        timeout=1000,
    )
    preview_workers = [worker for worker in workers if worker.fast_mode]
    assert preview_workers[-1].owner is preview_owner
    assert owners_by_mode[True] == [preview_owner]
    assert preview_owner.start_calls == preview_start_calls
    assert prepare_calls == []

    post_reset_preview_series = {
        str(name): np.asarray(values, dtype=float).copy()
        for name, values in dict(getattr(main_window._plot_tabs._main_plot, "_series", {}) or {}).items()
    }
    assert post_reset_preview_series

    species_slider = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert species_slider is not None
    species_preview_count = len([worker for worker in workers if worker.fast_mode])
    species_press_pos = _slider_handle_center(species_slider)
    qtbot.mousePress(species_slider, QtCore.Qt.LeftButton, pos=species_press_pos)
    species_slider.setValue(5000)
    qtbot.mouseRelease(species_slider, QtCore.Qt.LeftButton, pos=species_press_pos)
    qt_app.processEvents()
    qtbot.waitUntil(
        lambda: len([worker for worker in workers if worker.fast_mode]) > species_preview_count,
        timeout=1000,
    )
    qtbot.waitUntil(
        lambda: any(
            not np.array_equal(
                np.asarray(values, dtype=float),
                post_reset_preview_series.get(str(name), np.asarray([], dtype=float)),
            )
            for name, values in dict(getattr(main_window._plot_tabs._main_plot, "_series", {}) or {}).items()
        ),
        timeout=1000,
    )
    preview_workers = [worker for worker in workers if worker.fast_mode]
    assert preview_workers[-1].owner is preview_owner
    assert owners_by_mode[True] == [preview_owner]
    assert preview_owner.start_calls == preview_start_calls
    assert prepare_calls == []

    qtbot.waitUntil(lambda: all(not worker.isRunning() for worker in workers), timeout=1000)
    assert all(not worker.isRunning() for worker in workers)


def test_run_clicked_while_selected_runtime_is_warming_replays_without_second_click(
    main_window,
    qtbot,
    qt_app,
    monkeypatch,
):
    from kindred.core.simulation_plan import SimulationPlan

    ordinary_ready_gate = threading.Event()
    owners_by_mode: dict[bool, list[_RecordingOwner]] = {False: [], True: []}
    workers: list[object] = []

    class _DeferredOwner(_RecordingOwner):
        def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
            super().__init__(payload, fast_mode=bool(fast_mode))
            self.start_entered = threading.Event()

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append(
                {
                    "wait": bool(wait),
                    "thread_id": threading.get_ident(),
                }
            )
            self.start_entered.set()
            if self.fast_mode:
                self.ready = True
                return
            if ordinary_ready_gate.wait(timeout=5.0):
                self.ready = True

        def close(self, *, kill: bool = False) -> None:
            ordinary_ready_gate.set()
            super().close(kill=kill)

    class _InstantContainedWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        result_ready = QtCore.Signal(dict)
        error = QtCore.Signal(object)

        def __init__(
            self,
            *,
            owner,
            simulation_plan_payload,
            include_mechanism_in_result_payload=True,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.owner = owner
            self.simulation_plan_payload = dict(simulation_plan_payload)
            self.include_mechanism_in_result_payload = bool(include_mechanism_in_result_payload)
            self.running = False
            plan = SimulationPlan.from_payload(self.simulation_plan_payload)
            request = plan.to_execution_request().to_payload()
            self.fast_mode = plan.execution_mode == "preview"
            self.mechanism_text = str(request.get("mechanism_text") or "")
            self.initials = {
                str(name): float(value)
                for name, value in dict(request.get("initials") or {}).items()
            }
            workers.append(self)

        def start(self) -> None:
            self.running = True

            def _finish() -> None:
                t = np.linspace(0.0, 1.0, 8)
                initial_a = max(0.0, float(self.initials.get("A", 1.0)))
                initial_b = max(0.0, float(self.initials.get("B", 0.0)))
                y = np.vstack(
                    [
                        np.linspace(initial_a, initial_a * 0.5, t.size),
                        np.linspace(initial_b, initial_b + initial_a * 0.5, t.size),
                    ]
                )
                self.running = False
                self.progress.emit(100, "done")
                self.result_ready.emit(
                    {
                        "t": t,
                        "Y": y,
                        "species_names": ["A", "B"],
                        "mechanism": None,
                        "mechanism_text": self.mechanism_text,
                        "solver_config": {"solver": "BDF"},
                        "algebra_scalars": {},
                        "algebra_errors": [],
                    }
                )

            QtCore.QTimer.singleShot(0, _finish)

        def cancel(self) -> None:
            self.running = False

        def isRunning(self) -> bool:
            return bool(self.running)

        def wait(self, *_args, **_kwargs) -> bool:
            self.running = False
            return True

        def terminate(self) -> None:
            self.running = False

    def _owner_factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _DeferredOwner:
        owner = _DeferredOwner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        owners_by_mode[bool(fast_mode)].append(owner)
        return owner

    main_window.simulation_controller._contained_simulation_owner_factory = _owner_factory
    monkeypatch.setattr("kindred.gui.simulation_worker.ContainedSimulationWorker", _InstantContainedWorker)
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: None,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "parallel_batch_runtime_ready",
        lambda: True,
    )

    main_window.show()
    qtbot.waitUntil(lambda: main_window.isVisible(), timeout=1000)
    main_window._load_preset_mechanism("M1")
    qt_app.processEvents()
    _select_batch_rows(main_window, [0])
    qt_app.processEvents()

    main_window.simulation_controller.ensure_interactive_simulation_runtimes_available(wait=False)
    qtbot.waitUntil(
        lambda: bool(owners_by_mode[False])
        and owners_by_mode[False][0].start_entered.is_set()
        and not owners_by_mode[False][0].is_ready,
        timeout=2000,
    )

    main_window._simulation_run_ui_owner.set_runtime_backed_run_controls_ready(True)
    main_window._simulation_run_ui_owner.set_run_button_enabled(True)
    qtbot.waitUntil(lambda: main_window._run_btn.isEnabled(), timeout=1000)
    qtbot.mouseClick(main_window._run_btn, QtCore.Qt.LeftButton)
    qt_app.processEvents()

    assert not [worker for worker in workers if not worker.fast_mode]
    assert main_window.simulation_controller._pending_run_after_runtime_ready.active

    ordinary_ready_gate.set()
    qtbot.waitUntil(
        lambda: any(not worker.fast_mode for worker in workers),
        timeout=3000,
    )
    qtbot.waitUntil(
        lambda: main_window._run_btn.isEnabled()
        and bool(getattr(main_window._plot_tabs._main_plot, "_series", {}) or {}),
        timeout=3000,
    )

    explicit_workers = [worker for worker in workers if not worker.fast_mode]
    assert explicit_workers
    assert explicit_workers[-1].owner is owners_by_mode[False][0]
    assert not main_window.simulation_controller._pending_run_after_runtime_ready.active
