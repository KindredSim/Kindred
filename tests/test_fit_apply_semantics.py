import numpy as np
import pytest
from PySide6 import QtWidgets
from types import SimpleNamespace

from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult
from kindred.core.document_parameter_store import DocumentParameterStore
from kindred.gui.controllers.dataset_manager import DatasetFitSettings
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.mixins.fitting_mixin import FittingMixin
from kindred.gui.mixins.ports import FittingMixinPorts


def _make_fit_result() -> GlobalFitResult:
    y = np.linspace(1.0, 0.5, 5)
    model = np.linspace(1.0, 0.4, 5)
    residual = model - y

    return GlobalFitResult(
        success=True,
        shared_params={"k1": 1.23},
        dataset_params={"ds1": {"init:A": 2.0}},
        uncertainties=None,
        global_chi_squared=1.0,
        global_r_squared=0.9,
        dataset_info=[
            DatasetFitInfo(
                dataset_id="ds1",
                r_squared=0.9,
                chi_squared=0.1,
                rmse=0.1,
                mae=0.1,
                residuals=residual.copy(),
                n_points=int(residual.size),
                weight=1.0,
            )
        ],
        nfev=10,
        message="ok",
        covariance=None,
        objective_residuals=residual.copy(),
        model_series={"ds1": {"A": model.copy()}},
        residual_series={"ds1": {"A": residual.copy()}},
    )


def test_global_fit_does_not_persist_dataset_initials_until_apply(qt_app):
    calls = []

    def _dataset_settings_updater(dataset_id: str, updates: dict) -> None:
        calls.append((dataset_id, dict(updates)))

    t = np.linspace(0.0, 1.0, 5)
    y = np.linspace(1.0, 0.5, 5)
    model = np.linspace(1.0, 0.4, 5)
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 1.23, "min": 0.01, "max": 10.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": y.copy()},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": model.copy()}},
        dataset_settings_updater=_dataset_settings_updater,
    )
    try:
        window._handle_global_fit_complete({"result": _make_fit_result()})
        assert calls == []
    finally:
        window.close()


@pytest.mark.parametrize(
    ("scope_label", "expected_scope"),
    [
        ("Parameters only", "parameters"),
        ("Initial conditions only", "initial_conditions"),
        ("Parameters and initial conditions", "both"),
    ],
)
def test_apply_to_project_uses_selected_scope(qt_app, monkeypatch, scope_label, expected_scope):
    calls = []

    def _project_apply_callback(scope: str, shared_params: dict, dataset_params: dict) -> None:
        calls.append((str(scope), dict(shared_params), {str(k): dict(v) for k, v in dict(dataset_params).items()}))

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Ok,
    )

    t = np.linspace(0.0, 1.0, 5)
    y = np.linspace(1.0, 0.5, 5)
    model = np.linspace(1.0, 0.4, 5)
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 1.23, "min": 0.01, "max": 10.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": y.copy()},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": model.copy()}},
        project_apply_callback=_project_apply_callback,
    )
    try:
        window._handle_global_fit_complete({"result": _make_fit_result()})
        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        combo.setCurrentText(scope_label)
        button.click()
        assert len(calls) == 1
        scope, shared_params, dataset_params = calls[0]
        assert scope == expected_scope
        assert shared_params["k1"] == pytest.approx(1.23)
        assert dataset_params["ds1"]["init:A"] == pytest.approx(2.0)
    finally:
        window.close()


class _PlanMechanismEditor:
    def __init__(self, text: str) -> None:
        self._text = str(text)

    def reactions_text(self) -> str:
        return self._text

    def set_reactions_text(self, text: str) -> None:
        self._text = str(text)


class _NoRewritePlanMechanismEditor(_PlanMechanismEditor):
    def set_reactions_text(self, text: str) -> None:
        _ = text


class _PlanBatchStore:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = [
            {
                "set_id": str(row["set_id"]),
                "set_name": str(row["set_name"]),
                "values": {str(k): str(v) for k, v in dict(row["values"]).items()},
            }
            for row in rows
        ]

    def row_for_set_id(self, set_id: str) -> int | None:
        target = str(set_id or "").strip()
        for index, row in enumerate(self._rows):
            if row["set_id"] == target:
                return index
        return None

    def row_for_set(self, name: str) -> int | None:
        target = str(name or "").strip()
        for index, row in enumerate(self._rows):
            if row["set_name"] == target:
                return index
        return None

    def set_id_for_row(self, row: int) -> str:
        return str(self._rows[int(row)]["set_id"])

    def set_name_for_row(self, row: int) -> str:
        return str(self._rows[int(row)]["set_name"])

    def get_value(self, row: int, species: str) -> str:
        return str(self._rows[int(row)]["values"].get(str(species), "0.0"))

    def set_value(self, row: int, species: str, value: str) -> None:
        self._rows[int(row)]["values"][str(species)] = str(value)


class _PlanDatasetManager:
    def __init__(self, settings_by_dataset: dict[str, DatasetFitSettings]) -> None:
        self._settings_by_dataset = dict(settings_by_dataset)

    def get_fit_settings(self, name: str) -> DatasetFitSettings:
        return self._settings_by_dataset[str(name)]

    def update_fit_settings(self, name: str, settings: DatasetFitSettings) -> None:
        self._settings_by_dataset[str(name)] = settings


class _PlanHost(QtWidgets.QWidget, FittingMixin):
    def __init__(
        self,
        *,
        mechanism_text: str,
        authoritative_params: dict[str, float],
        batch_rows: list[dict[str, object]],
        settings_by_dataset: dict[str, DatasetFitSettings],
    ) -> None:
        super().__init__()
        self._fitting_ports = FittingMixinPorts(
            mechanism_editor=_PlanMechanismEditor(mechanism_text),
            dataset_manager=_PlanDatasetManager(settings_by_dataset),
            data_manager_getter=lambda: None,
            status_setter=lambda _text: None,
            temperature_getter=lambda: 298.15,
            num_points_getter=lambda: 100,
        )
        self._preview_session = SimpleNamespace(param_store=DocumentParameterStore())
        self._preview_session.param_store.sync_shared_params(dict(authoritative_params))
        self._batch_store = _PlanBatchStore(batch_rows)


class _NoRewritePlanHost(_PlanHost):
    def __init__(
        self,
        *,
        mechanism_text: str,
        authoritative_params: dict[str, float],
        batch_rows: list[dict[str, object]],
        settings_by_dataset: dict[str, DatasetFitSettings],
    ) -> None:
        super().__init__(
            mechanism_text=mechanism_text,
            authoritative_params=authoritative_params,
            batch_rows=batch_rows,
            settings_by_dataset=settings_by_dataset,
        )
        self._fitting_ports = FittingMixinPorts(
            mechanism_editor=_NoRewritePlanMechanismEditor(mechanism_text),
            dataset_manager=self._fitting_ports.dataset_manager,
            data_manager_getter=self._fitting_ports.data_manager_getter,
            status_setter=self._fitting_ports.status_setter,
            temperature_getter=self._fitting_ports.temperature_getter,
            num_points_getter=self._fitting_ports.num_points_getter,
        )


class _PlanStepHost(_PlanHost):
    def _update_variable_in_mechanism(
        self,
        name: str,
        value: float,
        *,
        source_text: str | None = None,
        commit: bool = True,
        metadata=None,
    ) -> str:
        _ = commit
        _ = metadata
        mechanism_text = str(source_text or self._fitting_ports.mechanism_editor.reactions_text())
        family = "".join(ch for ch in str(name) if not ch.isdigit())
        effective = abs(float(value))
        if family in {"Keq", "kf", "kr"} and abs(effective) < 1e-12:
            effective = 1e-12
        elif effective == 0.0:
            effective = 0.0
        return mechanism_text.replace(f"{family}=0", f"{family}={effective:.15g}", 1)


def test_apply_fit_results_to_project_allows_scalar_and_ic_updates_when_step_warning_present(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 1.0\nreaction: A -> B; k=0.2",
        authoritative_params={"alpha": 1.0},
        batch_rows=[
            {"set_id": "set-ds1", "set_name": "Set ds1", "values": {"A": "1.0"}},
            {"set_id": "set-ds2", "set_name": "Set ds2", "values": {"A": "1.0"}},
        ],
        settings_by_dataset={
            "ds1": DatasetFitSettings(initial_conditions={"A": 1.0}, batch_set="Set ds1", batch_set_id="set-ds1"),
            "ds2": DatasetFitSettings(initial_conditions={"A": 1.0}, batch_set="Set ds2", batch_set_id="set-ds2"),
        },
    )

    try:
        result = host._apply_fit_results_to_project(
            "both",
            {"alpha": 2.0, "kf2": 3.5},
            {"ds1": {"init:A": 2.5}, "ds2": {"init:A": 1.7}},
        )
    finally:
        host.close()

    assert isinstance(result, str)
    assert "kf2" in result
    assert "skipped" in result
    assert host._fitting_ports.mechanism_editor.reactions_text() == "alpha = 2\nreaction: A -> B; k=0.2"
    assert host._batch_store.get_value(0, "A") == "2.5"
    assert host._batch_store.get_value(1, "A") == "1.7"
    assert host._fitting_ports.dataset_manager.get_fit_settings("ds1").initial_conditions["A"] == pytest.approx(2.5)
    assert host._fitting_ports.dataset_manager.get_fit_settings("ds2").initial_conditions["A"] == pytest.approx(1.7)


def test_build_fit_project_apply_plan_reports_missing_step_parameter_as_warning_instead_of_true_noop(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="equilibrium: A <-> B ; kf=1, kr=2",
        authoritative_params={},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        plan = host._build_fit_project_apply_plan("parameters", {"kf2": 3.5}, {})
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.has_real_change is False
    assert plan.parameter_delta.needs_dsl_rewrite is False
    assert plan.parameter_delta.warning_messages == (
        "Step parameter 'kf2' no longer matches any writable step in the current mechanism text.",
    )
    assert len(plan.parameter_delta.step_outcomes) == 1
    outcome = plan.parameter_delta.step_outcomes[0]
    assert outcome.parameter_name == "kf2"
    assert outcome.found_target is False
    assert outcome.writable is False
    assert plan.needs_slider_guard is False
    assert plan.is_true_noop is False


def test_build_fit_project_apply_plan_reports_unwritable_derived_step_parameter_as_warning(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="equilibrium: A <-> B ; kr=2, K=3",
        authoritative_params={"kf1": 6.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        plan = host._build_fit_project_apply_plan("parameters", {"kf1": 9.0}, {})
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.has_real_change is False
    assert plan.parameter_delta.needs_dsl_rewrite is False
    assert plan.parameter_delta.warning_messages == (
        "Step parameter 'kf1' is no longer writable in the current mechanism text.",
    )
    assert len(plan.parameter_delta.step_outcomes) == 1
    outcome = plan.parameter_delta.step_outcomes[0]
    assert outcome.parameter_name == "kf1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == pytest.approx(6.0)
    assert plan.needs_slider_guard is False
    assert plan.is_true_noop is False


def test_build_fit_project_apply_plan_uses_current_text_owner_when_fit_metadata_is_stale(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="equilibrium: A <-> B ; kr=2, K=3",
        authoritative_params={"Keq1": 3.0},
        batch_rows=[],
        settings_by_dataset={},
    )
    host.variable_metadata = lambda: {
        "Keq1": {"type": "equilibrium", "index": 1, "role": "Keq", "editable": True},
        "kf1": {"type": "equilibrium", "index": 1, "role": "kf", "editable": True},
        "kr1": {"type": "equilibrium", "index": 1, "role": "kr", "editable": False, "derived": True},
    }

    try:
        plan = host._build_fit_project_apply_plan("parameters", {"Keq1": 6.0}, {})
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.has_real_change is True
    assert plan.parameter_delta.needs_dsl_rewrite is True
    assert plan.parameter_delta.warning_messages == ()
    assert plan.parameter_delta.updated_text == "equilibrium: A <-> B ; kr=2, Keq=6"
    assert len(plan.parameter_delta.step_outcomes) == 1
    outcome = plan.parameter_delta.step_outcomes[0]
    assert outcome.parameter_name == "Keq1"
    assert outcome.writable is True
    assert outcome.warning_reason is None


def test_build_fit_project_apply_plan_ignores_stale_fit_constraint_metadata_when_current_text_is_editable(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="equilibrium: A <-> B ; kf=6, K=3",
        authoritative_params={"Keq1": 3.0},
        batch_rows=[],
        settings_by_dataset={},
    )
    host.variable_metadata = lambda: {
        "Keq1": {
            "type": "equilibrium",
            "index": 1,
            "role": "Keq",
            "editable": False,
            "derived": True,
            "constraint_reason": "algebra",
        }
    }

    try:
        plan = host._build_fit_project_apply_plan("parameters", {"Keq1": 8.0}, {})
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.has_real_change is True
    assert plan.parameter_delta.needs_dsl_rewrite is True
    assert plan.parameter_delta.warning_messages == ()
    assert plan.parameter_delta.updated_text == "equilibrium: A <-> B ; kf=6, Keq=8"
    assert len(plan.parameter_delta.step_outcomes) == 1
    outcome = plan.parameter_delta.step_outcomes[0]
    assert outcome.parameter_name == "Keq1"
    assert outcome.writable is True
    assert outcome.warning_reason is None


def test_build_fit_project_apply_plan_preserves_scalar_backed_current_text_step_constraint(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 2\nequilibrium: A <-> B ; kf=6, K=3\n\n# Algebra\nparam Keq1 = alpha",
        authoritative_params={"alpha": 2.0, "Keq1": 2.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        plan = host._build_fit_project_apply_plan("parameters", {"Keq1": 8.0}, {})
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.has_real_change is False
    assert plan.parameter_delta.needs_dsl_rewrite is False
    assert plan.parameter_delta.warning_messages == (
        "Step parameter 'Keq1' is no longer writable in the current mechanism text.",
    )
    assert len(plan.parameter_delta.step_outcomes) == 1
    outcome = plan.parameter_delta.step_outcomes[0]
    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == pytest.approx(2.0)


def test_build_fit_project_apply_plan_preserves_plain_k_current_text_step_constraint(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="reaction: A -> B ; k=3\n\n# Algebra\nparam k1 = 4",
        authoritative_params={"k1": 4.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        plan = host._build_fit_project_apply_plan("parameters", {"k1": 8.0}, {})
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.has_real_change is False
    assert plan.parameter_delta.needs_dsl_rewrite is False
    assert plan.parameter_delta.warning_messages == (
        "Step parameter 'k1' is no longer writable in the current mechanism text.",
    )
    assert len(plan.parameter_delta.step_outcomes) == 1
    outcome = plan.parameter_delta.step_outcomes[0]
    assert outcome.parameter_name == "k1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == pytest.approx(4.0)


def test_apply_fit_results_to_project_allows_scalar_and_ic_updates_when_current_text_step_block_present(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 1.0\nequilibrium: A <-> B ; kf=6, K=3\n\n# Algebra\nparam Keq1 = 4",
        authoritative_params={"alpha": 1.0, "Keq1": 4.0},
        batch_rows=[
            {"set_id": "set-ds1", "set_name": "Set ds1", "values": {"A": "1.0"}},
            {"set_id": "set-ds2", "set_name": "Set ds2", "values": {"A": "1.0"}},
        ],
        settings_by_dataset={
            "ds1": DatasetFitSettings(initial_conditions={"A": 1.0}, batch_set="Set ds1", batch_set_id="set-ds1"),
            "ds2": DatasetFitSettings(initial_conditions={"A": 1.0}, batch_set="Set ds2", batch_set_id="set-ds2"),
        },
    )

    try:
        result = host._apply_fit_results_to_project(
            "both",
            {"alpha": 2.0, "Keq1": 8.0},
            {"ds1": {"init:A": 2.5}, "ds2": {"init:A": 1.7}},
        )
    finally:
        host.close()

    assert isinstance(result, str)
    assert "Keq1" in result
    assert "skipped" in result
    assert host._fitting_ports.mechanism_editor.reactions_text() == (
        "alpha = 2\nequilibrium: A <-> B ; kf=6, K=3\n\n# Algebra\nparam Keq1 = 4"
    )
    assert host._batch_store.get_value(0, "A") == "2.5"
    assert host._batch_store.get_value(1, "A") == "1.7"
    assert host._fitting_ports.dataset_manager.get_fit_settings("ds1").initial_conditions["A"] == pytest.approx(2.5)
    assert host._fitting_ports.dataset_manager.get_fit_settings("ds2").initial_conditions["A"] == pytest.approx(1.7)


def test_apply_fit_results_to_project_keeps_plain_k_block_when_current_text_constraint_present(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 1.0\nreaction: A -> B ; k=3\n\n# Algebra\nparam k1 = 4",
        authoritative_params={"alpha": 1.0, "k1": 4.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        result = host._apply_fit_results_to_project(
            "parameters",
            {"alpha": 2.0, "k1": 8.0},
            {},
        )
    finally:
        host.close()

    assert isinstance(result, str)
    assert "k1" in result
    assert "skipped" in result
    assert host._fitting_ports.mechanism_editor.reactions_text() == (
        "alpha = 2\nreaction: A -> B ; k=3\n\n# Algebra\nparam k1 = 4"
    )


def test_apply_fit_results_to_project_best_effort_applies_unrelated_step_scalar_and_ic_when_other_step_analysis_fails(
    qt_app,
):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="\n".join(
            [
                "alpha = 1.0",
                "sin = 2",
                "equilibrium: A <-> B ; kf=6, K=3",
                "equilibrium: B <-> C ; kf=4, K=5",
                "",
                "# Algebra",
                "param Keq2 = sin",
            ]
        ),
        authoritative_params={"alpha": 1.0, "Keq1": 3.0, "Keq2": 5.0},
        batch_rows=[
            {"set_id": "set-ds1", "set_name": "Set ds1", "values": {"A": "1.0"}},
        ],
        settings_by_dataset={
            "ds1": DatasetFitSettings(initial_conditions={"A": 1.0}, batch_set="Set ds1", batch_set_id="set-ds1"),
        },
    )

    try:
        result = host._apply_fit_results_to_project(
            "both",
            {"alpha": 2.0, "Keq1": 8.0, "Keq2": 9.0},
            {"ds1": {"init:A": 2.5}},
        )
    finally:
        host.close()

    assert isinstance(result, str)
    assert "Keq2" in result
    assert "skipped" in result
    assert host._fitting_ports.mechanism_editor.reactions_text() == "\n".join(
        [
            "alpha = 2",
            "sin = 2",
            "equilibrium: A <-> B ; kf=6, Keq=8",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param Keq2 = sin",
        ]
    )
    assert host._batch_store.get_value(0, "A") == "2.5"
    assert host._fitting_ports.dataset_manager.get_fit_settings("ds1").initial_conditions["A"] == pytest.approx(2.5)


def test_apply_fit_results_to_project_allows_same_step_constrained_K_when_other_step_analysis_fails(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="\n".join(
            [
                "alpha = 1.0",
                "sin = 2",
                "equilibrium: A <-> B ; kf=6, K=3",
                "equilibrium: B <-> C ; kf=4, K=5",
                "",
                "# Algebra",
                "param kf1 = 6",
                "param Keq2 = sin",
            ]
        ),
        authoritative_params={"alpha": 1.0, "Keq1": 3.0, "Keq2": 5.0},
        batch_rows=[
            {"set_id": "set-ds1", "set_name": "Set ds1", "values": {"A": "1.0"}},
        ],
        settings_by_dataset={
            "ds1": DatasetFitSettings(initial_conditions={"A": 1.0}, batch_set="Set ds1", batch_set_id="set-ds1"),
        },
    )

    try:
        result = host._apply_fit_results_to_project(
            "both",
            {"alpha": 2.0, "Keq1": 8.0, "Keq2": 9.0},
            {"ds1": {"init:A": 2.5}},
        )
    finally:
        host.close()

    assert isinstance(result, str)
    assert "Keq2" in result
    assert "skipped" in result
    assert host._fitting_ports.mechanism_editor.reactions_text() == "\n".join(
        [
            "alpha = 2",
            "sin = 2",
            "equilibrium: A <-> B ; kf=6, Keq=8",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param kf1 = 6",
            "param Keq2 = sin",
        ]
    )
    assert host._batch_store.get_value(0, "A") == "2.5"
    assert host._fitting_ports.dataset_manager.get_fit_settings("ds1").initial_conditions["A"] == pytest.approx(2.5)


def test_apply_fit_results_to_project_keeps_target_step_analysis_failure_block(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="\n".join(
            [
                "alpha = 1.0",
                "sin = 2",
                "equilibrium: A <-> B ; kf=6, K=3",
                "",
                "# Algebra",
                "param Keq1 = sin",
            ]
        ),
        authoritative_params={"alpha": 1.0, "Keq1": 3.0},
        batch_rows=[
            {"set_id": "set-ds1", "set_name": "Set ds1", "values": {"A": "1.0"}},
        ],
        settings_by_dataset={
            "ds1": DatasetFitSettings(initial_conditions={"A": 1.0}, batch_set="Set ds1", batch_set_id="set-ds1"),
        },
    )

    try:
        result = host._apply_fit_results_to_project(
            "both",
            {"alpha": 2.0, "Keq1": 8.0},
            {"ds1": {"init:A": 2.5}},
        )
    finally:
        host.close()

    assert isinstance(result, str)
    assert "Keq1" in result
    assert "skipped" in result
    assert host._fitting_ports.mechanism_editor.reactions_text() == "\n".join(
        [
            "alpha = 2",
            "sin = 2",
            "equilibrium: A <-> B ; kf=6, K=3",
            "",
            "# Algebra",
            "param Keq1 = sin",
        ]
    )
    assert host._batch_store.get_value(0, "A") == "2.5"
    assert host._fitting_ports.dataset_manager.get_fit_settings("ds1").initial_conditions["A"] == pytest.approx(2.5)


def test_apply_fit_results_to_project_requires_real_mechanism_rewrite_before_reporting_success(qt_app):
    _ = qt_app
    host = _NoRewritePlanHost(
        mechanism_text="alpha = 1.0\nreaction: A -> B; k=0.2",
        authoritative_params={"alpha": 1.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        result = host._apply_fit_results_to_project(
            "parameters",
            {"alpha": 2.0},
            {},
        )
    finally:
        host.close()

    assert result is False
    assert host._fitting_ports.mechanism_editor.reactions_text() == "alpha = 1.0\nreaction: A -> B; k=0.2"


def test_write_fit_results_to_mechanism_reports_already_current_when_dsl_needs_no_rewrite(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 2\nreaction: A -> B; k=0.2",
        authoritative_params={"alpha": 1.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        result = host._write_fit_results_to_mechanism({"alpha": 2.0})
    finally:
        host.close()

    assert result == "already_current"
    assert host._fitting_ports.mechanism_editor.reactions_text() == "alpha = 2\nreaction: A -> B; k=0.2"


def test_apply_fit_results_to_project_treats_already_current_parameter_state_as_success(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 2\nreaction: A -> B; k=0.2",
        authoritative_params={"alpha": 1.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        result = host._apply_fit_results_to_project(
            "parameters",
            {"alpha": 2.0},
            {},
        )
    finally:
        host.close()

    assert result is True
    assert host._fitting_ports.mechanism_editor.reactions_text() == "alpha = 2\nreaction: A -> B; k=0.2"


def test_apply_fit_results_to_project_skips_guard_for_stale_authority_already_current_parameter_state(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 2\nreaction: A -> B; k=0.2",
        authoritative_params={"alpha": 1.0},
        batch_rows=[],
        settings_by_dataset={},
    )
    guard_calls: list[str] = []
    host._guard_slider_transaction_invalidation = (  # type: ignore[attr-defined]
        lambda *, action_text: guard_calls.append(str(action_text)) or False
    )

    try:
        result = host._apply_fit_results_to_project(
            "parameters",
            {"alpha": 2.0},
            {},
        )
    finally:
        host.close()

    assert result is True
    assert guard_calls == []
    assert host._fitting_ports.mechanism_editor.reactions_text() == "alpha = 2\nreaction: A -> B; k=0.2"


def test_apply_fit_results_to_project_warns_when_requested_scalar_parameter_is_missing_from_current_text(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 1.0\nreaction: A -> B; k=0.2",
        authoritative_params={"beta": 1.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        result = host._apply_fit_results_to_project(
            "parameters",
            {"beta": 2.0},
            {},
        )
    finally:
        host.close()

    assert isinstance(result, str)
    assert "beta" in result
    assert "current mechanism text" in result
    assert host._fitting_ports.mechanism_editor.reactions_text() == "alpha = 1.0\nreaction: A -> B; k=0.2"


def test_apply_fit_results_to_project_warns_with_invalid_value_diagnosis_for_nonfinite_scalar_fit_result(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 1.0\nreaction: A -> B; k=0.2",
        authoritative_params={"alpha": 1.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        result = host._apply_fit_results_to_project(
            "parameters",
            {"alpha": float("nan")},
            {},
        )
    finally:
        host.close()

    assert isinstance(result, str)
    assert "alpha" in result
    assert "could not be applied" in result
    assert "non-finite" in result
    assert "no longer exists in the current mechanism text" not in result
    assert host._fitting_ports.mechanism_editor.reactions_text() == "alpha = 1.0\nreaction: A -> B; k=0.2"


def test_apply_fit_results_to_project_keeps_mixed_apply_success_when_parameter_dsl_is_already_current(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 2\nreaction: A -> B; k=0.2",
        authoritative_params={"alpha": 1.0},
        batch_rows=[
            {"set_id": "set-ds1", "set_name": "Set ds1", "values": {"A": "1.0"}},
        ],
        settings_by_dataset={
            "ds1": DatasetFitSettings(initial_conditions={"A": 1.0}, batch_set="Set ds1", batch_set_id="set-ds1"),
        },
    )

    try:
        result = host._apply_fit_results_to_project(
            "both",
            {"alpha": 2.0},
            {"ds1": {"init:A": 2.5}},
        )
    finally:
        host.close()

    assert result is True
    assert host._fitting_ports.mechanism_editor.reactions_text() == "alpha = 2\nreaction: A -> B; k=0.2"
    assert host._batch_store.get_value(0, "A") == "2.5"
    assert host._fitting_ports.dataset_manager.get_fit_settings("ds1").initial_conditions["A"] == pytest.approx(2.5)


def test_apply_fit_results_to_project_keeps_partial_success_when_missing_scalar_parameter_skips_but_ic_updates_apply(
    qt_app,
):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 1.0\nreaction: A -> B; k=0.2",
        authoritative_params={"beta": 1.0},
        batch_rows=[
            {"set_id": "set-ds1", "set_name": "Set ds1", "values": {"A": "1.0"}},
        ],
        settings_by_dataset={
            "ds1": DatasetFitSettings(initial_conditions={"A": 1.0}, batch_set="Set ds1", batch_set_id="set-ds1"),
        },
    )

    try:
        result = host._apply_fit_results_to_project(
            "both",
            {"beta": 2.0},
            {"ds1": {"init:A": 2.5}},
        )
    finally:
        host.close()

    assert isinstance(result, str)
    assert "Applied remaining valid project updates" in result
    assert "beta" in result
    assert host._fitting_ports.mechanism_editor.reactions_text() == "alpha = 1.0\nreaction: A -> B; k=0.2"
    assert host._batch_store.get_value(0, "A") == "2.5"
    assert host._fitting_ports.dataset_manager.get_fit_settings("ds1").initial_conditions["A"] == pytest.approx(2.5)


def test_apply_fit_results_to_project_best_effort_applies_unrelated_step_scalar_and_ic_when_other_step_uses_nonfinite_scalar_input(
    qt_app,
):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="\n".join(
            [
                "alpha = 1.0",
                "a = nan",
                "equilibrium: A <-> B ; kf=6, K=3",
                "equilibrium: B <-> C ; kf=4, K=5",
                "",
                "# Algebra",
                "param Keq2 = a",
            ]
        ),
        authoritative_params={"alpha": 1.0, "Keq1": 3.0, "Keq2": 5.0},
        batch_rows=[
            {"set_id": "set-ds1", "set_name": "Set ds1", "values": {"A": "1.0"}},
        ],
        settings_by_dataset={
            "ds1": DatasetFitSettings(initial_conditions={"A": 1.0}, batch_set="Set ds1", batch_set_id="set-ds1"),
        },
    )

    try:
        result = host._apply_fit_results_to_project(
            "both",
            {"alpha": 2.0, "Keq1": 8.0, "Keq2": 9.0},
            {"ds1": {"init:A": 2.5}},
        )
    finally:
        host.close()

    assert isinstance(result, str)
    assert "Keq2" in result
    assert "skipped" in result
    assert host._fitting_ports.mechanism_editor.reactions_text() == "\n".join(
        [
            "alpha = 2",
            "a = nan",
            "equilibrium: A <-> B ; kf=6, Keq=8",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param Keq2 = a",
        ]
    )
    assert host._batch_store.get_value(0, "A") == "2.5"
    assert host._fitting_ports.dataset_manager.get_fit_settings("ds1").initial_conditions["A"] == pytest.approx(2.5)


def test_apply_fit_results_to_project_keeps_step_block_when_scalar_name_matches_observable(qt_app):
    _ = qt_app
    mechanism_text = "\n".join(
        [
            "alpha = 1.0",
            "equilibrium: A <-> B ; kf=6, K=3",
            "",
            "# Algebra",
            "let alpha = [A]",
            "param Keq1 = 4",
        ]
    )
    host = _PlanHost(
        mechanism_text=mechanism_text,
        authoritative_params={"alpha": 1.0, "Keq1": 4.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        result = host._apply_fit_results_to_project(
            "parameters",
            {"alpha": 2.0, "Keq1": 8.0},
            {},
        )
    finally:
        host.close()

    assert isinstance(result, str)
    assert "Keq1" in result
    assert "skipped" in result
    assert host._fitting_ports.mechanism_editor.reactions_text() == "\n".join(
        [
            "alpha = 2",
            "equilibrium: A <-> B ; kf=6, K=3",
            "",
            "# Algebra",
            "let alpha = [A]",
            "param Keq1 = 4",
        ]
    )


def test_apply_fit_results_to_project_keeps_step_block_when_unused_builtin_shadow_scalar_input_present(qt_app):
    _ = qt_app
    mechanism_text = "\n".join(
        [
            "sin = 1.0",
            "equilibrium: A <-> B ; kf=6, K=3",
            "",
            "# Algebra",
            "param Keq1 = 4",
        ]
    )
    host = _PlanHost(
        mechanism_text=mechanism_text,
        authoritative_params={"sin": 1.0, "Keq1": 4.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        result = host._apply_fit_results_to_project(
            "parameters",
            {"sin": 2.0, "Keq1": 8.0},
            {},
        )
    finally:
        host.close()

    assert isinstance(result, str)
    assert "Keq1" in result
    assert "skipped" in result
    assert host._fitting_ports.mechanism_editor.reactions_text() == "\n".join(
        [
            "sin = 2",
            "equilibrium: A <-> B ; kf=6, K=3",
            "",
            "# Algebra",
            "param Keq1 = 4",
        ]
    )


def test_build_fit_project_apply_plan_treats_step_canonicalization_only_rewrite_as_true_noop(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="equilibrium: A <-> B ; k=1, kr=2",
        authoritative_params={"kf1": 1.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        plan = host._build_fit_project_apply_plan("parameters", {"kf1": 1.0}, {})
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.updated_text == "equilibrium: A <-> B ; kr=2, kf=1"
    assert plan.parameter_delta.has_real_change is False
    assert plan.parameter_delta.needs_dsl_rewrite is True
    assert plan.parameter_delta.warning_messages == ()
    assert len(plan.parameter_delta.step_outcomes) == 1
    outcome = plan.parameter_delta.step_outcomes[0]
    assert outcome.parameter_name == "kf1"
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is True
    assert outcome.canonicalization_only_change is True
    assert plan.needs_slider_guard is False
    assert plan.is_true_noop is True


def test_build_fit_project_apply_plan_treats_numeric_parameter_noop_with_text_formatting_diff_as_true_noop(
    qt_app,
):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 0.200000\nreaction: A -> B; k=1.0",
        authoritative_params={"alpha": 0.2},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        plan = host._build_fit_project_apply_plan("parameters", {"alpha": 0.2}, {})
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.has_real_change is False
    assert plan.parameter_delta.needs_dsl_rewrite is True
    assert plan.needs_slider_guard is False
    assert plan.needs_display_refresh is False
    assert plan.is_true_noop is True


def test_build_fit_project_apply_plan_uses_authoritative_parameter_precision_for_real_change(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 1000000.1234567\nreaction: A -> B; k=1.0",
        authoritative_params={"alpha": 1000000.1234567},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        plan = host._build_fit_project_apply_plan("parameters", {"alpha": 1000000.1234568}, {})
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.updated_text == "alpha = 1000000.1234568\nreaction: A -> B; k=1.0"
    assert plan.parameter_delta.has_real_change is True
    assert plan.parameter_delta.needs_dsl_rewrite is True
    assert plan.needs_slider_guard is True
    assert plan.is_true_noop is False


def test_build_fit_project_apply_plan_treats_signed_zero_parameter_difference_as_true_noop(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 0\nreaction: A -> B; k=1.0",
        authoritative_params={"alpha": 0.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        plan = host._build_fit_project_apply_plan("parameters", {"alpha": -0.0}, {})
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.updated_text == "alpha = 0\nreaction: A -> B; k=1.0"
    assert plan.parameter_delta.has_real_change is False
    assert plan.parameter_delta.needs_dsl_rewrite is False
    assert plan.needs_slider_guard is False
    assert plan.is_true_noop is True


def test_build_fit_project_apply_plan_treats_step_parameter_signed_zero_floor_as_real_change(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="equilibrium: A <-> B ; kf=1, K=0",
        authoritative_params={"Keq1": 0.0},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        plan = host._build_fit_project_apply_plan("parameters", {"Keq1": -0.0}, {})
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.updated_text == "equilibrium: A <-> B ; kf=1, Keq=1e-12"
    assert plan.parameter_delta.has_real_change is True
    assert plan.parameter_delta.needs_dsl_rewrite is True
    assert plan.parameter_delta.warning_messages == ()
    assert plan.needs_slider_guard is True
    assert plan.is_true_noop is False


def test_build_fit_project_apply_plan_marks_real_parameter_change_separately_from_dsl_rewrite(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 0.2\nreaction: A -> B; k=1.0",
        authoritative_params={"alpha": 0.2},
        batch_rows=[],
        settings_by_dataset={},
    )

    try:
        plan = host._build_fit_project_apply_plan("parameters", {"alpha": 0.55}, {})
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.has_real_change is True
    assert plan.needs_slider_guard is True
    assert plan.is_true_noop is False


def test_build_fit_project_apply_plan_separates_dataset_settings_sync_from_canonical_ic_change(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="reaction: A -> B; k=1.0",
        authoritative_params={},
        batch_rows=[
            {
                "set_id": "set-1",
                "set_name": "set1",
                "values": {"A": "2.5"},
            }
        ],
        settings_by_dataset={
            "ds1": DatasetFitSettings(
                initial_conditions={"A": 1.0},
                batch_set="set1",
                batch_set_id="set-1",
            )
        },
    )

    try:
        plan = host._build_fit_project_apply_plan(
            "initial_conditions",
            {},
            {"ds1": {"init:A": 2.5}},
        )
    finally:
        host.close()

    assert plan.parameter_delta is None
    assert plan.has_dataset_settings_sync is True
    assert plan.canonical_ic_affected_set_ids == ()
    assert plan.needs_display_refresh is False
    assert plan.is_true_noop is False


def test_build_fit_project_apply_plan_marks_canonical_ic_change_and_both_scope_matrix(qt_app):
    _ = qt_app
    host = _PlanHost(
        mechanism_text="alpha = 0.2\nreaction: A -> B; k=1.0",
        authoritative_params={"alpha": 0.2},
        batch_rows=[
            {
                "set_id": "set-1",
                "set_name": "set1",
                "values": {"A": "1.0"},
            }
        ],
        settings_by_dataset={
            "ds1": DatasetFitSettings(
                initial_conditions={"A": 1.0},
                batch_set="set1",
                batch_set_id="set-1",
            )
        },
    )

    try:
        plan = host._build_fit_project_apply_plan(
            "both",
            {"alpha": 0.2},
            {"ds1": {"init:A": 2.5}},
        )
    finally:
        host.close()

    assert plan.parameter_delta is not None
    assert plan.parameter_delta.has_real_change is False
    assert plan.needs_slider_guard is False
    assert plan.canonical_ic_affected_set_ids == ("set-1",)
    assert plan.needs_display_refresh is True
    assert plan.is_true_noop is False
