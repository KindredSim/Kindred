"""
Simulation Settings dialog.

Current dialog contract
-----------------------
- Exposes SciPy `solve_ivp` solver selection restricted to `Radau` and `BDF`,
  plus tolerances (rtol/atol) and the generated-symbolic-Jacobian toggle for
  supported stiff solvers.
- Exposes slider-preview defaults separately from the main run controls:
  preview solver, preview point count, and preview debounce timings.
- Does NOT expose temperature, t_end, or point-grid controls (those are surfaced
  directly in the main layout / DSL).

Returned schema
---------------
{
  "solver": str,
  "rtol": float,
  "atol": float,
  "use_sparse_jacobian": bool,
  "wegscheider_cyclicity_enabled": bool,
  "max_parallel_batch_workers": int,
  "batch_runtime_lane_budget": int,
  "limit_blas_threads_per_worker": bool,
  "slider_preview_solver": str,
  "slider_preview_points": int,
  "parameter_preview_debounce_ms": int,
  "equilibrium_preview_debounce_ms": int,
}
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

from PySide6 import QtCore, QtWidgets

from kindred.core.runtime_defaults import (
    MAX_PARALLEL_WORKERS_CEILING,
    PREVIEW_CACHE_CAP_DEFAULT,
    RESULT_CACHE_CAP_DEFAULT,
)
from kindred.core.validation import try_parse_int
from kindred.gui.ports import SimulationCacheControlsPort, SimulationCacheOpResult
from kindred.gui.project_schema import PROJECT_DEFAULTS

logger = logging.getLogger(__name__)

__all__ = ["SolverSettingsDialog"]


_SOLVERS = ["Radau", "BDF"]
_DEFAULT_RESULT_CACHE_CAP = int(RESULT_CACHE_CAP_DEFAULT)
_DEFAULT_PREVIEW_CACHE_CAP = int(PREVIEW_CACHE_CAP_DEFAULT)
_MAX_PARALLEL_WORKERS_SPIN_MAX = int(MAX_PARALLEL_WORKERS_CEILING)


class SolverSettingsDialog(QtWidgets.QDialog):
    settingsAccepted = QtCore.Signal(dict)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        cache_port: Optional[SimulationCacheControlsPort] = None,
    ) -> None:
        super().__init__(parent)
        self._cache_port = cache_port
        self.setWindowTitle("Simulation Settings")
        self.setModal(True)
        self._cache_controls_ready = False

        layout = QtWidgets.QVBoxLayout(self)

        max_input_width = 160

        def _make_header(text: str) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(str(text))
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            return label

        solver_section = QtWidgets.QWidget(self)
        solver_section_layout = QtWidgets.QVBoxLayout(solver_section)
        solver_section_layout.setContentsMargins(0, 0, 0, 0)
        solver_section_layout.setSpacing(6)

        solver_section_layout.addWidget(_make_header("Solver"))

        self._combo_solver = QtWidgets.QComboBox(self)
        self._combo_solver.addItems(_SOLVERS)
        self._combo_solver.setCurrentText("BDF")
        self._combo_solver.setMaximumWidth(max_input_width)
        self._combo_solver.setToolTip("ODE integration method. Radau and BDF are implicit and stiff-capable.")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Solver:", self))
        row.addWidget(self._combo_solver)
        row.addStretch(1)
        solver_section_layout.addLayout(row)

        self._spin_rtol = QtWidgets.QDoubleSpinBox(self)
        self._spin_rtol.setDecimals(12)
        self._spin_rtol.setRange(1e-16, 1.0)
        self._spin_rtol.setValue(1e-6)
        self._spin_rtol.setSingleStep(1e-6)
        self._spin_rtol.setMaximumWidth(180)
        self._spin_rtol.setToolTip("Relative error tolerance. Smaller values give more accurate but slower simulations.")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Relative tolerance (rtol):", self))
        row.addWidget(self._spin_rtol)
        row.addStretch(1)
        solver_section_layout.addLayout(row)

        self._spin_atol = QtWidgets.QDoubleSpinBox(self)
        self._spin_atol.setDecimals(12)
        self._spin_atol.setRange(1e-20, 1.0)
        self._spin_atol.setValue(1e-12)
        self._spin_atol.setSingleStep(1e-9)
        self._spin_atol.setMaximumWidth(180)
        self._spin_atol.setToolTip("Absolute error tolerance. Smaller values give more accurate but slower simulations.")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Absolute tolerance (atol):", self))
        row.addWidget(self._spin_atol)
        row.addStretch(1)
        solver_section_layout.addLayout(row)

        self._sparse_checkbox = QtWidgets.QCheckBox("Use generated symbolic Jacobian (Radau/BDF only)")
        self._sparse_checkbox.setChecked(bool(PROJECT_DEFAULTS["use_sparse_jacobian"]))
        self._sparse_checkbox.setToolTip(
            "Use a generated symbolic Jacobian when the mechanism is supported; unsupported cases use solver defaults."
        )
        solver_section_layout.addWidget(self._sparse_checkbox)

        self._wegscheider_checkbox = QtWidgets.QCheckBox("Thermodynamic cyclicity (Wegscheider)")
        self._wegscheider_checkbox.setChecked(bool(PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"]))
        self._wegscheider_checkbox.setToolTip(
            "Validate symbolic Wegscheider cycle constraints before simulation."
        )
        solver_section_layout.addWidget(self._wegscheider_checkbox)
        self._wegscheider_help = QtWidgets.QLabel(
            "Validate symbolic ln(kf/kr) cycle constraints and offer source-level resolution when needed."
        )
        self._wegscheider_help.setStyleSheet("font-size: 11px;")
        self._wegscheider_help.setWordWrap(True)
        solver_section_layout.addWidget(self._wegscheider_help)

        self._max_parallel_workers_spin = QtWidgets.QSpinBox(self)
        self._max_parallel_workers_spin.setRange(1, _MAX_PARALLEL_WORKERS_SPIN_MAX)
        self._ensure_parallel_worker_spin_capacity(
            int(PROJECT_DEFAULTS["max_parallel_batch_workers"])
        )
        self._max_parallel_workers_spin.setValue(int(PROJECT_DEFAULTS["max_parallel_batch_workers"]))
        self._max_parallel_workers_spin.setMaximumWidth(max_input_width)
        self._max_parallel_workers_spin.setMinimumWidth(80)
        self._max_parallel_workers_spin.setToolTip("Number of sets to simulate simultaneously.")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Max parallel workers:", self))
        row.addWidget(self._max_parallel_workers_spin)
        row.addStretch(1)
        solver_section_layout.addLayout(row)

        self._batch_runtime_lane_budget_spin = QtWidgets.QSpinBox(self)
        self._batch_runtime_lane_budget_spin.setRange(1, _MAX_PARALLEL_WORKERS_SPIN_MAX)
        self._batch_runtime_lane_budget_spin.setValue(int(PROJECT_DEFAULTS["batch_runtime_lane_budget"]))
        self._batch_runtime_lane_budget_spin.setMaximumWidth(max_input_width)
        self._batch_runtime_lane_budget_spin.setMinimumWidth(80)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Batch and fitting runtime lanes:", self))
        row.addWidget(self._batch_runtime_lane_budget_spin)
        row.addStretch(1)
        solver_section_layout.addLayout(row)

        self._limit_blas_checkbox = QtWidgets.QCheckBox("Limit BLAS threads per worker (recommended)")
        self._limit_blas_checkbox.setChecked(bool(PROJECT_DEFAULTS["limit_blas_threads_per_worker"]))
        self._limit_blas_checkbox.setToolTip("Restrict each worker to one BLAS thread to prevent CPU oversubscription.")
        solver_section_layout.addWidget(self._limit_blas_checkbox)

        solver_section_layout.addWidget(_make_header("Slider preview"))

        self._combo_slider_preview_solver = QtWidgets.QComboBox(self)
        self._combo_slider_preview_solver.addItems(_SOLVERS)
        self._combo_slider_preview_solver.setCurrentText("BDF")
        self._combo_slider_preview_solver.setMaximumWidth(max_input_width)
        self._combo_slider_preview_solver.setToolTip("ODE solver used for fast slider preview simulations.")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Slider preview solver:", self))
        row.addWidget(self._combo_slider_preview_solver)
        row.addStretch(1)
        solver_section_layout.addLayout(row)

        self._spin_slider_preview_points = QtWidgets.QSpinBox(self)
        self._spin_slider_preview_points.setRange(50, 20000)
        self._spin_slider_preview_points.setSingleStep(50)
        self._spin_slider_preview_points.setValue(100)
        self._spin_slider_preview_points.setMaximumWidth(max_input_width)
        self._spin_slider_preview_points.setToolTip("Number of time points for slider preview output.")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Slider preview points:", self))
        row.addWidget(self._spin_slider_preview_points)
        row.addStretch(1)
        solver_section_layout.addLayout(row)

        self._spin_parameter_preview_debounce_ms = QtWidgets.QSpinBox(self)
        self._spin_parameter_preview_debounce_ms.setRange(0, 1000)
        self._spin_parameter_preview_debounce_ms.setValue(80)
        self._spin_parameter_preview_debounce_ms.setSuffix(" ms")
        self._spin_parameter_preview_debounce_ms.setMaximumWidth(max_input_width)
        self._spin_parameter_preview_debounce_ms.setToolTip("Delay before launching a preview after a parameter slider change.")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Parameter slider debounce:", self))
        row.addWidget(self._spin_parameter_preview_debounce_ms)
        row.addStretch(1)
        solver_section_layout.addLayout(row)

        self._spin_equilibrium_preview_debounce_ms = QtWidgets.QSpinBox(self)
        self._spin_equilibrium_preview_debounce_ms.setRange(0, 1000)
        self._spin_equilibrium_preview_debounce_ms.setValue(150)
        self._spin_equilibrium_preview_debounce_ms.setSuffix(" ms")
        self._spin_equilibrium_preview_debounce_ms.setMaximumWidth(max_input_width)
        self._spin_equilibrium_preview_debounce_ms.setToolTip("Delay before launching a preview after an equilibrium constant slider change.")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Equilibrium K slider debounce:", self))
        row.addWidget(self._spin_equilibrium_preview_debounce_ms)
        row.addStretch(1)
        solver_section_layout.addLayout(row)

        layout.addWidget(solver_section)

        cache_section = QtWidgets.QWidget(self)
        cache_section.setObjectName("simulationCachingSection")
        cache_section.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        cache_layout = QtWidgets.QVBoxLayout(cache_section)
        cache_layout.setContentsMargins(0, 0, 0, 0)
        cache_layout.setSpacing(6)

        cache_layout.addWidget(_make_header("Simulation caching"))

        self._spin_result_cache_cap = QtWidgets.QSpinBox(self)
        self._spin_result_cache_cap.setRange(0, 1_000_000_000)
        self._spin_result_cache_cap.setValue(_DEFAULT_RESULT_CACHE_CAP)
        self._spin_result_cache_cap.setMaximumWidth(max_input_width)
        self._spin_result_cache_cap.setToolTip("Maximum number of full simulation results to keep in memory.")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Result cache cap:", self))
        row.addWidget(self._spin_result_cache_cap)
        row.addStretch(1)
        cache_layout.addLayout(row)

        self._spin_preview_cache_cap = QtWidgets.QSpinBox(self)
        self._spin_preview_cache_cap.setRange(0, 1_000_000_000)
        self._spin_preview_cache_cap.setValue(_DEFAULT_PREVIEW_CACHE_CAP)
        self._spin_preview_cache_cap.setMaximumWidth(max_input_width)
        self._spin_preview_cache_cap.setToolTip("Maximum number of slider preview results to keep in memory.")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Preview cache cap:", self))
        row.addWidget(self._spin_preview_cache_cap)
        row.addStretch(1)
        cache_layout.addLayout(row)

        self._label_result_cache_status = QtWidgets.QLabel("Result cache: 0/0, 0.0 MB", self)
        self._label_preview_cache_status = QtWidgets.QLabel("Preview cache: 0/0, 0.0 MB", self)
        cache_layout.addWidget(self._label_result_cache_status)
        cache_layout.addWidget(self._label_preview_cache_status)
        self._label_cache_status_error = QtWidgets.QLabel("", self)
        self._label_cache_status_error.setWordWrap(True)
        self._label_cache_status_error.setStyleSheet("font-weight: bold; font-size: 11px;")
        self._label_cache_status_error.hide()
        cache_layout.addWidget(self._label_cache_status_error)

        purge_row = QtWidgets.QHBoxLayout()
        self._btn_purge_results = QtWidgets.QPushButton("Purge result cache", self)
        self._btn_purge_results.setToolTip("Clear all cached full simulation results to free memory.")
        self._btn_purge_preview = QtWidgets.QPushButton("Purge preview cache", self)
        self._btn_purge_preview.setToolTip("Clear all cached slider preview results to free memory.")
        self._btn_purge_all = QtWidgets.QPushButton("Purge all caches", self)
        self._btn_purge_all.setToolTip("Clear all cached results and previews to free memory.")
        purge_row.addWidget(self._btn_purge_results)
        purge_row.addWidget(self._btn_purge_preview)
        purge_row.addWidget(self._btn_purge_all)
        purge_row.addItem(
            QtWidgets.QSpacerItem(
                0,
                0,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Minimum,
            )
        )
        cache_layout.addLayout(purge_row)

        layout.addWidget(cache_section)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        spacer = QtWidgets.QSpacerItem(
            0,
            0,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        layout.addItem(spacer)
        layout.setStretch(layout.count() - 1, 1)
        layout.addWidget(button_box)

        self._wire_cache_controls()
        self._cache_controls_ready = True
        self._refresh_cache_status()

    def _wire_cache_controls(self) -> None:
        def _apply_caps() -> None:
            if not bool(getattr(self, "_cache_controls_ready", False)):
                return
            ctrl = self._cache_port
            if ctrl is None:
                return
            result = ctrl.set_simulation_cache_caps(
                result_cap=int(self._spin_result_cache_cap.value()),
                preview_cap=int(self._spin_preview_cache_cap.value()),
                persist=True,
            )
            self._handle_cache_action_outcome(result)

        self._spin_result_cache_cap.valueChanged.connect(lambda _v: _apply_caps())
        self._spin_preview_cache_cap.valueChanged.connect(lambda _v: _apply_caps())

        self._btn_purge_results.clicked.connect(lambda: self._confirm_and_purge(which="result"))
        self._btn_purge_preview.clicked.connect(lambda: self._confirm_and_purge(which="preview"))
        self._btn_purge_all.clicked.connect(lambda: self._confirm_and_purge(which="all"))

    def _confirm_and_purge(self, *, which: str) -> None:
        ctrl = self._cache_port
        if ctrl is None:
            return
        which = str(which or "").strip().lower()
        if which not in {"result", "preview", "all"}:
            return

        if which == "result":
            title = "Purge result cache"
            message = "Purge the result cache?\n\nEvicted results must be recomputed by pressing Run."
        elif which == "preview":
            title = "Purge preview cache"
            message = "Purge the preview cache?\n\nEvicted previews will be recomputed on the next slider update."
        else:
            title = "Purge all caches"
            message = "Purge all simulation caches?\n\nResults and previews will need to be recomputed."

        resp = QtWidgets.QMessageBox.question(
            self,
            title,
            message,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        if which == "result":
            result = ctrl.purge_simulation_result_cache()
        elif which == "preview":
            result = ctrl.purge_simulation_preview_cache()
        else:
            result = ctrl.purge_simulation_all_caches()
        self._handle_cache_action_outcome(result)

    def _format_cache_line(self, *, label: str, used: int, cap: int, nbytes: int) -> str:
        mb = float(max(0, int(nbytes))) / (1024.0 * 1024.0)
        return f"{label} cache: {int(used)}/{int(cap)}, {mb:.1f} MB"

    def _set_cache_status_unavailable(self) -> None:
        self._label_result_cache_status.setText("Result cache: unavailable")
        self._label_preview_cache_status.setText("Preview cache: unavailable")

    def _set_cache_status_error(self, message: str, *, overwrite_status_labels: bool) -> None:
        if overwrite_status_labels:
            self._set_cache_status_unavailable()
        text = str(message or "").strip()
        self._label_cache_status_error.setText(text)
        self._label_cache_status_error.setVisible(bool(text))

    def _clear_cache_status_error(self) -> None:
        self._label_cache_status_error.setText("")
        self._label_cache_status_error.hide()

    def _handle_cache_action_outcome(self, outcome: SimulationCacheOpResult) -> None:
        if outcome.ok:
            self._refresh_cache_status()
            return
        if outcome.cache_state_changed:
            refreshed = self._refresh_cache_status()
            self._set_cache_status_error(
                outcome.message or "Cache action failed.",
                overwrite_status_labels=not refreshed,
            )
            return
        self._set_cache_status_error(
            outcome.message or "Cache action failed.",
            overwrite_status_labels=True,
        )

    def _cache_caps_from_controller(self) -> Optional[Tuple[int, int]]:
        ctrl = self._cache_port
        if ctrl is None:
            return None
        stats_outcome = ctrl.simulation_cache_stats()
        if not stats_outcome.ok or stats_outcome.stats is None:
            return None
        stats = stats_outcome.stats
        try:
            r_cap = int((stats.get("result") or {}).get("cap", 0))
            p_cap = int((stats.get("preview") or {}).get("cap", 0))
        except Exception:
            return None
        return r_cap, p_cap

    def _refresh_cache_status(self) -> bool:
        ctrl = self._cache_port
        if ctrl is None:
            self._set_cache_status_unavailable()
            self._set_cache_status_error("Simulation cache controls are unavailable.", overwrite_status_labels=False)
            return False
        stats_outcome = ctrl.simulation_cache_stats()
        if not stats_outcome.ok or stats_outcome.stats is None:
            self._set_cache_status_error(
                stats_outcome.message or "Failed to read simulation cache status.",
                overwrite_status_labels=True,
            )
            return False

        stats = stats_outcome.stats
        r = stats.get("result") or {}
        p = stats.get("preview") or {}
        self._label_result_cache_status.setText(
            self._format_cache_line(
                label="Result",
                used=int(r.get("used", 0) or 0),
                cap=int(r.get("cap", 0) or 0),
                nbytes=int(r.get("bytes", 0) or 0),
            )
        )
        self._label_preview_cache_status.setText(
            self._format_cache_line(
                label="Preview",
                used=int(p.get("used", 0) or 0),
                cap=int(p.get("cap", 0) or 0),
                nbytes=int(p.get("bytes", 0) or 0),
            )
        )
        self._clear_cache_status_error()
        return True

    def get_settings(self) -> Dict:
        solver = self._combo_solver.currentText()
        rtol = float(self._spin_rtol.value())
        atol = float(self._spin_atol.value())
        if not (rtol > 0.0):
            raise ValueError("rtol must be positive")
        if not (atol > 0.0):
            raise ValueError("atol must be positive")
        return {
            "solver": solver,
            "rtol": rtol,
            "atol": atol,
            "use_sparse_jacobian": bool(self._sparse_checkbox.isChecked()),
            "wegscheider_cyclicity_enabled": bool(self._wegscheider_checkbox.isChecked()),
            "max_parallel_batch_workers": int(self._max_parallel_workers_spin.value()),
            "batch_runtime_lane_budget": int(self._batch_runtime_lane_budget_spin.value()),
            "limit_blas_threads_per_worker": bool(self._limit_blas_checkbox.isChecked()),
            "slider_preview_solver": str(self._combo_slider_preview_solver.currentText()),
            "slider_preview_points": int(self._spin_slider_preview_points.value()),
            "parameter_preview_debounce_ms": int(self._spin_parameter_preview_debounce_ms.value()),
            "equilibrium_preview_debounce_ms": int(self._spin_equilibrium_preview_debounce_ms.value()),
            "result_cache_cap": int(self._spin_result_cache_cap.value()),
            "preview_cache_cap": int(self._spin_preview_cache_cap.value()),
        }

    def set_settings(self, cfg: Dict) -> None:
        cfg = dict(cfg or {})
        from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name

        def _coerce_positive_float(value: object, *, default: float) -> float:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return float(default)
            if not math.isfinite(parsed) or parsed <= 0.0:
                return float(default)
            return float(parsed)

        def _coerce_bool(value: object) -> bool:
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    return True
                if normalized in {"", "0", "false", "no", "off"}:
                    return False
                return False
            return bool(value)

        cfg.setdefault("solver", str(DEFAULT_SOLVER_NAME))
        cfg.setdefault("rtol", 1e-6)
        cfg.setdefault("atol", 1e-12)
        cfg.setdefault("use_sparse_jacobian", bool(PROJECT_DEFAULTS["use_sparse_jacobian"]))
        cfg.setdefault(
            "wegscheider_cyclicity_enabled",
            bool(PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"]),
        )
        cfg.setdefault("max_parallel_batch_workers", int(PROJECT_DEFAULTS["max_parallel_batch_workers"]))
        cfg.setdefault("batch_runtime_lane_budget", int(PROJECT_DEFAULTS["batch_runtime_lane_budget"]))
        cfg.setdefault(
            "limit_blas_threads_per_worker",
            bool(PROJECT_DEFAULTS["limit_blas_threads_per_worker"]),
        )
        cfg.setdefault("slider_preview_solver", "BDF")
        cfg.setdefault("slider_preview_points", 100)
        cfg.setdefault("parameter_preview_debounce_ms", 80)
        cfg.setdefault("equilibrium_preview_debounce_ms", 150)
        cfg.setdefault("result_cache_cap", _DEFAULT_RESULT_CACHE_CAP)
        cfg.setdefault("preview_cache_cap", _DEFAULT_PREVIEW_CACHE_CAP)
        solver_name, _warning = normalize_solver_name(cfg.get("solver", DEFAULT_SOLVER_NAME))
        self._combo_solver.setCurrentText(solver_name)

        self._spin_rtol.setValue(_coerce_positive_float(cfg.get("rtol"), default=1e-6))
        self._spin_atol.setValue(_coerce_positive_float(cfg.get("atol"), default=1e-12))

        self._sparse_checkbox.setChecked(_coerce_bool(cfg.get("use_sparse_jacobian")))
        self._wegscheider_checkbox.setChecked(_coerce_bool(cfg.get("wegscheider_cyclicity_enabled")))
        try:
            workers = int(cfg.get("max_parallel_batch_workers", int(PROJECT_DEFAULTS["max_parallel_batch_workers"])))
        except Exception:
            workers = int(PROJECT_DEFAULTS["max_parallel_batch_workers"])
        self._ensure_parallel_worker_spin_capacity(workers)
        self._max_parallel_workers_spin.setValue(max(1, workers))
        try:
            lane_budget = int(cfg.get("batch_runtime_lane_budget", int(PROJECT_DEFAULTS["batch_runtime_lane_budget"])))
        except Exception:
            lane_budget = int(PROJECT_DEFAULTS["batch_runtime_lane_budget"])
        self._batch_runtime_lane_budget_spin.setValue(
            max(1, min(_MAX_PARALLEL_WORKERS_SPIN_MAX, lane_budget))
        )
        self._limit_blas_checkbox.setChecked(
            _coerce_bool(cfg.get("limit_blas_threads_per_worker", bool(PROJECT_DEFAULTS["limit_blas_threads_per_worker"])))
        )
        slider_solver_name, _warning = normalize_solver_name(cfg.get("slider_preview_solver", DEFAULT_SOLVER_NAME))
        self._combo_slider_preview_solver.setCurrentText(slider_solver_name)
        try:
            slider_preview_points = int(cfg.get("slider_preview_points", 100))
        except Exception:
            slider_preview_points = 100
        self._spin_slider_preview_points.setValue(max(50, min(20000, slider_preview_points)))
        try:
            parameter_preview_debounce_ms = int(cfg.get("parameter_preview_debounce_ms", 80))
        except Exception:
            parameter_preview_debounce_ms = 80
        self._spin_parameter_preview_debounce_ms.setValue(max(0, min(1000, parameter_preview_debounce_ms)))
        try:
            equilibrium_preview_debounce_ms = int(cfg.get("equilibrium_preview_debounce_ms", 150))
        except Exception:
            equilibrium_preview_debounce_ms = 150
        self._spin_equilibrium_preview_debounce_ms.setValue(max(0, min(1000, equilibrium_preview_debounce_ms)))

        result_cap = cfg.get("result_cache_cap")
        preview_cap = cfg.get("preview_cache_cap")
        if isinstance(result_cap, int) and result_cap >= 0:
            self._spin_result_cache_cap.setValue(int(result_cap))
        else:
            self._spin_result_cache_cap.setValue(_DEFAULT_RESULT_CACHE_CAP)
        if isinstance(preview_cap, int) and preview_cap >= 0:
            self._spin_preview_cache_cap.setValue(int(preview_cap))
        else:
            self._spin_preview_cache_cap.setValue(_DEFAULT_PREVIEW_CACHE_CAP)

        # Align spinboxes with controller state if available (covers persisted settings).
        caps = self._cache_caps_from_controller()
        if caps is not None:
            r_cap, p_cap = caps
            r_parsed, r_ok = try_parse_int(r_cap)
            if r_ok:
                self._spin_result_cache_cap.setValue(int(r_parsed))
            p_parsed, p_ok = try_parse_int(p_cap)
            if p_ok:
                self._spin_preview_cache_cap.setValue(int(p_parsed))
        self._refresh_cache_status()

    def _ensure_parallel_worker_spin_capacity(self, workers: int) -> None:
        if self._max_parallel_workers_spin.maximum() != _MAX_PARALLEL_WORKERS_SPIN_MAX:
            self._max_parallel_workers_spin.setMaximum(_MAX_PARALLEL_WORKERS_SPIN_MAX)

    def _on_accept(self) -> None:
        try:
            cfg = self.get_settings()
        except Exception as exc:
            logger.warning("Invalid simulation settings: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Invalid settings", str(exc))
            return
        self.settingsAccepted.emit(cfg)
        self.accept()
