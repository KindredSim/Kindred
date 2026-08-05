"""
Data tab widget for the fitting window's visible sampling panel.

Communicates with FittingWindow via signals (outbound events) and public
methods (inbound queries/commands).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from PySide6 import QtWidgets
from PySide6.QtCore import Signal

from kindred.core.analysis.dataset_sampling import compute_windowed_indices
from kindred.gui.display_name_policy import (
    DATASET_LIST_LABEL_MAX_CHARS,
    compact_dataset_label,
)
from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter

# The historical presentation uses ``0`` for the visible “All” choice.  Keep
# that UI sentinel local until the current window wires sampling into its
# typed setup projection.
_SAMPLING_ALL_POINTS_SENTINEL = 0


def _canonical_x_mapping_mode(x_name: object, x_mapping_mode: object) -> str:
    """Return the presentation-level mapping mode for an X-axis selection."""

    axis_name = str(x_name or "t").strip() or "t"
    if axis_name == "t":
        return "auto"
    mode = (
        str(x_mapping_mode or "auto")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        or "auto"
    )
    if mode in ("monotone_only", "monotoneonly"):
        mode = "monotone"
    return mode if mode in ("auto", "monotone", "time_guided") else "auto"


class DataTab(QtWidgets.QWidget):
    """Visible sampling panel used by the unified dataset composition."""

    # ------------------------------------------------------------------
    # Signals (DataTab -> FittingWindow)
    # ------------------------------------------------------------------
    samplingApplied = Signal(str, dict)                # ds_id, config dict
    statusMessage = Signal(str)                        # status bar text
    validityChanged = Signal()

    def __init__(
        self,
        *,
        sampling_applied_config_getter: Callable[[str], Dict[str, float | int | str]],
        sampling_default_config_getter: Callable[
            [Mapping[str, ArrayLike]], Dict[str, float | int | str]
        ],
        fit_targets_full_t_getter: Callable[[str], Mapping[str, ArrayLike]],
        fit_targets_available_getter: Callable[[str], List[str]],
        fit_targets_full_series_getter: Callable[[str], Dict[str, np.ndarray]],
        fit_targets_selection_applied_getter: Callable[[str], List[str]],
        modeled_series_getter: Callable[[], set[str]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        # Store callable getters for shared state access.
        self._sampling_applied_config_getter = sampling_applied_config_getter
        self._sampling_default_config_getter = sampling_default_config_getter
        self._fit_targets_full_t_getter = fit_targets_full_t_getter
        self._fit_targets_available_getter = fit_targets_available_getter
        self._fit_targets_full_series_getter = fit_targets_full_series_getter
        self._fit_targets_selection_applied_getter = fit_targets_selection_applied_getter
        self._modeled_series_getter = modeled_series_getter

        # Private UI state.
        self._sampling_current_dataset_id: Optional[str] = None
        self._sampling_is_refreshing = False
        self._sampling_pending_by_dataset: dict[
            str, Dict[str, float | int | str]
        ] = {}
        self._sampling_invalid_raw: dict[tuple[str, str], str] = {}
        self._resolved_auto_modes: dict[str, str] = {}
        self._dataset_display_names: dict[str, str] = {}

        # Build layout.
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._sampling_panel_widget = self._create_sampling_panel()
        layout.addWidget(self._sampling_panel_widget, stretch=1)

    # ------------------------------------------------------------------
    # Public API (FittingWindow -> DataTab)
    # ------------------------------------------------------------------

    def select_dataset(
        self,
        dataset_id: str,
        display_name: str | None = None,
    ) -> None:
        """Load sampling controls for the given dataset (public entry point)."""
        ds_id = str(dataset_id or "").strip()
        if ds_id and display_name is not None:
            self._dataset_display_names[ds_id] = (
                str(display_name or "").strip() or ds_id
            )
        self._load_sampling_controls_for_dataset(ds_id or None)

    def set_resolved_auto_modes(
        self,
        modes: Mapping[str, str] | tuple[tuple[str, str], ...],
    ) -> None:
        """Project worker-resolved Auto modes without changing requested setup."""

        normalized = {str(dataset_id): str(mode) for dataset_id, mode in dict(modes).items()}
        if any(mode not in {"monotone", "time_guided"} for mode in normalized.values()):
            raise ValueError("Resolved Auto modes are invalid.")
        self._resolved_auto_modes = normalized
        self._refresh_resolved_auto_label()

    @property
    def invalid_editor_keys(self) -> tuple[tuple[str, str], ...]:
        """Stable dataset/field keys for invalid executable sampling text."""

        return tuple(sorted(self._sampling_invalid_raw))

    @property
    def invalid_raw_values(self) -> dict[tuple[str, str], str]:
        return dict(self._sampling_invalid_raw)

    def set_sampling_secondary_error(self, message: Optional[str], *, tooltip_text: Optional[str] = None) -> None:
        """Set or clear the secondary error message on the sampling footer."""
        if hasattr(self, "_sampling_footer"):
            self._sampling_footer.set_secondary_error(message, tooltip_text=tooltip_text)

    def sampling_validation_error(self, *, dataset_id: str, config: Dict[str, float | int | str]) -> Optional[str]:
        """Pure-computation validation of a sampling config. No UI side-effects."""
        return self._sampling_pending_validation_error(dataset_id=dataset_id, pending=config)

    # ------------------------------------------------------------------
    # Layout construction (moved from FittingWindow factories)
    # ------------------------------------------------------------------

    def _create_sampling_panel(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        container.setObjectName("global_fit_sampling_panel")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._sampling_header_label = QtWidgets.QLabel("Sampling (selected dataset: \u2014)")
        self._sampling_header_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._sampling_header_label)

        self._sampling_footer = ConfigPanelFooter(
            container,
            show_divider=False,
            show_reset=True,
            show_secondary_error=True,
            messages_position="after_body",
            apply_requires_no_error=True,
            button_order=("reset", "revert", "apply"),
            error_object_name="global_fit_sampling_error",
            secondary_error_object_name="global_fit_sampling_run_blocked",
            apply_object_name="global_fit_sampling_apply",
            revert_object_name="global_fit_sampling_revert",
            reset_object_name="global_fit_sampling_reset",
        )
        layout.addWidget(self._sampling_footer, stretch=1)
        self._sampling_footer.applyRequested.connect(self._apply_sampling_changes)
        self._sampling_footer.revertRequested.connect(self._revert_sampling_changes)
        self._sampling_footer.resetRequested.connect(self._reset_sampling_pending_to_defaults)

        # Stacked label-above-control layout for narrow (~240px) left panel.
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.setSpacing(4)

        self._sampling_x_axis_combo = QtWidgets.QComboBox(container)
        self._sampling_x_axis_combo.setObjectName("global_fit_sampling_x_axis")
        self._sampling_x_axis_combo.setEnabled(False)
        controls_layout.addWidget(QtWidgets.QLabel("X axis:"))
        controls_layout.addWidget(self._sampling_x_axis_combo)

        self._sampling_x_mode_combo = QtWidgets.QComboBox(container)
        self._sampling_x_mode_combo.setObjectName("global_fit_sampling_x_mode")
        self._sampling_x_mode_combo.setEnabled(False)
        self._sampling_x_mode_combo.setVisible(False)
        self._sampling_x_mode_combo.addItem("Auto", "auto")
        self._sampling_x_mode_combo.addItem("Monotone only (fast)", "monotone")
        self._sampling_x_mode_combo.addItem("Time-guided (non-monotone)", "time_guided")
        self._sampling_x_mode_label = QtWidgets.QLabel("X mapping:")
        self._sampling_x_mode_label.setVisible(False)
        controls_layout.addWidget(self._sampling_x_mode_label)
        controls_layout.addWidget(self._sampling_x_mode_combo)

        self._sampling_t_min_spin = QtWidgets.QDoubleSpinBox(container)
        self._sampling_t_min_spin.setObjectName("global_fit_sampling_t_min")
        self._sampling_t_min_spin.setDecimals(6)
        self._sampling_t_min_spin.setRange(-1e18, 1e18)
        self._sampling_t_min_spin.setSingleStep(1.0)
        self._sampling_t_min_spin.setEnabled(False)
        controls_layout.addWidget(QtWidgets.QLabel("t_min:"))
        controls_layout.addWidget(self._sampling_t_min_spin)

        self._sampling_t_max_spin = QtWidgets.QDoubleSpinBox(container)
        self._sampling_t_max_spin.setObjectName("global_fit_sampling_t_max")
        self._sampling_t_max_spin.setDecimals(6)
        self._sampling_t_max_spin.setRange(-1e18, 1e18)
        self._sampling_t_max_spin.setSingleStep(1.0)
        self._sampling_t_max_spin.setEnabled(False)
        controls_layout.addWidget(QtWidgets.QLabel("t_max:"))
        controls_layout.addWidget(self._sampling_t_max_spin)

        self._sampling_n_points_spin = QtWidgets.QSpinBox(container)
        self._sampling_n_points_spin.setObjectName("global_fit_sampling_n_points")
        self._sampling_n_points_spin.setRange(0, 0)
        self._sampling_n_points_spin.setSpecialValueText("All")
        self._sampling_n_points_spin.setValue(0)
        self._sampling_n_points_spin.setEnabled(False)
        controls_layout.addWidget(QtWidgets.QLabel("N:"))
        controls_layout.addWidget(self._sampling_n_points_spin)

        self._sampling_footer.body_layout.addLayout(controls_layout)

        self._sampling_used_label = QtWidgets.QLabel("Used: \u2014")
        self._sampling_used_label.setObjectName("global_fit_sampling_used_label")
        self._sampling_used_label.setStyleSheet("font-size: 11px;")
        self._sampling_footer.body_layout.addWidget(self._sampling_used_label)

        self._sampling_resolved_auto_label = QtWidgets.QLabel(
            "Current run: not resolved",
            container,
        )
        self._sampling_resolved_auto_label.setObjectName(
            "global_fit_sampling_resolved_auto"
        )
        self._sampling_resolved_auto_label.setStyleSheet("font-size: 11px;")
        self._sampling_resolved_auto_label.setVisible(False)
        self._sampling_footer.body_layout.addWidget(
            self._sampling_resolved_auto_label
        )

        self._sampling_t_min_spin.valueChanged.connect(self._on_sampling_controls_changed)
        self._sampling_t_max_spin.valueChanged.connect(self._on_sampling_controls_changed)
        self._sampling_n_points_spin.valueChanged.connect(self._on_sampling_controls_changed)
        self._sampling_x_axis_combo.currentIndexChanged.connect(self._on_sampling_controls_changed)
        self._sampling_x_mode_combo.currentIndexChanged.connect(self._on_sampling_controls_changed)
        for field, spin in (
            ("t_min", self._sampling_t_min_spin),
            ("t_max", self._sampling_t_max_spin),
            ("n_points", self._sampling_n_points_spin),
        ):
            line_edit = spin.lineEdit()
            if line_edit is not None:
                line_edit.textChanged.connect(
                    lambda _text, key=field: self._on_sampling_raw_text_changed(key)
                )

        return container

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Sampling — internal helpers (moved from FittingWindow)
    # ------------------------------------------------------------------

    def _selected_target_times(self, dataset_id: str) -> dict[str, np.ndarray]:
        """Return the selected model-target time series for one dataset.

        The getter deliberately exposes only the time vectors needed by this
        panel.  FittingWindow owns the raw-observation-ID to model-target
        selection and supplies the resulting mapping keyed by model target.
        Keeping those vectors separate is important: rows from different
        selected targets must not be flattened into one sampling population.
        """

        raw_series = self._fit_targets_full_t_getter(dataset_id)
        if not isinstance(raw_series, Mapping):
            return {}
        selected: dict[str, np.ndarray] = {}
        for raw_name, values in raw_series.items():
            name = str(raw_name or "").strip()
            if not name:
                continue
            selected[name] = np.asarray(values, dtype=float).reshape(-1)
        return selected

    @staticmethod
    def _sampling_time_bounds(
        target_times: Mapping[str, np.ndarray],
    ) -> tuple[float, float]:
        finite_ranges = [
            (
                float(np.min(finite)),
                float(np.max(finite)),
            )
            for values in target_times.values()
            if values.size
            for finite in (values[np.isfinite(values)],)
            if finite.size
        ]
        if not finite_ranges:
            return 0.0, 0.0
        return (
            min(low for low, _high in finite_ranges),
            max(high for _low, high in finite_ranges),
        )

    @staticmethod
    def _windowed_target_counts(
        *,
        target_times: Mapping[str, np.ndarray],
        t_min: float,
        t_max: float,
    ) -> dict[str, int]:
        return {
            name: int(
                compute_windowed_indices(
                    t=values,
                    t_min=t_min,
                    t_max=t_max,
                ).size
            )
            for name, values in target_times.items()
        }

    def _load_sampling_controls_for_dataset(self, dataset_id: Optional[str]) -> None:
        if not hasattr(self, "_sampling_t_min_spin"):
            return
        if not hasattr(self, "_sampling_footer"):
            return
        ds_id = str(dataset_id or "").strip()
        enabled = bool(ds_id)

        if hasattr(self, "_sampling_header_label") and self._sampling_header_label is not None:
            full_name = self._dataset_display_names.get(ds_id, ds_id)
            compact = compact_dataset_label(
                full_name,
                max_chars=DATASET_LIST_LABEL_MAX_CHARS,
            )
            title = (
                "Sampling (selected dataset: \u2014)"
                if not enabled
                else f"Sampling (selected dataset: {compact.display})"
            )
            self._sampling_header_label.setText(title)
            self._sampling_header_label.setToolTip(
                "" if not enabled else f"{compact.full}\nDataset ID: {ds_id}"
            )

        self._sampling_is_refreshing = True
        try:
            for widget in (
                self._sampling_x_axis_combo,
                self._sampling_x_mode_combo,
                self._sampling_t_min_spin,
                self._sampling_t_max_spin,
                self._sampling_n_points_spin,
            ):
                widget.setEnabled(enabled)
            if self._sampling_footer.reset_button is not None:
                self._sampling_footer.reset_button.setEnabled(enabled)
            if not enabled:
                self._sampling_current_dataset_id = None
                self._sampling_used_label.setText("Used: \u2014")
                self._sampling_footer.set_error(None)
                self._sampling_footer.set_secondary_error(None)
                self._sampling_footer.set_dirty(False)
                self._sampling_x_mode_combo.setVisible(False)
                self._sampling_x_mode_label.setVisible(False)
                self._sampling_resolved_auto_label.setVisible(False)
                try:
                    self._sampling_x_axis_combo.clear()
                    self._sampling_x_mode_combo.setCurrentIndex(0)
                except Exception:
                    return
                return

            target_times = self._selected_target_times(ds_id)
            t_min_full, t_max_full = self._sampling_time_bounds(target_times)

            applied = self._sampling_applied_config_getter(ds_id)
            cfg = self._sampling_pending_by_dataset.get(ds_id, applied)
            t_min_val = float(cfg.get("t_min", t_min_full))
            t_max_val = float(cfg.get("t_max", t_max_full))
            n_points = int(cfg.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL)))
            x_name = str(cfg.get("x_name") or "t").strip() or "t"
            x_mode = _canonical_x_mapping_mode(x_name, cfg.get("x_mapping_mode"))

            observed = set(self._fit_targets_available_getter(ds_id))
            modeled = self._modeled_series_getter()
            candidates = sorted({name for name in observed & modeled if name})

            self._sampling_x_axis_combo.blockSignals(True)
            try:
                self._sampling_x_axis_combo.clear()
                self._sampling_x_axis_combo.addItem("Time (t)", "t")
                for name in candidates:
                    self._sampling_x_axis_combo.addItem(str(name), str(name))
                idx_x = int(self._sampling_x_axis_combo.findData(x_name))
                if idx_x < 0:
                    idx_x = 0
                self._sampling_x_axis_combo.setCurrentIndex(idx_x)
            finally:
                self._sampling_x_axis_combo.blockSignals(False)

            self._sampling_x_mode_combo.blockSignals(True)
            try:
                idx_mode = int(self._sampling_x_mode_combo.findData(x_mode))
                if idx_mode < 0:
                    idx_mode = 0
                self._sampling_x_mode_combo.setCurrentIndex(idx_mode)
            finally:
                self._sampling_x_mode_combo.blockSignals(False)
            show_mode = x_name != "t"
            self._sampling_x_mode_combo.setVisible(bool(show_mode))
            self._sampling_x_mode_label.setVisible(bool(show_mode))
            self._sampling_x_mode_combo.setEnabled(bool(show_mode))

            self._sampling_t_min_spin.setRange(t_min_full, t_max_full)
            self._sampling_t_max_spin.setRange(t_min_full, t_max_full)
            self._sampling_t_min_spin.setValue(max(t_min_full, min(t_max_full, t_min_val)))
            self._sampling_t_max_spin.setValue(max(t_min_full, min(t_max_full, t_max_val)))

            target_lengths = [values.size for values in target_times.values()]
            max_points = min(target_lengths) if target_lengths else 0
            self._sampling_n_points_spin.setRange(
                int(_SAMPLING_ALL_POINTS_SENTINEL),
                max(int(_SAMPLING_ALL_POINTS_SENTINEL), int(max_points)),
            )
            self._sampling_n_points_spin.setValue(
                max(
                    int(_SAMPLING_ALL_POINTS_SENTINEL),
                    min(int(max_points), n_points),
                )
            )

            self._sampling_current_dataset_id = ds_id
            self._sampling_footer.set_error(None)
            for field, spin in (
                ("t_min", self._sampling_t_min_spin),
                ("t_max", self._sampling_t_max_spin),
                ("n_points", self._sampling_n_points_spin),
            ):
                raw = self._sampling_invalid_raw.get((ds_id, field))
                line_edit = spin.lineEdit()
                if raw is not None and line_edit is not None:
                    line_edit.setText(raw)
        finally:
            self._sampling_is_refreshing = False
        self._on_sampling_controls_changed()

    @staticmethod
    def _sampling_signature(
        config: Mapping[str, float | int | str],
    ) -> tuple[object, ...]:
        x_name = str(config.get("x_name") or "t").strip() or "t"
        return (
            float(config.get("t_min", 0.0)),
            float(config.get("t_max", 0.0)),
            int(config.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL))),
            x_name,
            _canonical_x_mapping_mode(x_name, config.get("x_mapping_mode")),
        )

    def _parse_sampling_numeric_field(
        self,
        field: str,
    ) -> float | int:
        spin = {
            "t_min": self._sampling_t_min_spin,
            "t_max": self._sampling_t_max_spin,
            "n_points": self._sampling_n_points_spin,
        }[field]
        line_edit = spin.lineEdit()
        text = str(line_edit.text() if line_edit is not None else "").strip()
        if field == "n_points":
            if not text or text.casefold() == "all":
                return int(_SAMPLING_ALL_POINTS_SENTINEL)
            value = int(text)
            if value < int(_SAMPLING_ALL_POINTS_SENTINEL):
                raise ValueError("N must be All or a nonnegative integer.")
            return value
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite.")
        return value

    def _capture_sampling_pending(self) -> Optional[Dict[str, float | int | str]]:
        ds_id = str(getattr(self, "_sampling_current_dataset_id", "") or "").strip()
        if not ds_id:
            return None
        applied = self._sampling_applied_config_getter(ds_id)
        pending = dict(self._sampling_pending_by_dataset.get(ds_id, applied))
        x_name = str(self._sampling_x_axis_combo.currentData() or "t").strip() or "t"
        pending["x_name"] = x_name
        pending["x_mapping_mode"] = _canonical_x_mapping_mode(
            x_name, self._sampling_x_mode_combo.currentData()
        )
        for field in ("t_min", "t_max", "n_points"):
            key = (ds_id, field)
            spin = {
                "t_min": self._sampling_t_min_spin,
                "t_max": self._sampling_t_max_spin,
                "n_points": self._sampling_n_points_spin,
            }[field]
            line_edit = spin.lineEdit()
            raw = str(line_edit.text() if line_edit is not None else "")
            try:
                pending[field] = self._parse_sampling_numeric_field(field)
            except (TypeError, ValueError, OverflowError):
                self._sampling_invalid_raw[key] = raw
            else:
                self._sampling_invalid_raw.pop(key, None)
        if (
            self._sampling_signature(pending) != self._sampling_signature(applied)
            or any(key[0] == ds_id for key in self._sampling_invalid_raw)
        ):
            self._sampling_pending_by_dataset[ds_id] = pending
        else:
            self._sampling_pending_by_dataset.pop(ds_id, None)
        return pending

    def _sampling_pending_values(self) -> Optional[Dict[str, float | int | str]]:
        return self._capture_sampling_pending()

    def _on_sampling_raw_text_changed(self, _field: str) -> None:
        if self._sampling_is_refreshing:
            return
        self._on_sampling_controls_changed()

    def _sampling_pending_validation_error(self, *, dataset_id: str, pending: Dict[str, float | int | str]) -> Optional[str]:
        ds_id = str(dataset_id or "").strip()
        target_times = self._selected_target_times(ds_id)
        if not target_times:
            return "At least one model target must be selected."
        t_min = float(pending.get("t_min", 0.0))
        t_max = float(pending.get("t_max", 0.0))
        n_points = int(pending.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL)))
        if t_min > t_max:
            return "t_min must be \u2264 t_max."
        windowed_counts = self._windowed_target_counts(
            target_times=target_times,
            t_min=t_min,
            t_max=t_max,
        )
        too_short = next(
            (
                (name, count)
                for name, count in windowed_counts.items()
                if count < 2
            ),
            None,
        )
        if too_short is not None:
            name, count = too_short
            return (
                f"Sampling window for target '{name}' must contain at least "
                f"2 points (found {count})."
            )
        if n_points != int(_SAMPLING_ALL_POINTS_SENTINEL):
            if n_points < 2:
                return "N must be All or \u2265 2."
            too_short_for_n = next(
                (
                    (name, count)
                    for name, count in windowed_counts.items()
                    if n_points > count
                ),
                None,
            )
            if too_short_for_n is not None:
                name, count = too_short_for_n
                return (
                    f"N must be \u2264 {count} for every selected target "
                    f"series ('{name}' has {count} windowed points)."
                )

        x_name = str(pending.get("x_name") or "t").strip() or "t"
        if x_name != "t":
            full_series = self._fit_targets_full_series_getter(ds_id) if ds_id else {}
            if not (isinstance(full_series, Mapping) and x_name in full_series):
                return f"X axis '{x_name}' is not available as an observed column for this dataset."
            modeled = self._modeled_series_getter()
            if modeled and x_name not in modeled:
                return f"X axis '{x_name}' is not a modeled series (species or algebra observable)."
            applied_targets = set(self._fit_targets_selection_applied_getter(ds_id) or [])
            if x_name in applied_targets:
                return f"X axis '{x_name}' cannot also be a fitted series. Remove it from Fit Targets or choose a different X."
        return None

    def _sampling_used_count_for_pending(self, *, dataset_id: str, pending: Dict[str, float | int | str]) -> Tuple[int, int]:
        ds_id = str(dataset_id or "").strip()
        target_times = self._selected_target_times(ds_id)
        total = sum(int(values.size) for values in target_times.values())
        t_min = float(pending.get("t_min", 0.0))
        t_max = float(pending.get("t_max", 0.0))
        n_points = int(pending.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL)))
        if t_min > t_max:
            return 0, total
        windowed_counts = self._windowed_target_counts(
            target_times=target_times,
            t_min=t_min,
            t_max=t_max,
        )
        if not windowed_counts or any(count < 2 for count in windowed_counts.values()):
            return 0, total
        if n_points == int(_SAMPLING_ALL_POINTS_SENTINEL):
            return sum(windowed_counts.values()), total
        if n_points >= 2 and all(n_points <= count for count in windowed_counts.values()):
            return n_points * len(windowed_counts), total
        return 0, total

    def _on_sampling_controls_changed(self) -> None:
        if self._sampling_is_refreshing:
            return
        ds_id = str(getattr(self, "_sampling_current_dataset_id", "") or "").strip()
        if not ds_id:
            return
        pending = self._sampling_pending_values()
        if not isinstance(pending, dict):
            return
        x_name = str(pending.get("x_name") or "t").strip() or "t"
        show_mode = x_name != "t"
        self._sampling_x_mode_combo.setVisible(bool(show_mode))
        self._sampling_x_mode_label.setVisible(bool(show_mode))
        self._sampling_x_mode_combo.setEnabled(bool(show_mode))
        used, total = self._sampling_used_count_for_pending(dataset_id=ds_id, pending=pending)
        self._sampling_used_label.setText(f"Used: {used}/{total} points")

        invalid_fields = sorted(
            field
            for dataset_id, field in self._sampling_invalid_raw
            if dataset_id == ds_id
        )
        error = (
            "Sampling values must be finite numbers (N may be All)."
            if invalid_fields
            else self._sampling_pending_validation_error(
                dataset_id=ds_id,
                pending=pending,
            )
        )
        if hasattr(self, "_sampling_footer"):
            self._sampling_footer.set_error(error)

        applied = self._sampling_applied_config_getter(ds_id)
        applied_x_name = str(applied.get("x_name") or "t").strip() or "t"
        pending_x_name = str(pending.get("x_name") or "t").strip() or "t"
        dirty = bool(
            ds_id in self._sampling_pending_by_dataset
            or invalid_fields
            or applied_x_name != pending_x_name
            or _canonical_x_mapping_mode(
                applied_x_name, applied.get("x_mapping_mode")
            )
            != _canonical_x_mapping_mode(
                pending_x_name, pending.get("x_mapping_mode")
            )
        )
        if hasattr(self, "_sampling_footer"):
            self._sampling_footer.set_dirty(bool(dirty))
        self._refresh_resolved_auto_label()
        self.validityChanged.emit()

    def _refresh_resolved_auto_label(self) -> None:
        if not hasattr(self, "_sampling_resolved_auto_label"):
            return
        ds_id = str(self._sampling_current_dataset_id or "").strip()
        config = self._sampling_pending_by_dataset.get(ds_id)
        if config is None and ds_id:
            config = self._sampling_applied_config_getter(ds_id)
        x_name = str((config or {}).get("x_name") or "t").strip() or "t"
        requested = _canonical_x_mapping_mode(
            x_name,
            (config or {}).get("x_mapping_mode"),
        )
        resolved = self._resolved_auto_modes.get(ds_id)
        visible = bool(x_name != "t" and requested == "auto")
        self._sampling_resolved_auto_label.setVisible(visible)
        resolved_label = {
            "monotone": "Monotone",
            "time_guided": "Time-guided",
        }.get(str(resolved or ""))
        self._sampling_resolved_auto_label.setText(
            "Current run: not resolved"
            if resolved_label is None
            else f"Current run: Auto \u2192 {resolved_label}"
        )
        self._sampling_resolved_auto_label.setToolTip(
            "" if not ds_id else f"Dataset ID: {ds_id}"
        )

    def _apply_sampling_changes(self) -> None:
        ds_id = str(getattr(self, "_sampling_current_dataset_id", "") or "").strip()
        if not ds_id:
            return
        pending = self._sampling_pending_values()
        if not isinstance(pending, dict):
            return
        pending["x_mapping_mode"] = _canonical_x_mapping_mode(
            pending.get("x_name"), pending.get("x_mapping_mode")
        )
        if any(key[0] == ds_id for key in self._sampling_invalid_raw):
            if hasattr(self, "_sampling_footer"):
                self._sampling_footer.set_error(
                    "Correct invalid sampling text before applying."
                )
            return
        error = self._sampling_pending_validation_error(dataset_id=ds_id, pending=pending)
        if error:
            if hasattr(self, "_sampling_footer"):
                self._sampling_footer.set_error(error)
            return

        # ORDERING: samplingApplied must use a direct (synchronous) connection so
        # FittingWindow updates _sampling_applied before _on_sampling_controls_changed
        # reads it back via the getter. Do not use Qt.QueuedConnection.
        self.samplingApplied.emit(ds_id, dict(pending))
        applied = self._sampling_applied_config_getter(ds_id)
        if self._sampling_signature(applied) == self._sampling_signature(pending):
            self._sampling_pending_by_dataset.pop(ds_id, None)
            for key in tuple(self._sampling_invalid_raw):
                if key[0] == ds_id:
                    self._sampling_invalid_raw.pop(key, None)
        self._on_sampling_controls_changed()

    def _revert_sampling_changes(self) -> None:
        ds_id = str(getattr(self, "_sampling_current_dataset_id", "") or "").strip()
        if not ds_id:
            return
        self._sampling_pending_by_dataset.pop(ds_id, None)
        for key in tuple(self._sampling_invalid_raw):
            if key[0] == ds_id:
                self._sampling_invalid_raw.pop(key, None)
        self._load_sampling_controls_for_dataset(ds_id)

    def _reset_sampling_pending_to_defaults(self) -> None:
        ds_id = str(getattr(self, "_sampling_current_dataset_id", "") or "").strip()
        if not ds_id:
            return
        for key in tuple(self._sampling_invalid_raw):
            if key[0] == ds_id:
                self._sampling_invalid_raw.pop(key, None)
        target_times = self._selected_target_times(ds_id)
        defaults = self._sampling_default_config_getter(target_times)
        self._sampling_is_refreshing = True
        try:
            self._sampling_t_min_spin.setValue(float(defaults.get("t_min", 0.0)))
            self._sampling_t_max_spin.setValue(float(defaults.get("t_max", 0.0)))
            self._sampling_n_points_spin.setValue(int(defaults.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL))))
            idx_mode = int(self._sampling_x_mode_combo.findData("auto"))
            if idx_mode < 0:
                idx_mode = 0
            self._sampling_x_mode_combo.setCurrentIndex(idx_mode)
            idx = int(self._sampling_x_axis_combo.findData("t"))
            if idx < 0:
                idx = 0
            self._sampling_x_axis_combo.setCurrentIndex(idx)
        finally:
            self._sampling_is_refreshing = False
        self._on_sampling_controls_changed()
