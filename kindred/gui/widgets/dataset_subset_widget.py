"""Reusable widget for viewing an arbitrary subset of datasets in a grid during fitting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
from PySide6 import QtCore, QtWidgets

from kindred.gui.widgets.dataset_overlay_panel import DatasetOverlayPanel
from kindred.gui.widgets.grid_plot_view import GridPlotView

__all__ = ["DatasetSubsetWidget"]


class DatasetSubsetWidget(QtWidgets.QWidget):
    """A live-updating grid view that can display any selected subset of datasets."""

    def __init__(
        self,
        *,
        dataset_entries: Sequence[Dict[str, Any]],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._dataset_entries = [dict(entry) for entry in (dataset_entries or [])]
        self._entry_lookup: Dict[str, Dict[str, Any]] = {
            str(entry.get("id")): entry for entry in self._dataset_entries if entry.get("id") is not None
        }

        self._model_series: Dict[str, Dict[str, np.ndarray]] = {}
        self._model_x_by_dataset: Dict[str, np.ndarray] = {}
        self._dataset_stats: Dict[str, Dict[str, float]] = {}
        self._last_union_species_for_grid_selection: Optional[Set[str]] = None

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter, stretch=1)

        self._selector = DatasetOverlayPanel()
        splitter.addWidget(self._selector)
        self._grid = GridPlotView()
        splitter.addWidget(self._grid)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self._selector.set_datasets(self._build_dataset_payloads())
        self._sync_overlay_selection(previous_selected_datasets=[], previous_enabled_species={}, allow_default_include=True)

        self._selector.selectionChanged.connect(lambda _ds: self.refresh())
        self._selector.styleChanged.connect(self.refresh)
        self.refresh()

    def set_dataset_entries(self, dataset_entries: Sequence[Dict[str, Any]]) -> None:
        had_any_series_before = self._entries_have_any_series(self._dataset_entries)
        previous_selected = []
        previous_enabled: Dict[str, Set[str]] = {}
        try:
            previous_selected = list(self._selector.selected_datasets())
            previous_enabled = {k: set(v) for k, v in (self._selector.selected_dataset_species() or {}).items()}
        except Exception:
            previous_selected = []
            previous_enabled = {}

        self._dataset_entries = [dict(entry) for entry in (dataset_entries or [])]
        self._entry_lookup = {
            str(entry.get("id")): entry for entry in self._dataset_entries if entry.get("id") is not None
        }
        self._selector.set_datasets(self._build_dataset_payloads())
        self._sync_overlay_selection(
            previous_selected_datasets=previous_selected,
            previous_enabled_species=previous_enabled,
            allow_default_include=not had_any_series_before,
        )
        self.refresh()

    def set_best_fit(
        self,
        *,
        model_series: Dict[str, Dict[str, np.ndarray]],
        dataset_stats: Optional[Dict[str, Dict[str, float]]] = None,
        model_x_by_dataset: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self._model_series = {
            str(ds_id): {str(k): np.asarray(v) for k, v in (series or {}).items()}
            for ds_id, series in (model_series or {}).items()
            if isinstance(series, dict)
        }
        self._model_x_by_dataset = {
            str(ds_id): np.asarray(values, dtype=float).reshape(-1)
            for ds_id, values in (model_x_by_dataset or {}).items()
        }
        self._dataset_stats = {
            str(ds_id): dict(stats) for ds_id, stats in (dataset_stats or {}).items() if isinstance(stats, dict)
        }
        self.refresh()

    def set_view_autorange_locked(self, locked: bool) -> None:
        """Prevent plot autorange zoom changes during live fit updates."""
        try:
            self._grid.set_autorange_locked(bool(locked))
        except Exception:
            return

    def refresh(self) -> None:
        selected = self._selector.selected_dataset_species()
        if not selected:
            any_series = False
            for entry in self._dataset_entries:
                species_data = entry.get("species_data") or entry.get("species") or {}
                if isinstance(species_data, dict) and any(str(k).strip() for k in species_data.keys()):
                    any_series = True
                    break
            if any_series:
                self._selector.set_status_messages(["No overlays selected."])
            else:
                self._selector.set_status_messages(["No series available to overlay. Select Fit Targets and Apply."])
            self._grid.clear_datasets()
            self._last_union_species_for_grid_selection = None
            return

        self._selector.set_status_messages([])
        datasets_to_show: List[Dict[str, Any]] = []
        union_species: Set[str] = set()

        ordered_selected_ids = [
            str(entry.get("id"))
            for entry in self._dataset_entries
            if entry.get("id") is not None and str(entry.get("id")) in selected
        ]
        for dataset_id in ordered_selected_ids:
            entry = self._entry_lookup.get(dataset_id)
            if entry is None:
                continue
            species_data = entry.get("species_data") or entry.get("species") or {}
            if not isinstance(species_data, dict):
                continue
            enabled_species = selected.get(dataset_id, set()) or set()
            filtered_data = {
                str(name): np.asarray(values, dtype=float).reshape(-1)
                for name, values in species_data.items()
                if str(name) in enabled_species
            }
            if not filtered_data:
                continue

            x_name = str(entry.get("x_name") or "t").strip() or "t"
            if x_name != "t":
                data_x = np.asarray(entry.get("x_obs", []), dtype=float).reshape(-1)
                x_label = x_name
                x_units = None
            else:
                data_x = np.asarray(entry.get("t", []), dtype=float).reshape(-1)
                x_label = "Time"
                x_units = "s"
            if data_x.size == 0:
                continue

            union_species.update(filtered_data.keys())
            model_for_ds = self._model_series.get(dataset_id, {}) if isinstance(self._model_series, dict) else {}
            filtered_model = {
                name: np.asarray(model_for_ds[name], dtype=float).reshape(-1)
                for name in filtered_data.keys()
                if name in model_for_ds
            }
            model_x = self._model_x_by_dataset.get(dataset_id)
            if filtered_model and (model_x is None or np.asarray(model_x).size == 0):
                model_x = data_x
            stats = self._dataset_stats.get(dataset_id, {}) if isinstance(self._dataset_stats, dict) else {}
            chi = stats.get("chi_squared")
            r2 = stats.get("r_squared")

            current_species = sorted(filtered_data.keys())[0]
            datasets_to_show.append(
                {
                    "name": str(entry.get("label") or dataset_id),
                    "data_x": np.asarray(data_x),
                    "data_y": np.asarray(filtered_data[current_species]),
                    "model_x": np.asarray(model_x) if (filtered_model and model_x is not None) else None,
                    "model_y": np.asarray(filtered_model.get(current_species)) if filtered_model else None,
                    "model_series": (dict(filtered_model) if filtered_model else None),
                    "chi_squared": float(chi) if chi is not None else None,
                    "r_squared": float(r2) if r2 is not None else None,
                    "all_species": dict(filtered_data),
                    "current_species": str(current_species),
                    "x_label": x_label,
                    "x_units": x_units,
                }
            )

        self._grid.set_datasets(datasets_to_show)
        if union_species:
            union_snapshot = set(union_species)
            if union_snapshot != (self._last_union_species_for_grid_selection or set()):
                self._grid.set_species_selection(sorted(union_snapshot))
                self._last_union_species_for_grid_selection = union_snapshot
        else:
            self._last_union_species_for_grid_selection = None

    def _build_dataset_payloads(self) -> Dict[str, dict]:
        payloads: Dict[str, dict] = {}
        for entry in self._dataset_entries:
            dataset_id = str(entry.get("id"))
            species_data = entry.get("species_data") or entry.get("species") or {}
            if not isinstance(species_data, dict):
                continue
            payloads[dataset_id] = {"species": {str(k): np.asarray(v) for k, v in species_data.items()}}
        return payloads

    @staticmethod
    def _entries_have_any_series(entries: Sequence[Dict[str, Any]]) -> bool:
        for entry in entries or []:
            species_data = entry.get("species_data") or entry.get("species") or {}
            if isinstance(species_data, dict) and any(str(name).strip() for name in species_data.keys()):
                return True
        return False

    def _sync_overlay_selection(
        self,
        *,
        previous_selected_datasets: List[str],
        previous_enabled_species: Dict[str, Set[str]],
        allow_default_include: bool,
    ) -> None:
        """
        Keep overlay selection and enabled species in sync with the current dataset payloads.

        This is used after Apply actions (fit targets) and add/remove session mutations where
        the available species keys per dataset can change.
        """
        include_ids = [
            str(entry.get("id"))
            for entry in self._dataset_entries
            if entry.get("id") is not None and bool(entry.get("include", True))
        ]
        ordered_ids = [str(entry.get("id")) for entry in self._dataset_entries if entry.get("id") is not None]
        self._selector.reconcile_selection(
            previous_selected_datasets=previous_selected_datasets,
            previous_enabled_species=previous_enabled_species,
            include_dataset_ids=include_ids,
            ordered_dataset_ids=ordered_ids,
            allow_default_include=allow_default_include,
        )
