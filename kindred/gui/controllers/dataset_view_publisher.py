"""Dataset plot/grid view publisher."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np

from kindred.core.analysis.global_fit_projection import (
    FitRenderDatasetProjection,
    FitRenderProjection,
)
from kindred.gui.controllers.dataset_errors import DatasetOwnerError
from kindred.gui.controllers.dataset_registry import DatasetRecord

logger = logging.getLogger(__name__)


class DatasetViewPublisher:
    """Own dataset plot/grid view state derived from committed registry records."""

    def __init__(
        self,
        plot_tabs,
        dataset_tab_simulation_owner,
    ) -> None:
        self._plot_tabs = plot_tabs
        self._dataset_tab_simulation_owner = dataset_tab_simulation_owner
        self._dataset_views: Dict[str, Dict[str, Any]] = {}
        self._fit_render_projection: Optional[FitRenderProjection] = None
        self._display_name_by_id: Dict[str, str] = {}
        self._dataset_panel_map: Dict[str, object] = {}

    def publish_records(self, records: Sequence[DatasetRecord]) -> None:
        for record in records:
            entry = self._prepare_dataset_entry(record)
            self._display_name_by_id[str(record.dataset_id)] = str(record.display_name)
            self._dataset_views[str(record.display_name)] = entry
            self._update_dataset_plot(str(record.display_name))
        if records:
            self._refresh_dataset_grid()

    def remove_record(self, record: DatasetRecord) -> bool:
        name = str(record.display_name)
        dataset_id = str(record.dataset_id)
        projection = self._fit_render_projection
        clear_projection = bool(
            isinstance(projection, FitRenderProjection)
            and dataset_id in {str(value) for value in projection.dataset_ids}
        )
        surviving_dataset_ids: tuple[str, ...] = ()
        if clear_projection and projection is not None:
            surviving_dataset_ids = tuple(
                str(value)
                for value in projection.dataset_ids
                if str(value) != dataset_id
            )
        removed = self._dataset_views.pop(name, None)
        self._display_name_by_id.pop(dataset_id, None)
        self._dataset_panel_map.pop(name, None)
        self._plot_tabs.remove_dataset_tab(name)
        if clear_projection:
            self.clear_fit_result_views(surviving_dataset_ids or (dataset_id,))
        else:
            self._refresh_dataset_grid()
        return removed is not None

    def clear_all(self) -> None:
        names_to_clear = list(
            dict.fromkeys(
                [str(name) for name in self._dataset_views.keys()]
                + [str(name) for name in self._dataset_panel_map.keys()]
            )
        )
        for name in names_to_clear:
            self._plot_tabs.remove_dataset_tab(str(name))
        self._dataset_views.clear()
        self._fit_render_projection = None
        self._display_name_by_id.clear()
        self._dataset_panel_map.clear()
        self._refresh_dataset_grid()

    def clear_fit_result_views(
        self,
        dataset_ids: Sequence[str],
    ) -> None:
        updated = False
        clear_ids = {
            str(dataset_id or "").strip()
            for dataset_id in dataset_ids
            if str(dataset_id or "").strip()
        }
        projection = self._fit_render_projection
        if projection is not None and clear_ids.intersection(projection.dataset_ids):
            clear_ids.update(str(dataset_id) for dataset_id in projection.dataset_ids)
            self._fit_render_projection = None
            updated = True
        for dataset_id in clear_ids:
            dataset_key = str(dataset_id or "").strip()
            if not dataset_key:
                continue
            dataset_name = self._display_name_by_id.get(dataset_key)
            if dataset_name is None:
                continue
            ds_entry = self._dataset_views.get(dataset_name)
            if ds_entry is None:
                continue
            self._update_dataset_plot(dataset_name)
            updated = True

        if updated:
            self._refresh_dataset_grid()

    def sync_fit_render_projection(
        self,
        projection: FitRenderProjection,
        *,
        dataset_ids: Optional[Sequence[str]] = None,
    ) -> None:
        if not isinstance(projection, FitRenderProjection):
            return
        previous_projection = self._fit_render_projection
        previous_dataset_ids = (
            tuple(str(dataset_id) for dataset_id in previous_projection.dataset_ids)
            if isinstance(previous_projection, FitRenderProjection)
            else ()
        )
        self._fit_render_projection = projection
        ordered_dataset_ids = tuple(dataset_ids if dataset_ids is not None else projection.dataset_ids)
        active_dataset_ids = {str(dataset_id) for dataset_id in projection.dataset_ids}
        dropped_dataset_ids = tuple(
            dataset_id for dataset_id in previous_dataset_ids if dataset_id not in active_dataset_ids
        )
        refresh_dataset_ids = tuple(
            dict.fromkeys(
                str(dataset_id)
                for dataset_id in (*ordered_dataset_ids, *dropped_dataset_ids)
                if str(dataset_id or "").strip()
            )
        )
        updated = False
        for raw_dataset_id in refresh_dataset_ids:
            dataset_id = str(raw_dataset_id or "").strip()
            if not dataset_id:
                continue
            dataset_name = self._display_name_by_id.get(dataset_id)
            if dataset_name is None:
                continue
            ds_entry = self._dataset_views.get(dataset_name)
            if ds_entry is None:
                continue
            dataset_projection = projection.datasets.get(dataset_id)
            if dataset_projection is None or dataset_projection.status != "ok":
                self._update_dataset_plot(dataset_name)
                updated = True
                continue
            self._update_dataset_plot(dataset_name)
            updated = True

        if updated:
            self._refresh_dataset_grid()

    def _prepare_dataset_entry(self, record: DatasetRecord) -> Dict[str, Any]:
        payload = record.payload
        species = payload.get("species", {})
        if not species:
            raise DatasetOwnerError("Dataset contains no numeric species columns.")

        t = np.asarray(payload["t"])
        all_species_data = {sp_name: np.asarray(sp_data) for sp_name, sp_data in species.items()}
        default_species = sorted(species.keys())[0]

        return {
            "name": str(record.display_name),
            "dataset_id": str(record.dataset_id),
            "t": t,
            "species": all_species_data,
            "series_name": default_species,
            "metadata": dict(payload.get("metadata") or {}),
        }

    def _clear_projection_if_contains(self, dataset_ids: Sequence[str]) -> bool:
        projection = self._fit_render_projection
        if projection is None:
            return False
        normalized_ids = {str(dataset_id).strip() for dataset_id in dataset_ids if str(dataset_id).strip()}
        if not normalized_ids.intersection(projection.dataset_ids):
            return False
        self._fit_render_projection = None
        return True

    def _projection_for_entry(self, entry: Dict[str, Any]) -> Optional[FitRenderDatasetProjection]:
        dataset_id = str(entry.get("dataset_id") or "").strip()
        if not dataset_id:
            return None
        projection = self._fit_render_projection
        if not isinstance(projection, FitRenderProjection):
            return None
        dataset_projection = projection.datasets.get(dataset_id)
        if not isinstance(dataset_projection, FitRenderDatasetProjection):
            return None
        if dataset_projection.status != "ok":
            return None
        return dataset_projection

    def _update_dataset_plot(self, name: str) -> None:
        entry = self._dataset_views.get(name)
        if not entry:
            return

        payload = self._dataset_render_payload(entry)
        species_map = payload["all_species"]
        if not isinstance(species_map, dict) or not species_map:
            return

        panel = self._plot_tabs.sync_dataset_tab(
            name,
            t=payload["data_x"],
            data_y=payload["data_y"],
            xlabel=payload["x_label"],
            ylabel=payload["current_species"],
            all_species=species_map,
            chi_squared=payload["chi_squared"],
            r_squared=payload["r_squared"],
            fit_render_projection=payload["fit_render_projection"],
        )
        if self._dataset_panel_map.get(name) is not panel:
            self._dataset_panel_map[name] = panel
            panel.simulateRequested.connect(lambda: self._on_dataset_simulate_requested(name))

    def _dataset_render_payload(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        projection = self._projection_for_entry(entry)
        base_species = entry.get("species") if isinstance(entry.get("species"), dict) else {}
        if projection is not None:
            projection_species = {
                str(species): np.asarray(values, dtype=float)
                for species, values in dict(projection.observed_series).items()
                if str(species)
            }
            species_map = projection_species or base_species
            data_x = projection.observed_x
            x_label = projection.observed_x_label
            stats = dict(projection.dataset_stats)
        else:
            species_map = base_species
            data_x = entry["t"]
            x_label = "Time"
            stats = {}

        current_species = str(entry.get("series_name") or "")
        if current_species not in species_map and species_map:
            current_species = sorted(species_map.keys())[0]
            entry["series_name"] = current_species
        data_y = species_map[current_species] if current_species in species_map else np.asarray([], dtype=float)
        return {
            "data_x": data_x,
            "x_label": x_label,
            "data_y": data_y,
            "all_species": species_map,
            "current_species": current_species,
            "fit_render_projection": projection,
            "chi_squared": stats.get("chi_squared"),
            "r_squared": stats.get("r_squared"),
        }

    def _on_dataset_simulate_requested(self, dataset_name: str) -> None:
        panel = self._dataset_panel_map.get(dataset_name)
        self._dataset_tab_simulation_owner.handle_simulate_requested(
            dataset_name=str(dataset_name),
            panel=panel,
        )

    def _refresh_dataset_grid(self) -> None:
        dataset_entries = []
        for entry in self._dataset_views.values():
            payload = self._dataset_render_payload(entry)
            dataset_entries.append(
                {
                    "name": entry["name"],
                    "t": entry["t"],
                    "data_x": payload["data_x"],
                    "x_label": payload["x_label"],
                    "data_y": payload["data_y"],
                    "fit_render_projection": payload["fit_render_projection"],
                    "chi_squared": payload["chi_squared"],
                    "r_squared": payload["r_squared"],
                    "all_species": payload["all_species"],
                    "current_species": payload["current_species"],
                }
            )
        self._plot_tabs.sync_dataset_grid(dataset_entries)
