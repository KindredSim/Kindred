from __future__ import annotations

from datetime import datetime
import platform
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from kindred import __version__ as KINDRED_VERSION


class SimulationProvenanceOwner:
    """Owns last-run simulation provenance and CTC state for GUI consumers."""

    def __init__(
        self,
        *,
        dataset_snapshot_getter: Callable[[], Dict[str, Any]],
        fit_metadata_getter: Callable[[], Optional[Dict[str, Any]]],
    ) -> None:
        self._dataset_snapshot_getter = dataset_snapshot_getter
        self._fit_metadata_getter = fit_metadata_getter
        self._last_simulation_provenance: Dict[str, Any] = {}
        self._last_simulation_ctc: Dict[str, float] = {}

    @property
    def last_simulation_provenance(self) -> Dict[str, Any]:
        return dict(self._last_simulation_provenance)

    @property
    def last_simulation_ctc(self) -> Dict[str, float]:
        return dict(self._last_simulation_ctc)

    def snapshot_datasets(self) -> Dict[str, Any]:
        return dict(self._dataset_snapshot_getter() or {})

    def last_fit_metadata(self) -> Optional[Dict[str, Any]]:
        value = self._fit_metadata_getter()
        return dict(value) if isinstance(value, dict) else None

    def set_last_simulation_provenance(self, provenance: Dict[str, Any]) -> None:
        self._last_simulation_provenance = dict(provenance)

    def set_last_simulation_ctc(self, ctc: Dict[str, float]) -> None:
        self._last_simulation_ctc = {str(key): float(value) for key, value in (ctc or {}).items()}

    def update_display_transaction_provenance(
        self,
        *,
        display_transaction: Mapping[str, Any] | None,
        display_sets: Sequence[Mapping[str, Any]] | None,
    ) -> Dict[str, Any]:
        provenance = dict(self._last_simulation_provenance or {})
        if isinstance(display_transaction, Mapping):
            provenance["display_transaction"] = dict(display_transaction)
        else:
            provenance.pop("display_transaction", None)
        if display_sets:
            provenance["display_sets"] = [
                dict(item) for item in display_sets if isinstance(item, Mapping)
            ]
        else:
            provenance.pop("display_sets", None)
        self._last_simulation_provenance = provenance
        return dict(provenance)

    def publish_simulation_completion_provenance(
        self,
        *,
        mechanism_text: str,
        solver_method: str,
        solver_label: str,
        solver_warning: str | None,
        solver_config: Mapping[str, Any],
        temperature_K: float,
        temperature_source: str,
        energy_unit: str | None,
        energy_mode: bool,
        simulation_time: float | str,
        num_points_requested: int,
        species_names: Sequence[str],
        t: Any,
        series: Mapping[str, Any],
        algebra_scalars: Mapping[str, Any] | None = None,
        dataset_overlays: Any = None,
        display_transaction: Mapping[str, Any] | None = None,
        display_sets: Sequence[Mapping[str, Any]] | None = None,
        solver_provenance: Mapping[str, Any] | None = None,
        warnings: Sequence[Mapping[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        species_list = [str(name) for name in species_names]
        provenance: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "kindred_version": KINDRED_VERSION,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "mechanism_dsl": str(mechanism_text),
            "solver": str(solver_method),
            "solver_label": str(solver_label),
            "solver_warning": str(solver_warning) if solver_warning else None,
            "rtol": solver_config.get("rtol", 1e-6),
            "atol": solver_config.get("atol", 1e-12),
            "temperature_K": float(temperature_K),
            "temperature_source": str(temperature_source),
            "energy_unit": energy_unit,
            "energy_mode": bool(energy_mode),
            "simulation_time": simulation_time,
            "num_points_requested": int(num_points_requested),
            "num_species": len(species_list),
            "num_points": len(t),
            "species_names": species_list,
            "datasets": self.snapshot_datasets(),
        }
        if algebra_scalars:
            provenance["algebra_scalars"] = dict(algebra_scalars)
        if dataset_overlays is not None:
            provenance["dataset_overlays"] = dataset_overlays
        if isinstance(display_transaction, Mapping):
            provenance["display_transaction"] = dict(display_transaction)
        if display_sets:
            provenance["display_sets"] = [
                dict(item) for item in display_sets if isinstance(item, Mapping)
            ]
        if isinstance(solver_provenance, Mapping) and solver_provenance:
            provenance["solver_provenance"] = dict(solver_provenance)
            symbolic_identity = solver_provenance.get("symbolic_jacobian_identity")
            if isinstance(symbolic_identity, Mapping):
                provenance["symbolic_jacobian_identity"] = dict(symbolic_identity)
            symbolic_status = solver_provenance.get("symbolic_jacobian_status")
            if isinstance(symbolic_status, Mapping):
                provenance["symbolic_jacobian_status"] = dict(symbolic_status)
            wegscheider_identity = solver_provenance.get("symbolic_wegscheider_identity")
            if isinstance(wegscheider_identity, Mapping):
                provenance["symbolic_wegscheider_identity"] = dict(wegscheider_identity)
            if "symbolic_jacobian" in solver_provenance:
                provenance["symbolic_jacobian"] = bool(solver_provenance.get("symbolic_jacobian"))
        if warnings:
            provenance["warnings"] = [dict(item) for item in warnings if isinstance(item, Mapping)]
        fit_meta = self.last_fit_metadata()
        if fit_meta:
            provenance["fit"] = fit_meta

        ctc_values: Dict[str, float] = {}
        ctc_metadata: Dict[str, Any] = {}
        for species_name, conc_array in series.items():
            values = np.asarray(conc_array, dtype=float)
            final_conc = values[-1]
            max_conc = np.max(np.abs(values))
            threshold = max(1e-10, 0.01 * max_conc)

            if abs(final_conc) < threshold:
                ctc_value, method, is_uniform, eps_used, tail_used = self.integrate_ctc(
                    t,
                    values,
                    uniformity_eps=1e-6,
                    tail_strategy="38",
                )
            else:
                deviation = np.abs(values - final_conc)
                ctc_value, method, is_uniform, eps_used, tail_used = self.integrate_ctc(
                    t,
                    deviation,
                    uniformity_eps=1e-6,
                    tail_strategy="38",
                )

            ctc_values[str(species_name)] = float(ctc_value)
            ctc_metadata = {
                "integration_method": method,
                "uniform_grid_detected": is_uniform,
                "uniformity_eps": eps_used,
                "tail_strategy": tail_used,
            }

        self.set_last_simulation_ctc(ctc_values)
        if ctc_metadata:
            provenance["ctc"] = ctc_metadata
        self.set_last_simulation_provenance(provenance)
        return provenance

    def integrate_ctc(
        self,
        t: Any,
        y: Any,
        *,
        uniformity_eps: float,
        tail_strategy: str,
    ) -> Tuple[float, str, bool, float, str]:
        from kindred.core.results import integrate_ctc as _integrate_ctc

        return _integrate_ctc(
            t,
            y,
            uniformity_eps=float(uniformity_eps),
            tail_strategy=str(tail_strategy),
        )
