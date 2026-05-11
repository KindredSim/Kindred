"""Coordinators for dataset visualization, parameter scanning, and fitting."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from kindred.core.simulator.dsl import extract_parameters_from_dsl
from kindred.gui.project_schema import PROJECT_DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class FitJob:
    """Snapshot of a dataset fit request."""

    dataset_name: str
    dataset: Dict[str, Any]
    param_names: List[str]
    objective: Callable[[np.ndarray], np.ndarray]
    t_exp: np.ndarray
    target_species: str


@dataclass
class DatasetFitSettings:
    """
    Per-dataset fitting settings.

    Attributes
    ----------
    weight : float
        Dataset weight/normalisation factor.
    initial_conditions : dict
        Initial concentration overrides {species: value}.
    fit_flags : dict
        Flags indicating whether an initial condition should be fitted.
    log10_flags : dict
        Flags indicating whether an initial condition is fitted in log10-space.
    bounds : dict
        Bounds for fittable initial conditions {species: (min, max)}.
    batch_set : str | None
        Name of the Batch Initial Conditions set used as the default initial
        conditions source for this dataset (selected at fit start).
    batch_set_id : str | None
        Stable identifier of the mapped Batch Initial Conditions set.
    """

    weight: float = 1.0
    initial_conditions: Dict[str, float] = field(default_factory=dict)
    fit_flags: Dict[str, bool] = field(default_factory=dict)
    log10_flags: Dict[str, bool] = field(default_factory=dict)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    batch_set: Optional[str] = None
    batch_set_id: Optional[str] = None

    def ensure_species(self, species_names: Sequence[str], defaults: Optional[Dict[str, float]] = None) -> None:
        """Ensure all species have entries for initials/flags/bounds."""
        defaults = defaults or {}
        for name in species_names:
            if name not in self.initial_conditions:
                self.initial_conditions[name] = float(defaults.get(name, 0.0))
            self.fit_flags.setdefault(name, False)
            self.log10_flags.setdefault(name, False)
            self.bounds.setdefault(name, (0.0, max(10.0, self.initial_conditions[name] * 10 or 10.0)))


class DatasetManagerError(RuntimeError):
    """Raised when dataset orchestration encounters an unrecoverable issue."""


class DatasetManager:
    """
    Coordinates dataset bookkeeping, parameter scanning, and fit execution.

    Parameters
    ----------
    plot_tabs : PlotTabsWidget
        Plot container used for dataset tabs/grid.
    dataset_resolver : callable
        Function that takes a dataset name and returns the raw dataset payload.
    mechanism_getter : callable, optional
        Function that returns the current mechanism DSL text from main window.
    simulation_runner : callable, optional
        Function that runs simulation and returns (t, species_dict).
    """

    def __init__(
        self,
        plot_tabs,
        dataset_resolver: Callable[[str], Optional[Dict[str, Any]]],
        mechanism_getter: Optional[Callable[[], str]] = None,
        simulation_runner: Optional[Callable[[str], Dict[str, Any]]] = None,
        solver_settings_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        self._plot_tabs = plot_tabs
        self._dataset_resolver = dataset_resolver
        self._mechanism_getter = mechanism_getter
        self._simulation_runner = simulation_runner
        self._solver_settings_getter = solver_settings_getter

        self._dataset_views: Dict[str, Dict[str, Any]] = {}
        self._param_scan_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._dataset_panel_map: Dict[str, object] = {}  # name -> panel
        self._fit_settings: Dict[str, DatasetFitSettings] = {}

    # ------------------------------------------------------------------
    # Dataset lifecycle
    # ------------------------------------------------------------------
    def register_dataset(self, name: str, data: Dict[str, Any]) -> None:
        """Track a newly loaded dataset and update views."""
        entry = self._prepare_dataset_entry(name, data)
        self._dataset_views[name] = entry
        self._fit_settings.setdefault(name, DatasetFitSettings())
        self._update_dataset_plot(name)
        self._refresh_dataset_grid()

    def remove_dataset(self, name: str) -> bool:
        """Remove dataset bookkeeping and views."""
        removed = self._dataset_views.pop(name, None)
        self._fit_settings.pop(name, None)
        self._dataset_panel_map.pop(name, None)
        self._plot_tabs.remove_dataset_tab(name)
        self._refresh_dataset_grid()
        return removed is not None

    def clear_all_datasets(self) -> None:
        """Clear all dataset bookkeeping and close all dataset-owned views."""
        names_to_clear = list(
            dict.fromkeys(
                [str(name) for name in self._dataset_views.keys()]
                + [str(name) for name in self._dataset_panel_map.keys()]
            )
        )
        for name in names_to_clear:
            self._plot_tabs.remove_dataset_tab(str(name))
        self._dataset_views.clear()
        self._dataset_panel_map.clear()
        self._fit_settings.clear()
        self._refresh_dataset_grid()

    def get_fit_settings(self, name: str) -> DatasetFitSettings:
        """Return fitting settings for a dataset, creating defaults if needed."""
        if name not in self._fit_settings:
            self._fit_settings[name] = DatasetFitSettings()
        return self._fit_settings[name]

    def update_fit_settings(self, name: str, settings: DatasetFitSettings) -> None:
        """Persist new fitting settings for a dataset."""
        self._fit_settings[name] = settings

    def iter_fit_settings(self) -> List[Tuple[str, DatasetFitSettings]]:
        """Return a stable snapshot of (dataset_name, settings) pairs."""
        return list(self._fit_settings.items())

    def sync_fit_result_views(
        self,
        model_series: Dict[str, Dict[str, np.ndarray]],
        *,
        dataset_stats: Optional[Dict[str, Dict[str, float]]] = None,
        dataset_ids: Optional[Sequence[str]] = None,
    ) -> None:
        """Apply fit result series/stats to dataset views owned by this manager."""
        ordered_dataset_ids = dataset_ids if dataset_ids is not None else model_series.keys()
        updated = False
        for dataset_id in ordered_dataset_ids:
            dataset_name = str(dataset_id)
            ds_entry = self._dataset_views.get(dataset_name)
            model_map = model_series.get(dataset_id, {})
            if not model_map:
                if ds_entry is None:
                    continue
                self._clear_fit_result_view(ds_entry)
                self._update_dataset_plot(dataset_name)
                updated = True
                continue
            if ds_entry is None:
                ds_entry = self._ensure_dataset_view_entry(dataset_name)
                if ds_entry is None:
                    continue
            self._apply_fit_result_view(
                ds_entry,
                model_map,
                dataset_stats=(dataset_stats or {}).get(dataset_name),
            )
            self._update_dataset_plot(dataset_name)
            updated = True

        if updated:
            self._refresh_dataset_grid()

    @staticmethod
    def _clear_fit_result_view(entry: Dict[str, Any]) -> None:
        entry["visible_species"] = None
        entry["model_x"] = None
        entry["model_y"] = None
        entry["model_series"] = None
        entry["chi_squared"] = None
        entry["r_squared"] = None

    def datasets_mapped_to_batch_sets(
        self, *, set_ids: Sequence[str], set_names: Sequence[str]
    ) -> List[str]:
        """Return dataset names mapped to any of the given batch set identifiers."""
        id_targets = {str(v) for v in (set_ids or []) if str(v)}
        name_targets = {str(v) for v in (set_names or []) if str(v)}
        affected: List[str] = []
        for dataset_name, settings in self.iter_fit_settings():
            mapped_id = str(getattr(settings, "batch_set_id", "") or "").strip()
            mapped_name = str(getattr(settings, "batch_set", "") or "").strip()
            if mapped_id and mapped_id in id_targets:
                affected.append(str(dataset_name))
                continue
            if mapped_name and mapped_name in name_targets:
                affected.append(str(dataset_name))
        return affected

    def unmap_batch_sets(self, *, set_ids: Sequence[str], set_names: Sequence[str]) -> List[str]:
        """Clear batch set mappings for datasets mapped to the given identifiers."""
        id_targets = {str(v) for v in (set_ids or []) if str(v)}
        name_targets = {str(v) for v in (set_names or []) if str(v)}
        affected: List[str] = []
        for dataset_name, settings in self.iter_fit_settings():
            mapped_id = str(getattr(settings, "batch_set_id", "") or "").strip()
            mapped_name = str(getattr(settings, "batch_set", "") or "").strip()
            if (mapped_id and mapped_id in id_targets) or (mapped_name and mapped_name in name_targets):
                settings.batch_set = None
                settings.batch_set_id = None
                affected.append(str(dataset_name))
        return affected

    def scan_mechanism_parameters(self, mechanism_text: str) -> List[Dict[str, Any]]:
        """
        Scan mechanism text for parameters with caching.

        Raises
        ------
        DatasetManagerError
            If no fittable parameters are found.
        """
        cleaned = mechanism_text.strip()
        if not cleaned:
            raise DatasetManagerError("Mechanism text is empty.")

        try:
            cfg = self._solver_settings_getter() if callable(getattr(self, "_solver_settings_getter", None)) else {}
        except Exception:
            cfg = {}

        mechanism_hash = self._param_scan_cache_key(cleaned, cfg)
        if mechanism_hash in self._param_scan_cache:
            return self._param_scan_cache[mechanism_hash]

        # Include parameter-algebra base params and exclude constrained targets.
        constrained: set[str] = set()
        scalar_params: Dict[str, float] = {}
        scalar_info: Dict[str, Dict[str, object]] = {}
        cfg_dict: Dict[str, Any] = cfg if isinstance(cfg, dict) else {}
        try:
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism
            from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism
            from kindred.core.simulator.wegscheider_symbolic import UnresolvedWegscheiderCyclicityError

            mech = parse_dsl_to_mechanism(cleaned, initials={})
            if isinstance(getattr(mech, "metadata", None), dict):
                mech.metadata["wegscheider_cyclicity_enabled"] = bool(
                    cfg_dict.get(
                        "wegscheider_cyclicity_enabled",
                        PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"],
                    )
                )
            _ = apply_parameter_algebra_to_mechanism(cleaned, mechanism=mech, require_mutable=False)
            constrained_meta = (getattr(mech, "metadata", {}) or {}).get("constrained_params") or {}
            if isinstance(constrained_meta, dict):
                constrained = {str(k) for k in constrained_meta.keys()}
            scalar_params_meta = (getattr(mech, "metadata", {}) or {}).get("scalar_params") or {}
            if isinstance(scalar_params_meta, dict):
                scalar_params = {str(k): float(v) for k, v in scalar_params_meta.items()}
            scalar_info_meta = (getattr(mech, "metadata", {}) or {}).get("scalar_param_info") or {}
            if isinstance(scalar_info_meta, dict):
                scalar_info = {str(k): dict(v) for k, v in scalar_info_meta.items() if isinstance(v, dict)}
        except UnresolvedWegscheiderCyclicityError as exc:
            raise DatasetManagerError(str(exc)) from exc
        except Exception:
            raise

        params: List[Dict[str, Any]] = []

        def _as_float(x: object) -> Optional[float]:
            try:
                return float(x()) if callable(x) else float(x)
            except (TypeError, ValueError, OverflowError):
                return None

        def _as_int(x: object) -> Optional[int]:
            try:
                return int(x)
            except (TypeError, ValueError, OverflowError):
                return None

        # Canonical step-index parameters from the mechanism step map.
        step_map = (getattr(mech, "metadata", {}) or {}).get("step_index_map") or []
        if isinstance(step_map, list) and step_map:
            rxns = list(getattr(mech, "reactions", []) or [])
            eqs = list(getattr(mech, "equilibria", []) or [])
            for entry in step_map:
                if not isinstance(entry, dict):
                    continue
                kind = str(entry.get("kind") or "")
                n = _as_int(entry.get("step_index"))
                if n is None:
                    continue
                context = str(entry.get("context") or "")
                has_Keq_param = bool(entry.get("has_Keq_param"))
                raw_derive_rate = str(entry.get("derive_rate") or "")
                derive_rate = raw_derive_rate if raw_derive_rate in {"kf", "kr"} else ("kr" if has_Keq_param else "")

                if kind == "reaction":
                    name = f"k{n}"
                    if name in constrained:
                        continue
                    idx = _as_int(entry.get("reaction_index", -1))
                    if idx is None:
                        continue
                    if not (0 <= idx < len(rxns)):
                        continue
                    value = _as_float(getattr(rxns[idx], "rate", None))
                    if value is None:
                        continue
                    min_bound, max_bound = self._suggest_parameter_bounds(name, value)
                    params.append(
                        {
                            "name": name,
                            "value": value,
                            "min": min_bound,
                            "max": max_bound,
                            "context": context,
                            "source": "Rate constant",
                        }
                    )
                elif kind == "equilibrium":
                    idx = _as_int(entry.get("equilibrium_index", -1))
                    if idx is None:
                        continue
                    if not (0 <= idx < len(eqs)):
                        continue
                    eq = eqs[idx]

                    # kfN
                    kf_name = f"kf{n}"
                    if kf_name not in constrained and derive_rate != "kf":
                        val = _as_float(getattr(eq, "kf", None))
                        if val is not None:
                            mn, mx = self._suggest_parameter_bounds(kf_name, val)
                            params.append(
                                {
                                    "name": kf_name,
                                    "value": val,
                                    "min": mn,
                                    "max": mx,
                                    "context": context,
                                    "source": "Forward rate",
                                }
                            )

                    # krN
                    kr_name = f"kr{n}"
                    if kr_name not in constrained and derive_rate != "kr":
                        val = _as_float(getattr(eq, "kr", None))
                        if val is not None:
                            mn, mx = self._suggest_parameter_bounds(kr_name, val)
                            params.append(
                                {
                                    "name": kr_name,
                                    "value": val,
                                    "min": mn,
                                    "max": mx,
                                    "context": context,
                                    "source": "Reverse rate",
                                }
                            )

                    # Show KeqN only when the source explicitly represents it or algebra uses it.
                    if has_Keq_param:
                        Keq_name = f"Keq{n}"
                        if Keq_name not in constrained:
                            meta = getattr(eq, "metadata", {}) or {}
                            val = _as_float(meta.get("Keq_input"))
                            if val is not None:
                                mn, mx = self._suggest_parameter_bounds(Keq_name, val)
                                params.append(
                                    {
                                        "name": Keq_name,
                                        "value": val,
                                        "min": mn,
                                        "max": mx,
                                        "context": context,
                                        "source": "Equilibrium constant",
                                    }
                                )
        else:
            # Fallback: use DSL extraction when structured parameter metadata is unavailable.
            parameter_defs = extract_parameters_from_dsl(cleaned)
            for definition in parameter_defs:
                name = definition.name
                step_index = getattr(definition, "step_index", None)
                match = re.match(r"^(kf|kr|Keq|k)\d*$", str(name))
                if match and step_index is not None:
                    family = match.group(1)
                    context_str = str(definition.context)
                    is_equilibrium = "<->" in context_str or "<=>" in context_str
                    if family == "k" and is_equilibrium:
                        family = "kf"
                    elif family == "kf" and not is_equilibrium:
                        family = "k"
                    elif family in {"kr", "Keq"} and not is_equilibrium:
                        family = "k"
                    name = f"{family}{int(step_index)}"
                if str(name) in constrained:
                    continue
                min_bound, max_bound = self._suggest_parameter_bounds(name, definition.value)
                params.append(
                    {
                        "name": name,
                        "value": definition.value,
                        "min": min_bound,
                        "max": max_bound,
                        "context": definition.context,
                        "source": definition.source,
                    }
                )

        # Add scalar base parameters (editable, dimensionless by default).
        for name, value in scalar_params.items():
            info = scalar_info.get(name) or {}
            if info.get("editable") is False:
                continue
            min_bound, max_bound = self._suggest_parameter_bounds(name, value)
            params.append(
                {
                    "name": name,
                    "value": value,
                    "min": min_bound,
                    "max": max_bound,
                    "context": "Algebra",
                    "source": "Scalar parameter",
                }
            )

        if not params:
            raise DatasetManagerError("No fittable parameters found in the mechanism text.")

        seen = set()
        unique_params: List[Dict[str, Any]] = []
        for param in params:
            if param["name"] in seen:
                continue
            seen.add(param["name"])
            unique_params.append(param)

        self._param_scan_cache[mechanism_hash] = unique_params
        return unique_params

    @staticmethod
    def _param_scan_cache_key(cleaned_mechanism: str, solver_cfg: Optional[Dict[str, Any]]) -> str:
        import hashlib
        import json

        cfg = solver_cfg or {}
        solver_key = {
            "wegscheider_cyclicity_enabled": bool(
                cfg.get(
                    "wegscheider_cyclicity_enabled",
                    PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"],
                )
            ),
        }
        payload = json.dumps(
            {"mechanism": str(cleaned_mechanism), "solver": solver_key},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        # MD5 here is used only as a deterministic cache key (not for security).
        return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()

    # ------------------------------------------------------------------
    # Fitting orchestration
    # ------------------------------------------------------------------
    def prepare_fit_job(
        self,
        config: Dict[str, Any],
        mechanism_text: str,
        state_network_text: str,
        temperature_K: float,
        preferred_target_species: Optional[str] = None,
    ) -> FitJob:
        """
        Prepare a fit job without executing it.

        Parameters
        ----------
        config : dict
            Fitting configuration including dataset name, parameters, bounds, method, etc.
        mechanism_text : str
            DSL mechanism text
        state_network_text : str
            State network DSL text (optional)
        temperature_K : float
            Temperature in Kelvin
        preferred_target_species : str, optional
            Species name to use for fitting. If None, will attempt to infer from
            visible series in the plot or fall back to first species with a warning.

        Returns
        -------
        FitJob
            Prepared fit job containing objective function and metadata.
        """
        dataset_name = config["dataset"]
        dataset = self._dataset_resolver(dataset_name)
        if not dataset:
            raise DatasetManagerError(f"Dataset '{dataset_name}' not found.")

        settings = self.get_fit_settings(dataset_name)
        species = dataset.get("species") or {}
        if not species:
            raise DatasetManagerError("Dataset has no species data.")

        # Determine target species with explicit fallback logic
        target_species = None

        # First priority: use preferred_target_species if provided
        if preferred_target_species:
            if preferred_target_species in species:
                target_species = preferred_target_species
                logger.info(f"Using user-specified target species: {target_species}")
            else:
                logger.warning(
                    f"Preferred target species '{preferred_target_species}' not found in dataset. "
                    f"Available species: {list(species.keys())}"
                )

        # Fallback: use first species key with a warning
        if target_species is None:
            target_species = list(species.keys())[0]
            logger.warning(
                f"No target species specified or selection unavailable. "
                f"Defaulting to first species: '{target_species}'. "
                f"Consider explicitly selecting a target species for fitting."
            )

        logger.info(f"Fitting target species: {target_species}")

        t_exp = np.asarray(dataset["t"])
        y_exp = species[target_species]

        from kindred.core.fitting_objective import build_fitting_objective

        param_names = list(config["parameters"].keys())
        mechanism_payload = mechanism_text

        def _has_state_network_header(text: str) -> bool:
            return bool(re.search(r"(?im)^\s*#\s*state\s+network\b", text or ""))

        state_network_clean = str(state_network_text or "").strip()
        if state_network_clean and not _has_state_network_header(mechanism_payload):
            mechanism_payload += "\n\n# State Network\n" + state_network_clean

        solver_settings: Dict[str, Any] = {}
        if callable(getattr(self, "_solver_settings_getter", None)):
            try:
                solver_settings = dict(self._solver_settings_getter() or {})
            except Exception:
                solver_settings = {}
        from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name

        solver_label = str(solver_settings.get("solver") or DEFAULT_SOLVER_NAME).strip() or DEFAULT_SOLVER_NAME
        solver_name, solver_warning = normalize_solver_name(solver_label)
        if solver_warning:
            logger.warning("Solver normalization: %s (requested=%r)", solver_warning, solver_label)
        rtol = float(solver_settings.get("rtol") or 1e-6)
        atol = float(solver_settings.get("atol") or 1e-12)
        wegscheider_enabled = bool(
            solver_settings.get(
                "wegscheider_cyclicity_enabled",
                PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"],
            )
        )
        solver_settings.setdefault("solver_label", str(solver_label))
        solver_settings["solver"] = str(solver_name)
        solver_settings["solver_warning"] = str(solver_warning) if solver_warning else None
        solver_settings["rtol"] = float(rtol)
        solver_settings["atol"] = float(atol)
        solver_settings["wegscheider_cyclicity_enabled"] = bool(wegscheider_enabled)

        objective = build_fitting_objective(
            mechanism_text=mechanism_payload,
            param_names=param_names,
            t_exp=t_exp,
            y_exp=y_exp,
            target_species=target_species,
            temperature_K=temperature_K,
            initials=dict(settings.initial_conditions),
            solver=solver_name,
            rtol=rtol,
            atol=atol,
            wegscheider_cyclicity_enabled=wegscheider_enabled,
        )

        return FitJob(
            dataset_name=dataset_name,
            dataset=dataset,
            param_names=param_names,
            objective=objective,
            t_exp=t_exp,
            target_species=target_species,
        )

    def finalize_fit_job(self, job: FitJob, result: Any) -> None:
        """Update dataset visuals using the finished fit result."""
        model_y = self._rebuild_fit_series(job.objective, job.param_names, result)

        entry = self._dataset_views.get(job.dataset_name)
        if entry is None:
            entry = self._prepare_dataset_entry(job.dataset_name, job.dataset)
            self._dataset_views[job.dataset_name] = entry

        if job.target_species in entry.get("species", {}):
            entry["series_name"] = job.target_species
            entry["data_y"] = entry["species"][job.target_species]

        entry["model_x"] = job.t_exp
        entry["model_y"] = model_y
        entry["model_series"] = {job.target_species: model_y} if model_y is not None else None
        entry["chi_squared"] = self._result_value(result, "chi_squared")
        entry["r_squared"] = self._result_value(result, "r_squared")

        self._update_dataset_plot(job.dataset_name)
        self._refresh_dataset_grid()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _rebuild_fit_series(self, objective, param_names: Sequence[str], result) -> Optional[np.ndarray]:
        """Regenerate fitted model series for visualization."""
        try:
            parameters = self._result_value(result, "parameters") or {}
            opt_vector = np.array([parameters[name] for name in param_names])
            residuals_opt = objective(opt_vector)
            model_y = getattr(objective, "last_model", None)
            if model_y is None and hasattr(objective, "y_exp"):
                model_y = objective.y_exp + residuals_opt
            if model_y is not None:
                return np.asarray(model_y)
        except Exception as exc:  # pragma: no cover - visualization best effort
            logger.warning("Failed to rebuild fitted series: %s", exc)
        return None

    def _prepare_dataset_entry(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dataset payload for visualization."""
        species = data.get("species", {})
        if not species:
            raise DatasetManagerError("Dataset contains no numeric species columns.")

        # Store ALL species data, not just the first one
        t = np.asarray(data["t"])

        # Convert all species to numpy arrays
        all_species_data = {sp_name: np.asarray(sp_data) for sp_name, sp_data in species.items()}

        # Default to first species alphabetically for initial display
        default_species = sorted(species.keys())[0]

        return {
            "name": name,
            "t": t,
            "species": all_species_data,  # Store ALL species
            "visible_species": None,  # Optional display filter {name: y_array}
            "series_name": default_species,  # Default display species
            "data_y": all_species_data[default_species],  # Default data
            "model_x": None,
            "model_y": None,
            "model_series": None,  # Optional multi-series model overlay {name: y_model}
            "chi_squared": None,
            "r_squared": None,
        }

    def _update_dataset_plot(self, name: str) -> None:
        entry = self._dataset_views.get(name)
        if not entry:
            return

        species_map = entry.get("visible_species") or entry.get("species") or {}
        if not isinstance(species_map, dict) or not species_map:
            return
        series_name = entry.get("series_name")
        if series_name not in species_map:
            series_name = sorted(species_map.keys())[0]
            entry["series_name"] = series_name
        entry["data_y"] = species_map[series_name]

        model_series = entry.get("model_series")
        model_x = entry.get("model_x")
        model_y = entry.get("model_y")
        panel = self._plot_tabs.sync_dataset_tab(
            name,
            t=entry["t"],
            data_y=entry["data_y"],
            model_x=model_x,
            model_y=model_y,
            ylabel=series_name,
            all_species=species_map,
            chi_squared=entry.get("chi_squared"),
            r_squared=entry.get("r_squared"),
            model_series=model_series if isinstance(model_series, dict) else None,
        )
        if self._dataset_panel_map.get(name) is not panel:
            self._dataset_panel_map[name] = panel
            panel.simulateRequested.connect(lambda: self._on_dataset_simulate_requested(name))

    def _ensure_dataset_view_entry(self, name: str) -> Optional[Dict[str, Any]]:
        entry = self._dataset_views.get(name)
        if entry is not None:
            return entry
        dataset = self._dataset_resolver(name)
        if dataset is None:
            return None
        entry = self._prepare_dataset_entry(name, dataset)
        self._dataset_views[name] = entry
        return entry

    def _apply_fit_result_view(
        self,
        entry: Dict[str, Any],
        model_map: Dict[str, np.ndarray],
        *,
        dataset_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        species_all = entry.get("species") if isinstance(entry.get("species"), dict) else {}
        visible_species = {sp: species_all[sp] for sp in model_map.keys() if sp in species_all}
        entry["visible_species"] = visible_species or None

        current_species = entry.get("series_name")
        if visible_species and current_species not in visible_species:
            current_species = sorted(visible_species.keys())[0]
            entry["series_name"] = current_species
        if visible_species and current_species in visible_species:
            entry["data_y"] = visible_species[current_species]

        entry["model_x"] = entry.get("t")
        entry["model_series"] = {sp: np.asarray(values) for sp, values in model_map.items()}
        entry["model_y"] = model_map.get(current_species)

        if isinstance(dataset_stats, dict):
            chi = dataset_stats.get("chi_squared")
            r2 = dataset_stats.get("r_squared")
            entry["chi_squared"] = float(chi) if chi is not None else entry.get("chi_squared")
            entry["r_squared"] = float(r2) if r2 is not None else entry.get("r_squared")

    def _on_dataset_simulate_requested(self, dataset_name: str) -> None:
        """
        Handle simulation request from a dataset panel using the main mechanism DSL.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Get the panel
        panel = self._dataset_panel_map.get(dataset_name)
        if panel is None:
            logger.error(f"Panel not found for dataset {dataset_name}")
            return

        # Get main mechanism text
        if self._mechanism_getter is None:
            panel.set_status("Error: No mechanism source configured")
            logger.error("No mechanism getter configured")
            return

        mechanism_text = self._mechanism_getter()
        if not mechanism_text or not mechanism_text.strip():
            panel.set_status("Error: No mechanism defined")
            logger.error("No mechanism text available")
            return

        # Dataset tabs share the main DSL-defined initial conditions.
        full_dsl = mechanism_text

        logger.info(f"Running simulation for dataset {dataset_name}")
        panel.set_status("Running simulation...")

        # Run simulation
        if self._simulation_runner is None:
            panel.set_status("Error: No simulation runner configured")
            logger.error("No simulation runner configured")
            return

        try:
            result = self._simulation_runner(full_dsl)

            if result and 't' in result and 'species' in result:
                t = result['t']
                species_data = result['species']

                # Plot results on panel
                panel.plot_simulation_results(t, species_data)
                panel.set_status("Simulation complete")
                logger.info(f"Simulation complete for dataset {dataset_name}")
            else:
                panel.set_status("Error: Invalid simulation result")
                logger.error("Simulation returned invalid result")
        except Exception as e:
            panel.set_status(f"Error: {str(e)}")
            logger.error(f"Simulation failed for dataset {dataset_name}: {e}")

    def _refresh_dataset_grid(self) -> None:
        dataset_entries = []
        for entry in self._dataset_views.values():
            species_map = entry.get("visible_species") or entry.get("species") or {}
            dataset_entries.append(
                {
                    "name": entry["name"],
                    "t": entry["t"],
                    "data_y": entry["data_y"],
                    "model_x": entry.get("model_x"),
                    "model_y": entry.get("model_y"),
                    "model_series": entry.get("model_series") if isinstance(entry.get("model_series"), dict) else None,
                    "chi_squared": entry.get("chi_squared"),
                    "r_squared": entry.get("r_squared"),
                    "all_species": species_map if isinstance(species_map, dict) else None,
                    "current_species": entry.get("series_name"),
                }
            )
        self._plot_tabs.sync_dataset_grid(dataset_entries)

    def _suggest_parameter_bounds(self, param_name: str, param_value: float) -> tuple:
        """Suggest parameter bounds using heuristics."""
        if param_name.startswith("k") and param_name[0].lower() == "k":
            if param_value > 0:
                min_bound = max(param_value * 0.01, 1e-10)
                max_bound = min(param_value * 100, 1e10)
            else:
                min_bound = 1e-10
                max_bound = 1e10
        elif param_name.startswith("Keq") and param_name[0].isupper():
            if param_value > 0:
                min_bound = max(param_value * 0.001, 1e-6)
                max_bound = min(param_value * 1000, 1e6)
            else:
                min_bound = 1e-6
                max_bound = 1e6
        elif param_name.startswith("A") and len(param_name) <= 2:
            if param_value > 0:
                min_bound = max(param_value * 0.1, 1e6)
                max_bound = min(param_value * 10, 1e16)
            else:
                min_bound = 1e6
                max_bound = 1e16
        elif param_name.startswith("Ea"):
            min_bound = max(param_value * 0.5, 0)
            max_bound = min(param_value * 1.5, 200)
        elif param_name.startswith("dG"):
            min_bound = max(param_value * 2, -100)
            max_bound = min(param_value * 2, 100)
        elif param_name.startswith("dH"):
            min_bound = max(param_value * 2, -150)
            max_bound = min(param_value * 2, 150)
        elif param_name.startswith("dS"):
            min_bound = max(param_value * 2, -100)
            max_bound = min(param_value * 2, 100)
        else:
            if param_value > 0:
                min_bound = param_value * 0.1
                max_bound = param_value * 10
            else:
                min_bound = -10
                max_bound = 10

        return min_bound, max_bound

    def _result_value(self, result: Any, attr: str):
        """Safely extract fields from FitResult or dict payloads."""
        if isinstance(result, dict):
            return result.get(attr)
        return getattr(result, attr, None)
