"""Mechanism parameter scan owner for fitting workflows."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from kindred.core.simulator.dsl import extract_parameters_from_dsl
from kindred.gui.controllers.dataset_errors import DatasetOwnerError
from kindred.gui.project_schema import PROJECT_DEFAULTS


class MechanismParameterScanOwner:
    """Scan mechanism text for fittable parameters with cache ownership."""

    def __init__(self, *, solver_settings_getter: Callable[[], Dict[str, Any]]) -> None:
        self._solver_settings_getter = solver_settings_getter
        self._param_scan_cache: Dict[str, List[Dict[str, Any]]] = {}

    def scan_mechanism_parameters(self, mechanism_text: str) -> List[Dict[str, Any]]:
        cleaned = mechanism_text.strip()
        if not cleaned:
            raise DatasetOwnerError("Mechanism text is empty.")

        cfg = self._solver_settings_getter()

        mechanism_hash = self._param_scan_cache_key(cleaned, cfg)
        if mechanism_hash in self._param_scan_cache:
            return self._param_scan_cache[mechanism_hash]

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
            apply_parameter_algebra_to_mechanism(cleaned, mechanism=mech, require_mutable=False)
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
            raise DatasetOwnerError(str(exc)) from exc
        except Exception:
            raise

        params: List[Dict[str, Any]] = []
        for definition in extract_parameters_from_dsl(cleaned):
            if getattr(definition, "editable", True) is False:
                continue
            name = definition.name
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
            raise DatasetOwnerError("No fittable parameters found in the mechanism text.")

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
    def _param_scan_cache_key(cleaned_mechanism: str, solver_cfg: Dict[str, Any] | None) -> str:
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
        return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()

    def _suggest_parameter_bounds(self, param_name: str, param_value: float) -> tuple:
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
