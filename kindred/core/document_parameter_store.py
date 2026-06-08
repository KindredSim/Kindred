"""
Canonical parameter ownership for a Kindred document.

This is intentionally Qt-free so it can be exercised in unit tests and
shared by GUI-facing controllers without living in the GUI layer.

Holds:
- shared_params: document-level parameter values (shared by default)
- set_local_overrides: per-set parameter diffs (optionally set-local)
- runtime_parameter_values(set_id): composed values
- param_fingerprint(set_id): deterministic hash for cache identity
- schema_id: hash of schema text for cache identity

Concentrations remain per-set and are NOT managed here.
"""

from __future__ import annotations

import hashlib
import math
from types import MappingProxyType
from typing import Dict, Iterator, Mapping, Optional


class DocumentParameterStore:
    """
    Canonical parameter ownership for shared and per-set parameter values.

    This is the single source of truth for parameter state in the document.
    """

    __slots__ = ("_shared_params", "_set_local_overrides", "_schema_text", "_schema_id")

    def __init__(self) -> None:
        self._shared_params: Dict[str, float] = {}
        self._set_local_overrides: Dict[str, Dict[str, float]] = {}
        self._schema_text: str = ""
        self._schema_id: str = ""

    @staticmethod
    def _normalized_local_override_set_id(set_id: Optional[str]) -> Optional[str]:
        """Return a valid per-set override key, or None for the shared/no-set sentinel."""
        set_id_s = str(set_id or "").strip()
        return set_id_s or None

    # ------------------------------------------------------------------
    # Schema identity
    # ------------------------------------------------------------------

    @property
    def schema_text(self) -> str:
        return self._schema_text

    @property
    def schema_id(self) -> str:
        return self._schema_id

    def set_schema(self, text: str) -> None:
        """Update the canonical schema text and recompute the schema identity hash."""
        self._schema_text = str(text or "")
        self._schema_id = hashlib.sha256(self._schema_text.encode("utf-8")).hexdigest()

    def _schema_parameter_names(self) -> set[str]:
        text = str(self._schema_text or "").strip()
        if not text:
            return set()
        mechanism = self._schema_mechanism()
        if mechanism is None:
            return set()
        names: set[str] = set()
        try:
            from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

            names = {str(name) for name in build_namespace_from_mechanism(mechanism).canonical_names if str(name)}
        except Exception:
            names = set()
        try:
            scalar_params = (getattr(mechanism, "metadata", {}) or {}).get("scalar_params") or {}
            scalar_info = (getattr(mechanism, "metadata", {}) or {}).get("scalar_param_info") or {}
            if isinstance(scalar_params, Mapping):
                for name in scalar_params.keys():
                    name_s = str(name)
                    if not name_s:
                        continue
                    info = scalar_info.get(name_s) if isinstance(scalar_info, Mapping) else None
                    if isinstance(info, Mapping) and info.get("derived") is True:
                        continue
                    names.add(name_s)
        except Exception:
            pass
        try:
            from kindred.core.simulation_preparation import _available_energy_binding_names

            names.update(str(name) for name in _available_energy_binding_names(mechanism) if str(name))
        except Exception:
            pass
        return names

    def _parameter_name_allowed(self, name: str) -> bool:
        name_s = str(name or "").strip()
        if not name_s:
            return False
        allowed_names = self._schema_parameter_names()
        if allowed_names:
            return name_s in allowed_names
        return False

    def parameter_names(self) -> list[str]:
        return sorted(self._schema_parameter_names())

    def _schema_mechanism(self) -> object | None:
        text = str(self._schema_text or "").strip()
        if not text:
            return None
        try:
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism
            from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism

            mechanism = parse_dsl_to_mechanism(text, initials={})
            apply_parameter_algebra_to_mechanism(
                text,
                mechanism=mechanism,
                require_mutable=False,
            )
            return mechanism
        except Exception:
            return None

    def _equilibrium_conflict_message(self, values: Mapping[str, float]) -> str:
        mechanism = self._schema_mechanism()
        if mechanism is None:
            return ""
        try:
            from kindred.core.equilibrium_rate_authority import (
                effective_reverse_rate_from_keq,
                normalize_existing_equilibrium_rate_authority,
            )
            from kindred.core.simulator.step_indexing import get_step_index_map
            from kindred.core.validation import try_parse_callable_finite_float
        except Exception:
            return ""

        def _finite_value(raw: object) -> Optional[float]:
            parsed = self._finite_float(raw)
            if parsed is not None:
                return parsed
            try:
                parsed_callable, ok = try_parse_callable_finite_float(raw)
            except Exception:
                return None
            return float(parsed_callable) if ok and math.isfinite(float(parsed_callable)) else None

        requested = {str(name): float(value) for name, value in dict(values or {}).items()}
        for entry in get_step_index_map(mechanism):
            if str(entry.get("kind") or "") != "equilibrium":
                continue
            try:
                eq_idx = int(entry.get("equilibrium_index"))
            except (TypeError, ValueError):
                continue
            step_idx_raw = entry.get("step_index")
            if isinstance(step_idx_raw, int):
                step_idx = int(step_idx_raw)
            elif isinstance(step_idx_raw, str) and step_idx_raw.isdigit():
                step_idx = int(step_idx_raw)
            else:
                continue
            kf_name = f"kf{step_idx}"
            kr_name = f"kr{step_idx}"
            keq_name = f"Keq{step_idx}"
            if kr_name not in requested or keq_name not in requested:
                continue
            try:
                eq = list(getattr(mechanism, "equilibria", []) or [])[eq_idx]
            except Exception:
                continue
            kf_value = requested.get(kf_name)
            if kf_value is None:
                kf_value = _finite_value(getattr(eq, "kf", None))
            if kf_value is None:
                continue
            keq_value = float(requested[keq_name])
            if not (math.isfinite(keq_value) and keq_value > 0.0):
                return f"equilibrium parameter {keq_name} must be positive."
            try:
                authority = normalize_existing_equilibrium_rate_authority(eq)
                reverse_std_ratio = _finite_value(authority.reverse_std_ratio)
            except Exception:
                reverse_std_ratio = 1.0
            if reverse_std_ratio is None or reverse_std_ratio <= 0.0:
                return f"equilibrium {keq_name} reverse standard ratio must be positive."
            expected_kr = float(effective_reverse_rate_from_keq(kf_value, keq_value, reverse_std_ratio))
            actual_kr = float(requested[kr_name])
            if not math.isclose(actual_kr, expected_kr, rel_tol=1e-9, abs_tol=1e-12):
                return (
                    f"Conflicting equilibrium parameter values for {kr_name} and {keq_name}: "
                    f"{kr_name}={actual_kr:.17g} is inconsistent with "
                    f"{keq_name}={keq_value:.17g} and {kf_name}={float(kf_value):.17g}."
                )
        return ""

    @staticmethod
    def _finite_float(value: object) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(parsed):
            return None
        return float(parsed)

    def _canonical_parameter_values(self, values: Mapping[str, object]) -> Dict[str, float]:
        allowed_names = self._schema_parameter_names()
        if not allowed_names:
            return {}
        canonical: Dict[str, float] = {}
        for key, value in dict(values or {}).items():
            name_s = str(key or "").strip()
            if not name_s or name_s not in allowed_names:
                continue
            value_f = self._finite_float(value)
            if value_f is None:
                continue
            canonical[name_s] = float(value_f)
        return canonical

    # ------------------------------------------------------------------
    # Shared parameters
    # ------------------------------------------------------------------

    @property
    def shared_params(self) -> Mapping[str, float]:
        """Immutable snapshot of the shared parameter baseline."""
        return MappingProxyType(self._canonical_parameter_values(self._shared_params))

    def clear_shared_params(self) -> None:
        self._shared_params.clear()

    # ------------------------------------------------------------------
    # Per-set local overrides
    # ------------------------------------------------------------------

    def _visible_local_override_items(self) -> Iterator[tuple[str, Dict[str, float]]]:
        for raw_set_id, workspace in self._set_local_overrides.items():
            set_id = self._normalized_local_override_set_id(raw_set_id)
            if set_id is None or not workspace:
                continue
            yield set_id, workspace

    @property
    def set_local_overrides(self) -> Dict[str, Dict[str, float]]:
        """Read-only access to the per-set parameter overrides."""
        return {
            set_id: self._canonical_parameter_values(workspace)
            for set_id, workspace in self._visible_local_override_items()
            if self._canonical_parameter_values(workspace)
        }

    def local_overrides_for_set(self, set_id: Optional[str]) -> Dict[str, float]:
        """Return a copy of the per-set overrides for *set_id*."""
        set_id_s = self._normalized_local_override_set_id(set_id)
        if set_id_s is None:
            return {}
        return self._canonical_parameter_values(self._set_local_overrides.get(set_id_s) or {})

    def set_ids_with_local_overrides(self) -> list[str]:
        return [set_id for set_id, _workspace in self._visible_local_override_items()]

    def has_any_local_overrides(self) -> bool:
        return any(True for _set_id, _workspace in self._visible_local_override_items())

    def has_local_overrides_for_set(self, set_id: Optional[str]) -> bool:
        set_id_s = self._normalized_local_override_set_id(set_id)
        if set_id_s is None:
            return False
        return bool(self._set_local_overrides.get(set_id_s))

    def clear_local_overrides_for_set(self, set_id: Optional[str]) -> bool:
        """Remove overrides for *set_id*. Returns True if there was something to remove."""
        set_id_s = self._normalized_local_override_set_id(set_id)
        if set_id_s is None or set_id_s not in self._set_local_overrides:
            return False
        self._set_local_overrides.pop(set_id_s, None)
        return True

    def clear_all_local_overrides(self) -> None:
        self._set_local_overrides.clear()

    # ------------------------------------------------------------------
    # Effective parameter composition
    # ------------------------------------------------------------------

    def runtime_parameter_values(self, set_id: Optional[str] = "") -> Dict[str, float]:
        """Return shared params merged with the per-set overrides for *set_id*."""
        result = dict(self._shared_params)
        set_id_s = self._normalized_local_override_set_id(set_id)
        if set_id_s is None:
            return self._canonical_parameter_values(result)
        overrides = self._set_local_overrides.get(set_id_s)
        if overrides:
            result.update(overrides)
        return self._canonical_parameter_values(result)

    # ------------------------------------------------------------------
    # Staging (per-set runtime parameter mutation)
    # ------------------------------------------------------------------

    def stage_runtime_parameter_value(self, set_id: Optional[str], name: str, value: float) -> bool:
        """
        Stage a single per-set parameter override.

        If the value matches the shared baseline within tolerance, the override
        is pruned (the parameter returns to the shared value for that set).

        Returns True if the store actually changed.
        """
        set_id_s = self._normalized_local_override_set_id(set_id)
        if set_id_s is None:
            return False
        name_s = str(name or "").strip()
        value_f = self._finite_float(value)
        if value_f is None or not self._parameter_name_allowed(name_s):
            return False
        baseline = self._shared_params.get(name_s)
        workspace = dict(self._set_local_overrides.get(set_id_s) or {})
        changed = False

        if baseline is not None and math.isclose(value_f, float(baseline), rel_tol=1e-12, abs_tol=1e-12):
            if name_s in workspace:
                workspace.pop(name_s, None)
                changed = True
        else:
            current = workspace.get(name_s)
            if current is None or not math.isclose(float(current), value_f, rel_tol=1e-12, abs_tol=1e-12):
                workspace[name_s] = value_f
                changed = True

        if workspace:
            candidate = dict(self._shared_params)
            candidate.update(workspace)
            if self._equilibrium_conflict_message(candidate):
                return False
            self._set_local_overrides[set_id_s] = {
                str(k): float(v) for k, v in workspace.items()
            }
        else:
            if self._equilibrium_conflict_message(self._shared_params):
                return False
            self._set_local_overrides.pop(set_id_s, None)

        return changed

    # ------------------------------------------------------------------
    # Shared parameter sync (with pruning)
    # ------------------------------------------------------------------

    def sync_shared_params(self, values: Dict[str, float]) -> bool:
        """
        Replace the shared baseline and prune per-set overrides.

        Per-set override entries that now match the new baseline (within tolerance)
        or reference parameters no longer in the baseline are removed.

        Returns True if per-set overrides changed as a side effect.
        """
        committed: Dict[str, float] = {}
        for key, value in dict(values or {}).items():
            key_s = str(key or "").strip()
            if not key_s:
                continue
            parsed = self._finite_float(value)
            if parsed is None:
                continue
            if not self._parameter_name_allowed(key_s):
                continue
            committed[key_s] = float(parsed)

        if self._equilibrium_conflict_message(committed):
            return False

        before = {
            str(sid): {str(k): float(v) for k, v in ws.items()}
            for sid, ws in self._set_local_overrides.items()
        }

        self._shared_params = committed

        pruned: Dict[str, Dict[str, float]] = {}
        for sid, workspace in list(self._set_local_overrides.items()):
            sid_s = self._normalized_local_override_set_id(sid)
            if sid_s is None:
                continue
            retained: Dict[str, float] = {}
            for name, val in list(dict(workspace or {}).items()):
                baseline = committed.get(str(name))
                if baseline is None:
                    continue
                val_f = float(val)
                if math.isclose(val_f, float(baseline), rel_tol=1e-12, abs_tol=1e-12):
                    continue
                retained[str(name)] = val_f
            candidate = dict(committed)
            candidate.update(retained)
            if retained and not self._equilibrium_conflict_message(candidate):
                pruned[sid_s] = retained
        self._set_local_overrides = pruned

        after = {
            str(sid): {str(k): float(v) for k, v in ws.items()}
            for sid, ws in self._set_local_overrides.items()
        }
        return before != after

    # ------------------------------------------------------------------
    # Commit / globalize
    # ------------------------------------------------------------------

    def commit_effective_as_shared(self, set_id: Optional[str]) -> Dict[str, float]:
        """
        Merge the focused set's effective values into shared params and preserve
        other sets' staged local overrides where they still differ from the new baseline.

        This is the "commit current set" operation: the focused set's effective values
        become the new shared baseline, the focused set's local overrides are discarded,
        and other sets remain dirty if their staged values still differ from the new
        canonical baseline.
        """
        set_id_s = self._normalized_local_override_set_id(set_id)
        if set_id_s is None:
            return dict(self._shared_params)
        effective = self.runtime_parameter_values(set_id_s)
        new_shared_params = {}
        for key, value in dict(effective or {}).items():
            key_s = str(key or "").strip()
            value_f = self._finite_float(value)
            if value_f is None:
                continue
            if not self._parameter_name_allowed(key_s):
                continue
            new_shared_params[key_s] = float(value_f)
        if self._equilibrium_conflict_message(new_shared_params):
            return dict(self._shared_params)
        preserved: Dict[str, Dict[str, float]] = {}
        for raw_sid, workspace in list(self._set_local_overrides.items()):
            sid_s = self._normalized_local_override_set_id(raw_sid)
            if sid_s is None or sid_s == set_id_s:
                continue
            retained: Dict[str, float] = {}
            for name, val in list(dict(workspace or {}).items()):
                if not self._parameter_name_allowed(str(name)):
                    continue
                baseline = new_shared_params.get(str(name))
                val_f = float(val)
                if baseline is None:
                    continue
                if math.isclose(val_f, float(baseline), rel_tol=1e-12, abs_tol=1e-12):
                    continue
                retained[str(name)] = val_f
                retained[str(name)] = val_f
            candidate = dict(new_shared_params)
            candidate.update(retained)
            if retained and not self._equilibrium_conflict_message(candidate):
                preserved[sid_s] = retained
        self._shared_params = new_shared_params
        self._set_local_overrides = preserved
        return dict(self._shared_params)

    # ------------------------------------------------------------------
    # Fingerprinting (for cache identity)
    # ------------------------------------------------------------------

    def param_fingerprint(self, set_id: Optional[str] = "") -> str:
        """Deterministic hash of effective parameter values for cache identity."""
        set_id_s = self._normalized_local_override_set_id(set_id)
        effective = (
            self._canonical_parameter_values(self._shared_params)
            if set_id_s is None
            else self.runtime_parameter_values(set_id_s)
        )
        items = sorted(effective.items())
        material = ";".join(f"{k}={v!r}" for k, v in items)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Bulk reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all parameter state."""
        self._shared_params.clear()
        self._set_local_overrides.clear()
        self._schema_text = ""
        self._schema_id = ""
