from __future__ import annotations

from collections import OrderedDict
import logging
import math
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from kindred.core.api.simulation import prepare_bound_mechanism
from kindred.core.simulator.dsl_text_update import (
    _dedupe_tokens_case_insensitive,
    _duplicate_canonical_step_token,
    _get_token_float,
    _is_equilibrium_k_token,
    _parse_mechanism_semicolon_kv,
    _remove_token_aliases,
    _serialize_mechanism_semicolon_kv,
    _set_token_float,
)

if TYPE_CHECKING:
    from kindred.core.simulation_preparation import BoundMechanism
    from kindred.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


class MainWindowVariableRuntime:
    """Owns MainWindow's prepared preview runtime and slider-variable metadata pipeline."""

    def __init__(self, main_window: "MainWindow") -> None:
        self._mw = main_window
        self._variable_metadata: Dict[str, Dict[str, object]] = {}
        self._slider_runtime: Optional[BoundMechanism] = None
        self._slider_runtime_dirty = True
        self._suppress_slider_runtime_invalidation = False

    def variable_metadata(self) -> Dict[str, Dict[str, object]]:
        return {str(name): dict(meta or {}) for name, meta in self._variable_metadata.items()}

    def mutable_variable_metadata(self) -> Dict[str, Dict[str, object]]:
        return self._variable_metadata

    def set_variable_metadata(self, metadata: Dict[str, Dict[str, object]] | None) -> None:
        normalized: Dict[str, Dict[str, object]] = {}
        for name, meta in dict(metadata or {}).items():
            normalized[str(name)] = dict(meta or {}) if isinstance(meta, dict) else {}
        self._variable_metadata = normalized

    def clear_variable_metadata(self) -> None:
        self._variable_metadata = {}

    def slider_runtime_dirty(self) -> bool:
        return bool(self._slider_runtime_dirty)

    def set_slider_runtime_dirty(self, value: bool) -> None:
        self._slider_runtime_dirty = bool(value)

    def suppress_slider_runtime_invalidation(self) -> bool:
        return bool(self._suppress_slider_runtime_invalidation)

    def set_suppress_slider_runtime_invalidation(self, value: bool) -> None:
        self._suppress_slider_runtime_invalidation = bool(value)

    def clear_prepared_slider_runtime(self, *, dirty: bool = True) -> None:
        self._slider_runtime = None
        self._slider_runtime_dirty = bool(dirty)

    def _normalize_visibility_scope_value(self, value: object) -> object:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                return str(value)
            return float(value)
        if isinstance(value, (list, tuple)):
            return tuple(self._normalize_visibility_scope_value(item) for item in value)
        if isinstance(value, dict):
            return tuple(
                (str(key), self._normalize_visibility_scope_value(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        return str(value)

    def _slider_visibility_scope_signature(
        self,
        variables: Dict[str, float],
        metadata: Dict[str, Dict[str, object]],
    ) -> tuple[tuple[str, tuple[tuple[str, object], ...]], ...]:
        # Visibility scope tracks the semantic slider universe, not raw DSL text or values.
        meta_keys = (
            "type",
            "index",
            "role",
            "label",
            "editable",
            "derived",
            "expr",
            "unit",
            "scale",
            "ts",
            "reactant",
            "product",
            "cm_id",
            "kind",
        )
        signature: list[tuple[str, tuple[tuple[str, object], ...]]] = []
        for name in sorted((str(name) for name in variables.keys()), key=str):
            meta = dict(metadata.get(str(name)) or {})
            scope_meta = tuple(
                (key, self._normalize_visibility_scope_value(meta[key]))
                for key in meta_keys
                if key in meta
            )
            signature.append((str(name), scope_meta))
        return tuple(signature)

    def invalidate_slider_runtime(self) -> None:
        if self.suppress_slider_runtime_invalidation():
            return
        self.clear_prepared_slider_runtime(dirty=True)

    def sanitize_mechanism_parameter_conflicts(
        self,
        text: str,
    ) -> tuple[str, "OrderedDict[str, float]", "OrderedDict[str, Dict[str, object]]"]:
        lines = text.split("\n")
        changed = False
        variables: OrderedDict[str, float] = OrderedDict()
        metadata: OrderedDict[str, Dict[str, object]] = OrderedDict()

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            lower = stripped.lower()
            prefix, tokens, comment = _parse_mechanism_semicolon_kv(line)
            if _duplicate_canonical_step_token(tokens) is not None:
                continue
            tokens = _dedupe_tokens_case_insensitive(tokens)

            original_line = line

            if "<->" in lower or "<=>" in lower:
                if sum(1 for key, _ in tokens if _is_equilibrium_k_token(key)) > 1:
                    continue
                k_explicit = any(_is_equilibrium_k_token(key) for key, _ in tokens)
                kf_val = _get_token_float(tokens, ("kf", "k"))
                kr_val = _get_token_float(tokens, ("kr",))
                k_val = _get_token_float(tokens, ("K",)) if k_explicit else None

                if kf_val is not None and (not math.isfinite(kf_val) or kf_val <= 0):
                    kf_val = None
                if kr_val is not None and (not math.isfinite(kr_val) or kr_val <= 0):
                    kr_val = None
                if k_val is not None and (not math.isfinite(k_val) or abs(k_val) < 1e-12):
                    k_val = None

                if kf_val is not None:
                    _set_token_float(tokens, "kf", kf_val, aliases=("k",))
                if kr_val is not None:
                    _set_token_float(tokens, "kr", kr_val)
                if k_explicit and k_val is not None:
                    _set_token_float(tokens, "K", k_val)

                _remove_token_aliases(tokens, ("k",))
                new_line = _serialize_mechanism_semicolon_kv(prefix, tokens, comment)
            elif "->" in lower:
                rate_val = _get_token_float(tokens, ("k", "kf"))
                if rate_val is not None:
                    _set_token_float(tokens, "k", rate_val)
                _remove_token_aliases(tokens, ("kf", "kr"))
                new_line = _serialize_mechanism_semicolon_kv(prefix, tokens, comment)
            else:
                continue

            if new_line != original_line:
                lines[idx] = new_line
                changed = True

        if not changed:
            return text, variables, metadata

        return "\n".join(lines), variables, metadata

    def extract_and_populate_variables(self, *, preserve_visibility: bool = False) -> None:
        mw = self._mw
        try:
            from kindred.core.batch_initial_conditions import (
                strip_named_reaction_dsl_initial_concentration_sets,
            )
            from kindred.gui.parameter_enumeration import enumerate_step_parameters_for_gui
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism
            from kindred.core.simulator.parameter_algebra import (
                apply_parameter_algebra_to_mechanism,
                solver_parameter_units_from_mechanism,
            )
            from kindred.core.units import UnitsModel

            mechanism_text = mw.mechanism_reactions_text_raw()
            if not mechanism_text.strip():
                return

            sanitized_text, baseline_variables, baseline_metadata = self.sanitize_mechanism_parameter_conflicts(
                mechanism_text
            )
            if baseline_variables or baseline_metadata:
                raise RuntimeError(
                    "Legacy line-counter-derived variables/metadata must not be used for sliders. "
                    "Slider parameters must be derived from mechanism.metadata['step_index_map'] via "
                    "enumerate_step_parameters_for_gui()."
                )
            if sanitized_text != mechanism_text:
                previous_authoritative_suppress = bool(
                    getattr(mw, "_suppress_authoritative_mechanism_input_change", False)
                )
                setattr(mw, "_suppress_authoritative_mechanism_input_change", True)
                try:
                    mw.set_mechanism_reactions_text_with_optional_undo(
                        sanitized_text,
                        "Normalize mechanism parameter tokens for sliders",
                        record_undo=False,
                    )
                finally:
                    setattr(mw, "_suppress_authoritative_mechanism_input_change", previous_authoritative_suppress)
                mechanism_text = sanitized_text
                self.set_slider_runtime_dirty(True)

            temperature_k = mw.temperature_spinbox_value()
            units = UnitsModel(temperature_K=temperature_k)

            state_network_dsl = mw.mechanism_state_network_dsl_raw()
            parse_mechanism_text = strip_named_reaction_dsl_initial_concentration_sets(mechanism_text)
            full_dsl = parse_mechanism_text
            if state_network_dsl.strip():
                full_dsl += "\n\n# State Network\n" + state_network_dsl.strip("\n")
            try:
                mechanism = parse_dsl_to_mechanism(full_dsl, initials={}, units=units)
                if isinstance(getattr(mechanism, "metadata", None), dict):
                    mechanism.metadata["wegscheider_cyclicity_enabled"] = bool(mw.wegscheider_cyclicity_enabled())
                _ = apply_parameter_algebra_to_mechanism(full_dsl, mechanism=mechanism, require_mutable=False)
            except Exception as exc:
                logger.warning("Could not parse mechanism for variable extraction: %s", exc)
                return

            unit_map = solver_parameter_units_from_mechanism(mechanism)

            variables, metadata = enumerate_step_parameters_for_gui(mechanism)
            for name, meta in list(metadata.items()):
                if isinstance(meta, dict) and meta.get("value_valid") is False:
                    mw._record_best_effort_failure(
                        "main_window.enumerate_step_parameters_for_gui.value_invalid",
                        message=(
                            f"Slider parameter {name!r} had a non-numeric/non-finite value; "
                            "using 0.0 and marking value_valid=False"
                        ),
                        max_logs=5,
                    )
            mw._step_index_map = (getattr(mechanism, "metadata", {}) or {}).get("step_index_map") or []
            for name in list(variables.keys()):
                meta = dict(metadata.get(name) or {})
                meta["unit"] = unit_map.get(name, "1")
                metadata[name] = meta

            scalar_params = (getattr(mechanism, "metadata", {}) or {}).get("scalar_params") or {}
            scalar_info = (getattr(mechanism, "metadata", {}) or {}).get("scalar_param_info") or {}
            if isinstance(scalar_params, dict):
                for name, value in scalar_params.items():
                    try:
                        nm = str(name)
                        variables[nm] = float(value)
                        info = scalar_info.get(nm) if isinstance(scalar_info, dict) else None
                        meta = {
                            "type": "scalar",
                            "index": 0,
                            "label": "Scalar parameter",
                            "line": (info.get("line") if isinstance(info, dict) else 0),
                            "role": "scalar",
                        }
                        meta["unit"] = unit_map.get(nm, "1")
                        if isinstance(info, dict):
                            if "expr" in info:
                                meta["expr"] = info["expr"]
                            if info.get("derived") is True:
                                meta["editable"] = False
                                meta["derived"] = True
                        metadata[nm] = meta
                    except Exception as exc:
                        mw._record_best_effort_failure(
                            "main_window.scalar_params.import",
                            message="Failed to import scalar solver parameter into slider variables",
                            exc=exc,
                        )
                        continue

            constrained = (getattr(mechanism, "metadata", {}) or {}).get("constrained_params") or {}
            if isinstance(constrained, dict):
                for name, info in constrained.items():
                    nm = str(name)
                    if nm not in variables:
                        continue
                    meta = dict(metadata.get(nm) or {})
                    meta["editable"] = False
                    meta["derived"] = True
                    if isinstance(info, dict):
                        if "expr" in info:
                            meta["expr"] = info["expr"]
                        if "line" in info:
                            meta["line"] = info["line"]
                        if "constraint_reason" in info:
                            meta["constraint_reason"] = info["constraint_reason"]
                    metadata[nm] = meta

            visibility_scope_signature = self._slider_visibility_scope_signature(variables, metadata)
            self.set_variable_metadata(metadata)

            if variables:
                mw.set_variable_sliders(
                    variables,
                    metadata=metadata,
                    preserve_visibility=bool(preserve_visibility),
                    visibility_scope_signature=visibility_scope_signature,
                )
                logger.info("Populated %s variable sliders", len(variables))
                mw._preview_session.sync_committed_slider_values(
                    {k: v for k, v in variables.items() if metadata.get(k, {}).get("editable") is not False}
                )
                self.update_parameter_table_from_sliders()
            else:
                logger.info("No variables found to populate sliders")
                mw.clear_variable_sliders()
                mw._preview_session.sync_committed_slider_values({})
                self.clear_variable_metadata()
                self.update_parameter_table_from_sliders()
            self.set_variable_metadata(metadata)
        except Exception as exc:
            logger.error("Error extracting variables: %s", exc, exc_info=True)
            try:
                mw.clear_variable_sliders()
            except RuntimeError as clear_exc:
                logger.debug("Failed to clear sliders after extraction failure: %s", clear_exc, exc_info=True)
                self.clear_prepared_slider_runtime(dirty=True)
            mw._preview_session.sync_committed_slider_values({})
            self.clear_variable_metadata()
            self.clear_prepared_slider_runtime(dirty=True)

    def update_parameter_table_from_sliders(self) -> None:
        mw = self._mw
        values = mw.variable_slider_values() or {}
        meta = self._variable_metadata or {}
        params: Dict[str, Tuple[float, str]] = {}
        for name, val in values.items():
            m = meta.get(name)
            unit = "1"
            if isinstance(m, dict) and m.get("unit"):
                unit = str(m.get("unit"))
            try:
                params[str(name)] = (float(val), unit)
            except Exception as exc:
                mw._record_best_effort_failure(
                    "main_window.param_table.float_value",
                    message=f"Failed to parse slider value for parameter table ({name})",
                    exc=exc,
                )
                continue
        try:
            mw.update_main_plot_parameter_summary(params)
        except Exception:
            return

    def prepare_slider_runtime(
        self,
        param_names: Optional[List[str]] = None,
        *,
        set_id: Optional[str] = None,
    ) -> Optional[BoundMechanism]:
        mw = self._mw

        if param_names is None or not param_names:
            param_names = list(mw.slider_overrides(set_id=set_id).keys())
        if not param_names:
            return None
        meta_map = self._variable_metadata or {}

        # Ensure constrained mechanism parameters are bound even if the user isn't directly editing them.
        try:
            mechanism_param_names = {k for k in meta_map.keys() if re.match(r"^(k|kf|kr|K)\d+$", str(k))}
            spec = mw._parameter_algebra_spec_for_ui(mechanism_param_names=mechanism_param_names)
            if spec is not None and getattr(spec, "param_statements", None):
                constrained = {p.name for p in spec.param_statements if re.match(r"^(k|kf|kr|K)\d+$", str(p.name))}
                if constrained:
                    param_names = sorted(set(param_names) | constrained)
        except Exception as exc:
            logger.debug("Failed to expand slider runtime params from algebra spec: %s", exc, exc_info=True)
            self.clear_prepared_slider_runtime(dirty=True)

        if self._slider_runtime is not None and not self._slider_runtime_dirty:
            current_params = set(self._slider_runtime.param_names)
            requested_params = set(param_names)
            if requested_params == current_params:
                return self._slider_runtime

        reactions_text = str(mw.mechanism_reactions_text_raw() or "")
        state_network_dsl = str(mw.mechanism_state_network_dsl_raw() or "")
        mechanism_text = reactions_text
        if state_network_dsl.strip():
            mechanism_text = (
                f"{mechanism_text}\n\n# State Network\n{state_network_dsl}"
                if mechanism_text.strip()
                else f"# State Network\n{state_network_dsl}"
            )
        temperature_K = mw._temperature_spinbox.value()

        try:
            runtime = prepare_bound_mechanism(
                mechanism_text=mechanism_text,
                param_names=list(param_names),
                temperature_K=temperature_K,
                initials={},
                use_advanced_dsl=True,
                wegscheider_cyclicity_enabled=bool(mw._wegscheider_cyclicity_enabled),
            )
        except Exception as exc:
            logger.error("Failed to prepare slider runtime: %s", exc)
            return None

        self._slider_runtime = runtime
        self._slider_runtime_dirty = False
        return runtime

    def apply_slider_overrides_to_bindings(
        self,
        runtime: Optional[BoundMechanism],
        *,
        set_id: Optional[str] = None,
    ) -> bool:
        mw = self._mw

        if runtime is None or not runtime.bindings:
            return False

        all_applied = True
        for name, value in mw.slider_overrides(set_id=set_id).items():
            binding = runtime.bindings.get(name)
            if binding is None:
                all_applied = False
                continue
            try:
                binding.set(float(value))
            except Exception as exc:
                all_applied = False
                logger.warning("Failed to update binding for %s: %s", name, exc)

        if not all_applied:
            logger.debug("Not all slider bindings could be updated; will re-parse on next run")
        return all_applied

    def is_energy_mode_mechanism(self, mechanism: object) -> bool:
        meta = getattr(mechanism, "metadata", {}) or {}
        if not isinstance(meta, dict):
            return False
        sn = meta.get("state_network")
        if not isinstance(sn, dict):
            return False
        states = sn.get("states") if isinstance(sn, dict) else None
        edges = sn.get("edges") if isinstance(sn, dict) else None
        return bool(states or edges)

    def dsl_has_computational_mode_generated_block(self, dsl_text: str) -> bool:
        if not dsl_text:
            return False
        try:
            from kindred.core.simulator.computational_mode import GENERATED_BLOCK_START
        except Exception:
            GENERATED_BLOCK_START = "# === Generated from Computational Mode ==="
        return str(GENERATED_BLOCK_START).strip() in str(dsl_text)

    def dsl_global_temperature_k(self, dsl_text: str) -> float | None:
        if not dsl_text:
            return None
        for raw in str(dsl_text).splitlines():
            before_comment, _, _comment = raw.partition("#")
            stripped = before_comment.strip()
            if not stripped:
                continue
            if not stripped.lower().startswith("t="):
                continue
            _key, _eq, rest = stripped.partition("=")
            try:
                val = float(rest.strip().split()[0])
            except Exception:
                return None
            if math.isfinite(val) and val > 0:
                return float(val)
            return None
        return None

    def sync_energy_mode_temperature_from_mechanism(self, mechanism: object) -> None:
        mw = self._mw
        meta = getattr(mechanism, "metadata", {}) or {}
        if not isinstance(meta, dict):
            return
        try:
            temperature_k = float(meta.get("temperature_K"))
        except Exception:
            return
        if not math.isfinite(temperature_k) or temperature_k <= 0:
            return
        mw.set_temperature_mode_indicator_text(f"Temperature: {temperature_k:.2f} K (from DSL)")
        mw.set_temperature_override_state(
            enabled=False,
            tooltip="Overridden by energy-mode DSL (T=...).",
        )

    def populate_energy_mode_variables_from_mechanism(
        self,
        mechanism: object,
        *,
        refresh_sliders: bool,
        preserve_visibility: bool = False,
    ) -> None:
        mw = self._mw

        def _equilibrium_value(eq_obj: object, role: str) -> float:
            raw_value = getattr(eq_obj, role, None)
            try:
                return float(raw_value() if callable(raw_value) else raw_value)
            except Exception:
                if role == "K":
                    meta_value = (getattr(eq_obj, "metadata", {}) or {}).get("K_input")
                    try:
                        return float(meta_value() if callable(meta_value) else meta_value)
                    except Exception:
                        return float("nan")
                return float("nan")

        meta = getattr(mechanism, "metadata", {}) or {}
        energy_unit = "kJ/mol"
        if isinstance(meta, dict) and meta.get("energy_unit"):
            energy_unit = str(meta.get("energy_unit"))

        from kindred.core.constants import R as r_j_per_mol_k

        try:
            from kindred.core.simulator.kinetics import rate_units
            from kindred.core.units import UnitsModel
        except Exception:
            UnitsModel = None  # type: ignore[assignment]
            rate_units = None  # type: ignore[assignment]

        unit_conv = UnitsModel(energy_unit=energy_unit) if UnitsModel is not None else None  # type: ignore[call-arg]

        temperature_k: float | None = None
        if isinstance(meta, dict) and meta.get("temperature_K") is not None:
            try:
                temperature_k = float(meta.get("temperature_K"))
            except Exception:
                temperature_k = None
        if temperature_k is None or (not math.isfinite(float(temperature_k))) or float(temperature_k) <= 0.0:
            try:
                temperature_k = float(mw.temperature_spinbox_value())
            except Exception:
                temperature_k = None

        ts_channels: list[dict[str, object]] = []
        for eq in list(getattr(mechanism, "equilibria", []) or []):
            eq_meta = getattr(eq, "metadata", {}) or {}
            if not isinstance(eq_meta, dict):
                continue
            if str(eq_meta.get("source") or "") != "state_network":
                continue
            reactant = str(eq_meta.get("reactant") or "")
            product = str(eq_meta.get("product") or "")
            ts = str(eq_meta.get("ts") or "")
            if not (reactant and product and ts):
                continue
            try:
                dg_act_fwd_j = float(eq_meta.get("dG_act_fwd_J_per_mol"))
                dg_eq_j = float(eq_meta.get("dG_eq_J_per_mol"))
            except Exception as exc:
                mw._record_best_effort_failure(
                    "main_window.energy_channels.read_dG",
                    message=f"Skipping energy-mode channel with invalid dG metadata ({reactant}->{product} via {ts})",
                    exc=exc,
                )
                continue
            if unit_conv is not None:
                dg_act_val = float(unit_conv.from_jmol(dg_act_fwd_j))
                dg_eq_val = float(unit_conv.from_jmol(dg_eq_j))
            else:
                dg_act_val = dg_act_fwd_j / 1000.0
                dg_eq_val = dg_eq_j / 1000.0
            try:
                kappa = float(eq_meta.get("kappa") or 1.0)
            except Exception:
                kappa = 1.0
            try:
                deg_ratio_fwd = float(eq_meta.get("degeneracy_ratio_fwd") or 1.0)
            except Exception:
                deg_ratio_fwd = 1.0
            try:
                deg_ratio_rev = float(eq_meta.get("degeneracy_ratio_rev") or 1.0)
            except Exception:
                deg_ratio_rev = 1.0
            try:
                molecularity_fwd = int(eq_meta.get("molecularity_fwd") or 1)
            except Exception:
                molecularity_fwd = 1
            try:
                molecularity_rev = int(eq_meta.get("molecularity_rev") or 1)
            except Exception:
                molecularity_rev = 1
            unit_kf = "1/s"
            unit_kr = "1/s"
            try:
                if rate_units is not None:
                    unit_kf = str(rate_units(int(molecularity_fwd)))
                    unit_kr = str(rate_units(int(molecularity_rev)))
            except Exception:
                unit_kf = "1/s"
                unit_kr = "1/s"

            std_ts = None
            std_react = None
            std_prod = None
            try:
                if eq_meta.get("std_conc_product_ts") is not None:
                    std_ts = float(eq_meta.get("std_conc_product_ts"))
            except Exception:
                std_ts = None
            try:
                if eq_meta.get("std_conc_product_reactant") is not None:
                    std_react = float(eq_meta.get("std_conc_product_reactant"))
            except Exception:
                std_react = None
            try:
                if eq_meta.get("std_conc_product_product") is not None:
                    std_prod = float(eq_meta.get("std_conc_product_product"))
            except Exception:
                std_prod = None
            label = f"{reactant}→{product} via {ts}"
            ts_channels.append(
                {
                    "kind": "ts_channel",
                    "reactant": reactant,
                    "product": product,
                    "ts": ts,
                    "label": label,
                    "dG_act_fwd": dg_act_val,
                    "dG_eq": dg_eq_val,
                    "kappa": kappa,
                    "degeneracy_ratio_fwd": deg_ratio_fwd,
                    "degeneracy_ratio_rev": deg_ratio_rev,
                    "molecularity_fwd": int(molecularity_fwd),
                    "molecularity_rev": int(molecularity_rev),
                    "std_conc_product_ts": std_ts,
                    "std_conc_product_reactant": std_react,
                    "std_conc_product_product": std_prod,
                    "kf": _equilibrium_value(eq, "kf"),
                    "kr": _equilibrium_value(eq, "kr"),
                    "K": _equilibrium_value(eq, "K"),
                    "unit_kf": unit_kf,
                    "unit_kr": unit_kr,
                }
            )

        ts_channels.sort(key=lambda c: (str(c.get("ts")), str(c.get("reactant")), str(c.get("product"))))

        fast_eq_channels: list[dict[str, object]] = []
        try:
            from kindred.core.simulator.computational_mode import (
                GENERATED_BLOCK_END,
                GENERATED_BLOCK_START,
                extract_marked_block,
            )
        except Exception:
            extract_marked_block = None  # type: ignore[assignment]
            GENERATED_BLOCK_START = ""
            GENERATED_BLOCK_END = ""

        reactions_text = mw.mechanism_reactions_text_raw()

        generated_body = None
        try:
            if callable(extract_marked_block) and reactions_text:
                generated_body = extract_marked_block(
                    reactions_text,
                    start_marker=GENERATED_BLOCK_START,
                    end_marker=GENERATED_BLOCK_END,
                )
        except Exception:
            generated_body = None

        def _parse_fast_side(side: str) -> dict[str, int]:
            out: dict[str, int] = {}
            for raw in [p.strip() for p in str(side or "").split("+") if p.strip()]:
                term = raw.replace(" ", "")
                match = re.match(r"^(\d+)?([A-Za-z_][A-Za-z0-9_]*)$", term)
                if not match:
                    raise ValueError("invalid term")
                coeff = int(match.group(1) or "1")
                name = str(match.group(2))
                out[name] = out.get(name, 0) + coeff
            return out

        def _canonical_fast_side(sto: dict[str, int]) -> str:
            parts: list[str] = []
            for name in sorted(sto.keys()):
                coeff = int(sto[name])
                parts.append(f"{name}" if coeff == 1 else f"{coeff}{name}")
            return "+".join(parts)

        if generated_body:
            for raw in str(generated_body).splitlines():
                before_comment, _, _comment = str(raw).partition("#")
                code = before_comment.strip()
                if not code or code.startswith("#"):
                    continue
                if not code.lower().startswith("equilibrium:"):
                    continue
                chunks = [c.strip() for c in code.split(";") if c.strip()]
                if not chunks:
                    continue
                prefix = chunks[0]
                if ":" not in prefix:
                    continue
                eqn = prefix.split(":", 1)[1].strip()

                tokens: dict[str, str] = {}
                for part in chunks[1:]:
                    if "=" not in part:
                        continue
                    key, value = part.split("=", 1)
                    tokens[str(key).strip()] = str(value).strip()

                try:
                    kf = float(tokens.get("kf") or "")
                    kr = float(tokens.get("kr") or "")
                except Exception as exc:
                    mw._record_best_effort_failure(
                        "main_window.energy_channels.fast_eq.read_k",
                        message="Skipping fast-equilibrium entry with invalid kf/kr tokens",
                        exc=exc,
                    )
                    continue
                if not (math.isfinite(kf) and kf > 0.0 and math.isfinite(kr) and kr > 0.0):
                    continue

                try:
                    if "<=>" in eqn:
                        lhs, rhs = [p.strip() for p in eqn.split("<=>", 1)]
                    else:
                        lhs, rhs = [p.strip() for p in eqn.split("<->", 1)]
                    sto_fwd = _parse_fast_side(lhs)
                    sto_rev = _parse_fast_side(rhs)
                    molecularity_fwd = int(sum(int(v) for v in sto_fwd.values()))
                    molecularity_rev = int(sum(int(v) for v in sto_rev.values()))
                    if molecularity_fwd < 1 or molecularity_rev < 1:
                        continue
                    cm_id = str(tokens.get("cm_id") or f"feq__{_canonical_fast_side(sto_fwd)}__{_canonical_fast_side(sto_rev)}")
                except Exception as exc:
                    mw._record_best_effort_failure(
                        "main_window.energy_channels.fast_eq.parse_equation",
                        message="Skipping fast-equilibrium entry due to invalid equilibrium equation tokens",
                        exc=exc,
                    )
                    continue

                unit_kf = "1/s"
                unit_kr = "1/s"
                try:
                    if rate_units is not None:
                        unit_kf = str(rate_units(int(molecularity_fwd)))
                        unit_kr = str(rate_units(int(molecularity_rev)))
                except Exception:
                    unit_kf = "1/s"
                    unit_kr = "1/s"

                dg_eq_val = None
                try:
                    if tokens.get("dG_eq") is not None:
                        dg_eq_val = float(tokens.get("dG_eq") or "")
                except Exception:
                    dg_eq_val = None

                std_ratio = 1.0
                k_thermo = float("nan")
                if dg_eq_val is not None and temperature_k is not None and math.isfinite(float(temperature_k)) and float(temperature_k) > 0.0:
                    try:
                        dg_eq_j = (
                            float(unit_conv.to_jmol(float(dg_eq_val)))
                            if unit_conv is not None
                            else float(dg_eq_val) * 1000.0
                        )
                        k_thermo = float(math.exp(-float(dg_eq_j) / (float(r_j_per_mol_k) * float(temperature_k))))
                        if math.isfinite(k_thermo) and k_thermo > 0.0:
                            kc = float(kf / kr)
                            std_ratio = float(kc / k_thermo)
                    except Exception:
                        std_ratio = 1.0
                        k_thermo = float("nan")
                elif temperature_k is not None and math.isfinite(float(temperature_k)) and float(temperature_k) > 0.0:
                    try:
                        kc = float(kf / kr)
                        dg_eq_j = -float(r_j_per_mol_k) * float(temperature_k) * math.log(kc)
                        dg_eq_val = (
                            float(unit_conv.from_jmol(float(dg_eq_j)))
                            if unit_conv is not None
                            else float(dg_eq_j) / 1000.0
                        )
                        k_thermo = float(math.exp(-float(dg_eq_j) / (float(r_j_per_mol_k) * float(temperature_k))))
                        std_ratio = 1.0
                    except Exception as exc:
                        mw._record_best_effort_failure(
                            "main_window.energy_channels.fast_eq.derive_dG_from_Kc",
                            message="Skipping fast-equilibrium entry due to failure deriving dG_eq from Kc",
                            exc=exc,
                        )
                        continue

                if dg_eq_val is None:
                    continue
                if not (math.isfinite(float(std_ratio)) and float(std_ratio) > 0.0):
                    std_ratio = 1.0

                fast_eq_channels.append(
                    {
                        "kind": "fast_equilibrium",
                        "cm_id": cm_id,
                        "label": eqn,
                        "dG_eq": float(dg_eq_val),
                        "kf_fixed": float(kf),
                        "kr": float(kr),
                        "K_thermo": float(k_thermo),
                        "std_ratio": float(std_ratio),
                        "molecularity_fwd": int(molecularity_fwd),
                        "molecularity_rev": int(molecularity_rev),
                        "unit_kf": unit_kf,
                        "unit_kr": unit_kr,
                    }
                )

        fast_eq_channels.sort(key=lambda c: (str(c.get("cm_id")), str(c.get("label"))))
        all_channels = list(ts_channels) + list(fast_eq_channels)
        mw._energy_mode_channels = list(all_channels)

        if not all_channels:
            mw.clear_variable_sliders()
            mw._preview_session.sync_committed_slider_values({})
            self.clear_variable_metadata()
            mw._energy_mode_channels = []
            try:
                mw.update_main_plot_parameter_summary({})
            except Exception as exc:
                logger.debug("Failed to clear plot parameter summary: %s", exc, exc_info=True)
                mw._plot_parameter_summary_stale = True
            return

        def _range(center: float, *, nonnegative: bool) -> tuple[float, float]:
            span = max(10.0, abs(center) * 0.5)
            lo = center - span
            hi = center + span
            if nonnegative:
                lo = max(0.0, lo)
            if hi <= lo:
                hi = lo + max(1.0, span)
            return float(lo), float(hi)

        variables: OrderedDict[str, float] = OrderedDict()
        vmeta: OrderedDict[str, Dict[str, object]] = OrderedDict()
        params: Dict[str, Tuple[float, str]] = {}

        for ch in ts_channels:
            reactant = str(ch["reactant"])
            product = str(ch["product"])
            ts = str(ch["ts"])
            label = str(ch["label"])
            dg_act = float(ch["dG_act_fwd"])
            dg_eq = float(ch["dG_eq"])

            var_act = f"dGact_fwd__{ts}__{reactant}__{product}"
            var_eq = f"dG_eq__{ts}__{reactant}__{product}"

            variables[var_act] = dg_act
            variables[var_eq] = dg_eq

            act_lo, act_hi = _range(dg_act, nonnegative=True)
            eq_lo, eq_hi = _range(dg_eq, nonnegative=False)
            vmeta[var_act] = {
                "type": "energy",
                "role": "dG_act_fwd",
                "ts": ts,
                "reactant": reactant,
                "product": product,
                "label": f"{label} (ΔG‡_fwd)",
                "unit": energy_unit,
                "scale": "linear",
                "min": act_lo,
                "max": act_hi,
            }
            vmeta[var_eq] = {
                "type": "energy",
                "role": "dG_eq",
                "ts": ts,
                "reactant": reactant,
                "product": product,
                "label": f"{label} (ΔG°)",
                "unit": energy_unit,
                "scale": "linear",
                "min": eq_lo,
                "max": eq_hi,
            }

            params[f"ΔG‡_fwd ({label})"] = (dg_act, energy_unit)
            params[f"ΔG° ({label})"] = (dg_eq, energy_unit)
            params[f"k_f ({label})"] = (float(ch["kf"]), str(ch.get("unit_kf") or "1/s"))
            params[f"k_r ({label})"] = (float(ch["kr"]), str(ch.get("unit_kr") or "1/s"))
            params[f"K ({label})"] = (float(ch["K"]), "1")

        def _slug(text: str) -> str:
            return re.sub(r"[^A-Za-z0-9_]+", "_", str(text or "").strip()).strip("_") or "fast_eq"

        for ch in fast_eq_channels:
            cm_id = str(ch.get("cm_id") or "")
            label = str(ch.get("label") or cm_id or "fast equilibrium")
            dg_eq = float(ch.get("dG_eq") or 0.0)
            kf_fixed = float(ch.get("kf_fixed") or float("nan"))
            std_ratio = float(ch.get("std_ratio") or 1.0)

            var_eq = f"dG_eq_fast__{_slug(cm_id)}"
            ch["var_eq"] = var_eq

            variables[var_eq] = dg_eq
            eq_lo, eq_hi = _range(dg_eq, nonnegative=False)
            vmeta[var_eq] = {
                "type": "energy",
                "role": "dG_eq_fast",
                "cm_id": cm_id,
                "label": f"{label} (ΔG° fast)",
                "unit": energy_unit,
                "scale": "linear",
                "min": eq_lo,
                "max": eq_hi,
                "kf_fixed": float(kf_fixed),
                "std_ratio": float(std_ratio),
            }

            k_thermo = float("nan")
            kr_val = float(ch.get("kr") or float("nan"))
            try:
                if temperature_k is not None and math.isfinite(float(temperature_k)) and float(temperature_k) > 0.0:
                    dg_eq_j = float(unit_conv.to_jmol(dg_eq)) if unit_conv is not None else float(dg_eq) * 1000.0
                    k_thermo = float(math.exp(-float(dg_eq_j) / (float(r_j_per_mol_k) * float(temperature_k))))
                    if math.isfinite(k_thermo) and k_thermo > 0.0 and math.isfinite(kf_fixed) and kf_fixed > 0.0:
                        kr_val = float(kf_fixed / (k_thermo * max(1e-300, float(std_ratio))))
            except Exception as exc:
                logger.debug("Failed to compute derived fast-equilibrium parameters: %s", exc, exc_info=True)
                k_thermo = float("nan")
                kr_val = float(ch.get("kr") or float("nan"))

            params[f"ΔG° ({label})"] = (dg_eq, energy_unit)
            params[f"k_f ({label})"] = (kf_fixed, str(ch.get("unit_kf") or "1/s"))
            params[f"k_r ({label})"] = (kr_val, str(ch.get("unit_kr") or "1/s"))
            params[f"K ({label})"] = (k_thermo, "1")

        self.set_variable_metadata(dict(vmeta))
        visibility_scope_signature = self._slider_visibility_scope_signature(dict(variables), dict(vmeta))

        if refresh_sliders:
            mw.set_variable_sliders(
                dict(variables),
                metadata=dict(vmeta),
                preserve_visibility=bool(preserve_visibility),
                visibility_scope_signature=visibility_scope_signature,
            )
            mw._preview_session.sync_committed_slider_values(dict(variables))

        try:
            mw.update_main_plot_parameter_summary(params)
        except Exception as exc:
            logger.debug("Failed to update plot parameter summary: %s", exc, exc_info=True)
            mw._plot_parameter_summary_stale = True
