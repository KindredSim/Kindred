from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping as MappingABC
import logging
import math
import re
from typing import TYPE_CHECKING, Dict, Tuple

from kindred.core.equilibrium_rate_authority import (
    effective_equilibrium_keq,
    effective_equilibrium_reverse_rate,
)
from kindred.core.simulator.computational_mode import GENERATED_BLOCK_START
from kindred.core.simulator.parameter_namespace import generated_fast_equilibrium_parameter_name

if TYPE_CHECKING:
    from kindred.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


class MainWindowVariableRuntime:
    """Owns MainWindow's slider-variable metadata and energy-mode UI pipeline."""

    def __init__(self, main_window: "MainWindow") -> None:
        self._mw = main_window
        self._variable_metadata: Dict[str, Dict[str, object]] = {}

    def variable_metadata(self) -> Dict[str, Dict[str, object]]:
        return {str(name): dict(meta or {}) for name, meta in self._variable_metadata.items()}

    def set_variable_metadata(self, metadata: Dict[str, Dict[str, object]] | None) -> None:
        normalized: Dict[str, Dict[str, object]] = {}
        for name, meta in dict(metadata or {}).items():
            normalized[str(name)] = dict(meta or {}) if isinstance(meta, dict) else {}
        self._variable_metadata = normalized

    def clear_variable_metadata(self) -> None:
        self._variable_metadata = {}

    def update_variable_metadata_entry(
        self,
        name: str,
        updates: MappingABC[str, object],
    ) -> Dict[str, object]:
        name_s = str(name)
        meta = dict(self._variable_metadata.get(name_s) or {})
        for key, value in dict(updates or {}).items():
            meta[str(key)] = value
        self._variable_metadata[name_s] = meta
        return dict(meta)

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

    def extract_and_populate_variables(
        self,
        *,
        preserve_visibility: bool = False,
    ) -> None:
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

            source = mw.canonical_mechanism_source()
            if not source.full_dsl.strip():
                return

            temperature_k = mw.temperature_spinbox_value()
            units = UnitsModel(temperature_K=temperature_k)

            parse_source = source.with_reactions_text(
                strip_named_reaction_dsl_initial_concentration_sets(str(source.reactions_text))
            )
            committed_schema_text = parse_source.full_dsl
            try:
                solver_owner = getattr(mw, "_simulation_solver_owner", None)
                if solver_owner is not None and callable(getattr(solver_owner, "wegscheider_cyclicity_enabled", None)):
                    wegscheider_enabled = bool(solver_owner.wegscheider_cyclicity_enabled())
                else:
                    wegscheider_enabled = bool(mw.wegscheider_cyclicity_enabled())

                def _build_structure_snapshot(full_dsl: str) -> object:
                    mechanism_obj = parse_dsl_to_mechanism(full_dsl, initials={}, units=units)
                    if isinstance(getattr(mechanism_obj, "metadata", None), dict):
                        mechanism_obj.metadata["wegscheider_cyclicity_enabled"] = wegscheider_enabled
                    _ = apply_parameter_algebra_to_mechanism(
                        full_dsl,
                        mechanism=mechanism_obj,
                        require_mutable=False,
                    )
                    return mechanism_obj

                mechanism_helpers = getattr(mw, "_mechanism_helpers", None)
                authoritative_structure_snapshot = getattr(
                    mechanism_helpers,
                    "authoritative_structure_snapshot",
                    None,
                )
                if callable(authoritative_structure_snapshot):
                    structure_snapshot = authoritative_structure_snapshot(
                        source=parse_source,
                        units_identity=(
                            "temperature_K",
                            f"{float(temperature_k):.17g}",
                            "wegscheider",
                            str(wegscheider_enabled),
                        ),
                        builder=_build_structure_snapshot,
                    )
                    mechanism = structure_snapshot.mechanism
                else:
                    mechanism = _build_structure_snapshot(parse_source.full_dsl)
            except Exception as exc:
                logger.warning("Could not parse mechanism for variable extraction: %s", exc)
                return

            variables, metadata = enumerate_step_parameters_for_gui(mechanism)
            unit_map = solver_parameter_units_from_mechanism(mechanism)

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

            current_metadata = self.variable_metadata()
            has_current_energy_sliders = any(
                isinstance(meta, dict) and meta.get("type") == "energy"
                for meta in dict(current_metadata or {}).values()
            )
            has_generated_fast_equilibrium = self.has_generated_computational_fast_equilibrium(
                mechanism
            )
            if (
                has_generated_fast_equilibrium
                or self.is_energy_mode_mechanism(mechanism)
            ) and (
                has_generated_fast_equilibrium
                or not variables
                or has_current_energy_sliders
            ):
                self.populate_energy_mode_variables_from_mechanism(
                    mechanism,
                    refresh_sliders=True,
                    preserve_visibility=bool(preserve_visibility),
                    base_variables=variables if has_generated_fast_equilibrium else None,
                    base_metadata=metadata if has_generated_fast_equilibrium else None,
                )
                return

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
                    {k: v for k, v in variables.items() if metadata.get(k, {}).get("editable") is not False},
                    schema_text=committed_schema_text,
                )
                self.update_parameter_table_from_sliders()
            else:
                logger.info("No variables found to populate sliders")
                mw.clear_variable_sliders()
                mw._preview_session.sync_committed_slider_values(
                    {},
                    schema_text=committed_schema_text,
                )
                self.clear_variable_metadata()
                self.update_parameter_table_from_sliders()
            self.set_variable_metadata(metadata)
        except Exception as exc:
            logger.error("Error extracting variables: %s", exc, exc_info=True)
            try:
                mw.clear_variable_sliders()
            except RuntimeError as clear_exc:
                logger.debug("Failed to clear sliders after extraction failure: %s", clear_exc, exc_info=True)
            mw._preview_session.sync_committed_slider_values({})
            self.clear_variable_metadata()

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
            plot = mw.main_plot()
            updater = getattr(plot, "update_parameters", None)
            if callable(updater):
                updater(dict(params))
        except Exception:
            return

    def is_energy_mode_mechanism(self, mechanism: object) -> bool:
        meta = getattr(mechanism, "metadata", {}) or {}
        if not isinstance(meta, MappingABC):
            return False
        sn = meta.get("state_network")
        if not isinstance(sn, MappingABC):
            return False
        states = sn.get("states")
        edges = sn.get("edges")
        return bool(states or edges)

    def has_generated_computational_fast_equilibrium(self, mechanism: object) -> bool:
        for eq in list(getattr(mechanism, "equilibria", []) or []):
            eq_meta = getattr(eq, "metadata", {}) or {}
            if not isinstance(eq_meta, MappingABC):
                continue
            if str(eq_meta.get("authority_source") or "") != "generated_computational_mode":
                continue
            if str(eq_meta.get("cm_id") or "").strip():
                return True
        return False

    def dsl_has_computational_mode_generated_block(self, dsl_text: str) -> bool:
        if not dsl_text:
            return False
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
        if not isinstance(meta, MappingABC):
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
        base_variables: MappingABC[str, object] | None = None,
        base_metadata: MappingABC[str, MappingABC[str, object]] | None = None,
    ) -> None:
        mw = self._mw

        def _equilibrium_value(eq_obj: object, role: str) -> float:
            effective_temperature = float(temperature_k) if temperature_k is not None else 298.15
            if role == "Keq":
                value = effective_equilibrium_keq(eq_obj, temperature_K=effective_temperature)
                return float(value) if value is not None else float("nan")
            if role == "kr":
                value = effective_equilibrium_reverse_rate(eq_obj, temperature_K=effective_temperature)
                return float(value) if value is not None else float("nan")
            raw_value = getattr(eq_obj, role, None)
            try:
                return float(raw_value() if callable(raw_value) else raw_value)
            except Exception:
                return float("nan")

        meta = getattr(mechanism, "metadata", {}) or {}
        energy_unit = "kJ/mol"
        if isinstance(meta, MappingABC) and meta.get("energy_unit"):
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
        if isinstance(meta, MappingABC) and meta.get("temperature_K") is not None:
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
            if not isinstance(eq_meta, MappingABC):
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
                    "Keq": _equilibrium_value(eq, "Keq"),
                    "unit_kf": unit_kf,
                    "unit_kr": unit_kr,
                }
            )

        ts_channels.sort(key=lambda c: (str(c.get("ts")), str(c.get("reactant")), str(c.get("product"))))

        step_context_by_equilibrium_index: dict[int, str] = {}
        if isinstance(meta, MappingABC):
            for entry in list(meta.get("step_index_map") or []):
                if not isinstance(entry, MappingABC):
                    continue
                if str(entry.get("kind") or "") != "equilibrium":
                    continue
                try:
                    eq_index = int(entry.get("equilibrium_index"))
                except Exception:
                    continue
                context = str(entry.get("context") or "").strip()
                if context:
                    step_context_by_equilibrium_index[eq_index] = context

        def _context_molecularity(context: str) -> tuple[int, int]:
            text = str(context or "")
            arrow = "<=>" if "<=>" in text else "<->" if "<->" in text else ""
            if not arrow:
                return 1, 1

            def _side_count(side: str) -> int:
                total = 0
                for raw in [part.strip() for part in str(side or "").split("+") if part.strip()]:
                    term = raw.replace(" ", "")
                    match = re.match(r"^(\d+)?[A-Za-z_][A-Za-z0-9_]*$", term)
                    if match is None:
                        return 1
                    total += int(match.group(1) or "1")
                return max(1, int(total or 1))

            lhs, rhs = [part.strip() for part in text.split(arrow, 1)]
            return _side_count(lhs), _side_count(rhs)

        fast_eq_channels: list[dict[str, object]] = []
        for eq_index, eq in enumerate(list(getattr(mechanism, "equilibria", []) or [])):
            eq_meta = getattr(eq, "metadata", {}) or {}
            if not isinstance(eq_meta, MappingABC):
                continue
            if str(eq_meta.get("authority_source") or "") != "generated_computational_mode":
                continue
            cm_id = str(eq_meta.get("cm_id") or "").strip()
            if not cm_id:
                mw._record_best_effort_failure(
                    "main_window.energy_channels.fast_eq.cm_id",
                    message="Skipping generated fast-equilibrium channel without cm_id metadata",
                    exc=None,
                )
                continue
            label = str(step_context_by_equilibrium_index.get(int(eq_index)) or cm_id or "fast equilibrium")
            try:
                dg_eq_j = float(eq_meta.get("dG_eq_J_per_mol"))
            except Exception as exc:
                mw._record_best_effort_failure(
                    "main_window.energy_channels.fast_eq.read_dG",
                    message=f"Skipping parsed fast-equilibrium channel with invalid dG metadata ({label})",
                    exc=exc,
                )
                continue
            dg_eq_val = (
                float(unit_conv.from_jmol(float(dg_eq_j)))
                if unit_conv is not None
                else float(dg_eq_j) / 1000.0
            )
            kf = _equilibrium_value(eq, "kf")
            kr = _equilibrium_value(eq, "kr")
            k_thermo = _equilibrium_value(eq, "Keq")
            if (
                (not math.isfinite(float(k_thermo)) or float(k_thermo) <= 0.0)
                and temperature_k is not None
                and math.isfinite(float(temperature_k))
                and float(temperature_k) > 0.0
            ):
                try:
                    k_thermo = float(math.exp(-float(dg_eq_j) / (float(r_j_per_mol_k) * float(temperature_k))))
                except Exception:
                    k_thermo = float("nan")
            if not (
                math.isfinite(float(kf))
                and float(kf) > 0.0
                and math.isfinite(float(kr))
                and float(kr) > 0.0
                and math.isfinite(float(k_thermo))
                and float(k_thermo) > 0.0
            ):
                continue
            try:
                std_ratio = float(eq_meta.get("std_ratio"))
            except Exception as exc:
                mw._record_best_effort_failure(
                    "main_window.energy_channels.fast_eq.std_ratio",
                    message=f"Skipping generated fast-equilibrium channel with invalid std_ratio metadata ({label})",
                    exc=exc,
                )
                continue
            if not (math.isfinite(float(std_ratio)) and float(std_ratio) > 0.0):
                mw._record_best_effort_failure(
                    "main_window.energy_channels.fast_eq.std_ratio",
                    message=f"Skipping generated fast-equilibrium channel with non-positive std_ratio metadata ({label})",
                    exc=None,
                )
                continue
            molecularity_fwd, molecularity_rev = _context_molecularity(label)
            try:
                molecularity_fwd = int(eq_meta.get("molecularity_fwd") or molecularity_fwd)
            except Exception:
                molecularity_fwd = int(molecularity_fwd)
            try:
                molecularity_rev = int(eq_meta.get("molecularity_rev") or molecularity_rev)
            except Exception:
                molecularity_rev = int(molecularity_rev)
            unit_kf = "1/s"
            unit_kr = "1/s"
            try:
                if rate_units is not None:
                    unit_kf = str(rate_units(int(molecularity_fwd)))
                    unit_kr = str(rate_units(int(molecularity_rev)))
            except Exception:
                unit_kf = "1/s"
                unit_kr = "1/s"

            fast_eq_channels.append(
                {
                    "kind": "fast_equilibrium",
                    "cm_id": cm_id,
                    "label": label,
                    "dG_eq": float(dg_eq_val),
                    "kf_fixed": float(kf),
                    "kr": float(kr),
                    "K_thermo": float(k_thermo),
                    "std_ratio": float(std_ratio),
                    "molecularity_fwd": int(max(1, int(molecularity_fwd))),
                    "molecularity_rev": int(max(1, int(molecularity_rev))),
                    "unit_kf": unit_kf,
                    "unit_kr": unit_kr,
                }
            )

        fast_eq_channels.sort(key=lambda c: (str(c.get("cm_id")), str(c.get("label"))))
        all_channels = list(ts_channels) + list(fast_eq_channels)
        mw._energy_mode_channels = list(all_channels)

        variables: OrderedDict[str, float] = OrderedDict()
        vmeta: OrderedDict[str, Dict[str, object]] = OrderedDict()
        params: Dict[str, Tuple[float, str]] = {}
        base_meta_map = dict(base_metadata or {})
        for raw_name, raw_value in dict(base_variables or {}).items():
            name = str(raw_name)
            try:
                value = float(raw_value)
            except Exception:
                continue
            meta_entry = dict(base_meta_map.get(name) or {})
            variables[name] = value
            vmeta[name] = meta_entry
            params[name] = (value, str(meta_entry.get("unit") or "1"))

        if not all_channels:
            self.set_variable_metadata(dict(vmeta))
            if refresh_sliders and variables:
                mw.set_variable_sliders(
                    dict(variables),
                    metadata=dict(vmeta),
                    preserve_visibility=bool(preserve_visibility),
                    visibility_scope_signature=self._slider_visibility_scope_signature(
                        dict(variables),
                        dict(vmeta),
                    ),
                )
                mw._preview_session.sync_committed_slider_values(
                    dict(variables),
                    schema_text=mw.canonical_mechanism_source().without_reaction_initial_concentrations().full_dsl,
                )
            else:
                mw.clear_variable_sliders()
                mw._preview_session.sync_committed_slider_values(
                    {},
                    schema_text=mw.canonical_mechanism_source().without_reaction_initial_concentrations().full_dsl,
                )
                self.clear_variable_metadata()
            mw._energy_mode_channels = []
            try:
                plot = mw.main_plot()
                updater = getattr(plot, "update_parameters", None)
                if callable(updater):
                    updater(dict(params))
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
            params[f"Keq ({label})"] = (float(ch["Keq"]), "1")

        for ch in fast_eq_channels:
            cm_id = str(ch.get("cm_id") or "")
            label = str(ch.get("label") or cm_id or "fast equilibrium")
            dg_eq = float(ch.get("dG_eq") or 0.0)
            kf_fixed = float(ch.get("kf_fixed") or float("nan"))
            std_ratio = float(ch.get("std_ratio") or 1.0)

            var_eq = generated_fast_equilibrium_parameter_name(
                {
                    "authority_source": "generated_computational_mode",
                    "cm_id": cm_id,
                }
            )
            if var_eq is None:
                continue
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
            params[f"Keq ({label})"] = (k_thermo, "1")

        self.set_variable_metadata(dict(vmeta))
        visibility_scope_signature = self._slider_visibility_scope_signature(dict(variables), dict(vmeta))

        if refresh_sliders:
            mw.set_variable_sliders(
                dict(variables),
                metadata=dict(vmeta),
                preserve_visibility=bool(preserve_visibility),
                visibility_scope_signature=visibility_scope_signature,
            )
            mw._preview_session.sync_committed_slider_values(
                dict(variables),
                schema_text=mw.canonical_mechanism_source().without_reaction_initial_concentrations().full_dsl,
            )

        try:
            plot = mw.main_plot()
            updater = getattr(plot, "update_parameters", None)
            if callable(updater):
                updater(dict(params))
        except Exception as exc:
            logger.debug("Failed to update plot parameter summary: %s", exc, exc_info=True)
            mw._plot_parameter_summary_stale = True
