from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MaterializedDisplayResult:
    series: Dict[str, np.ndarray]
    display_species: tuple[str, ...]
    owned_species: tuple[str, ...]
    algebra_scalars: Dict[str, object]


class SimulationResultMaterializationOwner:
    def __init__(
        self,
        *,
        ui: Any,
        record_nonfatal_exception: Callable[[str, BaseException], None],
    ) -> None:
        self._ui = ui
        self._record_nonfatal_exception = record_nonfatal_exception

    def materialize_completion_display_result(
        self,
        *,
        series: Mapping[str, Any] | None,
        finalized_species_names: Sequence[str] | None,
        owned_species: Sequence[str] | None,
        algebra_scalars: Mapping[str, object] | None = None,
    ) -> MaterializedDisplayResult | None:
        owned = tuple(str(name).strip() for name in (owned_species or ()) if str(name).strip())
        display_roster = tuple(
            str(name).strip() for name in (finalized_species_names or ()) if str(name).strip()
        )
        if not owned or not display_roster or not isinstance(series, Mapping):
            return None
        raw_series = dict(series or {})
        display_series: Dict[str, np.ndarray] = {}
        seen: set[str] = set()
        for name in display_roster:
            if name in seen or name not in raw_series:
                return None
            try:
                display_series[name] = np.asarray(raw_series[name], dtype=float).reshape(-1).copy()
            except Exception:
                return None
            seen.add(name)
        if any(name not in display_series for name in owned):
            return None
        return MaterializedDisplayResult(
            series=display_series,
            display_species=display_roster,
            owned_species=owned,
            algebra_scalars=dict(algebra_scalars or {}),
        )

    def resolve_completion_mechanism(
        self,
        *,
        mechanism: object | None,
        mechanism_text: str,
        solver_config: Mapping[str, Any],
        is_preview: bool,
        is_primary: bool,
    ) -> object | None:
        if mechanism is not None:
            return mechanism
        if bool(is_preview) or (not bool(is_primary)):
            return None
        mechanism_text_s = str(mechanism_text or "")
        if not mechanism_text_s.strip():
            return None
        try:
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism
            from kindred.core.units import UnitsModel

            temperature_value = (solver_config or {}).get("temperature_K")
            if temperature_value is None:
                return None
            temp_for_parse = float(temperature_value)
            return parse_dsl_to_mechanism(
                mechanism_text_s,
                initials={},
                units=UnitsModel(temperature_K=temp_for_parse, energy_unit="kJ/mol"),
            )
        except Exception:
            return None

    def update_primary_result_materialization_contract(
        self,
        *,
        mechanism: object | None,
        mechanism_text: str,
        solver_config: Mapping[str, Any],
        is_preview: bool,
        is_primary: bool,
    ) -> bool:
        _ = solver_config
        energy_mode = bool(
            mechanism is not None
            and (
                self._ui.variable_runtime.is_energy_mode_mechanism(mechanism)
                or self._ui.variable_runtime.dsl_has_computational_mode_generated_block(str(mechanism_text))
            )
        )
        if (not bool(is_primary)) or bool(is_preview):
            return energy_mode
        if (
            energy_mode
            and mechanism is not None
            and self._ui.solver.dsl_global_temperature_K(str(mechanism_text)) is not None
        ):
            self._ui.variable_runtime.sync_energy_mode_temperature_from_mechanism(mechanism)
        elif energy_mode:
            self._ui.mechanism_helpers.set_temperature_override_state(
                enabled=True,
                tooltip="Temperature for thermodynamic calculations (energy mode: add T=... to override).",
            )
            temperature_k = float(self._ui.solver.temperature_spinbox_value())
            self._ui.mechanism_helpers.set_temperature_mode_indicator_text(
                f"Temperature: {temperature_k:.2f} K (energy mode: set T=... in DSL)"
            )
        else:
            self._ui.mechanism_helpers.set_temperature_override_state(
                enabled=True,
                tooltip="Temperature for thermodynamic calculations",
            )
            self._ui.mechanism_helpers.update_temperature_mode_indicator()
        return energy_mode

    def remember_primary_result_mechanism(
        self,
        *,
        mechanism: object,
        mechanism_text: str,
        solver_config: Mapping[str, Any],
    ) -> None:
        self._ui.mechanism_helpers.remember_last_mechanism(
            mechanism,
            str(mechanism_text),
            dict(solver_config or {}),
        )
        try:
            self._ui.batch.sync_batch_species_columns(
                mechanism.species_names(),
                retain_active_cache_identity=True,
            )
        except Exception as exc:
            self._record_nonfatal_exception(
                "Failed to sync batch species columns after primary simulation completion",
                exc,
            )

    def refresh_primary_result_controls(
        self,
        *,
        mechanism: object | None,
        energy_mode: bool,
        slider_triggered: bool,
        is_primary: bool,
    ) -> None:
        if not bool(is_primary):
            return
        if bool(energy_mode) and mechanism is not None:
            self._ui.variable_runtime.populate_energy_mode_variables_from_mechanism(
                mechanism,
                refresh_sliders=bool((not self._ui.slider.suppress_slider_refresh()) and (not bool(slider_triggered))),
                preserve_visibility=True,
            )
            if bool(slider_triggered):
                self._ui.slider.set_slider_triggered_simulation(False)
            return
        if self._ui.slider.suppress_slider_refresh():
            if bool(slider_triggered):
                logger.debug("Suppressed slider refresh during live drag")
                self._ui.slider.set_slider_triggered_simulation(False)
            return
        if not bool(slider_triggered):
            self._ui.variable_runtime.extract_and_populate_variables(
                preserve_visibility=True
            )
            return
        self._ui.slider.set_slider_triggered_simulation(False)
        logger.debug("Skipped variable extraction (slider-triggered simulation)")
