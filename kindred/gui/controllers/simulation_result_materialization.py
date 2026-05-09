from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


class SimulationResultMaterializationOwner:
    def __init__(
        self,
        *,
        ui: Any,
        record_nonfatal_exception: Callable[[str, BaseException], None],
    ) -> None:
        self._ui = ui
        self._record_nonfatal_exception = record_nonfatal_exception

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

            temp_for_parse = float(
                (solver_config or {}).get("temperature_K") or self._ui.solver.temperature_spinbox_value()
            )
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
                self._ui.runtime.is_energy_mode_mechanism(mechanism)
                or self._ui.runtime.dsl_has_computational_mode_generated_block(str(mechanism_text))
            )
        )
        if (not bool(is_primary)) or bool(is_preview):
            return energy_mode
        if (
            energy_mode
            and mechanism is not None
            and self._ui.solver.dsl_global_temperature_K(str(mechanism_text)) is not None
        ):
            self._ui.runtime.sync_energy_mode_temperature_from_mechanism(mechanism)
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
                preserve_active_cache=True,
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
            self._ui.runtime.populate_energy_mode_variables_from_mechanism(
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
            self._ui.runtime.extract_and_populate_variables(
                preserve_visibility=True
            )
            return
        self._ui.slider.set_slider_triggered_simulation(False)
        logger.debug("Skipped variable extraction (slider-triggered simulation)")

