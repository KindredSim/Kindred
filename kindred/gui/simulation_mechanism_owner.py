from __future__ import annotations

from typing import Callable, Dict, Optional

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.validation import try_parse_finite_float


class SimulationMechanismOwner:
    """Thin Qt adapter for mechanism-session text and mechanism editor controls."""

    def __init__(
        self,
        *,
        mechanism_session_owner_getter: Callable[[], object | None],
        mechanism_editor_getter: Callable[[], object | None],
        preview_session: object,
        variable_runtime: object,
        mechanism_locked_getter: Callable[[], bool],
        try_lock_mechanism_editor: Callable[[], bool],
        apply_reactions_overrides_to_text: Callable[..., str],
        apply_state_network_overrides_to_dsl: Callable[..., str],
        apply_wegscheider_resolution_reactions_rewrite: Callable[[str], None] | None = None,
    ) -> None:
        self._mechanism_session_owner_getter = mechanism_session_owner_getter
        self._mechanism_editor_getter = mechanism_editor_getter
        self._preview_session = preview_session
        self._variable_runtime = variable_runtime
        self._mechanism_locked_getter = mechanism_locked_getter
        self._try_lock_mechanism_editor = try_lock_mechanism_editor
        self._apply_reactions_overrides_to_text = apply_reactions_overrides_to_text
        self._apply_state_network_overrides_to_dsl = apply_state_network_overrides_to_dsl
        self._apply_wegscheider_resolution_reactions_rewrite = apply_wegscheider_resolution_reactions_rewrite

    def auto_lock_for_run(self) -> bool:
        if not bool(self._mechanism_locked_getter()):
            return bool(self._try_lock_mechanism_editor())
        return True

    def is_mechanism_ready_for_run(self) -> bool:
        owner = self._mechanism_session_owner()
        return bool(owner.is_ready_for_explicit_run())

    def mechanism_reactions_text_raw(self) -> str:
        return str(self._mechanism_session_owner().canonical_reactions_text)

    def mechanism_state_network_dsl_raw(self) -> str:
        return str(self._mechanism_session_owner().canonical_state_network_dsl or "")

    def mechanism_source_for_run(self, *, fast_mode: bool) -> MechanismAuthoringSource:
        owner = self._mechanism_session_owner()
        if bool(fast_mode):
            return owner.preview_source()
        return owner.explicit_run_source()

    def mechanism_source_for_run_set(
        self,
        source: MechanismAuthoringSource,
        *,
        set_id: Optional[str] = None,
        apply_parameter_overrides: bool = True,
        strip_initial_concentrations: bool = False,
    ) -> MechanismAuthoringSource:
        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("source must be a MechanismAuthoringSource.")
        materialized = source
        if self.has_slider_overrides() and bool(apply_parameter_overrides):
            materialized = MechanismAuthoringSource.from_parts(
                reactions_text=self._apply_reactions_overrides_to_text(
                    materialized.reactions_text,
                    set_id=set_id,
                ),
                state_network_dsl=self._apply_state_network_overrides_to_dsl(
                    materialized.state_network_dsl,
                    set_id=set_id,
                ),
            )
        if bool(strip_initial_concentrations):
            materialized = materialized.without_reaction_initial_concentrations()
        return materialized

    def mechanism_slider_points_value(self) -> Optional[int]:
        try:
            return int(self._mechanism_editor().slider_points_value())
        except Exception:
            return None

    def mechanism_slider_solver_value(self) -> Optional[str]:
        try:
            value = self._mechanism_editor().slider_solver_value()
        except Exception:
            return None
        return str(value) if value is not None else None

    def set_variable_sliders(
        self,
        variables: Dict[str, float],
        *,
        metadata: Optional[Dict[str, Dict[str, object]]] = None,
        preserve_visibility: bool = False,
        visibility_scope_signature: object | None = None,
    ) -> None:
        self._mechanism_editor()._variable_sliders.set_variables(
            dict(variables),
            metadata=dict(metadata or {}),
            preserve_visibility=bool(preserve_visibility),
            visibility_scope_signature=visibility_scope_signature,
        )

    def variable_slider_values(self) -> Dict[str, float]:
        sliders = getattr(self._mechanism_editor(), "_variable_sliders", None)
        if sliders is None or not hasattr(sliders, "get_variables"):
            return {}
        values = sliders.get_variables() or {}
        return {str(name): float(value) for name, value in values.items()}

    def variable_metadata(self) -> Dict[str, Dict[str, object]]:
        return self._variable_runtime.variable_metadata()

    def clear_variable_sliders(self) -> None:
        sliders = getattr(self._mechanism_editor(), "_variable_sliders", None)
        if sliders is not None and hasattr(sliders, "clear"):
            sliders.clear()

    def has_slider_overrides(self) -> bool:
        return bool(self._preview_session.has_local_mechanism_workspaces())

    def simulation_schema_id(self, *, fast_mode: bool = False) -> str:
        param_store = self._preview_session.param_store
        schema_text = self.mechanism_source_for_run(fast_mode=bool(fast_mode)).full_dsl
        if str(param_store.schema_text or "") != schema_text:
            param_store.set_schema(schema_text)
        return str(param_store.schema_id or "")

    def simulation_param_fingerprint(self, set_id: Optional[str] = None, *, fast_mode: bool = False) -> str:
        self.simulation_schema_id(fast_mode=bool(fast_mode))
        target_set_id = str(set_id or "").strip()
        param_store = self._preview_session.param_store
        if not param_store.has_local_overrides_for_set(target_set_id):
            return ""
        return str(param_store.param_fingerprint(target_set_id) or "")

    def slider_overrides(self, set_id: Optional[str] = None) -> Dict[str, float]:
        raw = self._preview_session.slider_overrides(set_id=set_id)
        overrides: Dict[str, float] = {}
        for key, value in raw.items():
            parsed, ok = try_parse_finite_float(value)
            if not ok:
                continue
            overrides[str(key)] = float(parsed)
        return overrides

    def get_mechanism_text(self) -> str:
        return self._simulation_schema_text()

    def apply_wegscheider_resolution_reactions_rewrite(self, reactions_text: str) -> None:
        if self._apply_wegscheider_resolution_reactions_rewrite is None:
            raise RuntimeError("Wegscheider resolution Reactions rewrite is unavailable.")
        self._apply_wegscheider_resolution_reactions_rewrite(str(reactions_text))

    def _simulation_schema_text(self) -> str:
        return str(self._mechanism_session_owner().canonical_full_dsl)

    def _mechanism_session_owner(self) -> object:
        owner = self._mechanism_session_owner_getter()
        if owner is None:
            raise RuntimeError("Mechanism session owner is unavailable.")
        return owner

    def _mechanism_editor(self) -> object:
        editor = self._mechanism_editor_getter()
        if editor is None:
            raise RuntimeError("Mechanism editor is unavailable.")
        return editor
