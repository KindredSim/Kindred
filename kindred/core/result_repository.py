"""Owner-authorized result storage and bounded exact-key hydration cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Iterable, Mapping, TypeAlias

import numpy as np

from kindred.core.intervention_schedule import (
    intervention_schedule_identity_fingerprints,
)
from kindred.core.intervention_schedule_compiler import InterventionExecutionContract

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    MechanismTabDocument,
    SimulationExecutionSettings,
    TabScientificSettings,
    canonical_analysis_identity,
    canonical_initial_values,
    validate_simulation_execution_settings,
    validate_tab_scientific_settings,
)

__all__ = [
    "ExplicitResultEntry",
    "ExplicitResultKey",
    "explicit_result_key_for_tab",
    "explicit_result_key_from_values",
    "PreviewResultIdentity",
    "preview_result_identity_for_tab",
    "PreviewResultEntry",
    "PreviewResultKey",
    "ResultEntry",
    "ResultRepository",
    "SimulationResultProvenance",
    "SimulationResultValue",
    "scientific_identity",
    "standard_concentration_identity",
]


def _required_id(value: object, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must be nonblank.")
    return normalized


def standard_concentration_identity(prepared_dsl: object) -> str:
    """Return the C0 contribution to executable science, not source provenance.

    ``MechanismAuthoringSource`` remains on every result key so an exact source
    capture is retained for provenance.  C0 is additionally represented by its
    executable contribution: changing it is scientific only where an Eyring
    molecularity or an auto-derived equilibrium reverse rate consumes it.
    """

    mechanism_for_execution = getattr(prepared_dsl, "mechanism_for_execution", None)
    if not callable(mechanism_for_execution):
        raise TypeError("prepared_dsl must provide mechanism_for_execution().")
    mechanism = mechanism_for_execution()
    metadata = getattr(mechanism, "metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("prepared DSL mechanism metadata must be a mapping.")

    from kindred.core.kinetics import (
        EyringRateModel,
        forward_rate_model_for_equilibrium,
        forward_rate_model_for_reaction,
    )
    from kindred.core.equilibrium_rate_authority import (
        EquilibriumRateAuthorityKind,
        normalize_existing_equilibrium_rate_authority,
    )
    from kindred.core.mechanism_metadata import MechanismMetadataView

    view = MechanismMetadataView.from_metadata(metadata)
    consumes_c0 = False
    for reaction in tuple(getattr(mechanism, "reactions", ()) or ()):
        model = forward_rate_model_for_reaction(
            reaction,
            default_kappa=view.kappa_global,
            default_standard_conc_M=view.standard_conc_M,
        )
        if isinstance(model, EyringRateModel) and model.molecularity != 1.0:
            consumes_c0 = True
            break
    if not consumes_c0:
        for equilibrium in tuple(getattr(mechanism, "equilibria", ()) or ()):
            model = forward_rate_model_for_equilibrium(
                equilibrium,
                temperature_K=view.temperature_K,
                default_kappa=view.kappa_global,
                default_standard_conc_M=view.standard_conc_M,
            )
            if isinstance(model, EyringRateModel) and model.molecularity != 1.0:
                consumes_c0 = True
                break
            authority = normalize_existing_equilibrium_rate_authority(
                equilibrium
            )
            forward_order = math.fsum(
                float(value)
                for value in dict(getattr(equilibrium, "stoich_forward", {})).values()
            )
            reverse_order = math.fsum(
                float(value)
                for value in dict(getattr(equilibrium, "stoich_back", {})).values()
            )
            standard_state_exponent = math.fsum((forward_order, -reverse_order))
            if (
                authority.kind == EquilibriumRateAuthorityKind.KEQ
                and standard_state_exponent != 0.0
            ):
                consumes_c0 = True
                break
    if not consumes_c0:
        return "C0:inert"
    return f"C0:consumed:{float(view.standard_conc_M).hex()}"


def scientific_identity(prepared_dsl: object) -> str:
    """Return a deterministic identity for accepted executable science.

    Authoring text and provenance remain on the result key, but comments and
    formatting never enter this digest.  The payload is assembled from the
    accepted mechanism, retained ASTs, and executable schedule objects.  An
    inert standard concentration is removed recursively; a consumed value is
    retained by :func:`standard_concentration_identity`.
    """

    mechanism_for_execution = getattr(prepared_dsl, "mechanism_for_execution", None)
    if not callable(mechanism_for_execution):
        raise TypeError("prepared_dsl must provide mechanism_for_execution().")
    mechanism = mechanism_for_execution()
    c0_identity = standard_concentration_identity(prepared_dsl)
    metadata = getattr(mechanism, "metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("prepared DSL mechanism metadata must be a mapping.")
    from kindred.core.temperature import temperature_schedule_execution_identity

    schedule = metadata.get("temperature_schedule")
    schedule_identity = temperature_schedule_execution_identity(schedule)
    if schedule is not None and schedule_identity is None:
        raise TypeError(
            "scientific identity requires a serializable temperature schedule."
        )

    def canonical(value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): canonical(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                if str(key) not in {
                    "dsl_analysis_identity",
                    "dsl_source_origins",
                    "algebra_text",
                    "parameter_algebra_spec",
                }
            }
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return value
        if isinstance(value, Real):
            return {"float_hex": float(value).hex()}
        raise TypeError(
            "scientific identity cannot encode executable value "
            f"{value.__class__.__qualname__}."
        )

    from kindred.core.ode_builder import (
        build_effective_rate_coefficient_evaluator,
    )

    coefficient_authority = build_effective_rate_coefficient_evaluator(
        mechanism
    )
    reactions = []
    for reaction in tuple(getattr(mechanism, "reactions", ()) or ()):
        reactions.append(
            {
                "reactants": canonical(getattr(reaction, "reactants", {})),
                "products": canonical(getattr(reaction, "products", {})),
                "rate_orders": canonical(getattr(reaction, "rate_orders", {})),
            }
        )
    equilibria = []
    for equilibrium in tuple(getattr(mechanism, "equilibria", ()) or ()):
        equilibria.append(
            {
                "stoich_forward": canonical(getattr(equilibrium, "stoich_forward", {})),
                "stoich_back": canonical(getattr(equilibrium, "stoich_back", {})),
            }
        )
    intervention_for_execution = getattr(
        prepared_dsl,
        "intervention_schedule_for_execution",
        None,
    )
    if not callable(intervention_for_execution):
        raise TypeError(
            "prepared_dsl must provide intervention_schedule_for_execution()."
        )
    intervention = intervention_for_execution()
    (
        intervention_declarative_fingerprint,
        intervention_executable_fingerprint,
    ) = intervention_schedule_identity_fingerprints(intervention)
    payload: dict[str, object] = {
        "species": tuple(str(name) for name in getattr(mechanism, "species_names", lambda: ())()),
        "reactions": reactions,
        "equilibria": equilibria,
        "coefficient_authority": canonical(
            coefficient_authority.authority_snapshot
        ),
        "temperature_K": canonical(metadata.get("temperature_K", 298.15)),
        "temperature_schedule": canonical(schedule_identity),
        "intervention_schedule": {
            "declarative_fingerprint": intervention_declarative_fingerprint,
            "executable_fingerprint": intervention_executable_fingerprint,
        },
        "c0": c0_identity,
    }
    from kindred.core.algebra.simulation_series import algebra_block_to_source

    compiled = getattr(mechanism, "_compiled_algebra_series", None)
    payload["observable_program"] = (
        None
        if compiled is None
        else algebra_block_to_source(compiled.block)
    )
    parameter_spec_for_execution = getattr(
        prepared_dsl,
        "parameter_algebra_spec_for_execution",
        None,
    )
    if not callable(parameter_spec_for_execution):
        raise TypeError(
            "prepared_dsl must provide parameter_algebra_spec_for_execution()."
        )
    spec = parameter_spec_for_execution()
    block = getattr(spec, "parameter_block", None)
    payload["parameter_program"] = (
        None if block is None else algebra_block_to_source(block)
    )
    if c0_identity == "C0:inert":
        def strip_c0(value: object) -> object:
            if isinstance(value, Mapping):
                return {
                    key: strip_c0(item)
                    for key, item in value.items()
                    if str(key).lower() not in {"standard_conc_m", "std_ratio", "c0"}
                }
            if isinstance(value, list):
                return [strip_c0(item) for item in value]
            if isinstance(value, tuple):
                return [strip_c0(item) for item in value]
            return value

        payload = strip_c0(payload)  # type: ignore[assignment]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"science:{digest}"


def explicit_result_key_from_values(
    *,
    tab_id: str,
    set_id: str,
    mechanism_source: MechanismAuthoringSource,
    prepared_dsl: object,
    analysis_identity: str,
    initial_values: Iterable[tuple[str, float]],
    execution_settings: SimulationExecutionSettings,
    scientific_settings: TabScientificSettings,
) -> "ExplicitResultKey":
    """Construct one explicit key from one retained typed source identity."""

    return ExplicitResultKey(
        tab_id=tab_id,
        set_id=set_id,
        mechanism_source=mechanism_source,
        analysis_identity=analysis_identity,
        standard_concentration_identity=standard_concentration_identity(
            prepared_dsl
        ),
        scientific_identity=scientific_identity(prepared_dsl),
        initial_values=tuple(initial_values),
        execution_settings=execution_settings,
        scientific_settings=scientific_settings,
    )


def explicit_result_key_for_tab(
    tab: MechanismTabDocument,
    set_id: str,
) -> "ExplicitResultKey":
    """Construct the canonical key for one accepted document tab/set."""

    if not isinstance(tab, MechanismTabDocument):
        raise TypeError("tab must be a MechanismTabDocument.")
    prepared_dsl = tab.prepared_dsl
    if prepared_dsl is None:
        raise ValueError(f"Mechanism tab {tab.tab_id!r} has no accepted DSL.")
    return explicit_result_key_from_values(
        tab_id=tab.tab_id,
        set_id=set_id,
        mechanism_source=tab.mechanism_source,
        prepared_dsl=prepared_dsl,
        analysis_identity=canonical_analysis_identity(tab),
        initial_values=canonical_initial_values(tab, set_id),
        execution_settings=tab.resolved_execution_settings(set_id),
        scientific_settings=tab.scientific_settings,
    )


def preview_result_identity_for_tab(
    tab: MechanismTabDocument,
    set_id: str,
    *,
    parameter_values: Iterable[tuple[str, float]],
    initial_values: Iterable[tuple[str, float]] | None = None,
) -> "PreviewResultIdentity":
    """Construct Preview identity from the same accepted tab projection."""

    if not isinstance(tab, MechanismTabDocument):
        raise TypeError("tab must be a MechanismTabDocument.")
    prepared_dsl = tab.prepared_dsl
    if prepared_dsl is None:
        raise ValueError(f"Mechanism tab {tab.tab_id!r} has no accepted DSL.")
    return PreviewResultIdentity(
        tab_id=tab.tab_id,
        set_id=set_id,
        mechanism_source=tab.mechanism_source,
        analysis_identity=canonical_analysis_identity(tab),
        standard_concentration_identity=standard_concentration_identity(
            prepared_dsl
        ),
        parameter_values=tuple(parameter_values),
        initial_values=(
            canonical_initial_values(tab, set_id)
            if initial_values is None
            else tuple(initial_values)
        ),
        execution_settings=tab.resolved_execution_settings(set_id),
        scientific_settings=tab.scientific_settings,
    )


def _finite_pairs(
    values: Iterable[tuple[str, object]],
    *,
    field: str,
    nonnegative: bool = False,
) -> tuple[tuple[str, float], ...]:
    normalized: list[tuple[str, float]] = []
    names: set[str] = set()
    for raw_name, raw_value in values:
        name = _required_id(raw_name, f"{field} name")
        if name in names:
            raise ValueError(f"{field} contains duplicate name {name!r}.")
        if type(raw_value) is bool or not isinstance(raw_value, Real):
            raise TypeError(f"{field} value for {name!r} must be numeric.")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"{field} value for {name!r} must be finite.")
        if nonnegative and value < 0.0:
            raise ValueError(
                f"{field} value for {name!r} must be nonnegative."
            )
        names.add(name)
        normalized.append((name, value))
    return tuple(normalized)


def _frozen_vector(value: object, *, field: str) -> np.ndarray:
    candidate = np.asarray(value, dtype=float)
    if candidate.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional.")
    array = (
        value
        if (
            isinstance(value, np.ndarray)
            and value.dtype == np.dtype(float)
            and value.ndim == 1
            and not value.flags.writeable
        )
        else candidate.copy()
    )
    if array.size == 0:
        raise ValueError(f"{field} must not be empty.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite values.")
    array.setflags(write=False)
    return array


def _frozen_identity_vector(
    value: object,
    *,
    field: str,
    size: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{field} must be a one-dimensional integer array.")
    candidate = raw.astype(np.int64, copy=False)
    array = (
        value
        if (
            isinstance(value, np.ndarray)
            and value.dtype == np.dtype(np.int64)
            and value.ndim == 1
            and not value.flags.writeable
        )
        else candidate.copy()
    )
    if array.size != int(size):
        raise ValueError(f"{field} must align with result time.")
    if np.any(array < 0) or np.any(np.diff(array) < 0):
        raise ValueError(f"{field} must be ordered and nonnegative.")
    array.setflags(write=False)
    return array


def _validate_discontinuity_regions(
    time: np.ndarray,
    discontinuity_ids: np.ndarray,
    *,
    execution: InterventionExecutionContract,
    execution_start: float,
    execution_end: float,
) -> None:
    """Validate executor region identity against exact executed boundaries."""

    if int(discontinuity_ids[0]) != 0:
        raise ValueError("result discontinuity identity must start at zero.")
    expected_advances = {
        float(value)
        for value in execution.discontinuities
        if execution_start < float(value) <= execution_end
    }
    expected_advances.update(
        float(firing.root_time)
        for firing in execution.trigger_firings
        if execution_start < float(firing.root_time) <= execution_end
    )
    observed_advances: set[float] = set()
    for index in range(1, int(time.size)):
        delta = int(discontinuity_ids[index]) - int(
            discontinuity_ids[index - 1]
        )
        if delta not in {0, 1}:
            raise ValueError(
                "result discontinuity identity must advance one region at a time."
            )
        if delta == 1:
            observed_advances.add(float(time[index]))
    if observed_advances != expected_advances:
        raise ValueError(
            "result discontinuity regions conflict with executed intervention boundaries."
        )


def _executed_segment_catalog(
    provenance: SimulationResultProvenance,
) -> tuple[tuple[int, str, float, float, bool, bool], ...]:
    raw_catalog = provenance.solver.get("integration_segments")
    if not isinstance(raw_catalog, (tuple, list)) or not raw_catalog:
        raise ValueError(
            "solver provenance must contain the actual executed segment catalog."
        )
    catalog: list[tuple[int, str, float, float, bool, bool]] = []
    seen_ids: set[int] = set()
    seen_keys: set[str] = set()
    for position, raw_entry in enumerate(raw_catalog):
        if not isinstance(raw_entry, Mapping):
            raise TypeError("executed segment catalog entries must be mappings.")
        raw_id = raw_entry.get("segment_id")
        if type(raw_id) is not int or raw_id < 0:
            raise ValueError(
                "executed segment catalog IDs must be nonnegative integers."
            )
        segment_id = int(raw_id)
        if segment_id != position or segment_id in seen_ids:
            raise ValueError(
                "executed segment catalog IDs must be unique and contiguous."
            )
        segment_key = _required_id(
            raw_entry.get("segment_key"),
            "executed segment key",
        )
        if segment_key in seen_keys:
            raise ValueError("executed segment catalog keys must be unique.")
        try:
            start = float(raw_entry["t_start"])
            end = float(raw_entry["t_end"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "executed segment catalog spans must be explicit finite values."
            ) from exc
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError(
                "executed segment catalog spans must be finite and increasing."
            )
        identity = raw_entry.get("segment_identity")
        if not isinstance(identity, Mapping):
            raise TypeError(
                "executed segment catalog entries require direct segment identity."
            )
        if (
            identity.get("key") != segment_key
            or identity.get("kind") != "integration"
            or identity.get("ordinal") != segment_id
            or identity.get("t_start") != start
            or identity.get("t_end") != end
        ):
            raise ValueError(
                "executed segment catalog identity conflicts with its direct span."
            )
        temperature_boundary = raw_entry.get("temperature_step_boundary")
        starts_after_discontinuity = raw_entry.get(
            "starts_after_discontinuity"
        )
        if type(temperature_boundary) is not bool or type(
            starts_after_discontinuity
        ) is not bool:
            raise TypeError(
                "executed segment catalog boundary facts must be explicit booleans."
            )
        seen_ids.add(segment_id)
        seen_keys.add(segment_key)
        catalog.append(
            (
                segment_id,
                segment_key,
                start,
                end,
                temperature_boundary,
                starts_after_discontinuity,
            )
        )
    for previous, current in zip(catalog, catalog[1:]):
        if current[2] != previous[3]:
            raise ValueError(
                "executed segment catalog must describe one contiguous execution."
            )
    return tuple(catalog)


def _validate_rows_against_executed_segments(
    *,
    time: np.ndarray,
    segment_ids: np.ndarray,
    segment_keys: tuple[str, ...],
    catalog: tuple[tuple[int, str, float, float, bool, bool], ...],
) -> None:
    by_id = {entry[0]: entry for entry in catalog}
    for row, (raw_id, key) in enumerate(
        zip(segment_ids, segment_keys, strict=True)
    ):
        segment = by_id.get(int(raw_id))
        if segment is None or key != segment[1]:
            raise ValueError(
                "every result row must name one actual executed segment."
            )
        row_time = float(time[row])
        if row_time < segment[2] or row_time > segment[3]:
            raise ValueError(
                "result row time lies outside its executed segment span."
            )

    if (
        float(time[0]) != catalog[0][2]
        or int(segment_ids[0]) != catalog[0][0]
        or segment_keys[0] != catalog[0][1]
    ):
        raise ValueError(
            "the first public result row must identify execution start."
        )
    if float(time[-1]) != catalog[-1][3]:
        raise ValueError(
            "the final public result row must identify execution end."
        )

    for (
        segment_id,
        segment_key,
        start,
        _end,
        temperature_boundary,
        starts_after_discontinuity,
    ) in catalog:
        if segment_id == 0 or not (
            temperature_boundary or starts_after_discontinuity
        ):
            continue
        matching_rows = [
            row
            for row in range(int(time.size))
            if int(segment_ids[row]) == segment_id
            and segment_keys[row] == segment_key
            and float(time[row]) == start
        ]
        if len(matching_rows) != 1:
            raise ValueError(
                "every public execution boundary must publish one row in its new segment."
            )


def _validate_direct_solver_row_identity(
    *,
    provenance: SimulationResultProvenance,
    segment_ids: np.ndarray,
    segment_keys: tuple[str, ...],
    discontinuity_ids: np.ndarray,
) -> None:
    direct = provenance.solver
    retired = frozenset(
        {
            "boundary_sides",
            "boundary_phases",
            "result_boundary_sides",
            "result_boundary_phases",
        }
    ).intersection(direct)
    if retired:
        raise ValueError(
            "solver result contains retired boundary identity: "
            + ", ".join(sorted(retired))
        )
    expected = {
        "result_segment_ids": tuple(int(value) for value in segment_ids),
        "result_segment_keys": segment_keys,
        "result_discontinuity_ids": tuple(
            int(value) for value in discontinuity_ids
        ),
    }
    for field, value in expected.items():
        raw = direct.get(field)
        if raw is None or isinstance(raw, (str, bytes)):
            raise ValueError(
                f"solver result is missing direct row identity field {field!r}."
            )
        if tuple(raw) != value:  # type: ignore[arg-type]
            raise ValueError(
                f"explicit result identity conflicts with solver field {field!r}."
            )


def _freeze_provenance_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(name): _freeze_provenance_value(item)
                for name, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_provenance_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(
            _freeze_provenance_value(item)
            for item in sorted(value, key=repr)
        )
    if isinstance(value, np.ndarray):
        array = np.asarray(value).copy()
        array.setflags(write=False)
        return array
    return value


def _frozen_provenance_mapping(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(name): _freeze_provenance_value(item)
            for name, item in value.items()
        }
    )


@dataclass(frozen=True, slots=True)
class ExplicitResultKey:
    """Exact reusable scientific identity for one explicit Run target."""

    tab_id: str
    set_id: str
    mechanism_source: MechanismAuthoringSource
    analysis_identity: str
    standard_concentration_identity: str
    scientific_identity: str
    initial_values: tuple[tuple[str, float], ...]
    execution_settings: SimulationExecutionSettings
    scientific_settings: TabScientificSettings

    def __post_init__(self) -> None:
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "set_id", _required_id(self.set_id, "set_id"))
        if not isinstance(self.mechanism_source, MechanismAuthoringSource):
            raise TypeError("mechanism_source must be a MechanismAuthoringSource.")
        object.__setattr__(
            self,
            "analysis_identity",
            _required_id(self.analysis_identity, "analysis_identity"),
        )
        object.__setattr__(
            self,
            "standard_concentration_identity",
            _required_id(
                self.standard_concentration_identity,
                "standard_concentration_identity",
            ),
        )
        object.__setattr__(
            self,
            "scientific_identity",
            _required_id(self.scientific_identity, "scientific_identity"),
        )
        object.__setattr__(
            self,
            "initial_values",
            _finite_pairs(
                self.initial_values,
                field="initial_values",
                nonnegative=True,
            ),
        )
        if not isinstance(self.execution_settings, SimulationExecutionSettings):
            raise TypeError(
                "execution_settings must be a SimulationExecutionSettings value."
            )
        validate_simulation_execution_settings(self.execution_settings)
        validate_tab_scientific_settings(self.scientific_settings)

    @property
    def cache_identity(self) -> tuple[object, ...]:
        """Exact numerical-cache identity, deliberately excluding provenance."""

        return (
            self.tab_id,
            self.set_id,
            self.scientific_identity,
            self.initial_values,
            self.execution_settings,
            self.scientific_settings,
        )


@dataclass(frozen=True, slots=True)
class PreviewResultIdentity:
    """Generation-free scientific identity captured by the Slider owner."""

    tab_id: str
    set_id: str
    mechanism_source: MechanismAuthoringSource
    analysis_identity: str
    standard_concentration_identity: str
    parameter_values: tuple[tuple[str, float], ...]
    initial_values: tuple[tuple[str, float], ...]
    execution_settings: SimulationExecutionSettings
    scientific_settings: TabScientificSettings

    def __post_init__(self) -> None:
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "set_id", _required_id(self.set_id, "set_id"))
        if not isinstance(self.mechanism_source, MechanismAuthoringSource):
            raise TypeError("mechanism_source must be a MechanismAuthoringSource.")
        object.__setattr__(
            self,
            "analysis_identity",
            _required_id(self.analysis_identity, "analysis_identity"),
        )
        object.__setattr__(
            self,
            "standard_concentration_identity",
            _required_id(
                self.standard_concentration_identity,
                "standard_concentration_identity",
            ),
        )
        object.__setattr__(
            self,
            "parameter_values",
            _finite_pairs(self.parameter_values, field="parameter_values"),
        )
        object.__setattr__(
            self,
            "initial_values",
            _finite_pairs(
                self.initial_values,
                field="initial_values",
                nonnegative=True,
            ),
        )
        if not isinstance(self.execution_settings, SimulationExecutionSettings):
            raise TypeError(
                "execution_settings must be a SimulationExecutionSettings value."
            )
        validate_simulation_execution_settings(self.execution_settings)
        validate_tab_scientific_settings(self.scientific_settings)


@dataclass(frozen=True, slots=True)
class PreviewResultKey:
    """Exact latest-only identity minted by ``SimulationJobOwner``."""

    tab_id: str
    set_id: str
    mechanism_source: MechanismAuthoringSource
    analysis_identity: str
    standard_concentration_identity: str
    parameter_values: tuple[tuple[str, float], ...]
    initial_values: tuple[tuple[str, float], ...]
    execution_settings: SimulationExecutionSettings
    scientific_settings: TabScientificSettings
    generation: int
    kind: str = "preview"

    def __post_init__(self) -> None:
        identity = PreviewResultIdentity(
            tab_id=self.tab_id,
            set_id=self.set_id,
            mechanism_source=self.mechanism_source,
            analysis_identity=self.analysis_identity,
            standard_concentration_identity=self.standard_concentration_identity,
            parameter_values=self.parameter_values,
            initial_values=self.initial_values,
            execution_settings=self.execution_settings,
            scientific_settings=self.scientific_settings,
        )
        object.__setattr__(self, "tab_id", identity.tab_id)
        object.__setattr__(self, "set_id", identity.set_id)
        object.__setattr__(self, "mechanism_source", identity.mechanism_source)
        object.__setattr__(self, "analysis_identity", identity.analysis_identity)
        object.__setattr__(
            self,
            "standard_concentration_identity",
            identity.standard_concentration_identity,
        )
        object.__setattr__(self, "parameter_values", identity.parameter_values)
        object.__setattr__(self, "initial_values", identity.initial_values)
        object.__setattr__(
            self,
            "execution_settings",
            identity.execution_settings,
        )
        object.__setattr__(
            self,
            "scientific_settings",
            identity.scientific_settings,
        )
        if self.kind != "preview":
            raise ValueError("PreviewResultKey kind must be exactly 'preview'.")
        if type(self.generation) is not int:
            raise TypeError("Preview generation must be an int.")
        generation = self.generation
        if generation < 0:
            raise ValueError("Preview generation must be nonnegative.")
        object.__setattr__(self, "generation", generation)

    @classmethod
    def from_identity(
        cls,
        identity: PreviewResultIdentity,
        *,
        generation: int,
    ) -> PreviewResultKey:
        if not isinstance(identity, PreviewResultIdentity):
            raise TypeError("identity must be a PreviewResultIdentity.")
        return cls(
            tab_id=identity.tab_id,
            set_id=identity.set_id,
            mechanism_source=identity.mechanism_source,
            analysis_identity=identity.analysis_identity,
            standard_concentration_identity=identity.standard_concentration_identity,
            parameter_values=identity.parameter_values,
            initial_values=identity.initial_values,
            execution_settings=identity.execution_settings,
            scientific_settings=identity.scientific_settings,
            generation=generation,
        )

    @property
    def identity(self) -> PreviewResultIdentity:
        return PreviewResultIdentity(
            tab_id=self.tab_id,
            set_id=self.set_id,
            mechanism_source=self.mechanism_source,
            analysis_identity=self.analysis_identity,
            standard_concentration_identity=self.standard_concentration_identity,
            parameter_values=self.parameter_values,
            initial_values=self.initial_values,
            execution_settings=self.execution_settings,
            scientific_settings=self.scientific_settings,
        )


@dataclass(frozen=True, slots=True)
class SimulationResultProvenance:
    """Typed immutable provenance retained with one successful result."""

    solver: Mapping[str, Any]
    preparation_warnings: tuple[str, ...] = ()
    requested_algebra_outputs: tuple[str, ...] = ()
    succeeded_algebra_outputs: tuple[str, ...] = ()
    failed_algebra_outputs: tuple[str, ...] = ()
    omitted_algebra_outputs: tuple[str, ...] = ()
    algebra_errors: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.solver, Mapping):
            raise TypeError("solver provenance must be a mapping.")
        preparation_warnings = tuple(
            str(value) for value in self.preparation_warnings
        )
        requested = self._normalized_algebra_names(
            self.requested_algebra_outputs,
            field="requested_algebra_outputs",
        )
        succeeded = self._normalized_algebra_names(
            self.succeeded_algebra_outputs,
            field="succeeded_algebra_outputs",
        )
        failed = self._normalized_algebra_names(
            self.failed_algebra_outputs,
            field="failed_algebra_outputs",
        )
        omitted = self._normalized_algebra_names(
            self.omitted_algebra_outputs,
            field="omitted_algebra_outputs",
        )
        requested_set = set(requested)
        partitions = {
            "succeeded_algebra_outputs": set(succeeded),
            "failed_algebra_outputs": set(failed),
            "omitted_algebra_outputs": set(omitted),
        }
        for field, values in partitions.items():
            unexpected = values.difference(requested_set)
            if unexpected:
                raise ValueError(
                    f"{field} contains unrequested output(s): "
                    + ", ".join(sorted(unexpected))
                )
        overlap = (
            partitions["succeeded_algebra_outputs"]
            & partitions["failed_algebra_outputs"]
        ) | (
            partitions["succeeded_algebra_outputs"]
            & partitions["omitted_algebra_outputs"]
        ) | (
            partitions["failed_algebra_outputs"]
            & partitions["omitted_algebra_outputs"]
        )
        if overlap:
            raise ValueError(
                "algebra output outcome partitions overlap: "
                + ", ".join(sorted(overlap))
            )
        classified = set().union(*partitions.values())
        missing = requested_set.difference(classified)
        if missing:
            raise ValueError(
                "requested algebra outputs lack a terminal outcome: "
                + ", ".join(sorted(missing))
            )
        errors: list[Mapping[str, Any]] = []
        for value in self.algebra_errors:
            if not isinstance(value, Mapping):
                raise TypeError(
                    "algebra_errors must contain only mapping values."
                )
            errors.append(_frozen_provenance_mapping(value))
        object.__setattr__(
            self,
            "solver",
            _frozen_provenance_mapping(self.solver),
        )
        object.__setattr__(
            self,
            "preparation_warnings",
            preparation_warnings,
        )
        object.__setattr__(self, "requested_algebra_outputs", requested)
        object.__setattr__(self, "succeeded_algebra_outputs", succeeded)
        object.__setattr__(self, "failed_algebra_outputs", failed)
        object.__setattr__(self, "omitted_algebra_outputs", omitted)
        object.__setattr__(self, "algebra_errors", tuple(errors))

    @staticmethod
    def _normalized_algebra_names(
        values: Iterable[str],
        *,
        field: str,
    ) -> tuple[str, ...]:
        normalized = tuple(_required_id(value, field) for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{field} contains duplicate output names.")
        return normalized

    @property
    def completed_with_warnings(self) -> bool:
        """Whether prominent completion warning presentation is required."""

        return bool(
            self.preparation_warnings
            or self.failed_algebra_outputs
            or self.omitted_algebra_outputs
        )


@dataclass(frozen=True, slots=True, eq=False)
class SimulationResultValue:
    """Immutable numerical output with executor-issued execution identity."""

    t: np.ndarray
    series: Mapping[str, np.ndarray]
    segment_ids: np.ndarray
    segment_keys: tuple[str, ...]
    discontinuity_ids: np.ndarray
    intervention_execution: InterventionExecutionContract
    provenance: SimulationResultProvenance
    scalars: Mapping[str, float] = MappingProxyType({})
    parameters: tuple[tuple[str, float, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, SimulationResultProvenance):
            raise TypeError(
                "provenance must be a SimulationResultProvenance."
            )
        if not isinstance(
            self.intervention_execution, InterventionExecutionContract
        ):
            raise TypeError(
                "intervention_execution must be an InterventionExecutionContract."
            )
        time = _frozen_vector(self.t, field="result time")
        if time.size > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError("result time must be strictly increasing.")
        segment_ids = _frozen_identity_vector(
            self.segment_ids,
            field="result segment identity",
            size=time.size,
        )
        discontinuity_ids = _frozen_identity_vector(
            self.discontinuity_ids,
            field="result discontinuity identity",
            size=time.size,
        )
        segment_keys = tuple(
            _required_id(value, "result segment key")
            for value in self.segment_keys
        )
        if len(segment_keys) != time.size:
            raise ValueError("result segment keys must align with result time.")
        _validate_direct_solver_row_identity(
            provenance=self.provenance,
            segment_ids=segment_ids,
            segment_keys=segment_keys,
            discontinuity_ids=discontinuity_ids,
        )
        catalog = _executed_segment_catalog(self.provenance)
        _validate_rows_against_executed_segments(
            time=time,
            segment_ids=segment_ids,
            segment_keys=segment_keys,
            catalog=catalog,
        )
        _validate_discontinuity_regions(
            time,
            discontinuity_ids,
            execution=self.intervention_execution,
            execution_start=catalog[0][2],
            execution_end=catalog[-1][3],
        )

        series: dict[str, np.ndarray] = {}
        for raw_name, raw_values in dict(self.series or {}).items():
            name = _required_id(raw_name, "series name")
            values = _frozen_vector(raw_values, field=f"series {name!r}")
            if values.size != time.size:
                raise ValueError(
                    f"series {name!r} has {values.size} values for "
                    f"{time.size} time points."
                )
            series[name] = values
        if not series:
            raise ValueError("result must contain at least one series.")

        scalars: dict[str, float] = {}
        for raw_name, raw_value in dict(self.scalars or {}).items():
            name = _required_id(raw_name, "scalar name")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(f"scalar {name!r} must be finite.")
            scalars[name] = value

        parameters: list[tuple[str, float, str]] = []
        parameter_names: set[str] = set()
        for raw_name, raw_value, raw_unit in self.parameters:
            name = _required_id(raw_name, "parameter name")
            if name in parameter_names:
                raise ValueError(f"parameters contain duplicate name {name!r}.")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(f"parameter {name!r} must be finite.")
            parameter_names.add(name)
            parameters.append((name, value, str(raw_unit or "")))

        object.__setattr__(self, "t", time)
        object.__setattr__(self, "series", MappingProxyType(series))
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_keys", segment_keys)
        object.__setattr__(self, "discontinuity_ids", discontinuity_ids)
        object.__setattr__(self, "scalars", MappingProxyType(scalars))
        object.__setattr__(self, "parameters", tuple(parameters))


@dataclass(frozen=True, slots=True)
class ExplicitResultEntry:
    key: ExplicitResultKey
    result: SimulationResultValue


@dataclass(frozen=True, slots=True)
class PreviewResultEntry:
    key: PreviewResultKey
    result: SimulationResultValue


ResultEntry: TypeAlias = ExplicitResultEntry | PreviewResultEntry


class ResultRepository:
    """Store results while enforcing owner-issued Preview display eligibility."""

    def __init__(
        self,
        tab_id: str,
        *,
        cache_capacity: int,
    ) -> None:
        self._tab_id = _required_id(tab_id, "tab_id")
        if type(cache_capacity) is not int:
            raise TypeError("cache_capacity must be an int.")
        self._cache_capacity = cache_capacity
        if cache_capacity < 1:
            raise ValueError("cache_capacity must be at least one.")
        self._explicit_by_set_id: dict[str, ExplicitResultEntry] = {}
        self._preview_by_set_id: dict[str, PreviewResultEntry] = {}
        self._preview_display_key_by_set_id: dict[
            str,
            PreviewResultKey,
        ] = {}
        self._reference_by_set_id: dict[str, ExplicitResultEntry] = {}
        self._explicit_cache: OrderedDict[
            tuple[object, ...],
            SimulationResultValue,
        ] = OrderedDict()

    @property
    def tab_id(self) -> str:
        return self._tab_id

    @property
    def cache_capacity(self) -> int:
        return self._cache_capacity

    def resize_cache_capacity(self, cache_capacity: int) -> None:
        """Apply a positive capacity immediately and trim this repository's LRU."""

        if type(cache_capacity) is not int:
            raise TypeError("cache_capacity must be an int.")
        if cache_capacity < 1:
            raise ValueError("cache_capacity must be at least one.")
        self._cache_capacity = cache_capacity
        self._trim_explicit_cache()

    def explicit_entry(self, set_id: str) -> ExplicitResultEntry | None:
        return self._explicit_by_set_id.get(str(set_id or ""))

    def explicit_entries(self) -> tuple[ExplicitResultEntry, ...]:
        return tuple(self._explicit_by_set_id.values())

    def preview_entry(self, set_id: str) -> PreviewResultEntry | None:
        sid = str(set_id or "")
        entry = self._preview_by_set_id.get(sid)
        if (
            entry is None
            or self._preview_display_key_by_set_id.get(sid) != entry.key
        ):
            return None
        return entry

    def preview_entries(self) -> tuple[PreviewResultEntry, ...]:
        return tuple(
            entry
            for set_id, entry in self._preview_by_set_id.items()
            if self._preview_display_key_by_set_id.get(set_id) == entry.key
        )

    def reference_entry(self, set_id: str) -> ExplicitResultEntry | None:
        return self._reference_by_set_id.get(str(set_id or ""))

    def capture_preview_reference(self, key: PreviewResultKey) -> None:
        """Move the exact foreground Run to Reference before Preview debounce."""
        self._require_preview_key(key)
        explicit = self._explicit_by_set_id.pop(key.set_id, None)
        if explicit is not None:
            self._reference_by_set_id[key.set_id] = explicit

    def authorize_preview_display(self, key: PreviewResultKey) -> None:
        """Authorize exactly one owner-current key for display/export."""

        self._require_preview_key(key)
        self._preview_display_key_by_set_id[key.set_id] = key

    def store_preview(
        self,
        key: PreviewResultKey,
        result: SimulationResultValue,
    ) -> PreviewResultEntry:
        """Store one Preview result after owner authorization."""
        self._require_preview_key(key)
        if not isinstance(result, SimulationResultValue):
            raise TypeError("result must be a SimulationResultValue.")
        if self._preview_display_key_by_set_id.get(key.set_id) != key:
            raise ValueError(
                "Preview result key is not owner-authorized for display."
            )
        entry = PreviewResultEntry(key=key, result=result)
        self._preview_by_set_id[key.set_id] = entry
        self._explicit_by_set_id.pop(key.set_id, None)
        return entry

    def clear_preview(self, set_id: str, *, clear_reference: bool = True) -> None:
        sid = _required_id(set_id, "set_id")
        self._preview_by_set_id.pop(sid, None)
        self._preview_display_key_by_set_id.pop(sid, None)
        if clear_reference:
            self._reference_by_set_id.pop(sid, None)

    def put_explicit(
        self,
        key: ExplicitResultKey,
        result: SimulationResultValue,
    ) -> ExplicitResultEntry:
        self._require_explicit_key(key)
        if not isinstance(result, SimulationResultValue):
            raise TypeError("result must be a SimulationResultValue.")
        entry = ExplicitResultEntry(key=key, result=result)
        self._explicit_by_set_id[key.set_id] = entry
        self._preview_by_set_id.pop(key.set_id, None)
        self._preview_display_key_by_set_id.pop(key.set_id, None)
        self._reference_by_set_id.pop(key.set_id, None)
        cache_identity = key.cache_identity
        self._explicit_cache[cache_identity] = result
        self._explicit_cache.move_to_end(cache_identity)
        self._trim_explicit_cache()
        return entry

    def cached_result(self, key: ExplicitResultKey) -> SimulationResultValue | None:
        self._require_explicit_key(key)
        cache_identity = key.cache_identity
        result = self._explicit_cache.get(cache_identity)
        if result is not None:
            self._explicit_cache.move_to_end(cache_identity)
        return result

    def hydrate_cached(self, key: ExplicitResultKey) -> ExplicitResultEntry | None:
        """Hydrate one owner-authorized exact cache hit; a miss changes nothing."""
        result = self.cached_result(key)
        if result is None:
            retained = self._reference_by_set_id.get(key.set_id)
            if retained is None:
                retained = self._explicit_by_set_id.get(key.set_id)
            if retained is not None and retained.key == key:
                result = retained.result
        if result is None:
            return None
        entry = ExplicitResultEntry(key=key, result=result)
        self._explicit_by_set_id[key.set_id] = entry
        self._preview_by_set_id.pop(key.set_id, None)
        self._preview_display_key_by_set_id.pop(key.set_id, None)
        self._reference_by_set_id.pop(key.set_id, None)
        return entry

    def clear_semantic_set(self, set_id: str) -> None:
        """Clear visible semantic entries while retaining exact cache storage."""
        sid = _required_id(set_id, "set_id")
        self._explicit_by_set_id.pop(sid, None)
        self._preview_by_set_id.pop(sid, None)
        self._preview_display_key_by_set_id.pop(sid, None)
        self._reference_by_set_id.pop(sid, None)

    def clear_semantic_sets(self, set_ids: Iterable[str]) -> None:
        seen: set[str] = set()
        for set_id in set_ids:
            sid = _required_id(set_id, "set_id")
            if sid not in seen:
                seen.add(sid)
                self.clear_semantic_set(sid)

    def deauthorize_sets(
        self,
        set_ids: Iterable[str],
    ) -> None:
        """Clear display authorization while retaining exact reusable values."""

        targets = {
            _required_id(set_id, "set_id")
            for set_id in set_ids
        }
        self.clear_semantic_sets(targets)

    def clear_all_semantic_entries(self) -> None:
        self._explicit_by_set_id.clear()
        self._preview_by_set_id.clear()
        self._preview_display_key_by_set_id.clear()
        self._reference_by_set_id.clear()

    def purge_set(self, set_id: str) -> None:
        """Delete/history removal purges visible entries and cached storage."""

        sid = _required_id(set_id, "set_id")
        self.clear_semantic_set(sid)
        for cache_identity in tuple(self._explicit_cache):
            if cache_identity[1] == sid:
                self._explicit_cache.pop(cache_identity, None)

    def purge_sets(self, set_ids: Iterable[str]) -> None:
        for set_id in tuple(set_ids):
            self.purge_set(set_id)

    def clear(self) -> None:
        self.clear_all_semantic_entries()
        self._explicit_cache.clear()

    def _trim_explicit_cache(self) -> None:
        while len(self._explicit_cache) > self._cache_capacity:
            self._explicit_cache.popitem(last=False)

    def _require_explicit_key(self, key: ExplicitResultKey) -> None:
        if not isinstance(key, ExplicitResultKey):
            raise TypeError("key must be an ExplicitResultKey.")
        if key.tab_id != self._tab_id:
            raise ValueError(
                f"ExplicitResultKey belongs to tab {key.tab_id!r}, "
                f"not {self._tab_id!r}."
            )

    def _require_preview_key(self, key: PreviewResultKey) -> None:
        if not isinstance(key, PreviewResultKey):
            raise TypeError("key must be a PreviewResultKey.")
        if key.tab_id != self._tab_id:
            raise ValueError(
                f"PreviewResultKey belongs to tab {key.tab_id!r}, "
                f"not {self._tab_id!r}."
            )
