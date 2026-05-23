from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Optional, Sequence

from kindred.core.runtime_defaults import (
    USE_SPARSE_JACOBIAN_DEFAULT,
    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
)
from kindred.core.symbolic.identity import normalize_symbolic_identity_mapping

__all__ = [
    "SimulationIdentity",
    "SimulationScopeIdentity",
    "SimulationSolverIdentity",
    "canonical_initials_fingerprint",
    "contained_simulation_owner_identity",
    "coerce_simulation_identity",
    "coerce_simulation_scope_identity",
    "schedule_fingerprint_payload",
]


def _try_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _canonical_json_bytes(payload: object) -> bytes:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return serialized.encode("utf-8", "ignore")


def _sha256_text(value: object) -> str:
    text = "" if value is None else str(value)
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _symbolic_jacobian_structure_payload(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or not identity:
        return {}
    payload = {
        "kind": str(identity.get("kind") or ""),
        "backend_name": str(identity.get("backend_name") or ""),
        "backend_version": str(identity.get("backend_version") or ""),
        "profile_version": str(identity.get("profile_version") or ""),
        "source_fingerprint": str(identity.get("source_fingerprint") or ""),
        "structure_fingerprint": str(
            identity.get("structure_fingerprint")
            or identity.get("source_fingerprint")
            or ""
        ),
        "artifact_fingerprint": str(identity.get("artifact_fingerprint") or ""),
        "parameter_symbols": [
            str(name)
            for name in (identity.get("parameter_symbols") or ())
            if str(name)
        ],
    }
    return payload if payload["kind"] else {}


def _symbolic_wegscheider_source_payload(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or not identity:
        return {}
    source_fingerprint = str(identity.get("source_fingerprint") or "")
    if not source_fingerprint:
        return dict(identity)
    payload = {
        "kind": str(identity.get("kind") or ""),
        "backend_name": str(identity.get("backend_name") or ""),
        "backend_version": str(identity.get("backend_version") or ""),
        "profile_version": str(identity.get("profile_version") or ""),
        "source_fingerprint": source_fingerprint,
    }
    return payload if payload["kind"] else {}


def canonical_initials_fingerprint(initials: Mapping[str, Any] | None) -> str:
    if not isinstance(initials, Mapping):
        return ""
    payload: dict[str, float | str] = {}
    for raw_name, raw_value in initials.items():
        name = str(raw_name or "").strip()
        if not name:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            payload[name] = str(raw_value)
            continue
        if math.isfinite(value):
            payload[name] = float(value)
        else:
            payload[name] = str(raw_value)
    if not payload:
        return ""
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def schedule_fingerprint_payload(value: object) -> str:
    try:
        from kindred.core.intervention_schedule import coerce_intervention_schedule

        schedule = coerce_intervention_schedule(value)
    except Exception:
        return str(value or "")
    return "" if schedule is None else schedule.fingerprint


_DSL_PARAMETER_ASSIGNMENT_RE = re.compile(
    r"(?P<prefix>(?:^|[;,])\s*)"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*|Keq[0-9]*)"
    r"(?P<equals>\s*=\s*)"
    r"(?P<value>[^;,#\n]+)"
)
_DSL_PARAM_STATEMENT_RE = re.compile(
    r"^(?P<prefix>\s*param\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*)(?P<value>.+)$",
    re.IGNORECASE,
)
_DSL_REACTION_ARROW_RE = re.compile(r"(<->|<=>|->|=>|⇌|→|↔)")
_DSL_SEMICOLON_KEY_ALIASES = {
    "a": "A",
    "ea": "Ea",
    "dg_act": "dG_act",
    "dg_eq": "dG_eq",
    "k": "k",
    "kf": "kf",
    "kr": "kr",
    "keq": "Keq",
}
_STRUCTURAL_SEMICOLON_DIRECTIVES = {
    "A",
    "Ea",
    "dG_act",
    "dG_eq",
    "fast",
}
_MUTABLE_REACTION_RATE_DIRECTIVES = {"k", "kf", "kr"}
_MUTABLE_EQUILIBRIUM_DIRECTIVES = {"Keq"}
def _mutable_preview_parameter_names(parameter_names: Sequence[str] | object) -> set[str]:
    return {str(name) for name in (parameter_names or ()) if str(name)}


def _canonical_semicolon_directive_key(name: str) -> str:
    name_s = str(name or "").strip()
    if name_s == "K":
        return "K"
    return _DSL_SEMICOLON_KEY_ALIASES.get(name_s.lower(), name_s)


def _mutable_preview_assignment_token(
    name: str,
    parameter_names: set[str],
    *,
    step_index: int | None = None,
) -> str | None:
    name_s = str(name or "")
    if not name_s:
        return None
    if step_index is None:
        if name_s in parameter_names:
            return name_s
        return None

    canonical_name = _canonical_semicolon_directive_key(name_s)
    if canonical_name in _STRUCTURAL_SEMICOLON_DIRECTIVES:
        return None
    if canonical_name in _MUTABLE_EQUILIBRIUM_DIRECTIVES:
        indexed_keq = f"Keq{int(step_index)}"
        if indexed_keq in parameter_names:
            return "Keq"
        return None
    if canonical_name in _MUTABLE_REACTION_RATE_DIRECTIVES:
        indexed_k = f"k{int(step_index)}"
        if canonical_name in {"k", "kf"} and indexed_k in parameter_names:
            return "k"
        indexed_name = f"{canonical_name}{int(step_index)}"
        if indexed_name in parameter_names:
            return canonical_name
    return None


def _is_mutable_preview_assignment(name: str, parameter_names: set[str]) -> bool:
    return _mutable_preview_assignment_token(name, parameter_names) is not None


_NUMERIC_LITERAL_RE = re.compile(
    r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?"
)


def _is_preview_mutable_numeric_value(value: object) -> bool:
    value_text = str(value or "").split("#", 1)[0].strip()
    if not _NUMERIC_LITERAL_RE.fullmatch(value_text):
        return False
    try:
        return math.isfinite(float(value_text))
    except (TypeError, ValueError, OverflowError):
        return False


def _preview_structural_mechanism_digest(
    text: object,
    *,
    parameter_names: Sequence[str] = (),
) -> str:
    """Hash preview runtime structure while ignoring mutable parameter values."""

    mutable_names = _mutable_preview_parameter_names(parameter_names)

    def _mask_mutable_assignment(match: re.Match[str], *, step_index: int | None) -> str:
        name = str(match.group("name") or "")
        mutable_token = _mutable_preview_assignment_token(name, mutable_names, step_index=step_index)
        if mutable_token is None:
            return match.group(0)
        if not _is_preview_mutable_numeric_value(match.group("value")):
            return match.group(0)
        prefix = str(match.group("prefix") or "")
        if prefix.strip() in {";", ","}:
            prefix = "; "
        return f"{prefix}{mutable_token}=<param-value>"

    normalized_lines: list[str] = []
    step_index = 0
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        current_step_index: int | None = None
        if _DSL_REACTION_ARROW_RE.search(line):
            step_index += 1
            current_step_index = step_index
        param_statement = _DSL_PARAM_STATEMENT_RE.match(line)
        if param_statement is not None:
            prefix = str(param_statement.group("prefix") or "")
            name_match = re.match(r"\s*param\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*", prefix, re.IGNORECASE)
            name = str(name_match.group(1)) if name_match is not None else ""
            if _is_mutable_preview_assignment(name, mutable_names) and _is_preview_mutable_numeric_value(
                param_statement.group("value")
            ):
                normalized_lines.append(f"{prefix}<param-value>")
            else:
                normalized_lines.append(line)
            continue
        normalized_lines.append(
            _DSL_PARAMETER_ASSIGNMENT_RE.sub(
                lambda match: _mask_mutable_assignment(match, step_index=current_step_index),
                line,
            )
        )
    return _sha256_text("\n".join(normalized_lines))


@dataclass(frozen=True, slots=True)
class SimulationSolverIdentity:
    solver: str
    rtol: float
    atol: float
    grid_n: int
    temperature_K: float
    use_sparse_jacobian: bool
    wegscheider_cyclicity_enabled: bool

    @classmethod
    def from_solver_config(cls, solver_config: Mapping[str, Any] | None) -> "SimulationSolverIdentity":
        config = dict(solver_config or {})
        grid = config.get("grid") or {}
        return cls(
            solver=str(config.get("solver") or ""),
            rtol=_try_float(config.get("rtol", 1e-6), 1e-6),
            atol=_try_float(config.get("atol", 1e-12), 1e-12),
            grid_n=int((grid.get("N") or 0) if isinstance(grid, Mapping) else 0),
            temperature_K=_try_float(config.get("temperature_K", 298.15), 298.15),
            use_sparse_jacobian=bool(
                config.get("use_sparse_jacobian", USE_SPARSE_JACOBIAN_DEFAULT)
            ),
            wegscheider_cyclicity_enabled=bool(
                config.get(
                    "wegscheider_cyclicity_enabled",
                    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
                )
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "SimulationSolverIdentity":
        return cls(
            solver=str((payload or {}).get("solver") or ""),
            rtol=_try_float((payload or {}).get("rtol", 1e-6), 1e-6),
            atol=_try_float((payload or {}).get("atol", 1e-12), 1e-12),
            grid_n=int((payload or {}).get("grid_n") or 0),
            temperature_K=_try_float((payload or {}).get("temperature_K", 298.15), 298.15),
            use_sparse_jacobian=bool(
                (payload or {}).get(
                    "use_sparse_jacobian",
                    USE_SPARSE_JACOBIAN_DEFAULT,
                )
            ),
            wegscheider_cyclicity_enabled=bool(
                (payload or {}).get(
                    "wegscheider_cyclicity_enabled",
                    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
                )
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "solver": str(self.solver),
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "grid_n": int(self.grid_n),
            "temperature_K": float(self.temperature_K),
            "use_sparse_jacobian": bool(self.use_sparse_jacobian),
            "wegscheider_cyclicity_enabled": bool(self.wegscheider_cyclicity_enabled),
        }


@dataclass(frozen=True, slots=True)
class SimulationIdentity:
    schema_id: str
    param_fingerprint: str
    canonical_initials_fingerprint: str
    solver: SimulationSolverIdentity
    t_end: float
    intervention_schedule_fingerprint: str = ""
    preview_batch_cache_token: str = ""
    execution_flags: tuple[str, ...] = ()
    symbolic_jacobian_identity: Optional[dict[str, Any]] = None
    symbolic_wegscheider_identity: Optional[dict[str, Any]] = None
    version: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "symbolic_jacobian_identity",
            normalize_symbolic_identity_mapping(
                self.symbolic_jacobian_identity,
                label="symbolic Jacobian identity",
            ),
        )
        object.__setattr__(
            self,
            "symbolic_wegscheider_identity",
            normalize_symbolic_identity_mapping(
                self.symbolic_wegscheider_identity,
                label="symbolic Wegscheider identity",
            ),
        )

    @classmethod
    def build(
        cls,
        *,
        schema_id: str,
        param_fingerprint: str,
        solver_config: Mapping[str, Any] | None,
        t_end: float,
        canonical_initials_fingerprint: str = "",
        intervention_schedule_fingerprint: str = "",
        preview_batch_cache_token: str = "",
        execution_flags: Sequence[str] = (),
        symbolic_jacobian_identity: Mapping[str, Any] | None = None,
        symbolic_wegscheider_identity: Mapping[str, Any] | None = None,
    ) -> "SimulationIdentity":
        flags = tuple(sorted({str(flag) for flag in (execution_flags or ()) if str(flag)}))
        solver_identity = SimulationSolverIdentity.from_solver_config(solver_config)
        intervention_fp = str(intervention_schedule_fingerprint or "")
        symbolic_identity = normalize_symbolic_identity_mapping(
            symbolic_jacobian_identity,
            label="symbolic Jacobian identity",
        )
        wegscheider_identity = normalize_symbolic_identity_mapping(
            symbolic_wegscheider_identity,
            label="symbolic Wegscheider identity",
        )
        return cls(
            schema_id=str(schema_id or ""),
            param_fingerprint=str(param_fingerprint or ""),
            canonical_initials_fingerprint=str(canonical_initials_fingerprint or ""),
            solver=solver_identity,
            t_end=float(t_end),
            intervention_schedule_fingerprint=intervention_fp,
            preview_batch_cache_token=str(preview_batch_cache_token or ""),
            execution_flags=flags,
            symbolic_jacobian_identity=symbolic_identity,
            symbolic_wegscheider_identity=wegscheider_identity,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> Optional["SimulationIdentity"]:
        if not isinstance(payload, Mapping):
            return None
        solver_payload = payload.get("solver")
        if not isinstance(solver_payload, Mapping):
            return None
        return cls(
            version=int(payload.get("version") or 1),
            schema_id=str(payload.get("schema_id") or ""),
            param_fingerprint=str(payload.get("param_fingerprint") or ""),
            canonical_initials_fingerprint=str(payload.get("canonical_initials_fingerprint") or ""),
            solver=SimulationSolverIdentity.from_payload(solver_payload),
            t_end=_try_float(payload.get("t_end", 0.0), 0.0),
            intervention_schedule_fingerprint=str(payload.get("intervention_schedule_fingerprint") or ""),
            preview_batch_cache_token=str(payload.get("preview_batch_cache_token") or ""),
            execution_flags=tuple(str(flag) for flag in (payload.get("execution_flags") or ()) if str(flag)),
            symbolic_jacobian_identity=normalize_symbolic_identity_mapping(
                payload.get("symbolic_jacobian_identity"),
                label="symbolic Jacobian identity",
            ),
            symbolic_wegscheider_identity=normalize_symbolic_identity_mapping(
                payload.get("symbolic_wegscheider_identity"),
                label="symbolic Wegscheider identity",
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "version": int(self.version),
            "schema_id": str(self.schema_id),
            "param_fingerprint": str(self.param_fingerprint),
            "canonical_initials_fingerprint": str(self.canonical_initials_fingerprint),
            "solver": self.solver.to_payload(),
            "t_end": float(self.t_end),
            "intervention_schedule_fingerprint": str(self.intervention_schedule_fingerprint or ""),
            "preview_batch_cache_token": str(self.preview_batch_cache_token),
            "execution_flags": list(self.execution_flags),
        }
        if self.symbolic_jacobian_identity:
            payload["symbolic_jacobian_identity"] = dict(self.symbolic_jacobian_identity)
        if self.symbolic_wegscheider_identity:
            payload["symbolic_wegscheider_identity"] = dict(self.symbolic_wegscheider_identity)
        return payload

    def cache_key(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_payload())).hexdigest()

    def prepared_runtime_key(self) -> str:
        """Reuse key for the compiled/prepared mechanism cache.

        This key must include only structural factors that affect the compiled
        mechanism (schema, temperature, sparse jacobian, wegscheider).  Per-set
        parameter overrides are applied *after* cloning the cached prepared
        runtime, so ``param_fingerprint`` is intentionally excluded.
        """
        payload = {
            "version": int(self.version),
            "schema_id": str(self.schema_id),
            "temperature_K": float(self.solver.temperature_K),
            "use_sparse_jacobian": bool(self.solver.use_sparse_jacobian),
            "wegscheider_cyclicity_enabled": bool(self.solver.wegscheider_cyclicity_enabled),
        }
        if self.symbolic_jacobian_identity:
            structure_payload = _symbolic_jacobian_structure_payload(self.symbolic_jacobian_identity)
            if structure_payload:
                payload["symbolic_jacobian_structure_identity"] = structure_payload
        if self.symbolic_wegscheider_identity:
            source_payload = _symbolic_wegscheider_source_payload(self.symbolic_wegscheider_identity)
            if source_payload:
                payload["symbolic_wegscheider_source_identity"] = source_payload
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class SimulationScopeIdentity:
    entries: tuple[tuple[str, SimulationIdentity], ...]
    version: int = 1

    @classmethod
    def build(
        cls,
        *,
        queue_ids: Sequence[str],
        identity_by_set_id: Mapping[str, SimulationIdentity | Mapping[str, Any]],
    ) -> "SimulationScopeIdentity":
        entries: list[tuple[str, SimulationIdentity]] = []
        for raw_set_id in queue_ids or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id:
                continue
            identity = coerce_simulation_identity(identity_by_set_id.get(set_id))
            if identity is None:
                continue
            entries.append((set_id, identity))
        return cls(entries=tuple(entries))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> Optional["SimulationScopeIdentity"]:
        if not isinstance(payload, Mapping):
            return None
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, Sequence):
            return None
        entries: list[tuple[str, SimulationIdentity]] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                return None
            set_id = str(raw_entry.get("set_id") or "").strip()
            identity = coerce_simulation_identity(raw_entry.get("identity"))
            if not set_id or identity is None:
                return None
            entries.append((set_id, identity))
        return cls(entries=tuple(entries), version=int(payload.get("version") or 1))

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "entries": [
                {
                    "set_id": str(set_id),
                    "identity": identity.to_payload(),
                }
                for set_id, identity in self.entries
            ],
        }

    def cache_key(self) -> str:
        if len(self.entries) == 1:
            return self.entries[0][1].cache_key()
        return hashlib.sha256(_canonical_json_bytes(self.to_payload())).hexdigest()


def contained_simulation_owner_identity(
    *,
    execution_mode: str,
    owner_mechanism_text: str,
    solver_config: Mapping[str, Any] | None,
    t_end: float,
    set_id: str,
    parameter_names: Sequence[str] = (),
    simulation_identity: SimulationIdentity | Mapping[str, Any] | None = None,
    contained_child_blas_threads_limited: bool = True,
) -> dict[str, Any]:
    """Build the payload identity that decides contained runtime owner reuse.

    Preview owners may reuse a prepared mechanism across slider value changes,
    because slider values are applied as request-local parameter overrides.  The
    identity therefore includes structural mechanism/runtime dimensions and the
    parameter namespace, but not the current preview parameter values.
    """
    mode = "preview" if str(execution_mode or "").lower() == "preview" else "explicit"
    identity = coerce_simulation_identity(simulation_identity)
    solver_identity = (
        identity.solver
        if identity is not None
        else SimulationSolverIdentity.from_solver_config(solver_config)
    )
    payload: dict[str, Any] = {
        "version": 4,
        "execution_mode": mode,
        "solver": solver_identity.to_payload(),
        "t_end": float(t_end),
        "set_id": str(set_id or ""),
        "contained_child_blas_threads_limited": bool(contained_child_blas_threads_limited),
    }
    if mode == "preview":
        payload["structural_mechanism_digest"] = _preview_structural_mechanism_digest(
            owner_mechanism_text,
            parameter_names=parameter_names,
        )
    elif identity is None:
        payload["mechanism_digest"] = _sha256_text(owner_mechanism_text)
    if identity is not None:
        if mode != "preview":
            payload["schema_id"] = str(identity.schema_id)
            payload["simulation_identity_key"] = identity.prepared_runtime_key()
    if mode == "preview":
        payload["parameter_names"] = sorted(
            {str(name) for name in (parameter_names or ()) if str(name)}
        )
    return payload


def coerce_simulation_identity(value: object) -> Optional[SimulationIdentity]:
    if isinstance(value, SimulationIdentity):
        return value
    if isinstance(value, Mapping):
        return SimulationIdentity.from_payload(value)
    return None


def coerce_simulation_scope_identity(value: object) -> Optional[SimulationScopeIdentity]:
    if isinstance(value, SimulationScopeIdentity):
        return value
    if isinstance(value, Mapping):
        return SimulationScopeIdentity.from_payload(value)
    return None
