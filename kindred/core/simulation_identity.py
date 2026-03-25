from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "SimulationIdentity",
    "SimulationScopeIdentity",
    "SimulationSolverIdentity",
    "coerce_simulation_identity",
    "coerce_simulation_scope_identity",
]


def _try_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _canonical_json_bytes(payload: object) -> bytes:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return serialized.encode("utf-8", "ignore")


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
            use_sparse_jacobian=bool(config.get("use_sparse_jacobian")),
            wegscheider_cyclicity_enabled=bool(config.get("wegscheider_cyclicity_enabled")),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "SimulationSolverIdentity":
        return cls(
            solver=str((payload or {}).get("solver") or ""),
            rtol=_try_float((payload or {}).get("rtol", 1e-6), 1e-6),
            atol=_try_float((payload or {}).get("atol", 1e-12), 1e-12),
            grid_n=int((payload or {}).get("grid_n") or 0),
            temperature_K=_try_float((payload or {}).get("temperature_K", 298.15), 298.15),
            use_sparse_jacobian=bool((payload or {}).get("use_sparse_jacobian")),
            wegscheider_cyclicity_enabled=bool((payload or {}).get("wegscheider_cyclicity_enabled")),
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
    solver: SimulationSolverIdentity
    t_end: float
    preview_batch_cache_token: str = ""
    execution_flags: tuple[str, ...] = ()
    version: int = 1

    @classmethod
    def build(
        cls,
        *,
        schema_id: str,
        param_fingerprint: str,
        solver_config: Mapping[str, Any] | None,
        t_end: float,
        preview_batch_cache_token: str = "",
        execution_flags: Sequence[str] = (),
    ) -> "SimulationIdentity":
        flags = tuple(sorted({str(flag) for flag in (execution_flags or ()) if str(flag)}))
        return cls(
            schema_id=str(schema_id or ""),
            param_fingerprint=str(param_fingerprint or ""),
            solver=SimulationSolverIdentity.from_solver_config(solver_config),
            t_end=float(t_end),
            preview_batch_cache_token=str(preview_batch_cache_token or ""),
            execution_flags=flags,
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
            solver=SimulationSolverIdentity.from_payload(solver_payload),
            t_end=_try_float(payload.get("t_end", 0.0), 0.0),
            preview_batch_cache_token=str(payload.get("preview_batch_cache_token") or ""),
            execution_flags=tuple(str(flag) for flag in (payload.get("execution_flags") or ()) if str(flag)),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "schema_id": str(self.schema_id),
            "param_fingerprint": str(self.param_fingerprint),
            "solver": self.solver.to_payload(),
            "t_end": float(self.t_end),
            "preview_batch_cache_token": str(self.preview_batch_cache_token),
            "execution_flags": list(self.execution_flags),
        }

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
