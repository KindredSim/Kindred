from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .backend import SymbolicBackendMetadata
from .errors import UnsupportedSymbolicExpressionError
from .identity import normalize_symbolic_identity_mapping, symbolic_fingerprint


SYMBOLIC_JACOBIAN_IDENTITY_ATTR = "_kindred_symbolic_jacobian_identity"


def symbolic_jacobian_identity_payload(value: object) -> dict[str, Any] | None:
    payload = getattr(value, SYMBOLIC_JACOBIAN_IDENTITY_ATTR, None)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise UnsupportedSymbolicExpressionError("Symbolic Jacobian identity payload must be a JSON-safe mapping.")
    return normalize_symbolic_identity_mapping(payload, label="Symbolic Jacobian identity payload")


@dataclass(frozen=True, slots=True)
class SymbolicArtifactIdentity:
    kind: str
    backend_name: str
    backend_version: str
    profile_version: str
    source_fingerprint: str
    artifact_fingerprint: str
    fingerprint: str
    structure_fingerprint: str = ""
    evaluation_snapshot_fingerprint: str = ""
    parameter_symbols: tuple[str, ...] = ()

    @classmethod
    def none(cls, metadata: SymbolicBackendMetadata) -> "SymbolicArtifactIdentity":
        payload = {
            "kind": "none",
            "backend_name": metadata.backend_name,
            "backend_version": metadata.backend_version,
            "profile_version": metadata.profile_version,
            "source_fingerprint": "",
            "artifact_fingerprint": "",
            "structure_fingerprint": "",
            "evaluation_snapshot_fingerprint": "",
            "parameter_symbols": [],
        }
        return cls(
            kind="none",
            backend_name=metadata.backend_name,
            backend_version=metadata.backend_version,
            profile_version=metadata.profile_version,
            source_fingerprint="",
            artifact_fingerprint="",
            structure_fingerprint="",
            evaluation_snapshot_fingerprint="",
            parameter_symbols=(),
            fingerprint=symbolic_fingerprint(payload),
        )

    @classmethod
    def jacobian(
        cls,
        metadata: SymbolicBackendMetadata,
        *,
        source_fingerprint: str,
        artifact_fingerprint: str,
        structure_fingerprint: str = "",
        evaluation_snapshot_fingerprint: str = "",
        parameter_symbols: tuple[str, ...] | list[str] = (),
    ) -> "SymbolicArtifactIdentity":
        normalized_parameter_symbols = tuple(str(name) for name in (parameter_symbols or ()) if str(name))
        payload = {
            "kind": "jacobian",
            "backend_name": metadata.backend_name,
            "backend_version": metadata.backend_version,
            "profile_version": metadata.profile_version,
            "source_fingerprint": str(source_fingerprint or ""),
            "artifact_fingerprint": str(artifact_fingerprint or ""),
            "structure_fingerprint": str(structure_fingerprint or source_fingerprint or ""),
            "evaluation_snapshot_fingerprint": str(evaluation_snapshot_fingerprint or ""),
            "parameter_symbols": list(normalized_parameter_symbols),
        }
        return cls(
            kind="jacobian",
            backend_name=metadata.backend_name,
            backend_version=metadata.backend_version,
            profile_version=metadata.profile_version,
            source_fingerprint=str(source_fingerprint or ""),
            artifact_fingerprint=str(artifact_fingerprint or ""),
            structure_fingerprint=str(structure_fingerprint or source_fingerprint or ""),
            evaluation_snapshot_fingerprint=str(evaluation_snapshot_fingerprint or ""),
            parameter_symbols=normalized_parameter_symbols,
            fingerprint=symbolic_fingerprint(payload),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "profile_version": self.profile_version,
            "source_fingerprint": self.source_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "structure_fingerprint": self.structure_fingerprint,
            "evaluation_snapshot_fingerprint": self.evaluation_snapshot_fingerprint,
            "parameter_symbols": list(self.parameter_symbols),
            "fingerprint": self.fingerprint,
        }
