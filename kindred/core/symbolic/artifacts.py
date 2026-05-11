from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .backend import SymbolicBackendMetadata


SYMBOLIC_JACOBIAN_IDENTITY_ATTR = "_kindred_symbolic_jacobian_identity"


def _fingerprint(payload: Mapping[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def symbolic_jacobian_identity_payload(value: object) -> dict[str, Any] | None:
    payload = getattr(value, SYMBOLIC_JACOBIAN_IDENTITY_ATTR, None)
    if not isinstance(payload, Mapping):
        return None
    return {str(key): item for key, item in payload.items()}


@dataclass(frozen=True, slots=True)
class SymbolicArtifactIdentity:
    kind: str
    backend_name: str
    backend_version: str
    profile_version: str
    source_fingerprint: str
    artifact_fingerprint: str
    fingerprint: str

    @classmethod
    def none(cls, metadata: SymbolicBackendMetadata) -> "SymbolicArtifactIdentity":
        payload = {
            "kind": "none",
            "backend_name": metadata.backend_name,
            "backend_version": metadata.backend_version,
            "profile_version": metadata.profile_version,
            "source_fingerprint": "",
            "artifact_fingerprint": "",
        }
        return cls(
            kind="none",
            backend_name=metadata.backend_name,
            backend_version=metadata.backend_version,
            profile_version=metadata.profile_version,
            source_fingerprint="",
            artifact_fingerprint="",
            fingerprint=_fingerprint(payload),
        )

    @classmethod
    def jacobian(
        cls,
        metadata: SymbolicBackendMetadata,
        *,
        source_fingerprint: str,
        artifact_fingerprint: str,
    ) -> "SymbolicArtifactIdentity":
        payload = {
            "kind": "jacobian",
            "backend_name": metadata.backend_name,
            "backend_version": metadata.backend_version,
            "profile_version": metadata.profile_version,
            "source_fingerprint": str(source_fingerprint or ""),
            "artifact_fingerprint": str(artifact_fingerprint or ""),
        }
        return cls(
            kind="jacobian",
            backend_name=metadata.backend_name,
            backend_version=metadata.backend_version,
            profile_version=metadata.profile_version,
            source_fingerprint=str(source_fingerprint or ""),
            artifact_fingerprint=str(artifact_fingerprint or ""),
            fingerprint=_fingerprint(payload),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "profile_version": self.profile_version,
            "source_fingerprint": self.source_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "fingerprint": self.fingerprint,
        }
