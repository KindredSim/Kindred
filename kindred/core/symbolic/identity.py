from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from .errors import UnsupportedSymbolicExpressionError


def normalize_symbolic_identity_payload(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnsupportedSymbolicExpressionError("Symbolic identity payload values must be JSON-safe and finite.")
        return value
    if isinstance(value, Mapping):
        return {
            _normalize_identity_key(key): normalize_symbolic_identity_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize_symbolic_identity_payload(item) for item in value]
    raise UnsupportedSymbolicExpressionError(
        f"Symbolic identity payload values must be JSON-safe, got {type(value).__name__}."
    )


def _normalize_identity_key(key: object) -> str:
    if not isinstance(key, str):
        raise UnsupportedSymbolicExpressionError("Symbolic identity payload mapping keys must be JSON-safe strings.")
    return key


def symbolic_identity_json(payload: object) -> bytes:
    data = json.dumps(
        normalize_symbolic_identity_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return data.encode("utf-8")


def symbolic_fingerprint(payload: object) -> str:
    return hashlib.sha256(symbolic_identity_json(payload)).hexdigest()


def normalize_symbolic_identity_mapping(identity: object, *, label: str) -> dict[str, Any] | None:
    if identity is None:
        return None
    if not isinstance(identity, Mapping):
        raise UnsupportedSymbolicExpressionError(f"{label} must be a JSON-safe mapping.")
    if not identity:
        return None
    normalized = normalize_symbolic_identity_payload(dict(identity))
    if not isinstance(normalized, dict) or not normalized:
        return None
    return normalized
