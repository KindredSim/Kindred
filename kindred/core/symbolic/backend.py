from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

SYMBOLIC_PROFILE_VERSION = "kindred-symbolic-v1"


@dataclass(frozen=True, slots=True)
class SymbolicBackendMetadata:
    backend_name: str
    backend_version: str
    profile_version: str

    def to_payload(self) -> dict[str, str]:
        return {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "profile_version": self.profile_version,
        }


@lru_cache(maxsize=1)
def require_sympy() -> Any:
    import sympy

    return sympy


def get_symbolic_backend_metadata() -> SymbolicBackendMetadata:
    sympy = require_sympy()
    return SymbolicBackendMetadata(
        backend_name="sympy",
        backend_version=str(getattr(sympy, "__version__", "")),
        profile_version=SYMBOLIC_PROFILE_VERSION,
    )
