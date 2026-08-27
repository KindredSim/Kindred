from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def require_sympy() -> Any:
    import sympy

    return sympy
