"""
Environment compatibility shim.

Some sandboxed environments (including certain CI runners) may not provide
`os.urandom` / `/dev/urandom`. Python and NumPy rely on `os.urandom` during
initialization. When it is missing, imports can fail before the application or
tests even start.

This module is auto-imported by Python (via `site`) when present on `sys.path`.
It patches `os.urandom` only when it is unavailable, providing a deterministic
fallback suitable for non-cryptographic use (tests, headless audits).

Important:
- If `os.urandom` is available, this module is a no-op.
- This is not intended for cryptographic security; it only restores runtime
  operability in entropy-starved environments.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Callable


def _make_deterministic_urandom() -> Callable[[int], bytes]:
    lock = threading.Lock()
    counter = {"n": 0}

    def _urandom(n: int) -> bytes:
        try:
            size = int(n)
        except Exception:
            size = 0
        if size <= 0:
            return b""

        out = bytearray()
        while len(out) < size:
            with lock:
                counter["n"] += 1
                token = counter["n"]
            digest = hashlib.blake2b(
                f"kindred-deterministic-urandom:{token}".encode("utf-8"),
                digest_size=32,
            ).digest()
            out.extend(digest)
        return bytes(out[:size])

    return _urandom


try:
    os.urandom(1)
except NotImplementedError:
    fallback = _make_deterministic_urandom()
    os.urandom = fallback  # type: ignore[assignment]
    try:
        import random as _random

        _random._urandom = fallback  # type: ignore[attr-defined]
    except Exception:
        pass
