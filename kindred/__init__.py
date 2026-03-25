"""
Kindred package root.

Single importable namespace: `kindred`.

Contract:
- Defines `__version__` from the single authoritative version module.
- No I/O, no networking, no registry access.
- No reliance on current working directory.
- Do not introduce additional runtime behavior here.

This module intentionally keeps the surface minimal and deterministic.
"""

from __future__ import annotations

from kindred._version import __version__, get_version

__all__ = ["__version__", "get_version"]
