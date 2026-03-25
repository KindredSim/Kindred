from __future__ import annotations

__all__ = ["__version__", "get_version"]

# Single authoritative package version. Keep this in sync with the intended release.
__version__: str = "0.1.0"


def get_version() -> str:
    return __version__
