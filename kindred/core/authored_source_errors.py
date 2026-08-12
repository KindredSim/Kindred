"""Typed boundary for expected failures in user-authored Reactions source."""

from __future__ import annotations

__all__ = ["AuthoredSourceError"]


class AuthoredSourceError(ValueError):
    """An expected, user-correctable Reactions authoring failure."""
