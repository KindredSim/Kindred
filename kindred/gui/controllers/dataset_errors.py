"""Shared dataset-owner exceptions."""

from __future__ import annotations


class DatasetOwnerError(RuntimeError):
    """Raised when dataset ownership or fitting preparation cannot continue."""
