"""Dataset boundary exceptions."""

from __future__ import annotations


class DatasetError(RuntimeError):
    """Base class for user-visible dataset import and mapping failures."""


class DatasetImportAdmissionError(DatasetError):
    """A direct GUI-thread import admission check rejected the request."""
