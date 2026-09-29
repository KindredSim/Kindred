"""Dataset boundary exceptions."""

from __future__ import annotations


class DatasetImportAdmissionError(RuntimeError):
    """A direct GUI-thread import admission check rejected the request."""
