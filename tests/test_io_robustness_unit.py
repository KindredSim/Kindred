"""Unit tests for selected IO and logging edge cases."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestLoggingRobustness:
    """Test logging handles edge cases."""

    def test_logging_setup_multiple_times(self):
        """Test that calling setup_logging multiple times is safe."""
        from kindred.io.logging import setup_logging

        setup_logging(level="INFO")
        setup_logging(level="DEBUG")
        setup_logging(level="WARNING")

    def test_logging_with_invalid_level(self):
        """Test logging setup with invalid log level."""
        from kindred.io.logging import setup_logging

        try:
            setup_logging(level="INVALID_LEVEL")
        except ValueError:
            pass
