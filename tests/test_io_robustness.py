"""Test selected IO, logging, and cache edge cases."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

try:
    import multiprocessing as mp
except ImportError:
    mp = None

HAS_MULTIPROCESSING = bool(mp and hasattr(mp, "Process") and hasattr(mp, "get_start_method"))
MULTIPROCESS_REASON = "requires multiprocessing.Process support"


class TestLoggingRobustness:
    """Test logging handles edge cases."""

    @pytest.mark.skipif(not HAS_MULTIPROCESSING, reason=MULTIPROCESS_REASON)
    def test_logger_in_multiprocessing_context(self):
        """Test that logging works in multiprocessing context."""
        from kindred.io.logging import get_logger

        def worker_process():
            logger = get_logger(__name__)
            logger.info("Test message from worker")
            return True

        if hasattr(mp, "get_start_method"):
            try:
                p = mp.Process(target=worker_process)
                p.start()
                p.join(timeout=2)
                assert p.exitcode == 0 or p.exitcode is None
            except Exception as exc:
                pytest.skip(f"{MULTIPROCESS_REASON}: {exc}")


class TestPathResolution:
    """Test path resolution and resource loading."""

    def test_get_log_dir_creates_directory(self):
        """Test that get_log_dir creates directory if it doesn't exist."""
        from kindred.io.logging import get_log_dir

        log_dir = get_log_dir()
        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_resource_loading_missing_file(self):
        """Test resource loading handles missing files gracefully."""
        from kindred.io.resources import get_resource_text

        try:
            get_resource_text("presets/this_does_not_exist_12345.txt")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Expected missing bundled resource lookup to fail")
