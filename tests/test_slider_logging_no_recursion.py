from __future__ import annotations

from contextlib import suppress
import logging
import sys


class _StderrThatLogs:
    """Test double that reproduces stderr->logging re-entry."""

    def __init__(self, *, max_writes: int = 8) -> None:
        self._max_writes = int(max_writes)
        self._write_count = 0
        self._bridge_logger = logging.getLogger("tests.stderr_bridge")

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._write_count += 1
        if self._write_count > self._max_writes:
            raise RecursionError("stderr recursion guard tripped")
        self._bridge_logger.error("stderr->log bridge: %s", text.rstrip("\n"))
        return len(text)

    def flush(self) -> None:
        return None


def test_slider_logger_debug_does_not_recurse_when_sys_stderr_logs(monkeypatch):
    from kindred.gui.widgets.variable_sliders import slider_update_logger
    from kindred.io.logging import setup_logging

    root_logger = logging.getLogger()
    prev_root_handlers = list(root_logger.handlers)
    prev_root_level = int(root_logger.level)

    prev_slider_handlers = list(slider_update_logger.handlers)
    prev_slider_level = int(slider_update_logger.level)
    prev_slider_propagate = bool(slider_update_logger.propagate)

    try:
        for handler in list(slider_update_logger.handlers):
            slider_update_logger.removeHandler(handler)
        slider_update_logger.setLevel(logging.DEBUG)
        slider_update_logger.propagate = True

        fake_stderr = _StderrThatLogs(max_writes=8)
        monkeypatch.setattr(sys, "stderr", fake_stderr)

        setup_logging(level="INFO", console=True, file_handler=False)

        root_logger.setLevel(logging.DEBUG)
        stream_handlers = []
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.DEBUG)
                stream_handlers.append(handler)

        slider_update_logger.debug("slider_value_changed(K1 pos=%s -> %s)", 123, 4.56)

        assert stream_handlers, "Expected at least one console stream handler"
        assert all(handler.stream is sys.__stderr__ for handler in stream_handlers)
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            if handler not in prev_root_handlers:
                with suppress(Exception):
                    handler.close()
        for handler in prev_root_handlers:
            if handler not in root_logger.handlers:
                root_logger.addHandler(handler)
        root_logger.setLevel(prev_root_level)

        for handler in list(slider_update_logger.handlers):
            slider_update_logger.removeHandler(handler)
            if handler not in prev_slider_handlers:
                with suppress(Exception):
                    handler.close()
        for handler in prev_slider_handlers:
            if handler not in slider_update_logger.handlers:
                slider_update_logger.addHandler(handler)
        slider_update_logger.setLevel(prev_slider_level)
        slider_update_logger.propagate = prev_slider_propagate
