from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

from PySide6 import QtCore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerStopOutcome:
    still_running: bool
    used_terminate: bool


class FitWorkerStopPolicy:
    """Encapsulates the last-resort stop policy for fit workers."""

    def __init__(self, *, record_failure: Callable[[str], None]) -> None:
        self._record_failure = record_failure

    @staticmethod
    def is_running(worker: QtCore.QThread) -> bool:
        try:
            return bool(getattr(worker, "isRunning", lambda: False)())
        except Exception:
            return False

    def request_stop(self, worker: QtCore.QThread, *, context: str) -> None:
        if hasattr(worker, "cancel"):
            try:
                worker.cancel()
                return
            except Exception as exc:
                logger.debug("Failed to cancel worker during %s: %s", context, exc, exc_info=True)
        if hasattr(worker, "requestInterruption"):
            try:
                worker.requestInterruption()
            except Exception as exc:
                self._record_failure(f"{context}.requestInterruption")
                logger.debug("Failed to request interruption during %s: %s", context, exc, exc_info=True)
        if hasattr(worker, "quit"):
            try:
                worker.quit()
            except Exception as exc:
                self._record_failure(f"{context}.quit")
                logger.debug("Failed to quit worker during %s: %s", context, exc, exc_info=True)

    @staticmethod
    def wait_for_stop(worker: QtCore.QThread, *, timeout_ms: int, context: str) -> bool:
        wait_failed = False
        try:
            worker.wait(int(timeout_ms))
        except Exception as exc:
            logger.debug("Failed to wait for worker (%s): %s", context, exc, exc_info=True)
            wait_failed = True
        try:
            return bool(getattr(worker, "isRunning", lambda: False)())
        except Exception:
            return bool(wait_failed)

    def terminate_if_needed(self, worker: QtCore.QThread, *, timeout_ms: int, context: str) -> bool:
        try:
            worker.terminate()
        except Exception as exc:
            logger.debug("Failed to terminate worker during %s: %s", context, exc, exc_info=True)
            if hasattr(worker, "requestInterruption"):
                try:
                    worker.requestInterruption()
                except Exception as inner_exc:
                    self._record_failure(f"{context}.requestInterruption_after_terminate")
                    logger.debug(
                        "Failed to request interruption after terminate() failure during %s: %s",
                        context,
                        inner_exc,
                        exc_info=True,
                    )

        still_running = self.wait_for_stop(worker, timeout_ms=int(timeout_ms), context=f"{context}.wait_after_terminate")
        if still_running:
            logger.critical("Global fit worker refused to stop after terminate(); continuing teardown to avoid GUI freeze.")
        return still_running

    def stop_worker(self, worker: QtCore.QThread, *, timeout_ms: int, context: str) -> WorkerStopOutcome:
        self.request_stop(worker, context=context)
        still_running = self.wait_for_stop(worker, timeout_ms=int(timeout_ms), context=f"{context}.wait")
        used_terminate = False
        if still_running:
            used_terminate = True
            still_running = self.terminate_if_needed(worker, timeout_ms=int(timeout_ms), context=context)
        return WorkerStopOutcome(still_running=bool(still_running), used_terminate=bool(used_terminate))
