from __future__ import annotations

from kindred.gui.fitting.worker_lifecycle import FitWorkerStopPolicy


class _StuckWorker:
    def __init__(self) -> None:
        self.cancel_called = False
        self.request_interruption_called = False
        self.quit_called = False
        self.terminate_called = False
        self.wait_calls: list[int] = []
        self._running = True

    def cancel(self) -> None:
        self.cancel_called = True

    def requestInterruption(self) -> None:
        self.request_interruption_called = True

    def quit(self) -> None:
        self.quit_called = True

    def wait(self, msecs: int | None = None) -> bool:
        self.wait_calls.append(int(msecs or 0))
        return False

    def isRunning(self) -> bool:
        return self._running

    def terminate(self) -> None:
        self.terminate_called = True


def test_fit_worker_stop_policy_escalates_to_terminate_after_timeout():
    failures: set[str] = set()
    policy = FitWorkerStopPolicy(record_failure=failures.add)
    worker = _StuckWorker()

    outcome = policy.stop_worker(worker, timeout_ms=2000, context="hard_teardown")

    assert worker.cancel_called is True
    assert worker.wait_calls == [2000, 2000]
    assert worker.terminate_called is True
    assert outcome.used_terminate is True
    assert outcome.still_running is True
    assert failures == set()
