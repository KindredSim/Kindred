import threading
import time

import pytest


@pytest.mark.unit
def test_acquire_lock_polling_returns_false_when_cancelled_immediately():
    from kindred.gui import simulation_worker

    class _FakeLock:
        def __init__(self):
            self.acquire_calls = 0

        def acquire(self, *, timeout):  # match keyword usage from threading.Lock
            self.acquire_calls += 1
            raise AssertionError("acquire() should not be called when already cancelled")

    fake = _FakeLock()
    acquired = simulation_worker._acquire_lock_polling(fake, cancelled=lambda: True, timeout_s=0.001)  # type: ignore[arg-type]
    assert acquired is False
    assert fake.acquire_calls == 0


@pytest.mark.unit
def test_acquire_lock_polling_acquires_and_returns_true():
    from kindred.gui import simulation_worker

    lock = threading.Lock()
    acquired = simulation_worker._acquire_lock_polling(lock, cancelled=lambda: False, timeout_s=0.001)
    assert acquired is True
    lock.release()


@pytest.mark.unit
def test_release_lock_if_acquired_does_not_release_when_false():
    from kindred.gui import simulation_worker

    class _FakeLock:
        def release(self):
            raise AssertionError("release() should not be called when acquired=False")

    simulation_worker._release_lock_if_acquired(_FakeLock(), False)  # type: ignore[arg-type]


@pytest.mark.unit
def test_acquire_lock_polling_respects_cancellation_while_waiting():
    from kindred.gui import simulation_worker

    lock = threading.Lock()
    lock.acquire()

    cancelled_flag = threading.Event()
    result_holder: dict[str, object] = {}

    def _runner():
        result_holder["acquired"] = simulation_worker._acquire_lock_polling(
            lock,
            cancelled=cancelled_flag.is_set,
            timeout_s=0.001,
        )

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    time.sleep(0.01)
    cancelled_flag.set()
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert result_holder.get("acquired") is False
    lock.release()

