"""Calculator-owned process transport and settlement, entirely off the Qt GUI thread."""
from __future__ import annotations

from multiprocessing import get_context
from multiprocessing.connection import wait
import queue
import threading

from PySide6 import QtCore

from kindred.core.symbolic.calculator_worker import run_calculator_child


class CalculatorWorker(QtCore.QThread):
    result = QtCore.Signal(str, str, str, bool, str, object)
    failed = QtCore.Signal(str)

    def __init__(self, mechanism, source, *, parent):
        super().__init__(parent)
        self._inputs = (mechanism, source)
        self._queries = queue.SimpleQueue()
        self._shutdown = threading.Event()

    def submit(self, query: str) -> None:
        self._queries.put(query)

    def request_shutdown(self) -> None:
        self._shutdown.set()

    def run(self) -> None:
        process = None
        parent_pipe = child_pipe = None
        watchdog = None
        try:
            if self._shutdown.is_set():
                return
            spawn = get_context("spawn")
            parent_pipe, child_pipe = spawn.Pipe()
            process = spawn.Process(target=run_calculator_child, args=(child_pipe, *self._inputs), daemon=True)
            process.start()
            child_pipe.close()
            child_pipe = None

            def cancel_noncooperative_work():
                self._shutdown.wait()
                if process.is_alive():
                    process.terminate()
                    process.join(2.0)
                    if process.is_alive():
                        process.kill()

            watchdog = threading.Thread(target=cancel_noncooperative_work, name="Calculator cancellation")
            watchdog.start()
            while not self._shutdown.is_set():
                try:
                    query = self._queries.get_nowait()
                except queue.Empty:
                    pass
                else:
                    parent_pipe.send(query)
                ready = wait((parent_pipe, process.sentinel), timeout=0.05)
                if parent_pipe in ready:
                    query, result, context, error, copy_text, reuse_queries = parent_pipe.recv()
                    if not self._shutdown.is_set():
                        self.result.emit(query, result, context, error, copy_text, reuse_queries)
                if process.sentinel in ready:
                    if not self._shutdown.is_set():
                        self.failed.emit("Calculator child exited unexpectedly; the next explicit query can start a new context.")
                    break
        except Exception as exc:
            if not self._shutdown.is_set():
                self.failed.emit(f"Calculator process failed: {exc}")
        finally:
            self._shutdown.set()
            if watchdog is not None:
                watchdog.join()
            if process is not None and process.pid is not None:
                if process.is_alive():
                    process.terminate()
                    process.join(2.0)
                    if process.is_alive():
                        process.kill()
                process.join()
                process.close()
            if parent_pipe is not None:
                parent_pipe.close()
            if child_pipe is not None:
                child_pipe.close()
