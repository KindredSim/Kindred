"""Spreadsheet process and pipe lifetime, supervised off the GUI thread."""
from __future__ import annotations

from multiprocessing import get_context
from multiprocessing.connection import wait
import queue
import threading

from PySide6 import QtCore

from kindred.gui.spreadsheet_session import run_spreadsheet_child


class SpreadsheetWorker(QtCore.QThread):
    result = QtCore.Signal(str, object, str)
    failed = QtCore.Signal(str)

    def __init__(self, records, names, *, parent):
        super().__init__(parent)
        self._inputs = records, names
        self._requests = queue.SimpleQueue()
        self._shutdown = threading.Event()

    def submit(self, operation, sheet, args=()):
        self._requests.put((operation, sheet, args))

    def request_shutdown(self):
        self._shutdown.set()

    def run(self):
        process = parent_pipe = child_pipe = watchdog = None
        try:
            if self._shutdown.is_set():
                return
            spawn = get_context("spawn")
            parent_pipe, child_pipe = spawn.Pipe()
            process = spawn.Process(target=run_spreadsheet_child, args=(child_pipe, *self._inputs), daemon=True)
            process.start()
            child_pipe.close()
            child_pipe = None

            def stop_calculation():
                self._shutdown.wait()
                if process.is_alive():
                    process.terminate()
                    process.join(2.0)
                    if process.is_alive():
                        process.kill()

            watchdog = threading.Thread(target=stop_calculation, name="Spreadsheet cancellation")
            watchdog.start()
            while not self._shutdown.is_set():
                try:
                    request = self._requests.get_nowait()
                except queue.Empty:
                    pass
                else:
                    parent_pipe.send(request)
                ready = wait((parent_pipe, process.sentinel), timeout=0.05)
                if parent_pipe in ready:
                    operation, value, error = parent_pipe.recv()
                    if not self._shutdown.is_set():
                        self.result.emit(operation, value, error)
                if process.sentinel in ready:
                    if not self._shutdown.is_set():
                        self.failed.emit("Spreadsheet calculation stopped.")
                    break
        except Exception as exc:
            if not self._shutdown.is_set():
                self.failed.emit(f"Spreadsheet calculation failed: {exc}")
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
