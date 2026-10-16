"""Spreadsheet process and pipe lifetime, supervised off the GUI thread."""
from __future__ import annotations

import logging
from multiprocessing import Pipe, get_context
from multiprocessing.connection import wait
import queue
import threading

from PySide6 import QtCore

from kindred.gui.spreadsheet_session import run_spreadsheet_child

logger = logging.getLogger(__name__)


class SpreadsheetWorker(QtCore.QThread):
    result = QtCore.Signal(str, object, str)
    failed = QtCore.Signal(str)

    def __init__(self, records, names, *, parent):
        super().__init__(parent)
        self._inputs = records, names
        self._requests = queue.SimpleQueue()
        self._shutdown = threading.Event()
        # Wakes the supervisor as soon as a request or shutdown arrives.
        self._wake, self._waker = Pipe(duplex=False)

    def submit(self, operation, sheet, args=()):
        self._requests.put((operation, sheet, args))
        self._waker.send_bytes(b"")

    def request_shutdown(self):
        self._shutdown.set()
        self._waker.send_bytes(b"")

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
                ready = wait((parent_pipe, self._wake))
                if self._wake in ready:
                    while self._wake.poll():
                        self._wake.recv_bytes()
                    while not self._requests.empty():
                        parent_pipe.send(self._requests.get())
                if parent_pipe in ready:
                    try:
                        operation, value, error = parent_pipe.recv()
                    except EOFError:
                        # The child closed its end: it has exited.
                        if not self._shutdown.is_set():
                            self.failed.emit("Spreadsheet calculation stopped.")
                        break
                    if not self._shutdown.is_set():
                        self.result.emit(operation, value, error)
        except Exception:
            logger.exception("Spreadsheet calculation process failed.")
            if not self._shutdown.is_set():
                self.failed.emit("Spreadsheet calculation stopped.")
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
