"""Qt bridge owning one installed Windows scientific subprocess."""

from __future__ import annotations

import queue
import subprocess
import threading

from PySide6 import QtCore

from kindred.core.simulation_failure import factual_cause_from_exception
from kindred.gui.fitting import codec, launch, runtime

__all__ = ["GlobalFitWorker"]

_TERMINAL_EVENTS = frozenset({
    "candidate_ready", "factual_failure", "scientific_unavailable", "cancelled",
})
_LIVE_EVENTS = frozenset({
    "progress", "best_updated", "curves_updated",
})


class GlobalFitWorker(QtCore.QThread):
    """Serialize, receive and settle off the GUI thread; controls never wait."""

    execution_started = QtCore.Signal()
    progress = QtCore.Signal(int, str)
    best_updated = QtCore.Signal(object)
    curves_updated = QtCore.Signal(object)
    candidate_ready = QtCore.Signal(object)
    factual_failure = QtCore.Signal(object)
    scientific_unavailable = QtCore.Signal(object)
    cancelled = QtCore.Signal()

    def __init__(self, request: launch.FitExecutionRequest, *, parent=None) -> None:
        super().__init__(parent)
        self._request = request
        self._commands = queue.SimpleQueue()
        self._cancelled = threading.Event()
        self._shutdown = threading.Event()
        self.process_started = False
        self.terminal_projection = None

    def pause(self) -> None:
        self._commands.put("pause")

    def resume(self) -> None:
        self._commands.put("resume")

    def request_cancel(self) -> None:
        self._cancelled.set()
        # Arm before startup or blocked command delivery can delay cancellation.
        self._shutdown.set()
        self._commands.put("cancel")

    def request_stop(self) -> None:
        self._commands.put("stop")

    def run(self) -> None:
        process = None
        terminal = None
        projection = None
        errors = []
        error_lock = threading.Lock()
        settled = threading.Event()
        threads = []
        diagnostics = bytearray()

        def fail(exc):
            with error_lock:
                if not errors:
                    errors.append(exc)
            self._commands.put("cancel")
            self._commands.put(None)
            self._shutdown.set()

        def write_commands(initial):
            try:
                codec.write_frame(process.stdin, initial)
                while (command := self._commands.get()) is not None:
                    codec.write_frame(process.stdin, codec._encode_fit(command, {}))
            except BaseException as exc:
                fail(exc)
            finally:
                process.stdin.close()

        def drain_diagnostics():
            try:
                while chunk := process.stderr.read(8192):
                    diagnostics.extend(chunk)
                    del diagnostics[:-65536]
            except BaseException as exc:
                fail(exc)

        def watchdog():
            self._shutdown.wait()
            if settled.wait(5.0):
                return
            fail(RuntimeError("Fit process did not settle within five seconds of shutdown."))
            try:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
            except OSError as exc:
                if process.poll() is None:
                    fail(exc)

        def start_thread(target, name, args=()):
            thread = threading.Thread(target=target, name=name, args=args)
            thread.start()
            threads.append(thread)

        try:
            if self._cancelled.is_set():
                terminal = ("cancelled", ())
            else:
                origin = self._request.setup.origin
                request = codec._encode_fit(self._request, {"prepared": origin.prepared_dsl})
                initial = codec._encode_fit((
                    origin.mechanism_source, origin.scientific_settings, request,
                ), {})
                if not self._cancelled.is_set():
                    argv, env, cwd = runtime.fitting_process_command()
                    # Cancellation can arrive during encoding or source hashing.
                    if not self._cancelled.is_set():
                        process = subprocess.Popen(
                            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0, shell=False,
                            env=env, cwd=cwd,
                            creationflags=subprocess.CREATE_NO_WINDOW,
                        )
                if process is None:
                    terminal = ("cancelled", ())
                else:
                    self.process_started = True
                    self.execution_started.emit()
                    start_thread(watchdog, "fit-cleanup")
                    start_thread(drain_diagnostics, "fit-stderr")
                    start_thread(write_commands, "fit-writer", (initial,))
                    references = {"request": self._request, "origin": origin}
                    while (payload := codec.read_frame(process.stdout)) is not None:
                        if errors:
                            continue
                        try:
                            name, args = codec._decode_fit(payload, references)
                            if terminal is not None:
                                raise RuntimeError("Fit process sent output after its terminal event.")
                            if name == "candidate_ready":
                                candidate, curves, curve_failure = args
                                projection = (curves, curve_failure)
                                terminal = (name, (candidate,))
                            elif name in _TERMINAL_EVENTS:
                                terminal = (name, args)
                            elif name in _LIVE_EVENTS:
                                getattr(self, name).emit(*args)
                            else:
                                raise RuntimeError(f"Unknown Fit event: {name!r}.")
                            if terminal is not None:
                                self._commands.put(None)
                                self._shutdown.set()
                        except BaseException as exc:
                            fail(exc)
                    if terminal is None and not errors:
                        fail(RuntimeError("Fit process exited without a terminal value."))
        except BaseException as exc:
            fail(exc)
        finally:
            if process is not None:
                # Even malformed framing must not leave a child blocked on output.
                self._shutdown.set()
                self._commands.put(None)
                try:
                    while process.stdout.read(65536):
                        pass
                except OSError as exc:
                    fail(exc)
                process.wait()
                if process.returncode != 0:
                    fail(RuntimeError(
                        f"Fit process exited unexpectedly (exit code {process.returncode})."
                    ))
                # Closing stdin is the child's command-reader shutdown signal.
                # The writer owns it while active, so join before closing pipes.
                for thread in threads:
                    if thread.name != "fit-cleanup":
                        thread.join()
                settled.set()
                for thread in threads:
                    if thread.name == "fit-cleanup":
                        thread.join()
                for pipe in (process.stdin, process.stdout, process.stderr):
                    pipe.close()
            else:
                settled.set()

        if errors:
            detail = str(errors[0]) or type(errors[0]).__name__
            if diagnostics:
                detail += "\nFitting runtime diagnostics:\n" + diagnostics.decode("utf-8", errors="replace")
            terminal = ("factual_failure", (factual_cause_from_exception(
                RuntimeError(detail), domain="fit", origin="fit_process",
            ),))
        elif self._cancelled.is_set():
            terminal = ("cancelled", ())
        if terminal is None:
            terminal = ("factual_failure", (factual_cause_from_exception(
                RuntimeError("Fit bridge settled without a terminal value."),
                domain="fit", origin="fit_process",
            ),))
        name, args = terminal
        if name == "candidate_ready":
            self.terminal_projection = projection
        getattr(self, name).emit(*args)
