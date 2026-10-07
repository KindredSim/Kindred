"""Bounded local processes calling the complete shared simulation operation."""
from __future__ import annotations

from dataclasses import dataclass, replace
from multiprocessing import get_context
from multiprocessing.connection import wait
import time

from kindred.core.exceptions import SimulationCancelled
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.simulation_execution import execute_simulation
from kindred.core.simulation_preparation import build_simulation_execution_request, SimulationPreparationError
from kindred.core.simulation_preparation_stages import build_solver_config
from kindred.core.simulator.dsl_parse import prepare_dsl_request_set
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.core.units import UnitsModel
from kindred.scan.config import START, END, ScanProgress
from kindred.scan.json_export import JsonScanWriter, input_record, result_record
from kindred.scan.sampling import PreparedScanDesign


def execute_input(config, prepared, solver, run, cancellation):
    record = input_record(run)
    stage = "input_generation" if run.generation_error else "request_construction"
    try:
        if run.generation_error:
            raise ValueError(run.generation_error)
        request = build_simulation_execution_request(
            prepared_dsl=prepared, mechanism_source=MechanismAuthoringSource(config.source),
            initial_values=record["inputs"]["initials"].items(),
            parameter_values=record["inputs"]["parameters"],
            t_span=(run.values[START], run.values[END]), solver_config=solver,
            use_authored_stops=config.use_authored_stops,
        )
        stage = "execution"
        result = execute_simulation(request, cancellation_control=cancellation)
    except SimulationCancelled:
        record.update(simulation_status="cancelled", export_status="not_applicable", stage=stage)
    except Exception as exc:
        if isinstance(exc, SimulationPreparationError):
            stage = exc.stage
        record.update(simulation_status="failed", export_status="not_applicable", stage=stage,
                      error={"type": type(exc).__name__, "message": str(exc)})
    else:
        record = result_record(run, result, config.outputs)
    return record


def _worker(config, connection, cancel):
    try:
        try:
            prepared = prepare_dsl_request_set(config.source, units=UnitsModel(
                energy_unit=config.energy_unit, temperature_K=config.temperature_K,
                standard_conc_M=config.standard_conc_M))
            solver = build_solver_config(solver_input=config.solver, rtol=config.rtol, atol=config.atol,
                                         grid={"N": config.points}, use_analytic_jacobian=config.analytic_jacobian)
        except Exception as exc:
            connection.send(("bootstrap_failure", f"{type(exc).__name__}: {exc}"))
            return
        control = SimulationCancellationControl(cancel.is_set)
        while True:
            run = connection.recv()
            if run is None:
                return
            if cancel.is_set():
                connection.send(("not_started", run.run_id))
                continue
            connection.send(("started", run.run_id))
            record = execute_input(config, prepared, solver, run, control)
            connection.send(("result", record))
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


@dataclass
class _Worker:
    process: object
    connection: object
    run: object = None
    started: bool = False
    stopping: bool = False


def _pending(connection):
    """Whether input is buffered; a peer-closed pipe counts as drained.

    On Windows, PipeConnection.poll raises BrokenPipeError once the child has
    exited instead of reporting readable EOF as Linux does.
    """
    try:
        return connection.poll()
    except OSError:
        return False


def run_scan(config, prepared, cancel, progress_callback):
    """Run off the GUI thread. Own processes, generator and the sole file writer."""
    progress = ScanProgress(config.requested)
    process_cancel = None
    children = []
    writer = None
    design = None
    iterator = None
    writing = False
    exhausted = False
    last_update = 0.0

    def emit(force=False):
        nonlocal last_update
        now = time.monotonic()
        if force or now - last_update >= 0.2:
            progress_callback(replace(progress))
            last_update = now

    def check():
        if cancel.is_set():
            raise SimulationCancelled()

    def fail(message):
        progress.error = f"{progress.error}\n{message}" if progress.error else message
        progress.state = "incomplete"
        if process_cancel is not None:
            process_cancel.set()

    def dispatch(child):
        nonlocal exhausted
        child.run, child.started = None, False
        if not exhausted and not process_cancel.is_set():
            try:
                child.run = next(iterator)
            except StopIteration:
                exhausted = True
            except SimulationCancelled:
                process_cancel.set()
            except Exception as exc:
                fail(f"Input generation could not continue: {type(exc).__name__}: {exc}")
        if child.run is None:
            child.stopping = True
        child.connection.send(child.run)

    def consume(record):
        nonlocal writing
        state = record["simulation_status"]
        setattr(progress, state, getattr(progress, state) + 1)
        progress.running -= 1
        if state == "failed":
            progress.latest_failure = f"Run {record['run_id']} — {record.get('stage', 'execution')} — {record['error']['message']}"
        if record["export_status"] == "failed":
            progress.export_errors += 1
            progress.latest_failure = f"Run {record['run_id']} — unavailable exported outputs: {', '.join(record['missing_outputs'])}"
        if writing:
            try:
                writer.write(record)
                progress.saved += 1
            except Exception as exc:
                writing = False
                fail(f"JSON writing failed: {type(exc).__name__}: {exc}")

    try:
        emit(True)
        context = get_context("spawn")
        process_cancel = context.Event()
        design = PreparedScanDesign(config, prepared, check)
        check()
        writer = JsonScanWriter(config)
        writer.initialize(config, design)
        progress.path = writer.path
        writing = True
        iterator = iter(design.inputs())
        for _ in range(min(config.workers, config.requested)):
            check()
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(config, child, process_cancel), daemon=True)
            try:
                process.start()
            except BaseException:
                parent.close()
                child.close()
                raise
            child.close()
            state = _Worker(process, parent)
            children.append(state)
            dispatch(state)
        progress.phase = "Running"
        while children:
            if cancel.is_set():
                process_cancel.set()
                progress.phase = "Canceling"
            ready = wait([handle for child in children for handle in (child.connection, child.process.sentinel)], timeout=0.1)
            for child in list(children):
                connection_closed = False
                if child.connection in ready:
                    try:
                        kind, payload = child.connection.recv()
                    except (EOFError, OSError):
                        connection_closed = True
                    else:
                        if kind == "started":
                            child.started = True
                            progress.attempted += 1
                            progress.running += 1
                        elif kind == "result":
                            consume(payload)
                            dispatch(child)
                        elif kind == "not_started":
                            dispatch(child)
                        elif kind == "bootstrap_failure":
                            fail(f"Worker preparation failed: {payload}")
                            child.run = None
                # Drain the pipe before declaring a started attempt lost.
                if connection_closed or (child.process.sentinel in ready and not _pending(child.connection)):
                    if child.run is not None:
                        if child.started:
                            record = input_record(child.run)
                            record.update(simulation_status="failed", export_status="not_applicable", stage="worker",
                                          error={"type": "WorkerExit", "message": "Worker exited before returning its result."})
                            consume(record)
                        fail("A worker exited unexpectedly; remaining work was canceled without retries.")
                    child.process.join()
                    if child.process.exitcode and not progress.error:
                        fail(f"A local worker exited with code {child.process.exitcode}.")
                    child.connection.close()
                    child.process.close()
                    children.remove(child)
            emit()
        if not progress.error:
            progress.state = "complete" if progress.attempted == config.requested and not progress.cancelled else "cancelled"
        progress.phase = "Finalizing export"
        emit(True)
    except SimulationCancelled:
        progress.state = "cancelled"
        if process_cancel is not None:
            process_cancel.set()
    except Exception as exc:
        fail(f"{type(exc).__name__}: {exc}")
    finally:
        if process_cancel is not None:
            process_cancel.set()
        # On an outer infrastructure error, keep draining senders during cleanup.
        for child in children:
            try:
                if not child.stopping:
                    try:
                        child.connection.send(None)
                    except (BrokenPipeError, OSError):
                        pass
                while child.process.is_alive() or _pending(child.connection):
                    # wait() reports a peer-closed pipe as readable on every platform.
                    if wait([child.connection], timeout=0.1):
                        try:
                            kind, payload = child.connection.recv()
                            if kind == "started":
                                child.started = True
                                progress.attempted += 1
                                progress.running += 1
                            elif kind == "result":
                                consume(payload)
                                child.run = None
                                child.started = False
                        except (EOFError, OSError):
                            break
                    emit()
                child.process.join()
                if child.run is not None and child.started:
                    record = input_record(child.run)
                    record.update(simulation_status="failed", export_status="not_applicable", stage="worker",
                                  error={"type": "WorkerExit", "message": "Worker exited without a terminal result."})
                    consume(record)
            finally:
                child.connection.close()
                child.process.close()
        if iterator is not None:
            try:
                iterator.close()
            except Exception as exc:
                fail(f"Input generator cleanup failed: {type(exc).__name__}: {exc}")
        if writer is not None:
            progress.path = writer.path
            if writing:
                try:
                    progress.path = writer.finish(progress)
                except Exception as exc:
                    fail(f"JSON finalization failed: {type(exc).__name__}: {exc}")
            try:
                writer.close()
            except Exception as exc:
                fail(f"JSON closing failed: {type(exc).__name__}: {exc}")
        if design is not None:
            try:
                design.close()
            except Exception as exc:
                fail(f"Scan storage cleanup failed: {type(exc).__name__}: {exc}")
        progress.phase = "Finished"
        emit(True)
    return progress
