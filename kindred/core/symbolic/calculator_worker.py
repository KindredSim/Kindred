"""Same-runtime Calculator child; its input is admitted science, never source to admit."""
from __future__ import annotations

import sys
from html import escape

from .calculator import SymbolicCalculatorError, build_calculator_context, evaluate_symbolic_query

# Some SymPy algorithms grow without bound; the child ends at this limit
# instead of exhausting the machine.
_MEMORY_LIMIT = 2 * 2**30


def _limit_memory() -> None:
    """Cap this process's memory; a system that refuses leaves it uncapped."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        # 0x100 is JOB_OBJECT_LIMIT_PROCESS_MEMORY.
        info = Extended(BasicLimitInformation=Basic(LimitFlags=0x100), ProcessMemoryLimit=_MEMORY_LIMIT)
        job = kernel32.CreateJobObjectW(None, None)
        # 9 is JobObjectExtendedLimitInformation. The job keeps its limit while this
        # process belongs to it, whatever becomes of the handle.
        if job and kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())
    elif sys.platform.startswith("linux"):
        import resource

        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        soft = _MEMORY_LIMIT if hard == resource.RLIM_INFINITY else min(_MEMORY_LIMIT, hard)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        except (ValueError, OSError):
            pass


def run_calculator_child(connection, mechanism, source) -> None:
    _limit_memory()
    context = None
    try:
        while (query := connection.recv()) is not None:
            try:
                if context is None:
                    context = build_calculator_context(mechanism, mechanism_source=source)
                result = evaluate_symbolic_query(context, query)
            except SymbolicCalculatorError as exc:
                message = str(exc)
                connection.send((
                    query,
                    "<pre>" + escape(message) + "</pre>",
                    f"Query: {query}\nError: {message}\nMechanism source:\n{source}",
                    True,
                    message,
                    (),
                ))
            else:
                context_text = (f"Query: {query}\nError: {result.copy_text}\nMechanism source:\n{source}" if result.failed
                                else result.full_context_copy_text(context.authored_definitions))
                connection.send((
                    query,
                    result.result_html,
                    context_text,
                    result.failed,
                    result.copy_text,
                    result.reuse_queries,
                ))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()
