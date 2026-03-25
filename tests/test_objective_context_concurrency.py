import threading

import numpy as np

from kindred.core.objective import ObjectiveContext, ObjectiveWrapper


def test_objective_context_is_thread_local():
    ctx = ObjectiveContext()

    def base_fn(params):
        value = float(params[0])
        ctx.set_error(f"err-{value}", {"value": value})
        return np.array([value])

    wrapped = ObjectiveWrapper(base_fn, ctx)

    results = {}

    def worker(name, val):
        wrapped(np.array([val]))
        results[name] = (wrapped.last_error, wrapped.last_error_provenance)

    threads = [
        threading.Thread(target=worker, args=("a", 1.0)),
        threading.Thread(target=worker, args=("b", 2.0)),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert results["a"][0] == "err-1.0"
    assert results["b"][0] == "err-2.0"
    assert results["a"][1] != results["b"][1]
