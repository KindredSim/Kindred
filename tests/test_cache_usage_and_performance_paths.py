import numpy as np
from dataclasses import replace
import gc
import weakref

from kindred.core.cache import (
    cache_simulation,
    clear_cache,
    fingerprint_simulation_request,
    get_cache_stats,
)
from kindred.core.fitting_objective import build_fitting_objective
from kindred.core.simulation_preparation import prepare_bound_mechanism
from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.solvers import SimulationRequest, solve_ode


def _simple_mechanism():
    dsl_text = "\n".join([
        "reaction: A -> B; k=1.0",
        "initial: A=1.0",
        "initial: B=0.0",
    ])
    mech = parse_dsl_to_mechanism(dsl_text, initials={})
    rhs = build_ode_rhs_from_mechanism(mech)
    y0 = np.array([mech.species[name].initial_conc for name in mech.species_names()])
    return dsl_text, mech, rhs, y0


def test_fitting_objective_compiles_once(monkeypatch):
    """Fitting objective should parse/build once and reuse cached structures."""
    clear_cache()
    counts = {"parse": 0, "build": 0}

    orig_parse = parse_dsl_to_mechanism
    orig_build = build_ode_rhs_from_mechanism

    def counting_parse(*args, **kwargs):
        counts["parse"] += 1
        return orig_parse(*args, **kwargs)

    def counting_build(*args, **kwargs):
        counts["build"] += 1
        return orig_build(*args, **kwargs)

    monkeypatch.setattr("kindred.core.simulator.dsl.parse_dsl_to_mechanism", counting_parse)
    monkeypatch.setattr("kindred.core.ode_builder.build_ode_rhs_from_mechanism", counting_build)

    dsl_text, _, _, _ = _simple_mechanism()
    objective = build_fitting_objective(
        dsl_text,
        ["k"],
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 0.5, 0.25]),
        "B",
    )

    assert counts["parse"] == 1
    assert counts["build"] == 1

    objective(np.array([1.0]))
    objective(np.array([0.5]))

    assert counts["parse"] == 1
    assert counts["build"] == 1


def test_rate_binding_updates_do_not_reparse(monkeypatch):
    """Slider-style binding updates should not trigger extra parse/build."""
    clear_cache()
    counts = {"parse": 0, "build": 0}

    orig_parse = parse_dsl_to_mechanism
    orig_build = build_ode_rhs_from_mechanism

    def counting_parse(*args, **kwargs):
        counts["parse"] += 1
        return orig_parse(*args, **kwargs)

    def counting_build(*args, **kwargs):
        counts["build"] += 1
        return orig_build(*args, **kwargs)

    monkeypatch.setattr("kindred.core.simulator.dsl.parse_dsl_to_mechanism", counting_parse)
    monkeypatch.setattr("kindred.core.ode_builder.build_ode_rhs_from_mechanism", counting_build)

    dsl_text = "\n".join([
        "reaction: A -> B; k=1.0",
        "initial: A=1.0",
        "initial: B=0.0",
    ])
    bound = prepare_bound_mechanism(
        mechanism_text=dsl_text,
        param_names=["k1"],
        temperature_K=298.15,
        initials={},
        use_advanced_dsl=True,
    )
    assert counts["parse"] == 1
    assert counts["build"] == 1

    rhs = bound.rhs
    y0 = bound.y0
    for val in (0.5, 0.75, 1.25):
        bound.bindings["k1"].set(val)
        rhs(0.0, y0)

    assert counts["parse"] == 1
    assert counts["build"] == 1


def test_cache_hits_for_identical_requests():
    """Identical SimulationRequest fingerprints should hit the cache; variants should miss."""
    clear_cache()
    _, mech, rhs, y0 = _simple_mechanism()
    base_req = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 1.0),
        y0=y0,
        solver="LSODA",
        rtol=1e-6,
        atol=1e-12,
        grid={"N": 25},
    )

    fp_base = fingerprint_simulation_request(base_req)
    assert fp_base is not None

    @cache_simulation(maxsize=8)
    def run_cached(mechanism, *, request: SimulationRequest):
        return solve_ode(request)

    _ = run_cached(mech, _cache_fingerprint=fp_base, request=base_req)
    _ = run_cached(mech, _cache_fingerprint=fp_base, request=base_req)
    stats = get_cache_stats()
    assert stats.hits >= 1
    assert stats.misses == 1

    variant_req = replace(base_req, grid={"N": 50})
    fp_variant = fingerprint_simulation_request(variant_req)
    assert fp_variant is not None and fp_variant != fp_base

    _ = run_cached(mech, _cache_fingerprint=fp_variant, request=variant_req)
    stats_after_variant = get_cache_stats()
    assert stats_after_variant.misses >= 2
    assert stats_after_variant.hits >= 1


def test_cache_registered_caches_does_not_retain_ephemeral_wrappers():
    import kindred.core.cache as cache_mod

    baseline_len = len(cache_mod._registered_caches)

    def _make_ephemeral_cache():
        @cache_simulation(maxsize=1)
        def _tmp(mechanism, *, tag: str):
            return (mechanism, tag)

        return weakref.ref(_tmp)

    ref = _make_ephemeral_cache()
    del ref
    gc.collect()

    assert len(cache_mod._registered_caches) == baseline_len


def test_disk_cache_hit_path_enforces_lru_eviction(tmp_path):
    clear_cache()
    _, mech, _, _ = _simple_mechanism()

    calls = {"n": 0}

    @cache_simulation(maxsize=1, cache_dir=tmp_path)
    def run_cached(mechanism, *, tag: str):
        calls["n"] += 1
        return {"tag": tag}

    _ = run_cached(mech, tag="A")
    _ = run_cached(mech, tag="B")
    assert calls["n"] == 2

    clear_cache()
    assert get_cache_stats().evictions == 0

    _ = run_cached(mech, tag="B")
    assert calls["n"] == 2
    assert get_cache_stats().evictions == 0

    _ = run_cached(mech, tag="A")
    assert calls["n"] == 2
    assert get_cache_stats().evictions == 1
