from dataclasses import dataclass

import numpy as np

from kindred.core.cache import fingerprint_simulation_request, generate_mechanism_hash
from kindred.core.mechanism import Mechanism
from kindred.core.simulator.solvers import SimulationRequest
from kindred.core.temperature import TemperatureSchedule, coerce_temperature_schedule


@dataclass(frozen=True, repr=False)
class _PlainCallableSchedule:
    base_temp: float

    def __call__(self, _t: float) -> float:
        return float(self.base_temp)


def _mechanism_with_schedule(schedule) -> Mechanism:
    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)
    mechanism.metadata["temperature_schedule"] = schedule
    return mechanism


def test_coerce_temperature_schedule_accepts_to_dict_mapping():
    schedule = TemperatureSchedule.piecewise([(0.0, 1.0, 300.0), (1.0, 2.0, 350.0)])
    payload = schedule.to_dict()

    coerced = coerce_temperature_schedule(payload)
    assert coerced is not None
    assert callable(coerced)
    assert float(coerced(0.5)) == float(schedule(0.5))


def test_fingerprint_simulation_request_includes_temperature_schedule_fingerprint():
    def rhs(_t, y):
        return -y

    y0 = np.array([1.0], dtype=float)

    req_a = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 1.0),
        y0=y0,
        solver="BDF",
        grid={"N": 10},
        temperature_schedule=TemperatureSchedule.constant(300.0),
    )
    req_b = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 1.0),
        y0=y0,
        solver="BDF",
        grid={"N": 10},
        temperature_schedule=TemperatureSchedule.constant(310.0),
    )

    fp_a = fingerprint_simulation_request(req_a)
    fp_b = fingerprint_simulation_request(req_b)
    assert fp_a is not None and fp_b is not None
    assert fp_a != fp_b


def test_generate_mechanism_hash_is_deterministic_for_equivalent_plain_callable_schedules():
    hash_a = generate_mechanism_hash(_mechanism_with_schedule(_PlainCallableSchedule(300.0)))
    hash_b = generate_mechanism_hash(_mechanism_with_schedule(_PlainCallableSchedule(300.0)))
    hash_c = generate_mechanism_hash(_mechanism_with_schedule(_PlainCallableSchedule(310.0)))

    assert hash_a == hash_b
    assert hash_a != hash_c


def test_fingerprint_simulation_request_is_deterministic_for_equivalent_plain_callable_schedules():
    def rhs(_t, y):
        return -y

    y0 = np.array([1.0], dtype=float)

    req_a = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 1.0),
        y0=y0,
        solver="BDF",
        grid={"N": 10},
        temperature_schedule=_PlainCallableSchedule(300.0),
    )
    req_b = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 1.0),
        y0=y0,
        solver="BDF",
        grid={"N": 10},
        temperature_schedule=_PlainCallableSchedule(300.0),
    )
    req_c = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 1.0),
        y0=y0,
        solver="BDF",
        grid={"N": 10},
        temperature_schedule=_PlainCallableSchedule(310.0),
    )

    fp_a = fingerprint_simulation_request(req_a)
    fp_b = fingerprint_simulation_request(req_b)
    fp_c = fingerprint_simulation_request(req_c)

    assert fp_a is not None and fp_b is not None and fp_c is not None
    assert fp_a == fp_b
    assert fp_a != fp_c


def test_temp_response_fingerprint_is_deterministic_for_equivalent_schedules():
    schedule_a = TemperatureSchedule.response([0.0, 5.0, 10.0], [300.0, 600.0], tau=2.0)
    schedule_b = TemperatureSchedule.response([0.0, 5.0, 10.0], [300.0, 600.0], tau=2.0)
    schedule_c = TemperatureSchedule.response([0.0, 5.0, 10.0], [300.0, 600.0], tau=3.0)

    hash_a = generate_mechanism_hash(_mechanism_with_schedule(schedule_a))
    hash_b = generate_mechanism_hash(_mechanism_with_schedule(schedule_b))
    hash_c = generate_mechanism_hash(_mechanism_with_schedule(schedule_c))

    assert hash_a == hash_b
    assert hash_a != hash_c


def test_temp_response_and_temp_step_have_distinct_fingerprints():
    response = TemperatureSchedule.response([0.0, 5.0, 10.0], [300.0, 600.0], tau=2.0)
    step = TemperatureSchedule.piecewise([(0.0, 5.0, 300.0), (5.0, 10.0, 600.0)])

    assert response.fingerprint != step.fingerprint
