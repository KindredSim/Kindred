from kindred.core.temperature import TemperatureSchedule, coerce_temperature_schedule
import pytest

pytestmark = pytest.mark.unit

def test_coerce_temperature_schedule_accepts_to_dict_mapping():
    schedule = TemperatureSchedule.piecewise([(0.0, 1.0, 300.0), (1.0, 2.0, 350.0)])
    payload = schedule.to_dict()

    coerced = coerce_temperature_schedule(payload)
    assert coerced is not None
    assert callable(coerced)
    assert float(coerced(0.5)) == float(schedule(0.5))
def test_temp_response_fingerprint_is_deterministic_for_equivalent_schedules():
    schedule_a = TemperatureSchedule.response([0.0, 5.0, 10.0], [300.0, 600.0], tau=2.0)
    schedule_b = TemperatureSchedule.response([0.0, 5.0, 10.0], [300.0, 600.0], tau=2.0)
    schedule_c = TemperatureSchedule.response([0.0, 5.0, 10.0], [300.0, 600.0], tau=3.0)

    assert schedule_a.fingerprint == schedule_b.fingerprint
    assert schedule_a.fingerprint != schedule_c.fingerprint


def test_temp_response_and_temp_step_have_distinct_fingerprints():
    response = TemperatureSchedule.response([0.0, 5.0, 10.0], [300.0, 600.0], tau=2.0)
    step = TemperatureSchedule.piecewise([(0.0, 5.0, 300.0), (5.0, 10.0, 600.0)])

    assert response.fingerprint != step.fingerprint
