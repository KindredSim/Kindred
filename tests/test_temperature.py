"""
Tests for temperature schedule functionality.

Tests cover:
- Constant temperature schedules
- Piecewise temperature schedules
- Temperature DSL parsing
- Temperature schedule integration with solver
"""

import pytest
from kindred.core.temperature import (
    TemperatureSchedule,
    TemperatureScheduleError,
    TimeInterval,
)
from kindred.core.temperature_dsl import (
    parse_temperature_schedule,
    TemperatureDSLError,
)

pytestmark = pytest.mark.unit



class TestConstantTemperature:
    """Test constant temperature schedules."""

    def test_create_constant(self):
        """Test creating constant temperature schedule."""
        T = TemperatureSchedule.constant(298.15)

        assert T(0.0) == 298.15
        assert T(50.0) == 298.15
        assert T(1000.0) == 298.15

    def test_constant_negative_temperature(self):
        """Test that negative temperature raises error."""
        with pytest.raises(TemperatureScheduleError):
            TemperatureSchedule.constant(-10.0)

    def test_constant_zero_temperature(self):
        """Test that zero temperature raises error."""
        with pytest.raises(TemperatureScheduleError):
            TemperatureSchedule.constant(0.0)


class TestPiecewiseTemperature:
    """Test piecewise temperature schedules."""

    def test_create_piecewise(self):
        """Test creating piecewise temperature schedule."""
        intervals = [(0, 50, 298), (50, 100, 350)]
        T = TemperatureSchedule.piecewise(intervals)

        assert T(25.0) == 298.0
        assert T(75.0) == 350.0

    def test_piecewise_boundaries(self):
        """Test piecewise schedule at interval boundaries."""
        intervals = [(0, 50, 298), (50, 100, 350)]
        T = TemperatureSchedule.piecewise(intervals)

        assert T(0.0) == 298.0    # Start of first interval
        assert T(49.9) == 298.0   # End of first interval (exclusive)
        assert T(50.0) == 350.0   # Start of second interval
        assert T(100.0) == 350.0  # End of last interval (inclusive)

    def test_piecewise_extrapolation(self):
        """Test piecewise schedule beyond defined range."""
        intervals = [(0, 50, 298), (50, 100, 350)]
        T = TemperatureSchedule.piecewise(intervals)

        # Beyond last interval should use last temperature
        assert T(150.0) == 350.0

    def test_piecewise_three_intervals(self):
        """Test piecewise schedule with three intervals."""
        intervals = [(0, 50, 298), (50, 100, 350), (100, 150, 320)]
        T = TemperatureSchedule.piecewise(intervals)

        assert T(25.0) == 298.0
        assert T(75.0) == 350.0
        assert T(125.0) == 320.0

    def test_piecewise_validation_gap(self):
        """Test that gaps in intervals raise error."""
        intervals = [(0, 50, 298), (60, 100, 350)]  # Gap between 50 and 60
        with pytest.raises(TemperatureScheduleError) as exc_info:
            TemperatureSchedule.piecewise(intervals)

        assert "gap" in str(exc_info.value).lower()

    def test_piecewise_validation_overlap(self):
        """Test that overlapping intervals raise error."""
        intervals = [(0, 60, 298), (50, 100, 350)]  # Overlap between 50 and 60
        with pytest.raises(TemperatureScheduleError) as exc_info:
            TemperatureSchedule.piecewise(intervals)

        assert "overlap" in str(exc_info.value).lower()

    def test_piecewise_not_starting_at_zero(self):
        """Test that intervals not starting at t=0 raise error."""
        intervals = [(10, 50, 298), (50, 100, 350)]
        with pytest.raises(TemperatureScheduleError) as exc_info:
            TemperatureSchedule.piecewise(intervals)

        assert "t=0" in str(exc_info.value).lower()


class TestTemperatureDSL:
    """Test temperature schedule DSL parsing."""

    def test_parse_constant(self):
        """Test parsing constant temperature."""
        dsl = "temp_const: T=310.15"
        schedule = parse_temperature_schedule(dsl)

        assert schedule is not None
        assert schedule(0.0) == 310.15

    def test_parse_piecewise(self):
        """Test parsing piecewise temperature."""
        dsl = "temp_step: t=[0,50,100], T=[298,350]"
        schedule = parse_temperature_schedule(dsl)

        assert schedule is not None
        assert schedule(25.0) == 298.0
        assert schedule(75.0) == 350.0

    def test_parse_response(self):
        """Test parsing first-order response temperature."""
        dsl = "temp_response: t=[0,50,100], T=[298,350], tau=10"
        schedule = parse_temperature_schedule(dsl)

        assert schedule is not None
        assert schedule.schedule_type == "response"
        assert schedule(0.0) == pytest.approx(298.0)
        assert schedule(50.0) == pytest.approx(298.0)
        assert 298.0 < schedule(60.0) < 350.0
        assert schedule(1000.0) == pytest.approx(350.0, rel=0.0, abs=1e-6)

    def test_parse_no_schedule(self):
        """Test that DSL without temperature schedule returns None."""
        dsl = "reaction: A -> B; k=1.0"
        schedule = parse_temperature_schedule(dsl)

        assert schedule is None

    def test_parse_invalid_constant_syntax(self):
        """Test error for invalid constant temperature syntax."""
        dsl = "temp_const: 310.15"  # Missing T=
        with pytest.raises(TemperatureDSLError) as exc_info:
            parse_temperature_schedule(dsl)

        error_msg = str(exc_info.value)
        assert "example" in error_msg.lower()

    def test_parse_invalid_piecewise_syntax(self):
        """Test error for invalid piecewise syntax."""
        dsl = "temp_step: t=[0,50], temp=[298,350]"  # Should be T=, not temp=
        with pytest.raises(TemperatureDSLError) as exc_info:
            parse_temperature_schedule(dsl)

        error_msg = str(exc_info.value)
        assert "example" in error_msg.lower()

    def test_parse_piecewise_count_mismatch(self):
        """Test error for temperature count mismatch."""
        dsl = "temp_step: t=[0,50,100], T=[298]"  # Need 2 temperatures
        with pytest.raises(TemperatureDSLError) as exc_info:
            parse_temperature_schedule(dsl)

        error_msg = str(exc_info.value)
        assert "mismatch" in error_msg.lower()
        assert "example" in error_msg.lower()

    def test_parse_response_requires_positive_tau(self):
        """Test error for non-positive response tau."""
        dsl = "temp_response: t=[0,50], T=[298], tau=0"
        with pytest.raises(TemperatureDSLError) as exc_info:
            parse_temperature_schedule(dsl)

        assert "tau" in str(exc_info.value).lower()


class TestResponseTemperature:
    """Test first-order response schedules."""

    def test_create_response_schedule(self):
        schedule = TemperatureSchedule.response([0.0, 5.0, 10.0], [300.0, 600.0], tau=2.0)

        assert schedule.schedule_type == "response"
        assert schedule(0.0) == pytest.approx(300.0)
        assert schedule(5.0) == pytest.approx(300.0)
        assert 300.0 < schedule(6.0) < 600.0
        assert schedule(100.0) == pytest.approx(600.0, rel=0.0, abs=1e-6)

    def test_response_schedule_monotonically_approaches_setpoint(self):
        schedule = TemperatureSchedule.response([0.0, 5.0, 10.0], [300.0, 600.0], tau=2.0)

        early = schedule(5.5)
        mid = schedule(7.0)
        late = schedule(10.0)

        assert 300.0 < early < mid < late < 600.0

    def test_response_schedule_requires_positive_tau(self):
        with pytest.raises(TemperatureScheduleError):
            TemperatureSchedule.response([0.0, 5.0], [300.0], tau=0.0)

    def test_response_to_dict(self):
        schedule = TemperatureSchedule.response([0.0, 5.0, 10.0], [300.0, 600.0], tau=2.5)
        data = schedule.to_dict()

        assert data["type"] == "response"
        assert data["times"] == [0.0, 5.0, 10.0]
        assert data["temperatures"] == [300.0, 600.0]
        assert data["tau"] == 2.5

    def test_response_from_dict(self):
        schedule = TemperatureSchedule.from_dict(
            {
                "type": "response",
                "times": [0.0, 5.0, 10.0],
                "temperatures": [300.0, 600.0],
                "tau": 2.0,
            }
        )

        assert schedule.schedule_type == "response"
        assert schedule(0.0) == pytest.approx(300.0)
        assert 300.0 < schedule(6.0) < 600.0
        assert schedule(100.0) == pytest.approx(600.0, rel=0.0, abs=1e-6)


class TestTimeInterval:
    """Test TimeInterval dataclass."""

    def test_valid_interval(self):
        """Test creating valid interval."""
        interval = TimeInterval(0.0, 50.0, 298.15)

        assert interval.t_start == 0.0
        assert interval.t_end == 50.0
        assert interval.temperature == 298.15

    def test_contains(self):
        """Test interval.contains() method."""
        interval = TimeInterval(0.0, 50.0, 298.15)

        assert interval.contains(25.0)
        assert not interval.contains(50.0)  # Exclusive end
        assert not interval.contains(-10.0)
        assert not interval.contains(60.0)

    def test_contains_inclusive(self):
        """Test interval.contains_inclusive() method."""
        interval = TimeInterval(0.0, 50.0, 298.15)

        assert interval.contains_inclusive(50.0)  # Inclusive end

    def test_negative_start_time(self):
        """Test that negative start time raises error."""
        with pytest.raises(TemperatureScheduleError):
            TimeInterval(-10.0, 50.0, 298.15)

    def test_end_before_start(self):
        """Test that end time before start raises error."""
        with pytest.raises(TemperatureScheduleError):
            TimeInterval(50.0, 0.0, 298.15)

    def test_negative_temperature(self):
        """Test that negative temperature raises error."""
        with pytest.raises(TemperatureScheduleError):
            TimeInterval(0.0, 50.0, -10.0)


class TestTemperatureSerialization:
    """Test temperature schedule serialization."""

    def test_constant_to_dict(self):
        """Test serializing constant temperature to dict."""
        T = TemperatureSchedule.constant(298.15)
        data = T.to_dict()

        assert data["type"] == "constant"
        assert data["temperature"] == 298.15

    def test_piecewise_to_dict(self):
        """Test serializing piecewise temperature to dict."""
        intervals = [(0, 50, 298), (50, 100, 350)]
        T = TemperatureSchedule.piecewise(intervals)
        data = T.to_dict()

        assert data["type"] == "piecewise"
        assert len(data["intervals"]) == 2
        assert data["intervals"][0]["temperature"] == 298

    def test_constant_from_dict(self):
        """Test deserializing constant temperature from dict."""
        data = {"type": "constant", "temperature": 310.15}
        T = TemperatureSchedule.from_dict(data)

        assert T(0.0) == 310.15

    def test_piecewise_from_dict(self):
        """Test deserializing piecewise temperature from dict."""
        data = {
            "type": "piecewise",
            "intervals": [
                {"t_start": 0, "t_end": 50, "temperature": 298},
                {"t_start": 50, "t_end": 100, "temperature": 350}
            ]
        }
        T = TemperatureSchedule.from_dict(data)

        assert T(25.0) == 298.0
        assert T(75.0) == 350.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
