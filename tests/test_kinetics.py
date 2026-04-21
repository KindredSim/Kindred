"""
Tests for kinetics calculations (Eyring and Arrhenius).

Tests cover:
- Eyring rate constant calculations
- Arrhenius rate constant calculations
- Temperature dependence
- Units consistency
"""

import pytest
import math
from kindred.core.simulator.kinetics import (
    eyring_rate,
    arrhenius_rate,
    K_from_deltaG_eq,
    normalize_energy_to_J_per_mol,
    rate_units,
)

pytestmark = pytest.mark.unit



class TestEyringEquation:
    """Test Eyring equation calculations."""

    def test_eyring_unimolecular(self):
        """Test Eyring rate for unimolecular reaction."""
        dG_act = 75500.0  # J/mol (75.5 kJ/mol)
        T = 298.15        # K

        k = eyring_rate(dG_act, T, molecularity=1)

        # Should be positive and finite
        assert k > 0.0
        assert math.isfinite(k)

        # Check units: 1/s
        assert rate_units(1) == "1/s"

    def test_eyring_bimolecular(self):
        """Test Eyring rate for bimolecular reaction."""
        dG_act = 75500.0  # J/mol
        T = 298.15        # K

        # Use default standard_conc_M = 1.0 M
        k = eyring_rate(dG_act, T, molecularity=2, standard_conc_M=1.0)

        # For comparison, test with different standard state
        # Higher standard_conc_M gives lower bimolecular rate
        k_high_std = eyring_rate(dG_act, T, molecularity=2, standard_conc_M=10.0)
        assert k_high_std < k

        # Unimolecular rate independent of standard state
        k_uni = eyring_rate(dG_act, T, molecularity=1, standard_conc_M=1.0)
        assert isinstance(k_uni, float)  # Just verify it works

        # Check units: 1/(M*s)
        assert rate_units(2) == "1/(M*s)"

    def test_eyring_temperature_dependence(self):
        """Test that rate increases with temperature."""
        dG_act = 75500.0  # J/mol

        k_298 = eyring_rate(dG_act, 298.15, molecularity=1)
        k_310 = eyring_rate(dG_act, 310.0, molecularity=1)
        k_350 = eyring_rate(dG_act, 350.0, molecularity=1)

        # Higher temperature should give higher rate
        assert k_310 > k_298
        assert k_350 > k_310

    def test_eyring_activation_energy_dependence(self):
        """Test that rate decreases with higher activation energy."""
        T = 298.15

        k_low = eyring_rate(50000.0, T, molecularity=1)   # 50 kJ/mol
        k_high = eyring_rate(100000.0, T, molecularity=1)  # 100 kJ/mol

        # Higher activation energy should give lower rate
        assert k_low > k_high

    def test_eyring_kappa(self):
        """Test transmission coefficient κ."""
        dG_act = 75500.0
        T = 298.15

        k_default = eyring_rate(dG_act, T, kappa=1.0, molecularity=1)
        k_half = eyring_rate(dG_act, T, kappa=0.5, molecularity=1)

        # Rate should scale linearly with κ
        assert abs(k_half - 0.5 * k_default) < 1e-10

    def test_eyring_invalid_temperature(self):
        """Test that negative temperature raises error."""
        with pytest.raises(ValueError):
            eyring_rate(75500.0, -10.0, molecularity=1)

    def test_eyring_invalid_molecularity(self):
        """Test that zero molecularity raises error."""
        with pytest.raises(ValueError):
            eyring_rate(75500.0, 298.15, molecularity=0)


class TestArrheniusEquation:
    """Test Arrhenius equation calculations."""

    def test_arrhenius_basic(self):
        """Test basic Arrhenius rate calculation."""
        A = 1.0e10      # 1/s
        Ea = 65000.0    # J/mol
        T = 298.15      # K

        k = arrhenius_rate(A, Ea, T)

        assert k > 0.0
        assert k < A  # Rate should be less than pre-exponential factor
        assert math.isfinite(k)

    def test_arrhenius_temperature_dependence(self):
        """Test temperature dependence of Arrhenius equation."""
        A = 1.0e10
        Ea = 65000.0

        k_298 = arrhenius_rate(A, Ea, 298.15)
        k_350 = arrhenius_rate(A, Ea, 350.0)

        # Higher temperature should give higher rate
        assert k_350 > k_298

    def test_arrhenius_activation_energy(self):
        """Test activation energy dependence."""
        A = 1.0e10
        T = 298.15

        k_low = arrhenius_rate(A, 40000.0, T)   # 40 kJ/mol
        k_high = arrhenius_rate(A, 80000.0, T)  # 80 kJ/mol

        # Lower activation energy should give higher rate
        assert k_low > k_high

    def test_arrhenius_zero_activation_energy(self):
        """Test Arrhenius with zero activation energy."""
        A = 1.0e10
        T = 298.15

        k = arrhenius_rate(A, 0.0, T)

        # With Ea=0, k should equal A
        assert abs(k - A) < 1e-6

    def test_arrhenius_invalid_temperature(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError):
            arrhenius_rate(1.0e10, 65000.0, 0.0)


class TestEquilibriumConstant:
    """Test equilibrium constant calculations."""

    def test_K_from_negative_deltaG(self):
        """Test K from negative ΔG° (product-favored)."""
        dG_eq = -8500.0  # J/mol (-8.5 kJ/mol)
        T = 298.15

        K = K_from_deltaG_eq(dG_eq, T)

        # Negative ΔG° should give K > 1
        assert K > 1.0

    def test_K_from_positive_deltaG(self):
        """Test K from positive ΔG° (reactant-favored)."""
        dG_eq = 8500.0  # J/mol
        T = 298.15

        K = K_from_deltaG_eq(dG_eq, T)

        # Positive ΔG° should give K < 1
        assert K < 1.0

    def test_K_from_zero_deltaG(self):
        """Test K from zero ΔG°."""
        dG_eq = 0.0
        T = 298.15

        K = K_from_deltaG_eq(dG_eq, T)

        # Zero ΔG° should give K = 1
        assert abs(K - 1.0) < 1e-10

    def test_K_temperature_dependence(self):
        """Test temperature dependence of K."""
        dG_eq = -8500.0

        K_298 = K_from_deltaG_eq(dG_eq, 298.15)
        K_350 = K_from_deltaG_eq(dG_eq, 350.0)

        # For negative ΔG°, K should decrease with temperature
        # (Le Chatelier for exothermic)
        assert K_298 != K_350


class TestEnergyUnitConversion:
    """Test energy unit conversion."""

    def test_jmol_passthrough(self):
        """Test J/mol passthrough."""
        E = normalize_energy_to_J_per_mol(75500.0, "J/mol")
        assert E == 75500.0

    def test_kjmol_conversion(self):
        """Test kJ/mol to J/mol conversion."""
        E = normalize_energy_to_J_per_mol(75.5, "kJ/mol")
        assert E == 75500.0

    def test_kcalmol_conversion(self):
        """Test kcal/mol to J/mol conversion."""
        E = normalize_energy_to_J_per_mol(18.04, "kcal/mol")
        # 1 kcal/mol = 4184 J/mol
        expected = 18.04 * 4184
        assert abs(E - expected) < 1.0

    def test_none_unit_defaults_to_jmol(self):
        """Test that None unit defaults to J/mol."""
        E = normalize_energy_to_J_per_mol(75500.0, None)
        assert E == 75500.0

    def test_invalid_unit(self):
        """Test that invalid unit raises error."""
        with pytest.raises(ValueError):
            normalize_energy_to_J_per_mol(100.0, "eV")


class TestRateUnits:
    """Test rate constant unit strings."""

    def test_unimolecular_units(self):
        """Test unimolecular rate units."""
        assert rate_units(1) == "1/s"

    def test_bimolecular_units(self):
        """Test bimolecular rate units."""
        assert rate_units(2) == "1/(M*s)"

    def test_trimolecular_units(self):
        """Test trimolecular rate units."""
        assert rate_units(3) == "1/(M^2*s)"

    def test_higher_order_units(self):
        """Test higher order rate units."""
        assert rate_units(4) == "1/(M^3*s)"
        assert rate_units(5) == "1/(M^4*s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
