"""
State network to Mechanism converter.

This module converts validated StateNetwork graphs into Mechanism objects
by deriving rate constants from transition state theory.

Conversion contract
-------------------
- For each edge A-TS-B:
  - Forward barrier: ΔG‡_f = E(TS) - E(A)
  - Reverse barrier: ΔG‡_r = E(TS) - E(B)
  - Forward rate: k_f = (κ·k_B·T/h) · (q_TS/q_A) · exp(-ΔG‡_f/RT)
  - Reverse rate: k_r = (κ·k_B·T/h) · (q_TS/q_B) · exp(-ΔG‡_r/RT)

- Partition function ratio approximated by degeneracy ratio:
  - q_TS/q_A ≈ σ_TS/σ_A

- Standard state corrections applied for bimolecular steps

This module is deterministic and performs no I/O.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Set

from .state_model import StateNetwork, State, StateType
from ..equilibrium_rate_authority import EquilibriumRateInputContext
from ..mechanism import Mechanism
from ..mechanism_metadata import EquilibriumMetadataKeys
from ..constants import R, k_B, h

# Aliases for clarity
R_J_per_mol_K = R
h_planck = h

logger = logging.getLogger(__name__)

__all__ = [
    "StateNetworkConverter",
    "convert_state_network_to_mechanism",
]


# ------------------------------ converter class ------------------------------

class StateNetworkConverter:
    """
    Converts validated StateNetwork to Mechanism via transition state theory.

    Usage:
        converter = StateNetworkConverter(temperature_K=298.15, kappa=1.0, C0_M=1.0)
        mechanism = converter.convert(state_network, initials)
    """

    def __init__(
        self,
        temperature_K: float = 298.15,
        kappa: float = 1.0,
        C0_M: float = 1.0
    ):
        """
        Initialize converter.

        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin for rate constant calculations
        kappa : float
            Transmission coefficient (0 < κ ≤ 1)
        C0_M : float
            Standard concentration in M (typically 1.0)
        """
        if not (temperature_K > 0 and math.isfinite(temperature_K)):
            raise ValueError("temperature_K must be positive and finite")
        if not (0 < kappa <= 1):
            raise ValueError("kappa must be in (0, 1]")
        if not (C0_M > 0 and math.isfinite(C0_M)):
            raise ValueError("C0_M must be positive and finite")

        self.T = temperature_K
        self.kappa = kappa
        self.C0 = C0_M

        # Eyring prefactor: κ·k_B·T/h
        self.eyring_prefactor = kappa * k_B * temperature_K / h_planck
        # Track processed transition states to avoid duplicate reactions
        self._processed_ts: Set[str] = set()

    def convert(
        self,
        network: StateNetwork,
        initials: Optional[Dict[str, float]] = None
    ) -> Mechanism:
        """
        Convert StateNetwork to Mechanism.

        Parameters
        ----------
        network : StateNetwork
            Validated state network
        initials : dict, optional
            Initial concentrations {species: value}

        Returns
        -------
        Mechanism
            Mechanism with reactions derived from state network

        Raises
        ------
        ValueError
            If network is invalid or contains disconnected TS nodes
        """
        # Validate network first
        try:
            network.validate()
        except Exception as e:
            raise ValueError(f"State network validation failed: {e}") from e

        mechanism = Mechanism()
        initials = initials or {}

        # Get all states and edges
        states = {st.name: st for st in network.states()}
        edges = network.edges()

        if not states:
            raise ValueError("State network is empty")

        # Extract all ground states (potential reactant/product "sides")
        ground_states = {name: st for name, st in states.items() if st.kind == StateType.GS}

        # Add species implied by GS state membership.
        implied_species: set[str] = set()
        for gs in ground_states.values():
            for sp_name in self._state_stoich(gs).keys():
                implied_species.add(str(sp_name))
        for sp_name in sorted(implied_species):
            init_conc = float(initials.get(sp_name, 0.0))
            if sp_name not in mechanism.species:
                mechanism.add_species(sp_name, init_conc)

        # Process edges: find reaction pathways
        # Each TS must connect exactly 2 ground states (validated by network)
        self._processed_ts.clear()
        for edge in edges:
            # Determine which endpoint is TS
            a_state = states[edge.a]
            b_state = states[edge.b]

            if a_state.kind == StateType.TS and b_state.kind == StateType.TS:
                raise ValueError(f"Edge {edge.a}-{edge.b} connects two TS nodes (invalid)")

            if a_state.kind == StateType.TS:
                ts_name = edge.a
                other_name = edge.b
            elif b_state.kind == StateType.TS:
                ts_name = edge.b
                other_name = edge.a
            else:
                # Both ground states - direct connection (no TS)
                # Treat as equilibrium with barriers derived from energy difference
                logger.warning(f"Edge {edge.a}-{edge.b} connects two GS without TS (using simple equilibrium)")
                self._add_direct_equilibrium(mechanism, a_state, b_state)
                continue

            # Now we have ts_name (TS) and other_name (GS)
            # Find the other GS connected to this TS
            ts_neighbors = self._get_neighbors(network, ts_name)

            if len(ts_neighbors) != 2:
                raise ValueError(f"TS {ts_name} has degree {len(ts_neighbors)}, expected 2")

            # The other neighbor is the second GS
            gs_names = [n for n in ts_neighbors if n != other_name]
            if len(gs_names) != 1:
                raise ValueError(f"TS {ts_name} connectivity error")

            other_gs_name = gs_names[0]

            # Check if we've already processed this reaction pathway
            # (each TS is processed once, giving one reversible reaction)
            if self._is_processed(mechanism, other_name, other_gs_name, ts_name):
                continue

            # Add reaction: other_name <-> other_gs_name via ts_name
            ts_state = states[ts_name]
            gs1_state = states[other_name]
            gs2_state = states[other_gs_name]

            self._add_ts_reaction(mechanism, gs1_state, gs2_state, ts_state)
            self._mark_processed(ts_name, gs1_state.name, gs2_state.name)

        logger.info(
            f"Converted state network: {len(mechanism.species)} species, "
            f"{len(mechanism.reactions)} reactions, {len(mechanism.equilibria)} equilibria"
        )

        return mechanism

    def _get_neighbors(self, network: StateNetwork, name: str) -> List[str]:
        """Get neighbor state names."""
        neighbors = []
        for edge in network.edges():
            if edge.a == name:
                neighbors.append(edge.b)
            elif edge.b == name:
                neighbors.append(edge.a)
        return neighbors

    def _is_processed(self, mechanism: Mechanism, gs1: str, gs2: str, ts: str) -> bool:
        """Check if this reaction pathway has already been added."""
        return ts in self._processed_ts

    def _mark_processed(self, ts: str, gs1: str, gs2: str) -> None:
        """Mark a TS as processed to prevent duplicate equilibria."""
        self._processed_ts.add(ts)

    def _add_ts_reaction(
        self,
        mechanism: Mechanism,
        gs1: State,
        gs2: State,
        ts: State
    ):
        """
        Add reversible reaction gs1 <-> gs2 via transition state ts.

        Rate constants calculated via Eyring equation:
        k = (κ·k_B·T/h) · (q_TS/q_GS) · exp(-ΔG‡/RT)

        Where:
        - ΔG‡ = E(TS) - E(GS)
        - q_TS/q_GS ≈ σ_TS/σ_GS (degeneracy ratio)
        """
        # Canonicalize orientation for deterministic metadata/GUI mapping.
        if gs1.name > gs2.name:
            gs1, gs2 = gs2, gs1

        stoich_forward = self._state_stoich(gs1)
        stoich_back = self._state_stoich(gs2)
        m_fwd = self._molecularity(stoich_forward)
        m_rev = self._molecularity(stoich_back)

        std_ts = self._std_conc_product(ts, molecularity=1)
        std_react = self._std_conc_product(gs1, molecularity=m_fwd)
        std_prod = self._std_conc_product(gs2, molecularity=m_rev)
        std_ratio_fwd = std_ts / std_react
        std_ratio_rev = std_ts / std_prod

        # Forward: gs1 -> gs2
        dG_forward = ts.energy_jmol - gs1.energy_jmol  # Barrier
        deg_ratio_forward = ts.degeneracy / gs1.degeneracy
        k_forward = self._eyring_rate(dG_forward, deg_ratio_forward, std_ratio=std_ratio_fwd)

        # Reverse: gs2 -> gs1
        dG_reverse = ts.energy_jmol - gs2.energy_jmol  # Barrier
        deg_ratio_reverse = ts.degeneracy / gs2.degeneracy
        k_reverse = self._eyring_rate(dG_reverse, deg_ratio_reverse, std_ratio=std_ratio_rev)

        # Calculate equilibrium constant
        # K = k_forward / k_reverse = exp(-(E2 - E1)/RT)
        dG_eq = gs2.energy_jmol - gs1.energy_jmol
        K = math.exp(-(dG_eq) / (R_J_per_mol_K * self.T))

        logger.info(
            f"Adding TS reaction: {gs1.name} <-> {gs2.name} via {ts.name}\n"
            f"  Forward barrier: {dG_forward/1000:.2f} kJ/mol, k_f = {k_forward:.3e} s⁻¹\n"
            f"  Reverse barrier: {dG_reverse/1000:.2f} kJ/mol, k_r = {k_reverse:.3e} s⁻¹\n"
            f"  Equilibrium constant K = {K:.3e}"
        )

        meta = {
            "source": "state_network",
            "reactant": gs1.name,
            "product": gs2.name,
            "ts": ts.name,
            "temperature_K": float(self.T),
            "kappa": float(self.kappa),
            "molecularity_fwd": int(m_fwd),
            "molecularity_rev": int(m_rev),
            "std_conc_product_ts": float(std_ts),
            "std_conc_product_reactant": float(std_react),
            "std_conc_product_product": float(std_prod),
            "state_energy_reactant_J_per_mol": float(gs1.energy_jmol),
            "state_energy_product_J_per_mol": float(gs2.energy_jmol),
            "state_energy_ts_J_per_mol": float(ts.energy_jmol),
            "dG_act_fwd_J_per_mol": float(dG_forward),
            "dG_act_rev_J_per_mol": float(dG_reverse),
            "dG_eq_J_per_mol": float(dG_eq),
            "degeneracy_ratio_fwd": float(deg_ratio_forward),
            "degeneracy_ratio_rev": float(deg_ratio_reverse),
            "kf": float(k_forward),
            "kr": float(k_reverse),
            "Keq": float(K),
            EquilibriumMetadataKeys.USER_PROVIDED_KF: True,
            EquilibriumMetadataKeys.USER_PROVIDED_KR: True,
        }

        # Add as equilibrium to mechanism
        mechanism._add_equilibrium_with_authority_context(
            stoich_forward=stoich_forward,
            stoich_back=stoich_back,
            Keq=None,
            kf=k_forward,
            kr=k_reverse,
            fast=False,  # Not instantaneous equilibrium
            metadata=meta,
            authority_context=EquilibriumRateInputContext.GENERATED_STATE_NETWORK,
        )

    def _add_direct_equilibrium(
        self,
        mechanism: Mechanism,
        gs1: State,
        gs2: State
    ):
        """
        Add direct equilibrium between two ground states (no TS).

        This is a fallback for GS-GS edges. We use energy difference
        to calculate equilibrium constant and assign reasonable rates.
        """
        # Canonicalize orientation for deterministic metadata/GUI mapping.
        if gs1.name > gs2.name:
            gs1, gs2 = gs2, gs1

        stoich_forward = self._state_stoich(gs1)
        stoich_back = self._state_stoich(gs2)
        m_fwd = self._molecularity(stoich_forward)
        m_rev = self._molecularity(stoich_back)
        std_react = self._std_conc_product(gs1, molecularity=m_fwd)
        std_prod = self._std_conc_product(gs2, molecularity=m_rev)

        # Equilibrium constant from energy difference
        dG = gs2.energy_jmol - gs1.energy_jmol
        K = math.exp(-dG / (R_J_per_mol_K * self.T))

        # Use fast equilibrium defaults
        k_forward = 1e6  # Fast
        Kc = float(K * (std_prod / std_react))
        k_reverse = k_forward / Kc

        logger.info(
            f"Adding direct equilibrium: {gs1.name} <-> {gs2.name}\n"
            f"  ΔG = {dG/1000:.2f} kJ/mol, K = {K:.3e}"
        )

        meta = {
            "source": "state_network_direct",
            "reactant": gs1.name,
            "product": gs2.name,
            "temperature_K": float(self.T),
            "kappa": float(self.kappa),
            "molecularity_fwd": int(m_fwd),
            "molecularity_rev": int(m_rev),
            "std_conc_product_reactant": float(std_react),
            "std_conc_product_product": float(std_prod),
            "state_energy_reactant_J_per_mol": float(gs1.energy_jmol),
            "state_energy_product_J_per_mol": float(gs2.energy_jmol),
            "dG_eq_J_per_mol": float(dG),
            "kf": float(k_forward),
            "kr": float(k_reverse),
            "Keq": float(K),
            "std_ratio": float(std_prod / std_react),
            EquilibriumMetadataKeys.USER_PROVIDED_KF: True,
            EquilibriumMetadataKeys.USER_PROVIDED_KR: False,
        }

        mechanism._add_equilibrium_with_authority_context(
            stoich_forward=stoich_forward,
            stoich_back=stoich_back,
            Keq=K,
            kf=k_forward,
            kr=k_reverse,
            fast=True,
            metadata=meta,
            authority_context=EquilibriumRateInputContext.GENERATED_STATE_NETWORK,
        )

    def _eyring_rate(self, dG_barrier_J_per_mol: float, degeneracy_ratio: float, *, std_ratio: float = 1.0) -> float:
        """
        Calculate rate constant via Eyring equation.

        k = (κ·k_B·T/h) · (σ_TS/σ_GS) · exp(-ΔG‡/RT)

        For unimolecular reactions (GS -> GS), units are s⁻¹.
        """
        exponential_term = math.exp(-dG_barrier_J_per_mol / (R_J_per_mol_K * self.T))
        k = self.eyring_prefactor * degeneracy_ratio * exponential_term
        std_ratio = float(std_ratio)
        if not (math.isfinite(std_ratio) and std_ratio > 0.0):
            raise ValueError("std_ratio must be positive and finite")
        return float(k * std_ratio)

    @staticmethod
    def _molecularity(stoich: Dict[str, float]) -> int:
        tot = 0.0
        for v in stoich.values():
            tot += float(v)
        n = int(round(tot))
        if n < 1 or abs(tot - n) > 1e-9:
            raise ValueError("invalid molecularity (expected positive integer sum of stoichiometric coefficients)")
        return int(n)

    @staticmethod
    def _state_stoich(state: State) -> Dict[str, float]:
        if state.members:
            out: Dict[str, float] = {}
            for name in state.members:
                out[str(name)] = float(out.get(str(name), 0.0) + 1.0)
            return out
        return {str(state.name): 1.0}

    def _std_conc_product(self, state: State, *, molecularity: int) -> float:
        if state.std_conc_product_M is not None:
            v = float(state.std_conc_product_M)
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError("std_conc_product_M must be positive and finite")
            return float(v)
        if state.kind == StateType.TS:
            return float(self.C0)
        return float(self.C0) ** int(molecularity)


# ------------------------------ convenience function -------------------------

def convert_state_network_to_mechanism(
    network: StateNetwork,
    initials: Optional[Dict[str, float]] = None,
    temperature_K: float = 298.15,
    kappa: float = 1.0,
    C0_M: float = 1.0
) -> Mechanism:
    """
    Convert state network to mechanism (convenience function).

    Parameters
    ----------
    network : StateNetwork
        Validated state network
    initials : dict, optional
        Initial concentrations
    temperature_K : float
        Temperature in K
    kappa : float
        Transmission coefficient
    C0_M : float
        Standard concentration in M

    Returns
    -------
    Mechanism
        Mechanism with reactions derived from TS theory
    """
    converter = StateNetworkConverter(
        temperature_K=temperature_K,
        kappa=kappa,
        C0_M=C0_M
    )
    return converter.convert(network, initials)
