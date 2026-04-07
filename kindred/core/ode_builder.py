# kindred/core/ode_builder.py
"""
ODE system builder from Mechanism objects.

This module ties Mechanism data to kinetics and solver plumbing. It constructs
temperature-aware dy/dt = f(t, y, T=...) callables from mechanism stoichiometry
plus Arrhenius/Eyring models in kindred.core.kinetics. Equilibria stay reversible
and are evaluated as net forward minus reverse rates (no reaction duplication).
"""

from __future__ import annotations

import math
import logging
from typing import Callable, Dict, List, Optional, Tuple, Sequence

import numpy as np

from .constants import R, h, k_B
from .mechanism import Mechanism, Reaction, Equilibrium
from .rate_binding import RateBinding
from .kinetics import arrhenius_rate, eyring_rate, K_from_deltaG_eq
from .simulator.fast_eq import derive_equilibrium_rates

logger = logging.getLogger(__name__)

__all__ = ["RateBinding", "build_ode_rhs_from_mechanism"]


def build_ode_rhs_from_mechanism(
    mechanism: Mechanism
) -> Callable[..., np.ndarray]:
    """
    Build ODE right-hand side function from a Mechanism.

    Parameters
    ----------
    mechanism : Mechanism
        Mechanism with species, reactions, and equilibria

    Returns
    -------
    callable
        Function f(t, y, T=...) -> dy/dt where y is concentration vector
        in mechanism.species_names() order

    Notes
    -----
    - Rates come from Mechanism.rate bindings or Arrhenius/Eyring overrides in
      kinetics.py; T defaults to mechanism.metadata["temperature_K"] if omitted.
    - Reactions contribute: rate = k * ∏[reactants]^stoich
    - Equilibria remain a single reversible step with rate = v_fwd - v_rev
      (stoichiometry is not duplicated into two irreversible columns).
    - Stoichiometry determines species changes via matrix multiplication
    """
    species_names = mechanism.species_names()
    n_species = len(species_names)
    species_index = {name: idx for idx, name in enumerate(species_names)}
    from .mechanism_metadata import MechanismMetadataView

    mech_meta = MechanismMetadataView.from_metadata(getattr(mechanism, "metadata", {}) or {})
    default_temperature = float(mech_meta.temperature_K)
    default_standard_conc = float(mech_meta.standard_conc_M)

    def _evaluate_scalar(value: Optional[float | Callable[[], float]]) -> Optional[float]:
        if value is None:
            return None
        return float(value()) if callable(value) else float(value)

    def _eval_model(model_data: Optional[Dict[str, object]], T: float, molecularity: float) -> Optional[float]:
        if not model_data:
            return None
        model_type = model_data.get("type") or model_data.get("model")
        if model_type == "Arrhenius":
            A = model_data.get("A")
            Ea = model_data.get("Ea_J_per_mol") or model_data.get("Ea")
            if A is None or Ea is None:
                return None
            return arrhenius_rate(float(A), float(Ea), T)
        if model_type == "Eyring":
            dG = model_data.get("dG_act_J_per_mol")
            if dG is None:
                return None
            kappa = float(model_data.get("kappa", 1.0))
            std_c = float(model_data.get("standard_conc_M", default_standard_conc))
            return eyring_rate(float(dG), T, kappa=kappa, molecularity=int(round(molecularity)), standard_conc_M=std_c)
        return None

    def _compute_equilibrium_constant(meta: Dict[str, object], eq_obj: Equilibrium, T: float) -> Optional[float]:
        if meta.get("dG_eq_J_per_mol") is not None:
            return K_from_deltaG_eq(float(meta["dG_eq_J_per_mol"]), T)
        if meta.get("Keq_input") is not None:
            val = _evaluate_scalar(meta["Keq_input"])
            return float(val) if val is not None else None
        if eq_obj.Keq is not None:
            val = _evaluate_scalar(eq_obj.Keq)
            return float(val) if val is not None else None
        return None

    def _require_positive_finite_runtime_Keq(Keq_value: float) -> float:
        Keq_float = float(Keq_value)
        if not (Keq_float > 0.0) or not math.isfinite(Keq_float):
            raise ValueError("Equilibrium Keq must be positive and finite for runtime anchoring")
        return Keq_float

    class EquilibriumRateEvaluator:
        """
        Stateful evaluator for equilibrium forward/reverse rates.

        Replaces the previous nested `eq_rate_pair` closure to avoid fragile late-binding and
        to keep the equilibrium-rate policy testable and type-checkable.
        """

        def __init__(
            self,
            *,
            eq_obj: Equilibrium,
            meta: Dict[str, object],
            forward_model: Optional[Dict[str, object]],
            reverse_model: Optional[Dict[str, object]],
            explicit_rates_meta: Sequence[float],
            user_kf: bool,
            user_kr: bool,
            fast_flag: bool,
            fwd_mol: float,
            rev_mol: float,
            kf_binding: Optional[RateBinding],
            kr_binding: Optional[RateBinding],
            Keq_binding: Optional[RateBinding],
            kf_const: Optional[float],
            kr_const: Optional[float],
            Keq_const: Optional[float],
        ) -> None:
            self._eq = eq_obj
            self._meta = meta
            self._forward_model = forward_model
            self._reverse_model = reverse_model
            self._explicit_rates_meta = tuple(float(v) for v in explicit_rates_meta)
            self._user_kf = bool(user_kf)
            self._user_kr = bool(user_kr)
            self._fast_flag = bool(fast_flag)
            self._fwd_mol = float(fwd_mol)
            self._rev_mol = float(rev_mol)
            self._kf_binding = kf_binding
            self._kr_binding = kr_binding
            self._Keq_binding = Keq_binding
            self._kf_const = kf_const
            self._kr_const = kr_const
            self._Keq_const = Keq_const

        def __call__(self, T: float) -> Tuple[float, float]:
            Keq_T = _compute_equilibrium_constant(self._meta, self._eq, T)
            kf_val = _eval_model(self._forward_model, T, self._fwd_mol)
            if kf_val is None and self._kf_binding is not None:
                kf_val = float(self._kf_binding())
            elif kf_val is None and self._user_kf:
                kf_val = self._kf_const
            if (
                kf_val is None
                and self._kf_const is not None
                and (not self._user_kf)
                and self._forward_model is None
            ):
                kf_val = self._kf_const

            kr_val = _eval_model(self._reverse_model, T, self._rev_mol)
            if kr_val is None and self._kr_binding is not None:
                kr_val = float(self._kr_binding())
            elif kr_val is None and self._user_kr:
                kr_val = self._kr_const
            if (
                kr_val is None
                and self._kr_const is not None
                and (not self._user_kr)
                and self._reverse_model is None
            ):
                kr_val = self._kr_const

            explicit: List[float] = list(self._explicit_rates_meta)
            if kf_val is not None:
                explicit.append(float(kf_val))
            if kr_val is not None:
                explicit.append(float(kr_val))

            thermo_Keq = Keq_T if Keq_T is not None else self._Keq_const
            if thermo_Keq is None and self._Keq_binding is not None:
                thermo_Keq = float(self._Keq_binding())

            if self._fast_flag and (kf_val is None or kr_val is None) and (
                self._meta.get("dG_eq_J_per_mol") is not None or thermo_Keq is not None
            ):
                fe = derive_equilibrium_rates(
                    Keq=thermo_Keq,
                    dG_eq_J_per_mol=self._meta.get("dG_eq_J_per_mol"),
                    T=T,
                    explicit_rates=explicit or None,
                )
                if kf_val is None:
                    kf_val = fe.kf
                if kr_val is None:
                    kr_val = fe.kr
                if Keq_T is None:
                    Keq_T = fe.Keq

            if kr_val is None and kf_val is not None and Keq_T is not None and not self._user_kr:
                Keq_T = _require_positive_finite_runtime_Keq(Keq_T)
                kr_val = float(kf_val) / float(Keq_T)
            if kf_val is None and kr_val is not None and Keq_T is not None and not self._user_kf:
                Keq_T = _require_positive_finite_runtime_Keq(Keq_T)
                kf_val = float(kr_val) * float(Keq_T)

            if kf_val is None and kr_val is None and Keq_T is None:
                raise ValueError(
                    "Equilibrium missing usable kinetic and thermodynamic data "
                    "(need Keq/dG_eq or rates)"
                )
            if kf_val is None:
                if self._kf_const is not None:
                    kf_val = self._kf_const
                elif Keq_T is not None and kr_val is not None:
                    Keq_T = _require_positive_finite_runtime_Keq(Keq_T)
                    kf_val = float(kr_val) * float(Keq_T)
                elif Keq_T is not None and kr_val is None and not self._fast_flag:
                    # Preserve the long-standing programmatic Keq-only contract for
                    # non-fast equilibria by using the same deterministic anchor
                    # as the sparse Jacobian path.
                    Keq_T = _require_positive_finite_runtime_Keq(Keq_T)
                    kf_val = 1.0
                else:
                    raise ValueError("Equilibrium missing usable kf and thermodynamic data to derive it")
            if kr_val is None:
                if self._kr_const is not None:
                    kr_val = self._kr_const
                elif Keq_T is not None:
                    Keq_T = _require_positive_finite_runtime_Keq(Keq_T)
                    kr_val = float(kf_val) / float(Keq_T)
                else:
                    raise ValueError("Equilibrium missing usable kr and thermodynamic data to derive it")
            return float(kf_val), float(kr_val)

        def forward_rate(self, T: float) -> float:
            return self(T)[0]

        def reverse_rate(self, T: float) -> float:
            return self(T)[1]

    # Build list of all steps (irreversible reactions + equilibria)
    steps: List[Tuple[str, Reaction | Equilibrium]] = []
    for rxn in mechanism.reactions:
        steps.append(("reaction", rxn))
    for eq in mechanism.equilibria:
        steps.append(("equilibrium", eq))

    n_steps = len(steps)
    S = np.zeros((n_species, n_steps))
    exp_forward = np.zeros((n_steps, n_species), dtype=float)
    exp_reverse = np.zeros((n_steps, n_species), dtype=float)
    k_forward_base = np.zeros(n_steps, dtype=float)
    forward_dynamic: List[Tuple[int, Callable[[float], float]]] = []
    arrhenius_idx: List[int] = []
    arrhenius_A: List[float] = []
    arrhenius_Ea: List[float] = []
    eyring_idx: List[int] = []
    eyring_dG: List[float] = []
    eyring_scale: List[float] = []
    equilibrium_evaluators: List[Tuple[int, EquilibriumRateEvaluator]] = []

    for i_step, (step_type, step_obj) in enumerate(steps):
        if step_type == "reaction":
            rxn = step_obj  # type: ignore[assignment]
            vec = rxn.stoich_vector(species_names)
            S[:, i_step] = vec

            overrides = getattr(rxn, "overrides", {}) or {}
            model = overrides.get("model")
            rate_binding = rxn.rate if isinstance(rxn.rate, RateBinding) else None

            if rate_binding is not None:

                def rate_func(T: float, *, _binding=rate_binding) -> float:
                    return float(_binding())

                forward_dynamic.append((i_step, rate_func))
            elif model == "Arrhenius" and overrides.get("A") is not None and (
                overrides.get("Ea_J_per_mol") is not None or overrides.get("Ea") is not None
            ):
                A = float(overrides["A"])
                Ea = float(overrides.get("Ea_J_per_mol") or overrides.get("Ea"))

                arrhenius_idx.append(i_step)
                arrhenius_A.append(A)
                arrhenius_Ea.append(Ea)

            elif model == "Eyring" and overrides.get("dG_act_J_per_mol") is not None:
                dG_act = float(overrides["dG_act_J_per_mol"])
                kappa = float(overrides.get("kappa", 1.0))
                std_c = float(overrides.get("standard_conc_M", default_standard_conc))
                molecularity = rxn.order

                if std_c <= 0.0 or not np.isfinite(std_c):
                    raise ValueError("standard_conc_M must be positive and finite")
                if kappa <= 0.0 or not np.isfinite(kappa):
                    raise ValueError("kappa must be positive and finite")
                if int(molecularity) < 1:
                    raise ValueError("molecularity must be >= 1")

                n_mol = int(molecularity)
                scale = kappa / (float(std_c) ** (n_mol - 1) if n_mol >= 2 else 1.0)
                eyring_idx.append(i_step)
                eyring_dG.append(dG_act)
                eyring_scale.append(float(scale))

            else:
                # IMPORTANT: if rxn.rate is a mutable binding/callable (e.g. RateBinding),
                # preserve the callable so updates are reflected without rebuilding the RHS.
                rate_obj = rxn.rate
                if callable(rate_obj):
                    def rate_func(T: float, *, _rate=rate_obj) -> float:
                        v = _rate()
                        if v is None:
                            raise ValueError("Reaction missing rate constant")
                        return float(v)
                    forward_dynamic.append((i_step, rate_func))
                else:
                    k_const = _evaluate_scalar(rate_obj)
                    if k_const is None:
                        raise ValueError("Reaction missing rate constant")

                    k_forward_base[i_step] = float(k_const)

            for sp_name, stoich_coef in rxn.stoich.items():
                if stoich_coef < 0:  # Reactant
                    idx = species_index[sp_name]
                    exp_forward[i_step, idx] = abs(stoich_coef)

        elif step_type == "equilibrium":
            eq = step_obj  # type: ignore[assignment]
            fwd_vec = np.array(eq.forward_vector(species_names))
            back_vec = np.array(eq.back_vector(species_names))
            S[:, i_step] = back_vec - fwd_vec

            meta: Dict[str, object] = getattr(eq, "metadata", {}) or {}
            from .mechanism_metadata import EquilibriumMetadataView

            eq_meta = EquilibriumMetadataView.from_metadata(meta, default_fast=bool(eq.fast))
            forward_model = eq_meta.forward_model
            reverse_model = meta.get("reverse_model") if isinstance(meta.get("reverse_model"), dict) else None
            explicit_rates_meta = list(eq_meta.explicit_rates)
            user_kf = bool(eq_meta.user_provided_kf)
            user_kr = bool(eq_meta.user_provided_kr)
            fast_flag = bool(eq_meta.fast_equilibrium)
            fwd_mol = float(sum(eq.stoich_forward.values()))
            rev_mol = float(sum(eq.stoich_back.values()))

            kf_binding = eq.kf if isinstance(eq.kf, RateBinding) else None
            kr_binding = eq.kr if isinstance(eq.kr, RateBinding) else None
            Keq_binding = eq.Keq if isinstance(eq.Keq, RateBinding) else None

            # IMPORTANT: if we are in prepared/bound mode, eq.kf/eq.kr/eq.Keq can be
            # RateBinding objects that must be queried dynamically. Do not capture
            # their values at RHS-build time.
            kf_const = _evaluate_scalar(eq.kf) if kf_binding is None else None
            kr_const = _evaluate_scalar(eq.kr) if kr_binding is None else None
            Keq_const = _evaluate_scalar(eq.Keq) if Keq_binding is None else None

            evaluator = EquilibriumRateEvaluator(
                eq_obj=eq,
                meta=meta,
                forward_model=forward_model,
                reverse_model=reverse_model,
                explicit_rates_meta=explicit_rates_meta,
                user_kf=user_kf,
                user_kr=user_kr,
                fast_flag=fast_flag,
                fwd_mol=fwd_mol,
                rev_mol=rev_mol,
                kf_binding=kf_binding,
                kr_binding=kr_binding,
                Keq_binding=Keq_binding,
                kf_const=kf_const,
                kr_const=kr_const,
                Keq_const=Keq_const,
            )
            equilibrium_evaluators.append((i_step, evaluator))

            for sp_name, stoich_coef in eq.stoich_forward.items():
                exp_forward[i_step, species_index[sp_name]] = stoich_coef
            for sp_name, stoich_coef in eq.stoich_back.items():
                exp_reverse[i_step, species_index[sp_name]] = stoich_coef
        else:
            raise ValueError(f"Unknown step type: {step_type}")

    forward_mask = exp_forward != 0.0
    reverse_mask = exp_reverse != 0.0
    forward_buffer = np.ones_like(exp_forward)
    reverse_buffer = np.ones_like(exp_reverse)

    # Keep dynamic arrays stable across RHS calls; update only the entries that vary.
    k_forward_dyn = k_forward_base.copy()
    k_reverse_dyn = np.zeros(n_steps, dtype=float)
    has_reverse_terms = bool(equilibrium_evaluators)

    if forward_dynamic:
        forward_dynamic_idx_arr = np.fromiter(
            (idx for idx, _fn in forward_dynamic),
            dtype=int,
            count=len(forward_dynamic),
        )
        forward_dynamic_fns = [fn for _idx, fn in forward_dynamic]
    else:
        forward_dynamic_idx_arr = np.zeros(0, dtype=int)
        forward_dynamic_fns: List[Callable[[float], float]] = []

    if equilibrium_evaluators:
        equilibrium_idx_arr = np.fromiter(
            (idx for idx, _ev in equilibrium_evaluators),
            dtype=int,
            count=len(equilibrium_evaluators),
        )
        equilibrium_fns = [ev for _idx, ev in equilibrium_evaluators]
    else:
        equilibrium_idx_arr = np.zeros(0, dtype=int)
        equilibrium_fns: List[EquilibriumRateEvaluator] = []

    arrhenius_idx_arr = np.asarray(arrhenius_idx, dtype=int)
    arrhenius_A_arr = np.asarray(arrhenius_A, dtype=float)
    arrhenius_Ea_arr = np.asarray(arrhenius_Ea, dtype=float)
    eyring_idx_arr = np.asarray(eyring_idx, dtype=int)
    eyring_dG_arr = np.asarray(eyring_dG, dtype=float)
    eyring_scale_arr = np.asarray(eyring_scale, dtype=float)

    # Buffers for stable reversible net rate evaluation (avoids overflow and inf-inf cancellation).
    log_y = np.empty(n_species, dtype=float)
    log_k_forward_dyn = np.empty(n_steps, dtype=float)
    log_k_reverse_dyn = np.empty(n_steps, dtype=float)
    log_prod_forward = np.empty(n_steps, dtype=float)
    log_prod_reverse = np.empty(n_steps, dtype=float)
    log_rate_forward = np.empty(n_steps, dtype=float)
    log_rate_reverse = np.empty(n_steps, dtype=float)
    net_rates = np.empty(n_steps, dtype=float)

    def _stable_diff_exp_logs(log_f: np.ndarray, log_r: np.ndarray, out: np.ndarray) -> None:
        """
        Compute exp(log_f) - exp(log_r) stably from logs.

        This avoids `inf - inf -> nan` when both sides overflow, and reduces
        catastrophic cancellation when rates are nearly equal.
        """
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            out.fill(0.0)
            max_lr = np.maximum(log_f, log_r)
            min_lr = np.minimum(log_f, log_r)
            mask = np.isfinite(max_lr)
            if not np.any(mask):
                return
            d = min_lr[mask] - max_lr[mask]  # <= 0
            small = -np.expm1(d)  # 1 - exp(d), in [0, 1]
            log_mag = max_lr[mask] + np.log(small)  # log(0) -> -inf => mag 0
            mag = np.exp(log_mag)
            sign = np.where(log_f[mask] >= log_r[mask], 1.0, -1.0)
            out[mask] = sign * mag

    def ode_rhs(t: float, y: np.ndarray, *, T: Optional[float] = None) -> np.ndarray:
        """ODE right-hand side: dy/dt = S @ r(y, T)."""
        y_arr = y
        T_eval = default_temperature if T is None else float(T)
        k_forward = k_forward_dyn

        if arrhenius_idx_arr.size:
            if T_eval <= 0.0 or not np.isfinite(T_eval):
                raise ValueError("T must be positive and finite")
            k_forward[arrhenius_idx_arr] = arrhenius_A_arr * np.exp(-arrhenius_Ea_arr / (R * T_eval))

        if eyring_idx_arr.size:
            if T_eval <= 0.0 or not np.isfinite(T_eval):
                raise ValueError("T must be positive and finite")
            prefactor = (k_B * T_eval) / h
            k_forward[eyring_idx_arr] = eyring_scale_arr * prefactor * np.exp(-eyring_dG_arr / (R * T_eval))

        fwd_idx = forward_dynamic_idx_arr
        fwd_fns = forward_dynamic_fns
        for i in range(fwd_idx.size):
            k_forward[fwd_idx[i]] = fwd_fns[i](T_eval)

        if has_reverse_terms:
            eq_idx = equilibrium_idx_arr
            eq_fns = equilibrium_fns
            for i in range(eq_idx.size):
                idx = eq_idx[i]
                kf_val, kr_val = eq_fns[i](T_eval)
                k_forward[idx] = kf_val
                k_reverse_dyn[idx] = kr_val

        if not has_reverse_terms:
            forward_buffer.fill(1.0)
            np.power(y_arr, exp_forward, out=forward_buffer, where=forward_mask)
            rates_forward = k_forward * forward_buffer.prod(axis=1)
            return S @ rates_forward

        # Reversible/equilibrium path: work in log-space and combine stably.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_y.fill(-np.inf)
            pos = y_arr > 0.0
            log_y[pos] = np.log(y_arr[pos])

            log_k_forward_dyn.fill(-np.inf)
            np.log(k_forward_dyn, out=log_k_forward_dyn, where=(k_forward_dyn > 0.0))
            log_k_reverse_dyn.fill(-np.inf)
            np.log(k_reverse_dyn, out=log_k_reverse_dyn, where=(k_reverse_dyn > 0.0))

            forward_buffer.fill(0.0)
            np.multiply(exp_forward, log_y, out=forward_buffer, where=forward_mask)
            np.sum(forward_buffer, axis=1, out=log_prod_forward)
            np.add(log_k_forward_dyn, log_prod_forward, out=log_rate_forward)

            reverse_buffer.fill(0.0)
            np.multiply(exp_reverse, log_y, out=reverse_buffer, where=reverse_mask)
            np.sum(reverse_buffer, axis=1, out=log_prod_reverse)
            np.add(log_k_reverse_dyn, log_prod_reverse, out=log_rate_reverse)

        _stable_diff_exp_logs(log_rate_forward, log_rate_reverse, net_rates)
        return S @ net_rates

    return ode_rhs
