from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np

from kindred.core.exceptions import SimulationBuilderContractError

if TYPE_CHECKING:
    from kindred.core.simulation_preparation import BoundMechanism

__all__ = [
    "SimulationBuilder",
    "SimulationBuilderContractError",
    "coerce_simulation_builder",
    "prepare_bound_mechanism",
]

_REQUIRED_BUILDER_KWARGS = ("solver", "rtol", "atol")


class SimulationBuilder(Protocol):
    def __call__(
        self,
        mechanism_text: str,
        param_names: list[str],
        *,
        solver: str,
        rtol: float,
        atol: float,
    ) -> Callable[[dict[str, float]], dict[str, np.ndarray]]: ...


def coerce_simulation_builder(builder: Callable[..., Any]) -> SimulationBuilder:
    def _is_required_kwarg_contract_error(exc: TypeError) -> bool:
        msg = str(exc)
        if "unexpected keyword argument" not in msg:
            return False
        if not any(f"'{name}'" in msg for name in _REQUIRED_BUILDER_KWARGS):
            return False
        tb = exc.__traceback__
        return tb is not None and tb.tb_next is None

    def _wrapped(
        mechanism_text: str,
        param_names: list[str],
        *,
        solver: str,
        rtol: float,
        atol: float,
    ) -> Callable[[dict[str, float]], dict[str, np.ndarray]]:
        try:
            return builder(
                mechanism_text,
                list(param_names),
                solver=str(solver),
                rtol=float(rtol),
                atol=float(atol),
            )
        except TypeError as exc:
            if _is_required_kwarg_contract_error(exc):
                raise SimulationBuilderContractError() from exc
            raise

    return _wrapped


def prepare_bound_mechanism(*args, **kwargs) -> "BoundMechanism":
    from kindred.core.simulation_preparation import prepare_bound_mechanism as _impl

    return _impl(*args, **kwargs)
