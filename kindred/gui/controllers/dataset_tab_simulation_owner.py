"""Dataset-tab simulation request owner."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class DatasetTabSimulationOwner:
    """Own simulation behavior launched from committed dataset tabs."""

    def __init__(
        self,
        *,
        mechanism_getter: Callable[[], str],
        simulation_runner: Callable[[str], Dict[str, Any]],
    ) -> None:
        self._mechanism_getter = mechanism_getter
        self._simulation_runner = simulation_runner

    def handle_simulate_requested(self, *, dataset_name: str, panel: Any) -> None:
        if panel is None:
            logger.error("Panel not found for dataset %s", dataset_name)
            return

        mechanism_text = self._mechanism_getter()
        if not mechanism_text or not mechanism_text.strip():
            panel.set_status("Error: No mechanism defined")
            logger.error("No mechanism text available")
            return

        logger.info("Running simulation for dataset %s", dataset_name)
        panel.set_status("Running simulation...")
        try:
            result = self._simulation_runner(mechanism_text)
            if result and "t" in result and "species" in result:
                panel.plot_simulation_results(result["t"], result["species"])
                panel.set_status("Simulation complete")
                logger.info("Simulation complete for dataset %s", dataset_name)
            else:
                panel.set_status("Error: Invalid simulation result")
                logger.error("Simulation returned invalid result")
        except Exception as exc:
            panel.set_status(f"Error: {exc}")
            logger.error("Simulation failed for dataset %s: %s", dataset_name, exc)
