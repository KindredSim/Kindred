from __future__ import annotations

from typing import Callable

from kindred.core.batch_initial_conditions import (
    BatchInitialConditionsStore,
    batch_initial_conditions_store_is_true_placeholder,
)
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.gui.initial_conditions_import_owner import (
    InitialConditionsImportOwner,
    InitialConditionsImportStatus,
    InitialConditionsReconciliationPlan,
    InitialConditionsReconciliationResult,
)


class InitialConditionsSourceAcceptanceOwner:
    """Owns GUI source-acceptance policy around Initial Conditions import."""

    def __init__(
        self,
        *,
        import_owner: InitialConditionsImportOwner,
        show_reconciliation_error: Callable[[str], None],
    ) -> None:
        self._import_owner = import_owner
        self._show_reconciliation_error = show_reconciliation_error

    def prepare_for_authoritative_lock(
        self,
        source: MechanismAuthoringSource,
        *,
        prompt_overwrite: bool,
    ) -> InitialConditionsReconciliationPlan | None:
        return self._prepare_or_report(
            source,
            prompt_overwrite=bool(prompt_overwrite),
        )

    def apply_prepared_plan(
        self,
        plan: InitialConditionsReconciliationPlan,
    ) -> InitialConditionsReconciliationResult:
        return self._import_owner.apply_reconciliation_plan(plan)

    def accept_importable_source(
        self,
        source: MechanismAuthoringSource,
        *,
        prompt_overwrite: bool,
    ) -> MechanismAuthoringSource | None:
        plan = self._prepare_or_report(source, prompt_overwrite=bool(prompt_overwrite))
        if plan is None:
            return None
        result = self._import_owner.apply_reconciliation_plan(plan)
        return result.source

    def accept_project_source_after_batch_load(
        self,
        source: MechanismAuthoringSource,
        *,
        batch_store: BatchInitialConditionsStore,
        batch_payload_present: bool,
    ) -> MechanismAuthoringSource | None:
        if bool(batch_payload_present) and not batch_initial_conditions_store_is_true_placeholder(batch_store):
            return source.without_reaction_initial_concentrations()
        return self.accept_importable_source(source, prompt_overwrite=False)

    def _prepare_or_report(
        self,
        source: MechanismAuthoringSource,
        *,
        prompt_overwrite: bool,
    ) -> InitialConditionsReconciliationPlan | None:
        result = self._import_owner.prepare_reconciliation(
            source,
            prompt_overwrite=bool(prompt_overwrite),
        )
        if result.status == InitialConditionsImportStatus.CANCELLED_OVERWRITE:
            return None
        if result.status == InitialConditionsImportStatus.ERROR:
            detail = str(result.error_message or "").strip()
            message = "Initial Conditions could not be reconciled."
            if detail:
                message = f"{message}\n\n{detail}"
            self._show_reconciliation_error(message)
            return None
        return result
