"""Direct project File actions, disposition, replacement, and Close sequencing."""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import ProjectDocument
from kindred.core.project_presentation_state import default_workspace_presentation_state
from kindred.core.project_session import (
    PreparedProjectInstall,
    ProjectContentToken,
    ProjectSession,
)
from kindred.gui.controllers.config_controller import ConfigController
from kindred.gui.controllers.simulation_job_owner import SimulationJobOwner
from kindred.gui.controllers.interactive_preview_owner import InteractivePreviewOwner
from kindred.gui.dataset_import_staging import DatasetImportStaging
from kindred.gui.fit_session_owner import (
    FitReplacementDispositionSummary,
    FitSessionOwner,
)
from kindred.gui.workspace_tab_registry import (
    PreparedWorkspaceInstall,
    WorkspaceTabRegistry,
)
from kindred.io.kin_archive_store import ArchiveProject, KinArchiveStore

logger = logging.getLogger(__name__)

__all__ = [
    "AuthoringDiscardMode",
    "ProjectController",
    "ProjectTransientState",
]


@dataclass(frozen=True, slots=True)
class ProjectTransientState:
    """Exact transient authoring and active-work disposition state."""

    reactions_draft: bool = False
    notes_draft: bool = False
    invalid_initial_conditions: bool = False
    slider_drafts: bool = False
    active_run_or_preview: bool = False
    active_fit: bool = False
    active_import: bool = False
    active_xlsx_export: bool = False
    fit_disposition: FitReplacementDispositionSummary = field(
        default_factory=FitReplacementDispositionSummary
    )

    def __post_init__(self) -> None:
        for field_name in (
            "reactions_draft",
            "notes_draft",
            "invalid_initial_conditions",
            "slider_drafts",
            "active_run_or_preview",
            "active_fit",
            "active_import",
            "active_xlsx_export",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be bool.")
        if not isinstance(
            self.fit_disposition,
            FitReplacementDispositionSummary,
        ):
            raise TypeError(
                "fit_disposition must be FitReplacementDispositionSummary."
            )

    @property
    def has_authoring_edits(self) -> bool:
        return bool(
            self.reactions_draft
            or self.notes_draft
            or self.invalid_initial_conditions
        )

    @property
    def has_invalid_authoring(self) -> bool:
        return self.invalid_initial_conditions

    @property
    def has_active_work(self) -> bool:
        return bool(
            self.active_run_or_preview
            or self.active_fit
            or self.fit_disposition.active_worker
            or self.active_import
            or self.active_xlsx_export
        )

    @property
    def has_any(self) -> bool:
        return bool(
            self.has_authoring_edits
            or self.slider_drafts
            or self.has_active_work
            or self.fit_disposition.requires_disposition
        )


@dataclass(frozen=True, slots=True)
class _ProjectDispositionRequest:
    """One call-local combined disposition input."""

    action: str
    archive_dirty: bool
    transient: ProjectTransientState
    project_replacement: bool = False
    application_close: bool = False


class _DispositionChoice(Enum):
    SAVE = "save"
    DISCARD = "discard"
    RETURN_AND_FIX = "return_and_fix"
    CANCEL = "cancel"


class AuthoringDiscardMode(Enum):
    """Exact scope for one direct Integration authoring-discard action."""

    ALL_UNACCEPTED = "all_unaccepted"


@dataclass(frozen=True, slots=True)
class _DispositionResult:
    content_token: ProjectContentToken
    clean_idle: bool = False
    discard_all_authoring: bool = False
    discard_slider_drafts: bool = False


class ProjectController(QtCore.QObject):
    """The sole New/Open/Convert/Save/Save As/Close disposition owner."""

    projectStateChanged = QtCore.Signal()
    closePendingChanged = QtCore.Signal(bool)
    nativeCloseAuthorized = QtCore.Signal()
    hardExitAuthorized = QtCore.Signal()

    _APPLICATION_CLOSE_TIMEOUT_MS = 60_000

    def __init__(
        self,
        parent_widget: QtWidgets.QWidget,
        *,
        session: ProjectSession,
        tabs: WorkspaceTabRegistry,
        archive_store: KinArchiveStore,
        config: ConfigController,
        simulation_jobs: SimulationJobOwner,
        interactive_preview_owner: InteractivePreviewOwner,
        dataset_imports: DatasetImportStaging,
        fit_session: FitSessionOwner,
        xlsx_exports: object,
        build_workspace: Callable[
            [ProjectDocument, PreparedWorkspaceInstall],
            QtWidgets.QWidget,
        ],
        swap_workspace: Callable[[QtWidgets.QWidget], None],
        set_status_text: Callable[[str], None],
        settle_execution_candidate: Callable[
            [ProjectContentToken],
            ProjectContentToken | None,
        ],
        prepare_before_save: Callable[[bool], ProjectDocument | None],
        commit_saved_document: Callable[
            [ProjectDocument, ProjectContentToken],
            ProjectContentToken | None,
        ],
        inspect_transient_state: Callable[[], ProjectTransientState],
        discard_authoring: Callable[[AuthoringDiscardMode], None],
        discard_slider_drafts: Callable[[], None],
        replacement_timeout_seconds: float,
    ) -> None:
        if not isinstance(parent_widget, QtWidgets.QWidget):
            raise TypeError("parent_widget must be a QWidget.")
        super().__init__(parent_widget)
        if not isinstance(session, ProjectSession):
            raise TypeError("session must be a ProjectSession.")
        if not isinstance(tabs, WorkspaceTabRegistry):
            raise TypeError("tabs must be a WorkspaceTabRegistry.")
        if not isinstance(archive_store, KinArchiveStore):
            raise TypeError("archive_store must be a KinArchiveStore.")
        if not isinstance(config, ConfigController):
            raise TypeError("config must be a ConfigController.")
        if not isinstance(simulation_jobs, SimulationJobOwner):
            raise TypeError("simulation_jobs must be a SimulationJobOwner.")
        if not isinstance(interactive_preview_owner, InteractivePreviewOwner):
            raise TypeError("interactive_preview_owner must be an InteractivePreviewOwner.")
        if not isinstance(dataset_imports, DatasetImportStaging):
            raise TypeError("dataset_imports must be a DatasetImportStaging.")
        if not isinstance(fit_session, FitSessionOwner):
            raise TypeError("fit_session must be a FitSessionOwner.")
        if not callable(build_workspace):
            raise TypeError("build_workspace must be callable.")
        if not callable(swap_workspace):
            raise TypeError("swap_workspace must be callable.")
        for owner_name, owner in (
            ("simulation_jobs", simulation_jobs),
            ("interactive_preview_owner", interactive_preview_owner),
            ("dataset_imports", dataset_imports),
            ("fit_session", fit_session),
        ):
            for method_name in (
                "request_project_replacement_cancellation",
                "settle_project_replacement",
            ):
                if not callable(getattr(owner, method_name, None)):
                    raise TypeError(
                        f"{owner_name} must provide {method_name}()."
                    )
        for method_name in (
            "request_project_replacement_cancellation",
            "settle_project_replacement",
            "cancel_for_application_close",
        ):
            if not callable(getattr(xlsx_exports, method_name, None)):
                raise TypeError(
                    f"xlsx_exports must provide {method_name}()."
                )
        xlsx_settlement_changed = getattr(
            xlsx_exports,
            "application_close_settlement_changed",
            None,
        )
        if not callable(getattr(xlsx_settlement_changed, "connect", None)):
            raise TypeError(
                "xlsx_exports must provide application_close_settlement_changed."
            )
        if type(getattr(xlsx_exports, "application_close_settled", None)) is not bool:
            raise TypeError(
                "xlsx_exports must provide boolean application_close_settled."
            )
        if not callable(set_status_text):
            raise TypeError("set_status_text must be callable.")
        if not callable(settle_execution_candidate):
            raise TypeError("settle_execution_candidate must be callable.")
        if not callable(prepare_before_save):
            raise TypeError("prepare_before_save must be callable.")
        if not callable(commit_saved_document):
            raise TypeError("commit_saved_document must be callable.")
        if not callable(inspect_transient_state):
            raise TypeError("inspect_transient_state must be callable.")
        if not callable(discard_authoring):
            raise TypeError("discard_authoring must be callable.")
        if not callable(discard_slider_drafts):
            raise TypeError("discard_slider_drafts must be callable.")
        self._parent_widget = parent_widget
        self._session = session
        self._tabs = tabs
        self._archive_store = archive_store
        self._config = config
        self._simulation_jobs = simulation_jobs
        self._interactive_preview_owner = interactive_preview_owner
        self._dataset_imports = dataset_imports
        self._fit_session = fit_session
        self._xlsx_exports = xlsx_exports
        self._build_workspace = build_workspace
        self._swap_workspace = swap_workspace
        self._set_status_text = set_status_text
        self._settle_execution_candidate = settle_execution_candidate
        self._prepare_before_save = prepare_before_save
        self._commit_saved_document = commit_saved_document
        self._inspect_transient_state = inspect_transient_state
        self._discard_authoring = discard_authoring
        self._discard_slider_drafts = discard_slider_drafts
        self._replacement_timeout_seconds = self._positive_timeout(
            replacement_timeout_seconds,
            "replacement_timeout_seconds",
        )
        self._close_started = False
        self._close_pending = False
        self._close_deadline: float | None = None
        self._close_authorization: str | None = None
        self._close_timer = QtCore.QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._close_timer.timeout.connect(self._on_application_close_deadline)
        self._simulation_jobs.application_close_settlement_changed.connect(
            self.notify_close_progress
        )
        self._interactive_preview_owner.settled_changed.connect(
            self.notify_close_progress
        )
        self._dataset_imports.application_close_settlement_changed.connect(
            self.notify_close_progress
        )
        self._fit_session.application_close_settlement_changed.connect(
            self.notify_close_progress
        )
        self._xlsx_exports.application_close_settlement_changed.connect(
            self.notify_close_progress
        )

    @staticmethod
    def _positive_timeout(value: object, label: str) -> float:
        if type(value) is bool:
            raise TypeError(f"{label} must be numeric.")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{label} must be numeric.") from exc
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{label} must be finite and positive.")
        return parsed

    def _set_status(self, text: str) -> None:
        self._set_status_text(str(text))

    def _set_replacement_success_status(
        self,
        success_text: str,
        cleanup_failures: tuple[str, ...],
    ) -> None:
        """Publish exactly one terminal status for a committed replacement."""

        if cleanup_failures:
            self._set_status(
                f"{success_text}, but cleanup reported: "
                + "; ".join(cleanup_failures)
            )
            return
        self._set_status(success_text)

    @property
    def mutation_admitted(self) -> bool:
        """Whether a new user mutation may enter the project lifecycle."""

        # Close is a one-way lifecycle.  Keep the admission latched even
        # after early native authorization has cleared ``_close_pending`` so
        # the event loop cannot reopen a writable window before closeEvent.
        return not (self._close_started or self._close_pending)

    def _reject_mutation_during_close(self, action: str) -> bool:
        if self.mutation_admitted:
            return False
        self._set_status(f"{action} rejected: application Close is in progress.")
        return True

    def _show_error(self, title: str, message: str) -> None:
        logger.error("%s: %s", title, message)
        QtWidgets.QMessageBox.critical(self._parent_widget, title, message)

    def _content_token_is_current(
        self,
        token: ProjectContentToken,
        *,
        action: str,
    ) -> bool:
        if self._session.is_content_token_current(token):
            return True
        self._set_status(
            f"{action} was not completed because the current project changed."
        )
        return False

    def _transient_state(self) -> ProjectTransientState:
        state = self._inspect_transient_state()
        if not isinstance(state, ProjectTransientState):
            raise TypeError(
                "inspect_transient_state must return ProjectTransientState."
            )
        return state

    def _resolve_disposition(
        self,
        action: str,
        *,
        expected_token: ProjectContentToken,
        project_replacement: bool = False,
        application_close: bool = False,
    ) -> _DispositionResult | None:
        if not self._content_token_is_current(
            expected_token,
            action=action,
        ):
            return None
        request = _ProjectDispositionRequest(
            action=str(action),
            archive_dirty=self._session.is_dirty(
                self._tabs.workspace_presentation()
            ),
            transient=self._transient_state(),
            project_replacement=bool(project_replacement),
            application_close=bool(application_close),
        )
        if not request.archive_dirty and not request.transient.has_any:
            return _DispositionResult(
                content_token=expected_token,
                clean_idle=True,
            )
        choice = self._prompt_disposition(request)
        if not self._content_token_is_current(
            expected_token,
            action=action,
        ):
            return None
        if choice in {
            _DispositionChoice.CANCEL,
            _DispositionChoice.RETURN_AND_FIX,
        }:
            return None
        if choice is _DispositionChoice.SAVE:
            saved_token = self._save_project_from_token(
                expected_token,
                discard_invalid_initial_conditions=(
                    request.transient.invalid_initial_conditions
                ),
            )
            if saved_token is None:
                return None
            return _DispositionResult(
                content_token=saved_token,
                discard_slider_drafts=request.transient.slider_drafts,
            )
        if choice is _DispositionChoice.DISCARD:
            return _DispositionResult(
                content_token=expected_token,
                discard_all_authoring=request.transient.has_authoring_edits,
                discard_slider_drafts=request.transient.slider_drafts,
            )
        return None

    def _prompt_disposition(
        self,
        request: _ProjectDispositionRequest,
    ) -> _DispositionChoice:
        transient = request.transient
        details: list[str] = []
        if request.archive_dirty:
            details.append(
                "The project document or saved workspace presentation has "
                "unsaved changes."
            )
        if transient.reactions_draft:
            details.append("Visible Reactions text has not yet been accepted.")
        if transient.notes_draft:
            details.append("Visible Notes text has not yet been saved.")
        if transient.invalid_initial_conditions:
            details.append("One or more visible initial-condition cells are invalid.")
        if transient.slider_drafts:
            details.append("Slider Preview drafts are not project content.")
        if transient.active_run_or_preview:
            details.append("Run or Preview work is still active.")
        fit_disposition = transient.fit_disposition
        if transient.active_fit or fit_disposition.active_worker:
            details.append("Fit work is still active.")
        if fit_disposition.has_retained_work:
            if fit_disposition.retained_completed_fit:
                if fit_disposition.apply_ready:
                    details.append(
                        "Completed Fit results are retained and still have "
                        "values available to Apply."
                    )
                else:
                    details.append(
                        "Completed Fit results are retained; no values remain "
                        "available to Apply."
                    )
            elif fit_disposition.retained_diagnostic_only:
                details.append("Diagnostic-only Fit results are retained.")
            if fit_disposition.retained_nonconverged:
                details.append("The retained Fit did not converge.")
            if fit_disposition.retained_terminal and not (
                fit_disposition.retained_completed_fit
                or fit_disposition.retained_diagnostic_only
            ):
                details.append("A terminal Fit state is retained.")
            if fit_disposition.setup_validation_blocked:
                details.append("Fit setup edits still require correction.")
            if fit_disposition.shared_refit_retry:
                details.append("A Shared-refit retry remains staged.")
        if transient.active_import:
            details.append("A dataset import is still active.")
        if transient.active_xlsx_export:
            details.append("An XLSX export is still active.")
        if request.project_replacement and transient.has_active_work:
            details.append(
                "Continuing with this project replacement will stop active work."
            )
        details.append(f"Choose once how to proceed before {request.action}.")

        dialog = QtWidgets.QMessageBox(self._parent_widget)
        dialog.setWindowTitle("Unsaved Project Work")
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        dialog.setText("\n\n".join(details))

        choices: dict[QtWidgets.QAbstractButton, _DispositionChoice] = {}
        if (
            request.archive_dirty
            or transient.notes_draft
            or transient.reactions_draft
        ):
            if request.application_close:
                if transient.has_active_work:
                    label = (
                        "Discard Invalid Edits, Save, Stop Active Work, and Close"
                        if transient.has_invalid_authoring
                        else "Save, Stop Active Work, and Close"
                    )
                else:
                    label = (
                        "Discard Invalid Edits, Save, and Close"
                        if transient.has_invalid_authoring
                        else "Save and Close"
                    )
            else:
                label = (
                    "Discard Invalid Edits, Save, and Continue"
                    if transient.has_invalid_authoring
                    else "Save and Continue"
                )
            button = dialog.addButton(
                label,
                QtWidgets.QMessageBox.ButtonRole.AcceptRole,
            )
            choices[button] = _DispositionChoice.SAVE

        discard_parts: list[str] = []
        if request.archive_dirty:
            discard_parts.append("Project Changes")
        if transient.reactions_draft:
            discard_parts.append("Reactions Edit")
        if transient.notes_draft:
            discard_parts.append("Notes Edit")
        if transient.invalid_initial_conditions:
            discard_parts.append("Invalid Cell Edit(s)")
        if transient.slider_drafts:
            discard_parts.append("Slider Preview")
        if fit_disposition.requires_disposition:
            discard_parts.append("Retained Fit Work")
        if not discard_parts and not transient.has_active_work:
            raise RuntimeError("Disposition prompt has no discardable work.")
        if not discard_parts:
            discard_subject = ""
        elif len(discard_parts) == 1:
            discard_subject = discard_parts[0]
        elif len(discard_parts) == 2:
            discard_subject = " and ".join(discard_parts)
        else:
            discard_subject = (
                ", ".join(discard_parts[:-1])
                + ", and "
                + discard_parts[-1]
            )
        if request.application_close:
            if transient.has_active_work:
                discard_label = (
                    f"Discard {discard_subject}, Stop Active Work, and Close"
                    if discard_subject
                    else "Stop Active Work and Close"
                )
            else:
                discard_label = f"Discard {discard_subject} and Close"
        elif request.project_replacement and transient.has_active_work:
            discard_label = (
                f"Discard {discard_subject}, Stop Active Work, and Continue"
                if discard_subject
                else "Stop Active Work and Continue"
            )
        else:
            discard_label = (
                f"Discard {discard_subject} and Continue"
                if discard_subject
                else "Continue"
            )
        discard_button = dialog.addButton(
            discard_label,
            QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
        )
        choices[discard_button] = _DispositionChoice.DISCARD

        if transient.has_invalid_authoring:
            return_button = dialog.addButton(
                "Return and Fix",
                QtWidgets.QMessageBox.ButtonRole.RejectRole,
            )
            choices[return_button] = _DispositionChoice.RETURN_AND_FIX
            dialog.setDefaultButton(return_button)
        cancel_button = dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        choices[cancel_button] = _DispositionChoice.CANCEL
        if transient.has_active_work:
            dialog.setDefaultButton(cancel_button)
        dialog.exec()
        return choices.get(dialog.clickedButton(), _DispositionChoice.CANCEL)

    def _apply_discard(self, result: _DispositionResult) -> None:
        if result.discard_all_authoring:
            self._discard_authoring(AuthoringDiscardMode.ALL_UNACCEPTED)
        if result.discard_slider_drafts:
            self._discard_slider_drafts()

    def _settle_replacement_workers(self) -> bool:
        deadline = time.monotonic() + self._replacement_timeout_seconds
        owners = (
            ("Simulation", self._simulation_jobs),
            ("Interactive Preview", self._interactive_preview_owner),
            ("Dataset import", self._dataset_imports),
            ("Fit", self._fit_session),
            ("XLSX export", self._xlsx_exports),
        )

        # Admission fencing and cancellation are deliberately a complete
        # broadcast phase.  No owner is allowed to consume the shared
        # deadline before every other active owner has received its request.
        request_failures: set[str] = set()
        for label, owner in owners:
            try:
                owner.request_project_replacement_cancellation()
            except Exception:
                logger.exception(
                    "%s cancellation request failed during project replacement",
                    label,
                )
                request_failures.add(label)

        settlement_failures: set[str] = set(request_failures)
        for label, owner in owners:
            try:
                settled = bool(owner.settle_project_replacement(deadline))
            except Exception:
                logger.exception(
                    "%s settlement failed during project replacement",
                    label,
                )
                settled = False
            if not settled:
                settlement_failures.add(label)
        failed = tuple(
            label
            for label, _owner in owners
            if label in settlement_failures
        )
        if failed:
            logger.error(
                "Project replacement aborted after settlement deadline; incomplete: %s",
                ", ".join(failed),
            )
            self._set_status(
                "Project replacement was not completed because transient work "
                "did not stop before the deadline."
            )
            self._dataset_imports.resume_after_project_replacement()
            return False
        return True

    def _decode_project_candidate(
        self,
        project: ArchiveProject,
        *,
        path: str | None,
    ) -> tuple[PreparedProjectInstall, PreparedWorkspaceInstall]:
        if not isinstance(project, ArchiveProject):
            raise TypeError("project must be ArchiveProject.")
        prepared_session = self._session.prepare_install(
            project.document,
            project.presentation,
            path=path,
        )
        prepared_workspace = self._tabs.prepare_install(
            document=prepared_session.document,
            presentation=prepared_session.saved_baseline.presentation,
        )
        return prepared_session, prepared_workspace

    def _replace_project(
        self,
        prepared_session: PreparedProjectInstall,
        prepared_workspace: PreparedWorkspaceInstall,
        *,
        action: str,
        expected_token: ProjectContentToken,
    ) -> tuple[bool, tuple[str, ...]]:
        """Build, fence, commit and swap one detached project root."""

        if not self._content_token_is_current(
            expected_token,
            action="Project replacement",
        ):
            return False, ()
        previous_document = self._session.document
        workspace: QtWidgets.QWidget | None = None
        try:
            workspace = self._build_workspace(
                prepared_session.document,
                prepared_workspace,
            )
        except Exception as exc:
            logger.exception("Detached project workspace construction failed")
            self._set_status(
                "Project replacement was not completed because its detached "
                f"workspace could not be built: {exc}"
            )
            return False, ()
        if not isinstance(workspace, QtWidgets.QWidget):
            raise TypeError("build_workspace() must return a QWidget.")

        def discard_detached_workspace() -> None:
            """Release an unpublished root through its one Qt owner."""

            workspace.deleteLater()

        if not self._content_token_is_current(
            expected_token,
            action="Project replacement",
        ):
            discard_detached_workspace()
            return False, ()

        # Disposition is resolved only after the candidate root is complete;
        # cancellation therefore retires one detached root without touching
        # the current shell or semantic owners.
        disposition = self._resolve_disposition(
            action,
            expected_token=expected_token,
            project_replacement=True,
        )
        if disposition is None:
            discard_detached_workspace()
            return False, ()
        if not self._settle_replacement_workers():
            discard_detached_workspace()
            return False, ()

        if not self._content_token_is_current(
            disposition.content_token,
            action="Project replacement",
        ):
            discard_detached_workspace()
            self._dataset_imports.resume_after_project_replacement()
            return False, ()

        # The old result identities are purged while the old session, registry
        # and root are still current.  This is intentionally before either
        # semantic assignment or the one-root shell swap.
        cleanup_failures = list(
            self._purge_replaced_project_results(previous_document)
        )

        swap_invoked = False
        try:
            committed_token = self._session.commit_install(
                prepared_session,
                expected_token=disposition.content_token,
            )
            if committed_token is None:
                discard_detached_workspace()
                return False, tuple(dict.fromkeys(cleanup_failures))
            if not self._tabs.commit_install(
                prepared_workspace,
                expected_token=committed_token,
            ):
                discard_detached_workspace()
                raise RuntimeError(
                    "Workspace replacement registry rejected its current token."
                )
            swap_invoked = True
            try:
                self._swap_workspace(workspace)
            except Exception as exc:
                # Semantic publication has already occurred.  Keep the new
                # root owned by the callback's shell and surface the failure
                # without restoring the previous project.
                logger.exception("Project workspace root swap failed")
                cleanup_failures.append(f"workspace swap: {exc}")
        except Exception:
            if not swap_invoked:
                discard_detached_workspace()
            raise
        finally:
            self._dataset_imports.resume_after_project_replacement()
        self.projectStateChanged.emit()
        return True, tuple(dict.fromkeys(cleanup_failures))

    def _purge_replaced_project_results(
        self,
        previous_document: ProjectDocument,
    ) -> tuple[str, ...]:
        """Drop old result identities only after both prepared installs commit."""

        failures: list[str] = []
        for tab in previous_document.tabs:
            set_ids = tuple(
                initial_set.set_id
                for initial_set in tab.initial_condition_sets
            )
            try:
                self._simulation_jobs.purge_tab(tab.tab_id, set_ids)
            except Exception as exc:
                logger.exception(
                    "Old Simulation result state could not be purged for tab %s",
                    tab.tab_id,
                )
                failures.append(
                    f"result purge for tab {tab.tab_id}: {exc}"
                )
        return tuple(failures)

    def new_project(self) -> bool:
        if self._reject_mutation_during_close("New Project"):
            return False
        entry_token = self._session.capture_content_token()
        operative_token = self._settle_execution_candidate(entry_token)
        if operative_token is None:
            return False
        candidate = self._config.default_project_document()
        if not isinstance(candidate, ProjectDocument):
            raise TypeError(
                "ConfigController.default_project_document() must return "
                "ProjectDocument."
            )
        prepared_session, prepared_workspace = self._decode_project_candidate(
            ArchiveProject(
                document=candidate,
                presentation=default_workspace_presentation_state(candidate),
            ),
            path=None,
        )
        committed, cleanup_failures = self._replace_project(
            prepared_session,
            prepared_workspace,
            action="starting a new project",
            expected_token=operative_token,
        )
        if not committed:
            return False
        self._set_replacement_success_status(
            "New project",
            cleanup_failures,
        )
        return True

    def load_project(self) -> bool:
        if self._reject_mutation_during_close("Open Project"):
            return False
        entry_token = self._session.capture_content_token()
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self._parent_widget,
            "Open Project",
            "",
            "Kindred Project (*.kin)",
        )
        if not path:
            return False
        if not self._content_token_is_current(
            entry_token,
            action="Open Project",
        ):
            return False
        return self._open_project_candidate(
            path,
            remove_missing_recent=False,
            expected_token=entry_token,
        )

    def load_recent_project(self, path: str) -> bool:
        if self._reject_mutation_during_close("Open Recent Project"):
            return False
        entry_token = self._session.capture_content_token()
        return self._open_project_candidate(
            path,
            remove_missing_recent=True,
            expected_token=entry_token,
        )

    def _open_project_candidate(
        self,
        path: str,
        *,
        remove_missing_recent: bool,
        expected_token: ProjectContentToken,
    ) -> bool:
        if self._reject_mutation_during_close("Open Project"):
            return False
        absolute_path = os.path.abspath(os.fsdecode(os.fspath(path)))
        if remove_missing_recent and not os.path.isfile(absolute_path):
            try:
                self._config.remove_recent_project(absolute_path)
            except Exception:
                logger.warning(
                    "Stale recent-project removal failed",
                    exc_info=True,
                )
            QtWidgets.QMessageBox.warning(
                self._parent_widget,
                "File Not Found",
                f"File no longer exists:\n{absolute_path}",
            )
            return False
        if self._same_canonical_path(
            absolute_path,
            self._session.current_path,
        ):
            self._set_status(f"Project is already open: {absolute_path}")
            try:
                self._config.record_recent_project(absolute_path)
            except Exception:
                logger.warning("Recent-project promotion failed", exc_info=True)
            return True
        try:
            decoded = self._archive_store.load(absolute_path)
        except Exception as exc:
            self._show_error("Open Error", f"Failed to open project:\n\n{exc}")
            return False
        if not self._content_token_is_current(
            expected_token,
            action="Open Project",
        ):
            return False
        try:
            prepared_session, prepared_workspace = (
                self._decode_project_candidate(
                    decoded,
                    path=absolute_path,
                )
            )
        except Exception as exc:
            self._show_error(
                "Open Error",
                f"Failed to decode and validate the project:\n\n{exc}",
            )
            return False
        if not self._content_token_is_current(
            expected_token,
            action="Open Project",
        ):
            return False
        operative_token = self._settle_execution_candidate(expected_token)
        if operative_token is None:
            return False
        committed, cleanup_failures = self._replace_project(
            prepared_session,
            prepared_workspace,
            action="opening another project",
            expected_token=operative_token,
        )
        if not committed:
            return False
        self._set_replacement_success_status(
            f"Opened project: {absolute_path}",
            cleanup_failures,
        )
        try:
            self._config.record_recent_project(absolute_path)
        except Exception:
            logger.warning("Recent-project promotion failed", exc_info=True)
        return True

    @staticmethod
    def _same_canonical_path(
        first: str | os.PathLike[str],
        second: str | os.PathLike[str] | None,
    ) -> bool:
        if second is None:
            return False
        first_path = os.path.normcase(
            os.path.realpath(os.path.abspath(os.fsdecode(os.fspath(first))))
        )
        second_path = os.path.normcase(
            os.path.realpath(os.path.abspath(os.fsdecode(os.fspath(second))))
        )
        return first_path == second_path

    def is_current_project_path(self, path: str | os.PathLike[str]) -> bool:
        """Return whether ``path`` resolves to the active project archive."""

        return self._same_canonical_path(path, self._session.current_path)

    def save_project(self) -> bool:
        if self._reject_mutation_during_close("Save Project"):
            return False
        entry_token = self._session.capture_content_token()
        operative_token = self._settle_execution_candidate(entry_token)
        if operative_token is None:
            return False
        candidate = self._prepared_save_document(
            discard_invalid_initial_conditions=False,
        )
        if candidate is None:
            return False
        return (
            self._save_prepared_document(
                candidate,
                expected_token=operative_token,
                choose_destination=self._session.current_path is None,
            )
            is not None
        )

    def save_project_as(self) -> bool:
        if self._reject_mutation_during_close("Save Project As"):
            return False
        entry_token = self._session.capture_content_token()
        operative_token = self._settle_execution_candidate(entry_token)
        if operative_token is None:
            return False
        candidate = self._prepared_save_document(
            discard_invalid_initial_conditions=False,
        )
        if candidate is None:
            return False
        return (
            self._save_prepared_document(
                candidate,
                expected_token=operative_token,
                choose_destination=True,
            )
            is not None
        )

    def _save_project_from_token(
        self,
        expected_token: ProjectContentToken,
        *,
        discard_invalid_initial_conditions: bool,
    ) -> ProjectContentToken | None:
        """Save during one already-fenced project disposition."""

        if not self._content_token_is_current(
            expected_token,
            action="Save Project",
        ):
            return None
        operative_token = self._settle_execution_candidate(expected_token)
        if operative_token is None:
            return None
        candidate = self._prepared_save_document(
            discard_invalid_initial_conditions=(
                discard_invalid_initial_conditions
            ),
        )
        if candidate is None:
            return None
        return self._save_prepared_document(
            candidate,
            expected_token=operative_token,
            choose_destination=self._session.current_path is None,
        )

    def _prepared_save_document(
        self,
        *,
        discard_invalid_initial_conditions: bool,
    ) -> ProjectDocument | None:
        candidate = self._prepare_before_save(
            discard_invalid_initial_conditions
        )
        if candidate is None:
            return None
        if not isinstance(candidate, ProjectDocument):
            raise TypeError(
                "prepare_before_save must return ProjectDocument or None."
            )
        return candidate

    def _save_prepared_document(
        self,
        candidate: ProjectDocument,
        *,
        expected_token: ProjectContentToken,
        choose_destination: bool,
    ) -> ProjectContentToken | None:
        if not self._content_token_is_current(
            expected_token,
            action="Save Project",
        ):
            return None
        if not choose_destination:
            current_path = self._session.current_path
            if current_path is None:
                raise RuntimeError("Current project path is unavailable.")
            return self._write_project(
                current_path,
                candidate,
                expected_token=expected_token,
            )
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self._parent_widget,
            "Save Project As",
            self._session.current_path or "",
            "Kindred Project (*.kin)",
        )
        if not path:
            return None
        if not self._content_token_is_current(
            expected_token,
            action="Save Project",
        ):
            return None
        destination = self._save_as_path(path)
        if os.path.lexists(destination) and not self._confirm_overwrite(
            destination
        ):
            return None
        return self._write_project(
            destination,
            candidate,
            expected_token=expected_token,
        )

    @staticmethod
    def _save_as_path(path: str | os.PathLike[str]) -> str:
        absolute_path = os.path.abspath(os.fsdecode(os.fspath(path)))
        _directory, filename = os.path.split(absolute_path)
        if not filename:
            raise ValueError("Project save destination requires a file name.")
        if filename.casefold().endswith(".kin"):
            return absolute_path
        return f"{absolute_path}.kin"

    def _confirm_overwrite(self, path: str) -> bool:
        choice = QtWidgets.QMessageBox.question(
            self._parent_widget,
            "Replace Existing Project",
            f"A project already exists at:\n{path}\n\nReplace it?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return choice == QtWidgets.QMessageBox.StandardButton.Yes

    def _write_project(
        self,
        path: str,
        candidate: ProjectDocument,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        if not self._content_token_is_current(
            expected_token,
            action="Save Project",
        ):
            return None
        try:
            absolute_path = os.path.abspath(os.fsdecode(os.fspath(path)))
        except (TypeError, ValueError, OSError) as exc:
            self._show_error(
                "Save Error",
                f"Invalid project destination:\n\n{exc}",
            )
            return None
        committed_token = self._commit_saved_document(
            candidate,
            expected_token,
        )
        if committed_token is None:
            self._set_status(
                "The selected save candidate was superseded before publication."
            )
            return None
        if not isinstance(committed_token, ProjectContentToken):
            raise TypeError(
                "commit_saved_document must return ProjectContentToken or None."
            )
        if (
            not self._session.is_content_token_current(committed_token)
            or self._session.document != candidate
        ):
            self._set_status(
                "The selected save candidate changed before archive publication."
            )
            return None
        try:
            presentation = self._tabs.workspace_presentation()
            self._archive_store.save(
                absolute_path,
                candidate,
                presentation,
            )
        except Exception as exc:
            self._show_error("Save Error", f"Failed to save project:\n\n{exc}")
            return None
        if not self._content_token_is_current(
            committed_token,
            action="Save Project",
        ):
            self._set_status(
                "The selected snapshot was written, but the current project "
                "changed and remains unsaved."
            )
            return None
        if not self._session.mark_saved(
            absolute_path,
            presentation,
            expected_token=committed_token,
        ):
            self._set_status(
                "The selected snapshot was written, but the current project "
                "changed before save publication."
            )
            return None
        self._discard_authoring(AuthoringDiscardMode.ALL_UNACCEPTED)
        self._set_status(f"Saved project: {absolute_path}")
        try:
            self._config.record_recent_project(absolute_path)
        except Exception:
            logger.warning("Recent-project update failed", exc_info=True)
        self.projectStateChanged.emit()
        return committed_token

    def request_close(self) -> bool:
        """Start application Close once; subsequent requests only advance it."""

        if self._close_authorization is not None:
            return True
        if self._close_started:
            self.notify_close_progress()
            return self._close_authorization is not None
        entry_token = self._session.capture_content_token()
        operative_token = self._settle_execution_candidate(entry_token)
        if operative_token is None:
            return False
        disposition = self._resolve_disposition(
            "closing Kindred",
            expected_token=operative_token,
            application_close=True,
        )
        if disposition is None:
            return False
        if not self._content_token_is_current(
            disposition.content_token,
            action="Close",
        ):
            return False
        self._apply_discard(disposition)
        if not self._session.is_content_token_current(
            disposition.content_token
        ):
            return False
        if disposition.clean_idle and self._async_owners_are_settled():
            try:
                fit_idle_closed = bool(
                    self._fit_session.close_idle_for_application_close()
                )
            except Exception:
                logger.exception(
                    "Idle Fit cleanup failed during application Close"
                )
                fit_idle_closed = False
            if fit_idle_closed:
                if not self._session.is_content_token_current(
                    disposition.content_token
                ):
                    self._set_status(
                        "Close was not completed because the current project changed."
                    )
                    return False
                self._close_started = True
                self._close_authorization = "native"
                self.nativeCloseAuthorized.emit()
                return True
            # A failed synchronous idle-window teardown is still an owner
            # cleanup problem, not a second Close policy.  Fall through to
            # the already-established single 60-second bounded state machine;
            # its Fit owner cancellation/settlement path gets one chance to
            # finish disposal before the hard-exit deadline.
            self._set_status(
                "Idle Fit cleanup did not finish; waiting for the bounded Close path."
            )
        self._close_started = True
        self._close_pending = True
        self._close_deadline = (
            time.monotonic()
            + (float(self._APPLICATION_CLOSE_TIMEOUT_MS) / 1000.0)
        )
        self.closePendingChanged.emit(True)
        try:
            self._simulation_jobs.cancel_for_application_close()
        except Exception:
            logger.exception(
                "Simulation admission fencing/cancellation failed during Close"
            )
        try:
            self._interactive_preview_owner.cancel_for_application_close()
        except Exception:
            logger.exception(
                "Interactive Preview admission fencing/cancellation failed during Close"
            )
        try:
            self._dataset_imports.cancel_for_application_close()
        except Exception:
            logger.exception(
                "Dataset-import admission fencing/cancellation failed during Close"
            )
        try:
            self._fit_session.cancel_for_application_close()
        except Exception:
            logger.exception(
                "Fit admission fencing/cancellation failed during Close"
            )
        try:
            self._xlsx_exports.cancel_for_application_close()
        except Exception:
            logger.exception(
                "XLSX export admission fencing/cancellation failed during Close"
            )
        remaining_ms = max(
            1,
            math.ceil(
                (
                    float(self._close_deadline)
                    - time.monotonic()
                )
                * 1000.0
            ),
        )
        self._close_timer.start(remaining_ms)
        self.notify_close_progress()
        return self._close_authorization is not None

    @QtCore.Slot()
    def notify_close_progress(self) -> None:
        if not self._close_pending:
            return
        deadline = self._close_deadline
        if deadline is None:
            return
        if time.monotonic() >= deadline:
            self._authorize_application_close(hard_exit=True)
            return
        if self._async_owners_are_settled():
            self._authorize_application_close(hard_exit=False)

    @QtCore.Slot()
    def _on_application_close_deadline(self) -> None:
        if not self._close_pending:
            return
        deadline = self._close_deadline
        if deadline is None:
            return
        if time.monotonic() < deadline:
            self._close_timer.start(
                max(
                    1,
                    math.ceil(
                        (float(deadline) - time.monotonic()) * 1000.0
                    ),
                )
            )
            return
        self._authorize_application_close(hard_exit=True)

    def _async_owners_are_settled(self) -> bool:
        """Read each asynchronous owner's explicit Close settlement truth."""

        settled = True
        for owner_label, owner in (
            ("Simulation", self._simulation_jobs),
            ("Interactive Preview", self._interactive_preview_owner),
            ("Dataset import", self._dataset_imports),
            ("Fit", self._fit_session),
            ("XLSX export", self._xlsx_exports),
        ):
            try:
                owner_settled = owner.application_close_settled
            except Exception:
                logger.exception(
                    "%s settlement could not be read during Close",
                    owner_label,
                )
                settled = False
                continue
            if type(owner_settled) is not bool:
                logger.error(
                    "%s returned malformed application Close settlement",
                    owner_label,
                )
                settled = False
            elif not owner_settled:
                settled = False
        return settled

    def _authorize_application_close(self, *, hard_exit: bool) -> None:
        if not self._close_pending or self._close_authorization is not None:
            return
        if hard_exit:
            self._close_pending = False
            self._close_timer.stop()
            self.closePendingChanged.emit(False)
            self._close_authorization = "hard_exit"
            self.hardExitAuthorized.emit()
            return
        self._close_pending = False
        self._close_timer.stop()
        self.closePendingChanged.emit(False)
        self._close_authorization = "native"
        self.nativeCloseAuthorized.emit()

    def consume_native_close_authorization(self) -> bool:
        """Consume the one native-close authorization at the physical boundary."""

        if self._close_authorization != "native":
            return False
        self._close_authorization = None
        return True
