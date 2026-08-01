"""Direct project File actions, disposition, replacement, and Close sequencing."""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import ProjectDocument
from kindred.core.project_presentation_state import (
    default_workspace_presentation_state,
)
from kindred.core.project_session import PreparedProjectInstall, ProjectSession
from kindred.gui.controllers.config_controller import ConfigController
from kindred.gui.controllers.simulation_job_owner import SimulationJobOwner
from kindred.gui.dataset_import_staging import DatasetImportStaging
from kindred.gui.fit_session_owner import FitSessionOwner
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
    """Exact unsaved Integration state relevant to one project disposition."""

    reactions_draft: bool = False
    notes_draft: bool = False
    invalid_reactions: bool = False
    invalid_initial_conditions: bool = False
    slider_drafts: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "reactions_draft",
            "notes_draft",
            "invalid_reactions",
            "invalid_initial_conditions",
            "slider_drafts",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be bool.")
        if self.invalid_reactions and not self.reactions_draft:
            raise ValueError("invalid_reactions requires reactions_draft.")

    @property
    def has_authoring_edits(self) -> bool:
        return bool(
            self.reactions_draft
            or self.notes_draft
            or self.invalid_initial_conditions
        )

    @property
    def has_invalid_authoring(self) -> bool:
        return bool(self.invalid_reactions or self.invalid_initial_conditions)

    @property
    def has_any(self) -> bool:
        return bool(self.has_authoring_edits or self.slider_drafts)


@dataclass(frozen=True, slots=True)
class _ProjectDispositionRequest:
    """One call-local combined disposition input."""

    action: str
    archive_dirty: bool
    transient: ProjectTransientState


class _DispositionChoice(Enum):
    SAVE = "save"
    DISCARD = "discard"
    RETURN_AND_FIX = "return_and_fix"
    CANCEL = "cancel"


class AuthoringDiscardMode(Enum):
    """Exact scope for one direct Integration authoring-discard action."""

    INVALID_ONLY = "invalid_only"
    ALL_UNACCEPTED = "all_unaccepted"


@dataclass(frozen=True, slots=True)
class _DispositionResult:
    discard_all_authoring: bool = False
    discard_slider_drafts: bool = False


class ProjectController(QtCore.QObject):
    """The sole New/Open/Convert/Save/Save As/Close disposition owner."""

    documentReplaced = QtCore.Signal()
    projectSaved = QtCore.Signal()
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
        dataset_imports: DatasetImportStaging,
        fit_session: FitSessionOwner,
        xlsx_exports: object,
        set_status_text: Callable[[str], None],
        settle_execution_candidate: Callable[[], bool],
        prepare_before_save: Callable[[], ProjectDocument | None],
        commit_saved_document: Callable[[ProjectDocument], None],
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
        if not isinstance(dataset_imports, DatasetImportStaging):
            raise TypeError("dataset_imports must be a DatasetImportStaging.")
        if not isinstance(fit_session, FitSessionOwner):
            raise TypeError("fit_session must be a FitSessionOwner.")
        for method_name in (
            "cancel_and_wait_for_project_replacement",
            "cancel_for_application_close",
            "owned_qthreads",
        ):
            if not callable(getattr(xlsx_exports, method_name, None)):
                raise TypeError(
                    f"xlsx_exports must provide {method_name}()."
                )
        xlsx_threads_changed = getattr(
            xlsx_exports,
            "owned_qthreads_changed",
            None,
        )
        if not callable(getattr(xlsx_threads_changed, "connect", None)):
            raise TypeError(
                "xlsx_exports must provide owned_qthreads_changed."
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
        self._dataset_imports = dataset_imports
        self._fit_session = fit_session
        self._xlsx_exports = xlsx_exports
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
        self._simulation_jobs.owned_qthreads_changed.connect(
            self.notify_close_progress
        )
        self._dataset_imports.owned_qthreads_changed.connect(
            self.notify_close_progress
        )
        self._fit_session.owned_qthreads_changed.connect(
            self.notify_close_progress
        )
        self._xlsx_exports.owned_qthreads_changed.connect(
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

    def _show_error(self, title: str, message: str) -> None:
        logger.error("%s: %s", title, message)
        QtWidgets.QMessageBox.critical(self._parent_widget, title, message)

    def _transient_state(self) -> ProjectTransientState:
        state = self._inspect_transient_state()
        if not isinstance(state, ProjectTransientState):
            raise TypeError(
                "inspect_transient_state must return ProjectTransientState."
            )
        return state

    def _disposition_request(self, action: str) -> _ProjectDispositionRequest:
        return _ProjectDispositionRequest(
            action=str(action),
            archive_dirty=self._session.is_dirty(
                self._tabs.workspace_presentation()
            ),
            transient=self._transient_state(),
        )

    def _resolve_disposition(
        self,
        action: str,
    ) -> _DispositionResult | None:
        request = self._disposition_request(action)
        if not request.archive_dirty and not request.transient.has_any:
            return _DispositionResult()
        choice = self._prompt_disposition(request)
        if choice in {
            _DispositionChoice.CANCEL,
            _DispositionChoice.RETURN_AND_FIX,
        }:
            return None
        if choice is _DispositionChoice.SAVE:
            if request.transient.has_invalid_authoring:
                self._discard_authoring(AuthoringDiscardMode.INVALID_ONLY)
            if not self.save_project():
                return None
            return _DispositionResult(
                discard_slider_drafts=request.transient.slider_drafts,
            )
        if choice is _DispositionChoice.DISCARD:
            return _DispositionResult(
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
        if transient.invalid_reactions:
            details.append("Visible Reactions text is invalid.")
        elif transient.reactions_draft:
            details.append("Visible Reactions text has not yet been accepted.")
        if transient.notes_draft:
            details.append("Visible Notes text has not yet been saved.")
        if transient.invalid_initial_conditions:
            details.append("One or more visible initial-condition cells are invalid.")
        if transient.slider_drafts:
            details.append("Slider Preview drafts are not project content.")
        details.append(f"Choose once how to proceed before {request.action}.")

        dialog = QtWidgets.QMessageBox(self._parent_widget)
        dialog.setWindowTitle("Unsaved Project Work")
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        dialog.setText("\n\n".join(details))

        choices: dict[QtWidgets.QAbstractButton, _DispositionChoice] = {}
        if (
            request.archive_dirty
            or transient.notes_draft
            or (
                transient.reactions_draft
                and not transient.invalid_reactions
            )
        ):
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
            discard_parts.append(
                "Invalid Reactions"
                if transient.invalid_reactions
                else "Reactions Edit"
            )
        if transient.notes_draft:
            discard_parts.append("Notes Edit")
        if transient.invalid_initial_conditions:
            discard_parts.append("Invalid Cell Edit(s)")
        if transient.slider_drafts:
            discard_parts.append("Slider Preview")
        if not discard_parts:
            raise RuntimeError("Disposition prompt has no discardable work.")
        if len(discard_parts) == 1:
            discard_subject = discard_parts[0]
        elif len(discard_parts) == 2:
            discard_subject = " and ".join(discard_parts)
        else:
            discard_subject = (
                ", ".join(discard_parts[:-1])
                + ", and "
                + discard_parts[-1]
            )
        discard_label = f"Discard {discard_subject} and Continue"
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
        dialog.exec()
        return choices.get(dialog.clickedButton(), _DispositionChoice.CANCEL)

    def _apply_discard(self, result: _DispositionResult) -> None:
        if result.discard_all_authoring:
            self._discard_authoring(AuthoringDiscardMode.ALL_UNACCEPTED)
        if result.discard_slider_drafts:
            self._discard_slider_drafts()

    def _settle_replacement_workers(self) -> bool:
        deadline = time.monotonic() + self._replacement_timeout_seconds
        try:
            simulation_stopped = bool(
                self._simulation_jobs.cancel_and_wait_for_project_replacement(
                    deadline
                )
            )
        except Exception:
            logger.exception(
                "Simulation settlement failed during project replacement"
            )
            simulation_stopped = False
        try:
            imports_stopped = bool(
                self._dataset_imports.cancel_and_settle_for_project_replacement(
                    deadline
                )
            )
        except Exception:
            logger.exception(
                "Dataset import settlement failed during project replacement"
            )
            imports_stopped = False
        try:
            fit_stopped = bool(
                self._fit_session.close_for_project_replacement(deadline)
            )
        except Exception:
            logger.exception("Fit settlement failed during project replacement")
            fit_stopped = False
        try:
            xlsx_stopped = bool(
                self._xlsx_exports.cancel_and_wait_for_project_replacement(
                    deadline
                )
            )
        except Exception:
            logger.exception(
                "XLSX export settlement failed during project replacement"
            )
            xlsx_stopped = False
        failed = tuple(
            label
            for label, succeeded in (
                ("Simulation", simulation_stopped),
                ("Dataset import", imports_stopped),
                ("Fit", fit_stopped),
                ("XLSX export", xlsx_stopped),
            )
            if not succeeded
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

    def _prepare_project_install(
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

    def _commit_project_install(
        self,
        prepared_session: PreparedProjectInstall,
        prepared_workspace: PreparedWorkspaceInstall,
        disposition: _DispositionResult,
    ) -> bool:
        if not self._settle_replacement_workers():
            return False
        previous_document = self._session.document
        try:
            self._apply_discard(disposition)
            self._session.commit_install(prepared_session)
            self._tabs.commit_install(prepared_workspace)
        finally:
            self._dataset_imports.resume_after_project_replacement()
        self._purge_replaced_project_results(previous_document)
        self.documentReplaced.emit()
        self.projectStateChanged.emit()
        return True

    def _purge_replaced_project_results(
        self,
        previous_document: ProjectDocument,
    ) -> None:
        """Drop old result identities only after both prepared installs commit."""

        for tab in previous_document.tabs:
            set_ids = tuple(
                initial_set.set_id
                for initial_set in tab.initial_condition_sets
            )
            try:
                self._simulation_jobs.purge_tab(tab.tab_id, set_ids)
            except Exception:
                logger.exception(
                    "Old Simulation result state could not be purged for tab %s",
                    tab.tab_id,
                )

    def new_project(self) -> bool:
        candidate = self._config.default_project_document()
        if not isinstance(candidate, ProjectDocument):
            raise TypeError(
                "ConfigController.default_project_document() must return "
                "ProjectDocument."
            )
        prepared_session, prepared_workspace = self._prepare_project_install(
            ArchiveProject(
                document=candidate,
                presentation=default_workspace_presentation_state(candidate),
            ),
            path=None,
        )
        if not self._settle_execution_candidate():
            return False
        disposition = self._resolve_disposition("starting a new project")
        if disposition is None:
            return False
        if not self._commit_project_install(
            prepared_session,
            prepared_workspace,
            disposition,
        ):
            return False
        self._set_status("New project")
        return True

    def load_project(self) -> bool:
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self._parent_widget,
            "Open Project",
            "",
            "Kindred Project (*.kin)",
        )
        if not path:
            return False
        return self._open_project_candidate(
            path,
            remove_missing_recent=False,
        )

    def load_recent_project(self, path: str) -> bool:
        return self._open_project_candidate(
            path,
            remove_missing_recent=True,
        )

    def open_project_path(self, path: str) -> bool:
        return self._open_project_candidate(
            path,
            remove_missing_recent=False,
        )

    def _open_project_candidate(
        self,
        path: str,
        *,
        remove_missing_recent: bool,
    ) -> bool:
        absolute_path = os.path.abspath(os.fspath(path))
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
        try:
            decoded = self._archive_store.load(absolute_path)
        except Exception as exc:
            self._show_error("Open Error", f"Failed to open project:\n\n{exc}")
            return False
        try:
            prepared_session, prepared_workspace = (
                self._prepare_project_install(
                    decoded,
                    path=absolute_path,
                )
            )
        except Exception as exc:
            self._show_error(
                "Open Error",
                f"Failed to prepare project installation:\n\n{exc}",
            )
            return False
        if not self._settle_execution_candidate():
            return False
        disposition = self._resolve_disposition("opening another project")
        if disposition is None:
            return False
        if not self._commit_project_install(
            prepared_session,
            prepared_workspace,
            disposition,
        ):
            return False
        self._set_status(f"Opened project: {absolute_path}")
        try:
            self._config.record_recent_project(absolute_path)
        except Exception:
            logger.warning("Recent-project promotion failed", exc_info=True)
        return True

    def save_project(self) -> bool:
        candidate = self._prepared_save_document()
        if candidate is None:
            return False
        path = self._session.current_path
        if path is None:
            return self._choose_save_path_and_write(candidate)
        return self._write_project(path, candidate)

    def save_project_as(self) -> bool:
        candidate = self._prepared_save_document()
        if candidate is None:
            return False
        return self._choose_save_path_and_write(candidate)

    def _prepared_save_document(self) -> ProjectDocument | None:
        candidate = self._prepare_before_save()
        if candidate is None:
            return None
        if not isinstance(candidate, ProjectDocument):
            raise TypeError(
                "prepare_before_save must return ProjectDocument or None."
            )
        return candidate

    def _choose_save_path_and_write(
        self,
        candidate: ProjectDocument,
    ) -> bool:
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self._parent_widget,
            "Save Project As",
            self._session.current_path or "",
            "Kindred Project (*.kin)",
        )
        if not path:
            return False
        return self._write_project(path, candidate)

    def _write_project(
        self,
        path: str,
        candidate: ProjectDocument,
    ) -> bool:
        absolute_path = os.path.abspath(os.fspath(path))
        presentation = self._tabs.workspace_presentation()
        try:
            self._archive_store.save(
                absolute_path,
                candidate,
                presentation,
            )
        except Exception as exc:
            self._show_error("Save Error", f"Failed to save project:\n\n{exc}")
            return False
        self._commit_saved_document(candidate)
        self._session.mark_saved(absolute_path, presentation)
        self._set_status(f"Saved project: {absolute_path}")
        try:
            self._config.record_recent_project(absolute_path)
        except Exception:
            logger.warning("Recent-project update failed", exc_info=True)
        self.projectSaved.emit()
        self.projectStateChanged.emit()
        return True

    def request_close(self) -> bool:
        """Start application Close once; subsequent requests only advance it."""

        if self._close_started:
            self.notify_close_progress()
            return self._close_authorization is not None
        if not self._settle_execution_candidate():
            return False
        disposition = self._resolve_disposition("closing Kindred")
        if disposition is None:
            return False
        self._apply_discard(disposition)
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
        running, enumeration_complete = self._running_owned_qthreads()
        if not enumeration_complete:
            return
        if not running:
            self._authorize_application_close(hard_exit=False)
            return
        deadline = self._close_deadline
        if deadline is not None and time.monotonic() >= deadline:
            self._authorize_application_close(hard_exit=True)

    @QtCore.Slot()
    def _on_application_close_deadline(self) -> None:
        if not self._close_pending:
            return
        deadline = self._close_deadline
        if deadline is None:
            return
        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            self._close_timer.start(max(1, math.ceil(remaining * 1000.0)))
            return
        running, enumeration_complete = self._running_owned_qthreads()
        if running:
            self._authorize_application_close(hard_exit=True)
            return
        if not enumeration_complete:
            self._set_status(
                "Close is waiting because owned QThread status could not be "
                "determined safely."
            )
            self._close_timer.start(100)
            return
        self._authorize_application_close(hard_exit=False)

    def _running_owned_qthreads(
        self,
    ) -> tuple[tuple[QtCore.QThread, ...], bool]:
        running: list[QtCore.QThread] = []
        seen: set[int] = set()
        complete = True
        try:
            simulation_threads = self._simulation_jobs.owned_qthreads()
        except Exception:
            logger.exception(
                "Simulation QThread enumeration failed during Close"
            )
            complete = False
        else:
            complete = (
                self._append_running_qthreads(
                    simulation_threads,
                    owner_label="Simulation",
                    running=running,
                    seen=seen,
                )
                and complete
            )
        try:
            import_threads = self._dataset_imports.owned_qthreads()
        except Exception:
            logger.exception(
                "Dataset-import QThread enumeration failed during Close"
            )
            complete = False
        else:
            complete = (
                self._append_running_qthreads(
                    import_threads,
                    owner_label="Dataset import",
                    running=running,
                    seen=seen,
                )
                and complete
            )
        try:
            fit_threads = self._fit_session.owned_qthreads()
        except Exception:
            logger.exception("Fit QThread enumeration failed during Close")
            complete = False
        else:
            complete = (
                self._append_running_qthreads(
                    fit_threads,
                    owner_label="Fit",
                    running=running,
                    seen=seen,
                )
                and complete
            )
        try:
            xlsx_threads = self._xlsx_exports.owned_qthreads()
        except Exception:
            logger.exception(
                "XLSX export QThread enumeration failed during Close"
            )
            complete = False
        else:
            complete = (
                self._append_running_qthreads(
                    xlsx_threads,
                    owner_label="XLSX export",
                    running=running,
                    seen=seen,
                )
                and complete
            )
        return tuple(running), complete

    @staticmethod
    def _append_running_qthreads(
        candidates: Iterable[object],
        *,
        owner_label: str,
        running: list[QtCore.QThread],
        seen: set[int],
    ) -> bool:
        complete = True
        try:
            owned_threads = tuple(candidates)
        except (TypeError, RuntimeError):
            logger.exception(
                "%s returned an unreadable owned-QThread collection",
                owner_label,
            )
            return False
        for candidate in owned_threads:
            if not isinstance(candidate, QtCore.QThread):
                logger.error(
                    "%s returned a non-QThread owned worker during Close",
                    owner_label,
                )
                complete = False
                continue
            identity = id(candidate)
            if identity in seen:
                continue
            seen.add(identity)
            try:
                is_running = bool(candidate.isRunning())
            except RuntimeError:
                logger.exception(
                    "%s QThread status could not be read during Close",
                    owner_label,
                )
                complete = False
                continue
            if is_running:
                running.append(candidate)
        return complete

    def _authorize_application_close(self, *, hard_exit: bool) -> None:
        if not self._close_pending or self._close_authorization is not None:
            return
        self._close_pending = False
        self._close_timer.stop()
        self._close_authorization = "hard_exit" if hard_exit else "native"
        self.closePendingChanged.emit(False)
        if hard_exit:
            self.hardExitAuthorized.emit()
        else:
            self.nativeCloseAuthorized.emit()
