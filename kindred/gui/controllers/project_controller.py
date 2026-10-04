"""Direct project file actions, disposition, and Close sequencing."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import ProjectDocument
from kindred.core.project_presentation_state import (
    WorkspacePresentationState,
    default_workspace_presentation_state,
    validate_workspace_presentation_state,
)
from kindred.core.project_session import (
    ProjectContentToken,
    ProjectSession,
)
from kindred.gui.controllers.config_controller import ConfigController
from kindred.gui.controllers.simulation_job_owner import SimulationJobOwner
from kindred.gui.dataset_import_staging import DatasetImportStaging
from kindred.gui.fit_session_owner import FitSessionOwner
from kindred.gui.workspace_tab_registry import WorkspaceTabRegistry
from kindred.io.kin_archive_store import KinArchiveStore
from kindred.io.paths import same_local_path

logger = logging.getLogger(__name__)

__all__ = [
    "ProjectController",
    "ProjectTransientState",
]


@dataclass(frozen=True, slots=True)
class ProjectTransientState:
    """Exact transient authoring and active-work disposition state."""

    reactions_draft: bool = False
    notes_draft: bool = False
    initial_condition_drafts: bool = False
    run_defaults_drafts: bool = False
    # Mechanism tab names holding an invalid Initial Conditions cell or an
    # invalid Run-settings draft; Save discards both with one choice.
    invalid_authoring_tabs: tuple[str, ...] = ()
    slider_drafts: bool = False
    active_run_or_preview: bool = False
    active_fit: bool = False
    active_import: bool = False
    active_xlsx_export: bool = False
    active_scan: bool = False
    active_calculator: bool = False
    active_spreadsheet: bool = False

    @property
    def has_authoring_edits(self) -> bool:
        return bool(
            self.reactions_draft
            or self.notes_draft
            or self.initial_condition_drafts
            or self.run_defaults_drafts
            or self.has_invalid_authoring
        )

    @property
    def has_invalid_authoring(self) -> bool:
        return bool(self.invalid_authoring_tabs)

    @property
    def has_active_work(self) -> bool:
        return bool(
            self.active_run_or_preview
            or self.active_fit
            or self.active_import
            or self.active_xlsx_export
            or self.active_scan
            or self.active_calculator
            or self.active_spreadsheet
        )

    @property
    def has_any(self) -> bool:
        return bool(
            self.has_authoring_edits
            or self.slider_drafts
            or self.has_active_work
        )


@dataclass(frozen=True, slots=True)
class _ProjectDispositionRequest:
    """One call-local combined disposition input."""

    action: str
    archive_dirty: bool
    transient: ProjectTransientState
    application_close: bool = False


class _DispositionChoice(Enum):
    SAVE = "save"
    DISCARD = "discard"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class _DispositionResult:
    content_token: ProjectContentToken
    clean_idle: bool = False
    discard_all_authoring: bool = False
    discard_slider_drafts: bool = False


class ProjectController(QtCore.QObject):
    """The sole Save/Save As/Close disposition owner."""

    projectStateChanged = QtCore.Signal(bool)
    closeStarted = QtCore.Signal()
    nativeCloseAuthorized = QtCore.Signal()

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
        scan: object,
        calculator_owners: Callable[[], tuple[object, ...]],
        spreadsheet_editor: Callable[[], object | None],
        set_status_text: Callable[[str], None],
        admit_incoming_document: Callable[
            [ProjectDocument],
            ProjectDocument | None,
        ],
        capture_visible_tab_inputs: Callable[[], None],
        prepare_before_save: Callable[
            [bool],
            tuple[
                ProjectDocument,
                tuple[tuple[str, tuple[str, ...]], ...],
            ]
            | None,
        ],
        commit_saved_document: Callable[
            [
                ProjectDocument,
                tuple[tuple[str, tuple[str, ...]], ...],
                ProjectContentToken,
            ],
            ProjectContentToken | None,
        ],
        inspect_transient_state: Callable[[], ProjectTransientState],
        discard_authoring: Callable[[], None],
        discard_slider_drafts: Callable[[], None],
    ) -> None:
        super().__init__(parent_widget)
        self._parent_widget = parent_widget
        self._session = session
        self._tabs = tabs
        self._archive_store = archive_store
        self._config = config
        self._simulation_jobs = simulation_jobs
        self._dataset_imports = dataset_imports
        self._fit_session = fit_session
        self._xlsx_exports = xlsx_exports
        self._scan = scan
        self._calculator_owners = calculator_owners
        self._spreadsheet_editor = spreadsheet_editor
        self._set_status_text = set_status_text
        self._admit_incoming_document = admit_incoming_document
        self._capture_visible_tab_inputs = capture_visible_tab_inputs
        self._prepare_before_save = prepare_before_save
        self._commit_saved_document = commit_saved_document
        self._inspect_transient_state = inspect_transient_state
        self._discard_authoring = discard_authoring
        self._discard_slider_drafts = discard_slider_drafts
        self._close_started = False
        self._close_pending = False
        self._native_close_authorized = False
        self._simulation_jobs.application_close_settlement_changed.connect(
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
        self._scan.application_close_settlement_changed.connect(self.notify_close_progress)

    def _set_status(self, text: str) -> None:
        self._set_status_text(str(text))

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

    def _resolve_disposition(
        self,
        action: str,
        *,
        application_close: bool = False,
    ) -> _DispositionResult | None:
        self._capture_visible_tab_inputs()
        expected_token = self._session.capture_content_token()
        request = _ProjectDispositionRequest(
            action=str(action),
            archive_dirty=self._session.is_dirty(
                self._tabs.workspace_presentation()
            ),
            transient=self._inspect_transient_state(),
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
        if choice is _DispositionChoice.CANCEL:
            return None
        if choice is _DispositionChoice.SAVE:
            saved_token = self._save_project_from_token(
                expected_token,
                discard_invalid_authoring=request.transient.has_invalid_authoring,
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
            details.append("Reactions text has not yet been accepted.")
        if transient.notes_draft:
            details.append("Notes text has not yet been saved.")
        if transient.initial_condition_drafts:
            details.append("Initial Conditions edits have not yet been saved.")
        if transient.has_invalid_authoring:
            tabs = ", ".join(repr(name) for name in transient.invalid_authoring_tabs)
            details.append(
                f"Initial Conditions or Run settings on mechanism tab(s) {tabs} are invalid."
            )
        if transient.run_defaults_drafts:
            details.append("Run-default edits have not yet been saved.")
        if transient.slider_drafts:
            details.append("Slider Preview drafts are not project content.")
        if transient.active_run_or_preview:
            details.append("Run or Preview work is still active.")
        if transient.active_fit:
            details.append("Fit work is still active.")
        if transient.active_import:
            details.append("A dataset import is still active.")
        if transient.active_xlsx_export:
            details.append("An XLSX export is still active.")
        if transient.active_scan:
            details.append("A Chemical Space Scan is still active.")
        if transient.active_calculator:
            details.append("A Calculator query is still active.")
        if transient.active_spreadsheet:
            details.append("The spreadsheet editor has temporary data. Closing discards it; Save saves only project data already created.")
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
            or transient.initial_condition_drafts
            or transient.run_defaults_drafts
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
        if transient.initial_condition_drafts:
            discard_parts.append("Initial Conditions Edit")
        if transient.run_defaults_drafts:
            discard_parts.append("Run Defaults Edit")
        if transient.has_invalid_authoring:
            discard_parts.append("Invalid Edit(s)")
        if transient.slider_drafts:
            discard_parts.append("Slider Preview")
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

        cancel_button = dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        choices[cancel_button] = _DispositionChoice.CANCEL
        if transient.has_active_work:
            dialog.setDefaultButton(cancel_button)
        dialog.exec()
        return choices.get(dialog.clickedButton(), _DispositionChoice.CANCEL)

    def _apply_discard(self, result: _DispositionResult) -> None:
        if result.discard_all_authoring:
            self._discard_authoring()
        if result.discard_slider_drafts:
            self._discard_slider_drafts()

    def is_current_project_path(self, path: str | os.PathLike[str]) -> bool:
        """Return whether ``path`` resolves to the active project archive."""

        return same_local_path(path, self._session.current_path)

    def _replace_project(
        self,
        document: ProjectDocument,
        presentation: WorkspacePresentationState,
        *,
        baseline_document: ProjectDocument,
        path: str | None,
        expected_token: ProjectContentToken,
        action: str,
    ) -> bool:
        if not self._content_token_is_current(expected_token, action=action):
            return False
        self._tabs.replace_project(
            document,
            presentation,
            path=path,
            baseline_document=baseline_document,
            expected_token=expected_token,
        )
        status = (
            "Created new project." if path is None else f"Loaded project: {path}"
        )
        if path is not None:
            try:
                self._config.record_recent_project(path)
            except Exception as exc:
                logger.warning("Recent-project update failed", exc_info=True)
                status += f"; recent-project persistence failed: {exc}"
        self._set_status(status)
        self.projectStateChanged.emit(True)
        return True

    def new_project(self) -> bool:
        """Replace the current project after project-action admission."""

        if self._reject_mutation_during_close("New Project"):
            return False
        disposition = self._resolve_disposition("New Project")
        if disposition is None:
            return False
        self._apply_discard(disposition)
        document = self._config.default_project_document()
        presentation = default_workspace_presentation_state(document)
        return self._replace_project(
            document,
            presentation,
            baseline_document=document,
            path=None,
            expected_token=disposition.content_token,
            action="New Project",
        )

    def load_project(self) -> bool:
        """Choose and directly install one validated project archive."""

        if self._reject_mutation_during_close("Open Project"):
            return False
        # Capture the visible editor before opening the native picker.  The
        # picker may take focus away; canceling it must leave this exact draft
        # owned by the current project with no publication or disposition.
        self._capture_visible_tab_inputs()
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self._parent_widget,
            "Open Project",
            self._session.current_path or "",
            "Kindred Project (*.kin)",
        )
        if not path:
            return False
        return self.load_recent_project(path)

    def load_recent_project(
        self,
        path: str | os.PathLike[str],
    ) -> bool:
        """Admit and directly install one selected project archive."""

        if self._reject_mutation_during_close("Open Project"):
            return False
        if not os.path.isfile(path):
            self._show_error(
                "Open Project Error",
                f"Could not open project:\n\nThe file does not exist:\n{os.fspath(path)}",
            )
            return False
        disposition = self._resolve_disposition("Open Project")
        if disposition is None:
            return False
        try:
            archive = self._archive_store.load(path)
        except Exception as exc:
            logger.exception("Project archive load failed")
            self._show_error("Open Project Error", f"Could not open project:\n\n{exc}")
            return False
        try:
            admitted_document = self._admit_incoming_document(archive.document)
        except Exception as exc:
            logger.exception("Incoming project admission failed")
            self._show_error(
                "Open Project Error",
                f"Could not admit the incoming project:\n\n{exc}",
            )
            return False
        if admitted_document is None:
            return False
        decoded_tab_ids = tuple(tab.tab_id for tab in archive.document.tabs)
        admitted_tab_ids = tuple(tab.tab_id for tab in admitted_document.tabs)
        if admitted_tab_ids != decoded_tab_ids:
            raise ValueError(
                "Incoming project admission must preserve decoded tab IDs."
            )
        try:
            presentation = validate_workspace_presentation_state(
                admitted_document,
                archive.presentation,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.exception("Incoming project presentation validation failed")
            self._show_error(
                "Open Project Error",
                f"Could not open project:\n\n{exc}",
            )
            return False
        self._apply_discard(disposition)
        absolute_path = os.path.abspath(os.fsdecode(os.fspath(path)))
        return self._replace_project(
            admitted_document,
            presentation,
            baseline_document=archive.document,
            path=absolute_path,
            expected_token=disposition.content_token,
            action="Open Project",
        )

    def save_project(self) -> bool:
        if self._reject_mutation_during_close("Save Project"):
            return False
        self._capture_visible_tab_inputs()
        expected_token = self._session.capture_content_token()
        prepared = self._prepare_before_save(False)
        if prepared is None:
            return False
        candidate, materialized_by_tab = prepared
        return (
            self._save_prepared_document(
                candidate,
                materialized_by_tab,
                expected_token=expected_token,
                choose_destination=self._session.current_path is None,
            )
            is not None
        )

    def save_project_as(self) -> bool:
        if self._reject_mutation_during_close("Save Project As"):
            return False
        self._capture_visible_tab_inputs()
        expected_token = self._session.capture_content_token()
        prepared = self._prepare_before_save(False)
        if prepared is None:
            return False
        candidate, materialized_by_tab = prepared
        return (
            self._save_prepared_document(
                candidate,
                materialized_by_tab,
                expected_token=expected_token,
                choose_destination=True,
            )
            is not None
        )

    def _save_project_from_token(
        self,
        expected_token: ProjectContentToken,
        *,
        discard_invalid_authoring: bool,
    ) -> ProjectContentToken | None:
        """Save during one already-admitted project disposition."""

        prepared = self._prepare_before_save(discard_invalid_authoring)
        if prepared is None:
            return None
        candidate, materialized_by_tab = prepared
        return self._save_prepared_document(
            candidate,
            materialized_by_tab,
            expected_token=expected_token,
            choose_destination=self._session.current_path is None,
        )

    def _save_prepared_document(
        self,
        candidate: ProjectDocument,
        materialized_by_tab: tuple[tuple[str, tuple[str, ...]], ...],
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
                materialized_by_tab,
                expected_token=expected_token,
            )
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self._parent_widget,
            "Save Project As",
            self._session.current_path or "",
            "Kindred Project (*.kin)",
            options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite,
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
            materialized_by_tab,
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
        materialized_by_tab: tuple[tuple[str, tuple[str, ...]], ...],
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
        if self._scan.owns_destination(absolute_path):
            self._show_error("Save Error", "Cannot overwrite the destination of an active Chemical Space Scan.")
            return None
        committed_token = self._commit_saved_document(
            candidate,
            materialized_by_tab,
            expected_token,
        )
        if committed_token is None:
            self._set_status(
                "The selected save candidate was superseded before publication."
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
        self._session.mark_saved(
            absolute_path,
            presentation,
            expected_token=committed_token,
        )
        status = f"Saved project: {absolute_path}"
        try:
            self._config.record_recent_project(absolute_path)
        except Exception as exc:
            logger.warning("Recent-project update failed", exc_info=True)
            status += f"; recent-project persistence failed: {exc}"
        self._set_status(status)
        self.projectStateChanged.emit(False)
        return committed_token

    def request_close(self) -> bool:
        """Start application Close once; subsequent requests only advance it."""

        if self._native_close_authorized:
            return True
        if self._close_started:
            self.notify_close_progress()
            return self._native_close_authorized
        disposition = self._resolve_disposition(
            "closing Kindred",
            application_close=True,
        )
        if disposition is None:
            return False
        self._apply_discard(disposition)
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
            try:
                scan_idle_closed = self._scan.close_idle_for_application_close()
            except Exception:
                logger.exception("Idle Scan cleanup failed during application Close")
                scan_idle_closed = False
            if fit_idle_closed and scan_idle_closed:
                self._close_started = True
                self._native_close_authorized = True
                self.nativeCloseAuthorized.emit()
                return True
            # A failed synchronous idle-window teardown is still an owner
            # cleanup problem. Continue through the normal owner settlement
            # path so Close remains pending until every owner is settled.
            self._set_status(
                "Idle window cleanup did not finish; waiting for active work to settle."
            )
        self._close_started = True
        self._close_pending = True
        self.closeStarted.emit()
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
        try:
            self._scan.cancel_for_application_close()
        except Exception:
            logger.exception("Chemical Space Scan cancellation failed during Close")
        for calculator in self._calculator_owners():
            calculator.cancel_for_application_close()
        spreadsheet = self._spreadsheet_editor()
        if spreadsheet is not None:
            spreadsheet.cancel_for_application_close()
        self.notify_close_progress()
        return self._native_close_authorized

    @QtCore.Slot()
    def notify_close_progress(self) -> None:
        if not self._close_pending:
            return
        if self._async_owners_are_settled():
            self._authorize_application_close()

    def _async_owners_are_settled(self) -> bool:
        """Read each asynchronous owner's explicit Close settlement truth."""

        settled = True
        spreadsheet = self._spreadsheet_editor()
        if spreadsheet is not None and not spreadsheet.application_close_settled:
            settled = False
        for owner_label, owner in (
            ("Simulation", self._simulation_jobs),
            ("Dataset import", self._dataset_imports),
            ("Fit", self._fit_session),
            ("XLSX export", self._xlsx_exports),
            ("Chemical Space Scan", self._scan),
            *(("Calculator", owner) for owner in self._calculator_owners()),
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
            if not owner_settled:
                settled = False
        return settled

    def _authorize_application_close(self) -> None:
        if not self._close_pending or self._native_close_authorized:
            return
        self._close_pending = False
        self._native_close_authorized = True
        self.nativeCloseAuthorized.emit()

    def consume_native_close_authorization(self) -> bool:
        """Consume the one native-close authorization at the physical boundary."""

        if not self._native_close_authorized:
            return False
        self._native_close_authorized = False
        return True
