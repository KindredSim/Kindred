"""Small ordered lessons and their session/navigation owner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6 import QtCore, QtWidgets

from kindred.gui.display_name_policy import count_text
from kindred.gui.tutorials.overlay import Highlight, TutorialOverlay


@dataclass(frozen=True)
class Step:
    name: str
    title: str
    instruction: str
    target: Callable[[], Highlight | tuple[Highlight, ...] | None]
    complete: Callable[[], bool] | None = None
    success: str = "Completed. You can continue when ready."
    manual: bool = False
    waiting: str = "Complete the action above to continue."
    link: tuple[str, str] | None = None


class Lesson(QtCore.QObject):
    """An ordered lesson with session-local observations and no mutation authority."""

    changed = QtCore.Signal()
    ended = QtCore.Signal(str)
    revisitRequested = QtCore.Signal(str)
    title = ""
    description = ""

    def __init__(self) -> None:
        super().__init__()
        self.steps: tuple[Step, ...] = ()
        self.position = 0
        self._connections: list[tuple[object, Callable]] = []

    def observe(self, signal, callback: Callable | None = None) -> None:
        slot = (lambda *_args: self.changed.emit()) if callback is None else callback
        signal.connect(slot)
        self._connections.append((signal, slot))

    def enter(self, position: int) -> None:
        self.position = position

    @property
    def step(self) -> Step:
        return self.steps[self.position]

    def blocked(self) -> tuple[str, str, Callable[[], None] | None] | None:
        return None

    def working(self) -> bool:
        return False

    def close(self) -> None:
        for signal, slot in reversed(self._connections):
            try:
                signal.disconnect(slot)
            except RuntimeError:
                # Qt may already have destroyed a transient dialog or tab.
                pass
        self._connections.clear()


class TutorialController(QtCore.QObject):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        lessons: tuple[tuple[str, str, Callable[[], Lesson]], ...],
    ):
        super().__init__(parent)
        self._window = parent
        self._lessons = lessons
        self.lesson: Lesson | None = None
        self.position = 0
        self._awaiting_outcome = False
        self._visited: set[str] = set()
        self._verified: set[str] = set()
        self._skipped: set[str] = set()
        self._chooser: QtWidgets.QDialog | None = None
        self.overlay = TutorialOverlay(parent)
        self.overlay.card.previousRequested.connect(self.previous)
        self.overlay.card.nextRequested.connect(self.next)
        self.overlay.card.skipRequested.connect(self.skip)
        self.overlay.card.exitRequested.connect(self.stop)
        self.overlay.card.recoveryRequested.connect(self._recover)
        self.overlay.changed.connect(self.schedule_refresh)
        self._recovery: Callable[[], None] | None = None
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        # Observe retained lesson references, never arbitrary Qt construction events.
        self._geometry_timer = QtCore.QTimer(self)
        self._geometry_timer.setInterval(100)
        self._geometry_timer.timeout.connect(self.refresh)

    def open(self) -> None:
        if self.lesson is not None:
            self.overlay.expand()
            self.refresh()
            if self.overlay.card.isVisible():
                self.overlay.card.raise_()
            return
        if self._chooser is not None:
            self._chooser.raise_()
            self._chooser.activateWindow()
            return
        dialog = QtWidgets.QDialog(self._window)
        self._chooser = dialog
        dialog.setWindowTitle("Interactive Tutorials")
        layout = QtWidgets.QVBoxLayout(dialog)
        heading = QtWidgets.QLabel("Learn Kindred by Doing", dialog)
        font = heading.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 3)
        heading.setFont(font)
        layout.addWidget(heading)
        choices = QtWidgets.QListWidget(dialog)
        for title, _description, _factory in self._lessons:
            choices.addItem(title)
        layout.addWidget(choices)
        description = QtWidgets.QLabel(dialog)
        description.setWordWrap(True)
        description.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        choices.currentRowChanged.connect(
            lambda row: description.setText(self._lessons[row][1]) if row >= 0 else None
        )
        layout.addWidget(description)
        buttons = QtWidgets.QDialogButtonBox(dialog)
        start = buttons.addButton(
            "Start Tutorial", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
        )
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        choices.itemDoubleClicked.connect(lambda _item: dialog.accept())
        layout.addWidget(buttons)
        choices.setCurrentRow(0)
        start.setEnabled(bool(self._lessons))
        dialog.resize(500, 340)
        accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
        index = choices.currentRow()
        self._chooser = None
        dialog.deleteLater()
        if accepted and index >= 0:
            self.start(self._lessons[index][2]())

    def start(self, lesson: Lesson) -> None:
        self.stop()
        self.lesson = lesson
        self.position = 0
        self._visited.clear()
        self._verified.clear()
        self._skipped.clear()
        lesson.changed.connect(self.schedule_refresh)
        lesson.ended.connect(self._ended)
        lesson.revisitRequested.connect(self.revisit)
        self._geometry_timer.start()
        self._enter_step()
        self.refresh()

    def stop(self) -> None:
        self._geometry_timer.stop()
        self._refresh_timer.stop()
        lesson, self.lesson = self.lesson, None
        self._recovery = None
        if lesson is not None:
            lesson.changed.disconnect(self.schedule_refresh)
            lesson.ended.disconnect(self._ended)
            lesson.revisitRequested.disconnect(self.revisit)
            lesson.close()
            lesson.deleteLater()
        self.overlay.close()

    def _ended(self, reason: str) -> None:
        self.stop()
        QtWidgets.QMessageBox.information(self._window, "Tutorial Ended", reason)

    def next(self) -> None:
        lesson = self.lesson
        if lesson is None:
            return
        position = self.position
        self.refresh()
        if (
            self.lesson is not lesson
            or self.position != position
            or not self.overlay.card.next_button.isEnabled()
        ):
            return
        if lesson.step.complete is None:
            self._skipped.discard(lesson.step.name)
        self._advance()

    def _advance(self) -> None:
        lesson = self.lesson
        if self.position == len(lesson.steps) - 1:
            self.stop()
            return
        self._visited.add(lesson.step.name)
        self.position += 1
        self._enter_step()
        # Navigation and its visible instructions change together. A queued
        # repaint left the old Next button acting on the newly entered step.
        self.refresh()

    def _enter_step(self) -> None:
        self._awaiting_outcome = False
        self.lesson.enter(self.position)

    def skip(self) -> None:
        if self.lesson is None:
            return
        self._skipped.add(self.lesson.step.name)
        self._advance()

    def revisit(self, name: str) -> None:
        if self.lesson is not None:
            self._verified.discard(name)
            self._skipped.discard(name)
            self.position = next(
                index
                for index, step in enumerate(self.lesson.steps)
                if step.name == name
            )
            self._enter_step()
            self.refresh()

    def previous(self) -> None:
        if self.lesson is not None and self.position > 0:
            self.position -= 1
            self._enter_step()
            self.refresh()

    def _recover(self) -> None:
        if self._recovery is not None:
            self._recovery()
            self.schedule_refresh()

    def schedule_refresh(self, *_args) -> None:
        if self.lesson is not None and not self._refresh_timer.isActive():
            self._refresh_timer.start(0)

    def refresh(self) -> None:
        lesson = self.lesson
        if lesson is None:
            return
        if not self._window.isVisible() or self._window.isMinimized():
            self.overlay.hide()
            return
        blocked = lesson.blocked()
        if self.lesson is None:  # A closed practice tab ends its lesson.
            return
        step = lesson.step
        highlights = ()
        self._recovery = None
        recovery_text = ""
        complete = False
        observed = False
        if blocked is not None:
            status, recovery_text, self._recovery = blocked
        else:
            target = step.target()
            highlights = (target,) if isinstance(target, Highlight) else target or ()
            if step.complete is not None:
                observed = step.complete()
                if observed:
                    self._verified.add(step.name)
                    self._skipped.discard(step.name)
                else:
                    self._awaiting_outcome = True
            complete = step.complete is None or step.name in self._verified
            skipped = step.name in self._skipped
            status = (
                (step.success if step.complete is not None else "Continue when ready.")
                if complete
                else (
                    "Working — waiting for the result."
                    if lesson.working()
                    else step.waiting
                )
            )
            if step.manual:
                status = "Manual step — check the outcome yourself, then Continue. This action is not automatically verified."
            if skipped:
                status = "Skipped — completion was not verified. You can continue or do this step now."
                complete = True
            hidden = tuple(item for item in highlights if item.visible_rect().isEmpty())
            if hidden:
                reveals = tuple(
                    item.reveal for item in hidden if item.reveal is not None
                )
                if reveals:

                    def reveal_hidden():
                        for reveal in reveals:
                            reveal()

                    self._recovery = reveal_hidden
                recovery_text = (
                    "Show This Control" if self._recovery is not None else ""
                )
                if complete and (step.complete is not None or skipped):
                    status += " The control is currently outside the visible area."
                else:
                    status = "The control is hidden or outside the visible area. Reveal it to continue."
                    complete = False
                highlights = tuple(item for item in highlights if item not in hidden)
        if self.position == len(lesson.steps) - 1 and self._skipped:
            skipped = count_text(len(self._skipped), "step").capitalize()
            status = (
                f"End of lesson. {skipped} {'was' if len(self._skipped) == 1 else 'were'} skipped "
                "without verification. Previous lets you revisit them."
            )
        self.overlay.present(
            highlights,
            title=step.title,
            body=step.instruction,
            progress=f"{lesson.title} · {self.position + 1} / {len(lesson.steps)}",
            status=status,
            complete=complete,
            previous=self.position > 0,
            skipped=step.name in self._skipped,
            final=self.position == len(lesson.steps) - 1,
            recovery=recovery_text,
            manual=step.manual,
            link=step.link,
        )
        if (
            self.overlay.card.auto_advance.isChecked()
            and step.complete is not None
            and not step.manual
            and observed
            and self._awaiting_outcome
            and step.name not in self._visited
            and step.name not in self._skipped
            and blocked is None
            and not self.overlay.collapsed
            and QtWidgets.QApplication.activeModalWidget() is None
            and QtWidgets.QApplication.activePopupWidget() is None
        ):
            self._advance()
