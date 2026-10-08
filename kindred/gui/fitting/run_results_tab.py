"""Physical Fit Results and Summary views.

The session owner retains the admitted Fit candidate and Apply policy. These
views retain only rendered text and immutable curve projections for display
and copying; actions remain ordinary requests to the owner.
"""

from __future__ import annotations

import math
import html
from collections.abc import Callable, Mapping, Sequence

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from kindred.core.analysis.fit_evaluation_outcome import ScientificUnavailable
from kindred.core.quantity_units import format_unit_for_display
from kindred.core.validation import six_digit_text
from kindred.gui.display_name_policy import (
    TAB_LABEL_MAX_CHARS,
    compact_dataset_label,
    dataset_alias,
)
from kindred.gui.fitting.reporting import (
    FitValueEntry, fit_candidate_presentation, fit_live_presentation, fit_overview,
)
from kindred.gui.fitting.launch import FitExecutionRequest, FitSetupSnapshot
from kindred.gui.ui_helpers import ControlRowsLayout
from kindred.gui.fitting.execution import FitCandidate, FitCurveDataset, FitLiveUpdate
from kindred.gui.widgets.grid_plot_view import GridPlotView

__all__ = ["FitTrackerPanel", "ResultsSummaryDialog", "RunResultsTab"]

ColorResolver = Callable[[str], object]



class _FitValueLabel(QtWidgets.QLabel):
    """Copy literal selections without the wrapping characters we inserted."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.copy_break_positions: list[int] = []

    def _copy_selection(self) -> None:
        if not self.hasSelectedText():
            return
        start = self.selectionStart()
        # QTextCursor and QLabel selection offsets count UTF-16 code units.
        selected = self.selectedText().encode("utf-16-le")
        end = start + len(selected) // 2
        for position in reversed(self.copy_break_positions):
            if start <= position < end:
                offset = (position - start) * 2
                selected = selected[:offset] + selected[offset + 2:]
        mime = QtCore.QMimeData()
        mime.setText(selected.decode("utf-16-le"))
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self._copy_selection()
            event.accept()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        if not self.copy_break_positions:
            super().contextMenuEvent(event)
            for menu in self.findChildren(
                QtWidgets.QMenu,
                options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly,
            ):
                if menu.isVisible():
                    menu.setToolTipsVisible(True)
                    copy, select_all = (
                        action for action in menu.actions() if not action.isSeparator()
                    )
                    copy.setToolTip("Copy the selected result text.")
                    select_all.setToolTip("Select all text in this result entry.")
            return
        menu = QtWidgets.QMenu()
        menu.setToolTipsVisible(True)
        copy = menu.addAction("Copy", self._copy_selection)
        copy.setEnabled(self.hasSelectedText())
        copy.setToolTip("Copy the selected result text.")
        select_all = menu.addAction("Select All", self._select_all)
        select_all.setToolTip("Select all text in this result entry.")
        menu.exec(event.globalPos())

    def _select_all(self) -> None:
        document = QtGui.QTextDocument()
        document.setHtml(self.text())
        self.setSelection(0, document.characterCount() - 1)


def _cause_chain(cause: object) -> str:
    """Each message in a failure chain with its error code, outermost first."""
    parts: list[str] = []
    while cause is not None:
        code = getattr(cause, "code", None)
        parts.append(str(getattr(cause, "message", cause)) + (f" [{code}]" if code else ""))
        cause = getattr(cause, "cause", None)
    return " Caused by: ".join(parts)


class ResultsSummaryDialog(QtWidgets.QDialog):
    """A Summary popup with Results and Provenance tabs, each a selectable document."""

    def __init__(
        self,
        documents: tuple[str, str] = ("", ""),
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("Results Summary")
        self.resize(560, 480)
        layout = QtWidgets.QVBoxLayout(self)
        self._info_label = QtWidgets.QLabel("Review the most recent fit run.")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)
        self._tabs = QtWidgets.QTabWidget(self)
        self._browsers: list[QtWidgets.QTextBrowser] = []
        self._displayed_html = ["", ""]
        self._pending_html: list[str | None] = [None, None]
        for index, title in enumerate(("Results", "Provenance")):
            browser = QtWidgets.QTextBrowser(self)
            browser.setOpenLinks(False)
            browser.setOpenExternalLinks(False)
            browser.setObjectName(f"results_summary_{title.lower()}")
            browser.setReadOnly(True)
            browser.setWordWrapMode(QtGui.QTextOption.WrapMode.WordWrap)
            browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            browser.selectionChanged.connect(lambda index=index: self._flush_pending(index))
            browser.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            self._tabs.addTab(browser, title)
            self._browsers.append(browser)
        layout.addWidget(self._tabs, stretch=1)
        self.refresh(documents)

    def refresh(self, documents: tuple[str, str], *, reset: bool = False) -> None:
        """Keep each reader's position; defer only the latest update during a selection."""
        for index, document in enumerate(documents):
            document = document or "<p>No summary available.</p>"
            browser = self._browsers[index]
            if reset:
                self._pending_html[index] = None
                self._replace_document(index, document, preserve_scroll=False)
            elif document == self._displayed_html[index]:
                self._pending_html[index] = None
            elif browser.textCursor().hasSelection():
                self._pending_html[index] = document
            else:
                self._pending_html[index] = None
                self._replace_document(index, document, preserve_scroll=True)
        self._info_label.setText("An update is waiting. Clear the selection to show it."
                                 if any(item is not None for item in self._pending_html)
                                 else "Review the most recent fit run.")

    def _replace_document(self, index: int, document: str, *, preserve_scroll: bool) -> None:
        browser = self._browsers[index]
        horizontal = browser.horizontalScrollBar()
        vertical = browser.verticalScrollBar()
        positions = (horizontal.value(), vertical.value()) if preserve_scroll else (0, 0)
        # setHtml clears selection; block that signal during this synchronous replacement.
        with QtCore.QSignalBlocker(browser):
            browser.setHtml(document)
        self._displayed_html[index] = document
        horizontal.setValue(positions[0])
        vertical.setValue(positions[1])

    def _flush_pending(self, index: int) -> None:
        if not self._browsers[index].textCursor().hasSelection() and self._pending_html[index] is not None:
            self.refresh(tuple(
                pending if pending is not None else displayed
                for pending, displayed in zip(self._pending_html, self._displayed_html)
            ))


class FitTrackerPanel(QtWidgets.QWidget):
    """Wrapping fit facts and a bounded viewport of complete fitted values."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("global_fit_tracker_panel")
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self._overview: tuple[str, ...] = ()
        self._outcome = self._phase = self._model = ""
        self._fitted_value_count = self._dataset_count = self._calls = 0
        self._objective = "not evaluated"
        self._values_html = ""
        self._setup: FitSetupSnapshot | None = None
        self._entries: tuple[FitValueEntry, ...] = ()
        self._entry_labels: list[_FitValueLabel] = []
        self._entry_layout_key: tuple | None = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(2)
        self._summary_widget = QtWidgets.QWidget(self)
        self._summary_layout = ControlRowsLayout(self._summary_widget)
        self._summary_labels: list[QtWidgets.QLabel] = []
        layout.addWidget(self._summary_widget)
        self._values_scroll = QtWidgets.QScrollArea(self)
        self._values_scroll.setObjectName("global_fit_tracker_values")
        self._values_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._values_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._values_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._values_scroll.setStyleSheet("QScrollArea#global_fit_tracker_values { border: 0; padding: 0; margin: 0; }")
        self._values_scroll.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._values_scroll.setAccessibleName("Fitted values. Use the arrow or Page keys to scroll.")
        self._values_widget = QtWidgets.QWidget()
        self._values_layout = ControlRowsLayout(self._values_widget)
        self._values_layout.setSpacing(8)
        self._values_layout.setContentsMargins(0, 4, 0, 4)
        self._values_scroll.setWidget(self._values_widget)
        self._values_scroll.viewport().installEventFilter(self)
        layout.addWidget(self._values_scroll)
        self.clear()

    def clear(self) -> None:
        self._overview = ()
        self._outcome = self._phase = self._values_html = ""
        self._objective = "not evaluated"
        self._calls = 0
        self._set_entries(())
        self._layout_entries()
        for label in self._summary_labels:
            label.hide()
        self.hide()

    def start(self, setup: FitSetupSnapshot, dataset_count: int, fitted_value_count: int) -> None:
        self.clear()
        self._model, self._setup = setup.origin.tab_name, setup
        self._fitted_value_count, self._dataset_count = fitted_value_count, dataset_count
        self._values_scroll.verticalScrollBar().setValue(0)
        self.set_phase("Starting", "Waiting for the first complete candidate")

    def restart(self) -> None:
        """Start again with the setup and counts of the current run."""
        self.start(self._setup, self._dataset_count, self._fitted_value_count)

    def set_phase(self, phase: str, detail: str) -> None:
        self._phase = phase
        self._overview = fit_overview(phase, self._model, self._objective,
                                     self._dataset_count, self._fitted_value_count)
        self._outcome = detail
        self._render()

    def update_from_best(self, update: FitLiveUpdate, *, dataset_labels: Mapping[str, str],
                         initial_labels: Mapping[str, str]) -> None:
        entries, self._values_html = fit_live_presentation(update, self._setup, dataset_labels, initial_labels)
        self._set_entries(entries)
        self._objective = six_digit_text(update.cost)
        self._calls = update.iteration
        phase, detail = self._phase, self._outcome
        if phase in {"Starting", "Running"}:
            phase, detail = "Running", f"Objective calls {update.iteration} · Best complete candidate"
        self.set_phase(phase, detail)

    def set_summary(self, overview: tuple[str, ...], entries: tuple[FitValueEntry, ...]) -> None:
        self._overview, self._outcome = overview, ""
        self._phase = ""
        self._set_entries(entries)
        self._render()

    def summary_html(self) -> str:
        if not self._overview:
            return "<h2>Fit Ready</h2><p>Run a fit to obtain results.</p>"
        return (f"<h2>{html.escape(" | ".join(self._overview))}</h2><p>{html.escape(self._outcome)}</p>"
                f"<p>Objective calls: {self._calls}</p>" + self._values_html)

    def _set_entries(self, entries: tuple[FitValueEntry, ...]) -> None:
        same_identity = tuple(entry[0] for entry in entries) == tuple(entry[0] for entry in self._entries)
        self._entries = entries
        if not same_identity:
            self._entry_layout_key = None
            while self._values_layout.count():
                self._values_layout.takeAt(0).widget().deleteLater()
            self._entry_labels = []
            for _entry in entries:
                label = _FitValueLabel(self._values_widget)
                label.setTextFormat(Qt.TextFormat.RichText)
                label.setWordWrap(True)
                label.setMargin(0)
                label.setIndent(0)
                label.setStyleSheet("border: 0; padding: 0; margin: 0;")
                label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
                self._values_layout.addWidget(label)
                label.show()
                self._entry_labels.append(label)
        self._values_scroll.setVisible(bool(entries))

    def _layout_entries(self) -> None:
        if not self._entries:
            self._values_scroll.setFixedHeight(0)
            return
        scroll = self._values_scroll.verticalScrollBar()
        position = scroll.value()
        gutter = self._values_scroll.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent)
        width = max(1, self._values_scroll.maximumViewportSize().width() - gutter)
        key = (self._entries, width, gutter,
               tuple(label.font().toString() for label in self._entry_labels))
        if key == self._entry_layout_key:
            return
        self._entry_layout_key = key
        row_height = max(label.fontMetrics().lineSpacing() + 6 for label in self._entry_labels)
        for label, (_identity, symbol, plain_symbol, context, value, unit) in zip(self._entry_labels, self._entries):
            label.copy_break_positions.clear()
            metrics = label.fontMetrics()
            number = six_digit_text(value)
            suffix = "" if unit == "1" else " " + format_unit_for_display(unit)
            document = QtGui.QTextDocument()
            document.setDefaultFont(label.font())
            document.setDocumentMargin(0)
            document.setHtml(suffix)
            numeric_width = (
                metrics.horizontalAdvance("−8.88888e+308")
                + math.ceil(document.idealWidth())
            )
            document.setHtml(symbol + " = ")
            symbol_width = math.ceil(document.idealWidth()) + 4
            context_budget = max(0, min(metrics.horizontalAdvance("M" * 28), width - symbol_width - numeric_width - 24))
            compact = metrics.elidedText(" ".join(context.splitlines()), Qt.TextElideMode.ElideRight, context_budget)
            prefix = html.escape(compact) + " " if compact else ""
            numeric_html = ' = <span style="white-space: nowrap">' + html.escape(number) + suffix + '</span>'
            if symbol_width + numeric_width > width:
                # Rich text retains scientific formatting; explicit opportunities
                # let an authored unbroken name wrap without splitting its number.
                document.setHtml(symbol)
                cursor = QtGui.QTextCursor(document)
                while cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextCharacter):
                    label.copy_break_positions.append(cursor.position())
                    cursor.insertText("\u200b")
                cursor.insertHtml(numeric_html)
                label.setText(document.toHtml())
            else:
                label.setText(prefix + symbol + numeric_html)
            accessible_unit = f" {unit}" if unit != "1" else ""
            label.setAccessibleName(f"{context} {plain_symbol} = {number}{accessible_unit}")
            label.setToolTip(label.accessibleName())
            entry_width = min(width, metrics.horizontalAdvance(compact + (" " if compact else "")) + symbol_width + numeric_width)
            label.setFixedWidth(entry_width)
            label.setMinimumHeight(0)
            label.setMaximumHeight(16777215)
            label.setFixedHeight(max(row_height, label.heightForWidth(entry_width)))
        self._values_layout.invalidate()
        height = self._values_layout.heightForWidth(width)
        cap = 3 * row_height + 2 * self._values_layout.spacing() + 8
        self._values_widget.resize(width, height)
        self._values_layout.setGeometry(QtCore.QRect(0, 0, width, height))
        self._values_scroll.setFixedHeight(min(height, cap))
        scroll.setValue(position)

    def _render(self) -> None:
        if not self._overview:
            self.hide()
            return
        width = max(1, self.contentsRect().width() - 16)
        fragments = list(self._overview)
        model_index = len(fragments) - 1
        if self._phase and self._calls:
            fragments[2:2] = [f"Objective calls {self._calls}", "Best complete candidate"]
            model_index += 2
        if self._outcome and self._outcome != f"Objective calls {self._calls} · Best complete candidate":
            fragments.append(self._outcome)
        while len(self._summary_labels) < len(fragments):
            label = QtWidgets.QLabel(self._summary_widget)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setMargin(0)
            label.setIndent(0)
            label.setStyleSheet("border: 0; padding: 0; margin: 0;")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._summary_layout.addWidget(label)
            self._summary_labels.append(label)
        for index, label in enumerate(self._summary_labels):
            label.setVisible(index < len(fragments))
            if index >= len(fragments):
                continue
            text = fragments[index]
            metrics = label.fontMetrics()
            # Bind only the authored model context; decisive facts wrap intact.
            shown = metrics.elidedText(" ".join(text.splitlines()), Qt.TextElideMode.ElideRight,
                                      min(width, metrics.horizontalAdvance("M" * 24))) if index == model_index else text
            label.setText(shown)
            label.setAccessibleName(text)
            label.setToolTip(text)
            label.setFixedWidth(min(width, metrics.horizontalAdvance(shown) + 4))
            label.setMinimumHeight(0)
            label.setMaximumHeight(16777215)
            label.setFixedHeight(max(metrics.lineSpacing(), label.heightForWidth(label.width())))
        self._summary_widget.setAccessibleName(" | ".join(fragments))
        self._summary_layout.invalidate()
        height = self._summary_layout.heightForWidth(width)
        self._summary_widget.setFixedHeight(height)
        self._summary_layout.setGeometry(QtCore.QRect(0, 0, width, height))
        self.show()
        self._layout_entries()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self._values_scroll.viewport() and event.type() == QtCore.QEvent.Type.Resize:
            self._layout_entries()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._render()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.FontChange and hasattr(self, "_summary_widget"):
            self._render()


class RunResultsTab(QtWidgets.QWidget):
    """Physical Results action surface with no scientific custody."""

    copyAvailableChanged = Signal(bool)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        color_resolver: ColorResolver | None = None,
        observation_style: tuple[str, float, float],
    ) -> None:
        super().__init__(parent)
        self._observation_style = observation_style
        self._color_resolver = color_resolver
        self._summary_dialog: ResultsSummaryDialog | None = None
        self._summary_body = ""
        self._summary_provenance = ""
        self._progress = 0
        self._dark_mode = False
        self._copy_allowed = True
        self._curves: tuple[FitCurveDataset, ...] = ()
        self._dataset_tab_ids: list[tuple[str, str]] = []
        self._dataset_labels: dict[str, str] = {}
        self._dataset_plot_views: dict[tuple[str, str], GridPlotView] = {}
        self._all_datasets_plot_view: GridPlotView | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        self._tracker_panel = FitTrackerPanel(self)
        layout.addWidget(self._tracker_panel)
        self._empty_label = QtWidgets.QLabel("Run a fit to see results here.", self)
        self._empty_label.setWordWrap(True)
        layout.addWidget(self._empty_label)

        self._render_diagnostic_label = QtWidgets.QLabel(self)
        self._render_diagnostic_label.setWordWrap(True)
        self._render_diagnostic_label.hide()
        layout.addWidget(self._render_diagnostic_label)

        self._subtabs = QtWidgets.QTabBar(self)
        self._subtabs.setObjectName("global_fit_results_subtabs")
        self._subtabs.setDocumentMode(True)
        self._subtabs.setDrawBase(False)
        self._subtabs.hide()
        layout.addWidget(self._subtabs)
        self._subtab_stack = QtWidgets.QStackedWidget(self)
        self._subtab_stack.setObjectName("global_fit_results_subtab_stack")
        layout.addWidget(self._subtab_stack, stretch=1)
        self._subtabs.currentChanged.connect(self._on_subtab_changed)

    def set_live_best(self, update: FitLiveUpdate, *, dataset_labels: Mapping[str, str],
                      initial_labels: Mapping[str, str]) -> None:
        self._empty_label.hide()
        self._tracker_panel.update_from_best(update, dataset_labels=dataset_labels,
                                             initial_labels=initial_labels)
        self._refresh_summary()

    def set_live_curves(self, curves: Sequence[FitCurveDataset]) -> None:
        self._install_plot_projection(curves)
        self._refresh_summary()

    def set_terminal_presentation(
        self,
        candidate: FitCandidate | None,
        *,
        curves: Sequence[FitCurveDataset] = (),
        curve_failure: str | None = None,
    ) -> bool:
        """Consume one call-local typed terminal presentation."""
        if candidate is None:
            self._summary_body = self._summary_provenance = ""
            self._tracker_panel.clear()
            self._render_diagnostic_label.clear()
            self._render_diagnostic_label.hide()
        else:
            overview, self._summary_body, self._summary_provenance, entries = fit_candidate_presentation(candidate)
            self._tracker_panel.set_summary(overview, entries)
        self._install_plot_projection(curves)
        self._refresh_summary()
        if curve_failure is not None:
            self._render_diagnostic_label.setText(
                "Fit curves unavailable: " + str(curve_failure)
            )
            self._render_diagnostic_label.show()
        return self.copy_eligible()

    def set_progress(self, value: int) -> None:
        self._progress = value
        self._refresh_summary()

    def _summary_html(self) -> tuple[str, str]:
        """The Results and Provenance documents."""
        body = self._summary_body or self._tracker_panel.summary_html()
        if not self._summary_body:
            body += f"<p>Progress: {self._progress}%</p>"
        return body, self._summary_provenance

    def _refresh_summary(self) -> None:
        if self._summary_dialog is not None and self._summary_dialog.isVisible():
            self._summary_dialog.refresh(self._summary_html())

    def render_results_summary(self) -> None:
        """Show the current physical execution/terminal projection in one document."""
        if self._summary_dialog is not None and self._summary_dialog.isVisible():
            self._refresh_summary()
            self._summary_dialog.raise_()
            self._summary_dialog.activateWindow()
            return
        dialog = ResultsSummaryDialog(self._summary_html(), parent=self)
        dialog.destroyed.connect(
            lambda: setattr(self, "_summary_dialog", None)
            if self._summary_dialog is dialog else None
        )
        self._summary_dialog = dialog
        dialog.show()

    def clear_candidate(self, *, keep_plots: bool = False) -> None:
        """Clear only physical result widgets; the owner controls custody."""

        self._tracker_panel.clear()
        self._summary_body = self._summary_provenance = ""
        self._empty_label.setText("Run a fit to see results here.")
        self._empty_label.show()
        self._render_diagnostic_label.clear()
        self._render_diagnostic_label.hide()
        self._install_plot_projection(())
        if not keep_plots:
            self._dataset_tab_ids = []
            self._dataset_labels = {}
            self._clear_subtabs()
        self._refresh_summary()

    def set_terminal_failure(self, failure: object) -> None:
        self._empty_label.hide()
        self._tracker_panel.clear()
        dataset_id = getattr(failure, "details", {}).get("dataset")
        dataset_label = self._dataset_labels.get(dataset_id, dataset_id)
        if dataset_label is not None and tuple(self._dataset_labels.values()).count(dataset_label) > 1:
            dataset_label = f"{dataset_label} ({dataset_id})"
        caption = f"Dataset: {dataset_label}" if dataset_label is not None else ""
        message = getattr(failure, "message", str(failure))
        self.set_execution_status("Failed", f"{caption}: {message}" if caption else message)
        self._summary_body = (
            "<h2>Fit Failed</h2>" + (f"<p>{html.escape(caption)}</p>" if caption else "")
            + "<p>" + html.escape(_cause_chain(failure)).replace("\n", "<br>") + "</p>"
        )
        self._summary_provenance = ""
        self._refresh_summary()

    def set_terminal_unavailable(self, cause: ScientificUnavailable) -> None:
        self._empty_label.hide()
        self._tracker_panel.clear()
        self.set_execution_status("No valid candidate found",
                                  cause.factual_cause.message if cause.factual_cause is not None else cause.stage)
        self._summary_body = ("<h2>No Valid Candidate Found</h2><p>"
                              + html.escape(_cause_chain(cause.factual_cause)).replace("\n", "<br>") + "</p>")
        self._summary_provenance = ""
        self._refresh_summary()

    def copy_eligible(self) -> bool:
        return self._copy_allowed and bool(self._curves)

    def set_copy_allowed(self, allowed: bool) -> None:
        self._copy_allowed = bool(allowed)
        for plot_view in self._dataset_plot_views.values():
            plot_view.set_copy_allowed(self._copy_allowed)
        if self._all_datasets_plot_view is not None:
            self._all_datasets_plot_view.set_copy_allowed(self._copy_allowed)
        self.copyAvailableChanged.emit(self.copy_eligible())

    def copy_active_to_clipboard(self) -> None:
        """Copy only the dataset scope represented by the active Results tab."""

        view_id = self._active_plot_view_id()
        if view_id is None:
            raise ValueError("No Fit curves are available to copy.")
        if view_id[0] == "aggregate":
            view = self._all_datasets_plot_view
            curves = self._curves
        else:
            view = self._dataset_plot_views.get(view_id)
            curves = tuple(
                curve for curve in self._curves if curve.dataset_id == view_id[1]
            )
        if view is None or not curves:
            raise ValueError("No Fit curves are available to copy.")
        view.copy_projection_to_clipboard(curves)

    def copy_dataset_to_clipboard(self, dataset_id: str) -> None:
        """Copy the one dataset represented by a clicked Fit plot."""

        identity = str(dataset_id or "").strip()
        curves = tuple(
            curve for curve in self._curves if curve.dataset_id == identity
        )
        view = self._subtab_stack.currentWidget()
        if view is None or len(curves) != 1:
            raise ValueError("The selected Fit dataset has no curves to copy.")
        view.copy_projection_to_clipboard(curves)

    def set_observation_style(self, shape: str, size: float, opacity: float) -> None:
        self._observation_style = (shape, size, opacity)
        for plot_view in self._dataset_plot_views.values():
            plot_view.set_observation_style(shape, size, opacity)
        if self._all_datasets_plot_view is not None:
            self._all_datasets_plot_view.set_observation_style(shape, size, opacity)
        self._refresh_summary()

    def set_waiting(self, request: FitExecutionRequest) -> None:
        self.clear_candidate(keep_plots=True)
        self._progress = 0
        self._tracker_panel.start(
            request.setup,
            len(request.dispatch.included_dataset_ids),
            len(request.dispatch.ordinary_optimizer_layout.opt_param_keys()),
        )
        if self._summary_dialog is not None and self._summary_dialog.isVisible():
            self._summary_dialog.refresh(self._summary_html(), reset=True)

    def restart_waiting(self) -> None:
        """Wait again for the running request the tracker already holds."""
        self.clear_candidate(keep_plots=True)
        self._progress = 0
        self._tracker_panel.restart()
        if self._summary_dialog is not None and self._summary_dialog.isVisible():
            self._summary_dialog.refresh(self._summary_html(), reset=True)

    def set_execution_status(self, phase: str, detail: str) -> None:
        self._empty_label.hide()
        self._tracker_panel.set_phase(phase, detail)
        self._refresh_summary()

    def set_dark_mode(self, enabled: bool) -> None:
        self._dark_mode = bool(enabled)
        for plot_view in self._dataset_plot_views.values():
            plot_view.set_dark_mode(self._dark_mode)
        if self._all_datasets_plot_view is not None:
            self._all_datasets_plot_view.set_dark_mode(self._dark_mode)
        self._refresh_summary()

    def prepare(self, request: FitExecutionRequest) -> None:
        self._ensure_subtabs(request)
        ids = request.dispatch.included_dataset_ids
        for identity in self._dataset_tab_ids + ([("aggregate", "all")] if len(ids) > 1 else []):
            aggregate = identity[0] == "aggregate"
            view = self._all_datasets_plot_view if aggregate else self._dataset_plot_views.get(identity)
            if view is None:
                view = self._create_plot_view(f"global_fit_results_plot_{identity[1]}")
                if aggregate:
                    self._all_datasets_plot_view = view
                else:
                    self._dataset_plot_views[identity] = view
                self._subtab_stack.addWidget(view)
            view.prepare(request, ids if aggregate else (identity[1],))
        self._empty_label.hide()
        for index in range(self._subtabs.count()):
            if self._subtabs.tabData(index)["view_identity"] == ("aggregate", "all"):
                self._subtabs.setCurrentIndex(index)
                break
        self._refresh_active_plot_view()

    def finish_preparation(self) -> None:
        self.layout().activate()
        size = self._subtab_stack.contentsRect().size()
        for index in range(self._subtab_stack.count()):
            view = self._subtab_stack.widget(index)
            view.resize(size)
            view.finish_preparation()
        # Paint the selected, displayed canvas before the owner queues launch.
        view = self._subtab_stack.currentWidget()
        if view is not None:
            view._canvas.viewport().repaint()

    def _ensure_subtabs(self, request: FitExecutionRequest) -> None:
        mapped = {item.dataset_id: item for item in request.setup.origin.mapped_datasets}
        curves = tuple(mapped[identity] for identity in request.dispatch.included_dataset_ids)
        ids = [("dataset", curve.dataset_id) for curve in curves]
        labels = {curve.dataset_id: curve.display_name for curve in curves}
        if ids == self._dataset_tab_ids and labels == self._dataset_labels:
            return
        selected = self._active_plot_view_id()
        self._dataset_tab_ids = ids
        self._dataset_labels = labels
        for identity in tuple(self._dataset_plot_views):
            if identity not in ids:
                self._remove_plot_view(self._dataset_plot_views.pop(identity))
        if len(curves) < 2 and self._all_datasets_plot_view is not None:
            self._remove_plot_view(self._all_datasets_plot_view)
            self._all_datasets_plot_view = None
        blocker = QtCore.QSignalBlocker(self._subtabs)
        while self._subtabs.count():
            self._subtabs.removeTab(self._subtabs.count() - 1)
        label_counts: dict[str, int] = {}
        for label in labels.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        for index, (curve, view_identity) in enumerate(zip(curves, ids, strict=True)):
            full_label = curve.display_name
            if label_counts[full_label] > 1:
                full_label = f"{full_label} ({curve.dataset_id})"
            compact_label = compact_dataset_label(full_label, max_chars=TAB_LABEL_MAX_CHARS)
            tab_index = self._subtabs.addTab(compact_label.display)
            self._subtabs.setTabToolTip(
                tab_index, f"{compact_label.full}\nDataset ID: {curve.dataset_id}"
            )
            self._subtabs.setTabData(
                tab_index,
                {
                    "view_identity": view_identity,
                    "alias": dataset_alias(index),
                    "full_label": compact_label.full,
                },
            )
        if len(curves) > 1:
            index = self._subtabs.addTab("All Datasets")
            self._subtabs.setTabData(index, {"view_identity": ("aggregate", "all")})
        self._subtabs.setVisible(len(curves) > 1)
        identities = ids + ([("aggregate", "all")] if len(curves) > 1 else [])
        self._subtabs.setCurrentIndex(identities.index(selected) if selected in identities else 0)
        del blocker

    def _create_plot_view(self, object_name: str) -> GridPlotView:
        plot_view = GridPlotView(
            color_resolver=self._color_resolver,
            observation_style=self._observation_style,
            parent=self,
        )
        plot_view.setObjectName(object_name)
        plot_view.copyFitDataRequested.connect(self.copy_dataset_to_clipboard)
        plot_view.set_dark_mode(self._dark_mode)
        plot_view.set_copy_allowed(self._copy_allowed)
        return plot_view

    def _install_plot_projection(self, curves: Sequence[FitCurveDataset]) -> None:
        values = tuple(curves)
        ids = [item.dataset_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("Fit curve datasets must have unique dataset IDs.")
        for index in range(self._subtab_stack.count()):
            self._subtab_stack.widget(index).cancel_copy()
        self._curves = values
        if values:
            self._empty_label.hide()
            self._refresh_active_plot_view()
        else:
            for index in range(self._subtab_stack.count()):
                self._subtab_stack.widget(index).set_projection(())
        self.copyAvailableChanged.emit(self.copy_eligible())

    def _active_plot_view_id(self) -> tuple[str, str] | None:
        index = self._subtabs.currentIndex()
        if index < 0:
            return None
        return self._subtabs.tabData(index)["view_identity"]

    def _refresh_active_plot_view(self) -> None:
        view_id = self._active_plot_view_id()
        if view_id is not None:
            self._refresh_plot_view(view_id)

    def _refresh_plot_view(self, view_id: tuple[str, str]) -> None:
        if view_id[0] == "aggregate":
            plot_view = self._all_datasets_plot_view
            curves = self._curves
        else:
            plot_view = self._dataset_plot_views.get(view_id)
            curves = tuple(
                curve for curve in self._curves if curve.dataset_id == view_id[1]
            )
        if plot_view is None:
            return
        self._subtab_stack.setCurrentWidget(plot_view)
        plot_view.set_projection(curves)
        self.copyAvailableChanged.emit(self.copy_eligible())

    def _on_subtab_changed(self, index: int) -> None:
        self._refresh_active_plot_view()
        self._refresh_summary()
        self.copyAvailableChanged.emit(self.copy_eligible())

    def _clear_subtabs(self) -> None:
        with QtCore.QSignalBlocker(self._subtabs):
            while self._subtabs.count():
                self._subtabs.removeTab(self._subtabs.count() - 1)
        while self._subtab_stack.count():
            self._remove_plot_view(self._subtab_stack.widget(0))
        self._dataset_plot_views.clear()
        self._all_datasets_plot_view = None
        self._subtabs.hide()

    def _remove_plot_view(self, view: GridPlotView) -> None:
        view.set_projection(())
        self._subtab_stack.removeWidget(view)
        view.deleteLater()
