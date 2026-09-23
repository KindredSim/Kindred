from __future__ import annotations

import html
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.symbolic_calculator_owner import SymbolicCalculatorViewState
from kindred.gui.ui_helpers import clamp_initial_size
from kindred.core.symbolic.calculator_help import calculator_help_html


class _CalculatorResultView(QtWidgets.QTextBrowser):
    def minimumSizeHint(self) -> QtCore.QSize:
        size = super().minimumSizeHint()
        size.setHeight(
            3 * self.fontMetrics().lineSpacing()
            + 2 * self.frameWidth()
            + int(2 * self.document().documentMargin())
        )
        return size


class SymbolicCalculatorPanel(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        on_evaluate: Callable[[str], SymbolicCalculatorViewState],
        on_copy_result: Callable[[], str],
        on_copy_context: Callable[[], str],
        on_use_result: Callable[[str, int], str],
        on_cancel: Callable[[], SymbolicCalculatorViewState],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("symbolicCalculatorPanel")
        self._on_evaluate = on_evaluate
        self._on_copy_result = on_copy_result
        self._on_copy_context = on_copy_context
        self._on_use_result = on_use_result
        self._on_cancel = on_cancel
        self._latest_query = ""
        self._latest_is_error = False
        self._help_dialog: QtWidgets.QDialog | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._status_label = QtWidgets.QLabel("", self)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        query_row = QtWidgets.QHBoxLayout()
        query_row.setSpacing(6)
        self._query_edit = QtWidgets.QLineEdit(self)
        self._query_edit.setObjectName("symbolicCalculatorQueryEdit")
        self._query_edit.setPlaceholderText("dA/dt, dA/dB, odes(), jacobian(), factor(dA/dt)")
        self._query_edit.returnPressed.connect(self.evaluate_current_query)
        query_row.addWidget(self._query_edit, stretch=1)
        self._evaluate_button = QtWidgets.QPushButton("Evaluate", self)
        self._evaluate_button.setObjectName("symbolicCalculatorEvaluateButton")
        self._evaluate_button.setToolTip(
            "Calculate your expression using the reactions and definitions last applied in this tab."
        )
        self._evaluate_button.clicked.connect(self.evaluate_current_query)
        query_row.addWidget(self._evaluate_button)
        self._cancel_button = QtWidgets.QPushButton("Cancel", self)
        self._cancel_button.setObjectName("symbolicCalculatorCancelButton")
        self._cancel_button.setToolTip("Stop the calculation that is running.")
        self._cancel_button.clicked.connect(self.cancel_current_query)
        query_row.addWidget(self._cancel_button)
        layout.addLayout(query_row)

        self._result_view = _CalculatorResultView(self)
        self._result_view.setObjectName("symbolicCalculatorResultView")
        self._result_view.setReadOnly(True)
        self._result_view.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        result_font = QtGui.QFont("Courier New", 10)
        self._result_view.setFont(result_font)
        layout.addWidget(self._result_view, stretch=1)

        use_row = QtWidgets.QHBoxLayout()
        self._solution_combo = QtWidgets.QComboBox(self)
        self._solution_combo.setObjectName("symbolicCalculatorSolutionCombo")
        self._solution_combo.setToolTip("Choose the answer you want to work with next.")
        self._solution_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._solution_combo.setMinimumContentsLength(20)
        use_row.addWidget(self._solution_combo, stretch=1)
        self._use_button = QtWidgets.QPushButton("Use result", self)
        self._use_button.setObjectName("symbolicCalculatorUseButton")
        self._use_button.clicked.connect(self.use_result)
        use_row.addWidget(self._use_button)
        layout.addLayout(use_row)

        copy_row = QtWidgets.QHBoxLayout()
        copy_row.setSpacing(6)
        self._copy_result_button = QtWidgets.QPushButton("Copy Result", self)
        self._copy_result_button.setObjectName("symbolicCalculatorCopyResultButton")
        self._copy_result_button.setToolTip(
            "Copy every answer, including conditions and explanations, to the clipboard. "
            "Your selection does not limit what gets copied."
        )
        self._copy_result_button.clicked.connect(self.copy_compact_result)
        copy_row.addWidget(self._copy_result_button)
        self._copy_context_button = QtWidgets.QPushButton("Copy Context", self)
        self._copy_context_button.setObjectName("symbolicCalculatorCopyContextButton")
        self._copy_context_button.setToolTip(
            "Copy the results together with your expression, reactions, definitions "
            "and the assumptions used in the calculation."
        )
        self._copy_context_button.clicked.connect(self.copy_full_context)
        copy_row.addWidget(self._copy_context_button)
        self._help_button = QtWidgets.QPushButton("Help", self)
        self._help_button.setObjectName("symbolicCalculatorHelpButton")
        self._help_button.setToolTip("Find commands and worked examples for using Calculator.")
        self._help_button.clicked.connect(self.show_help)
        copy_row.addWidget(self._help_button)
        copy_row.addStretch(1)
        layout.addLayout(copy_row)

        self.render_state(
            SymbolicCalculatorViewState(
                available=False,
                reason="Canonical mechanism unavailable.",
            )
        )

    def inputs_enabled(self) -> bool:
        return bool(self._query_edit.isEnabled() and self._evaluate_button.isEnabled())

    def render_state(self, state: SymbolicCalculatorViewState) -> None:
        available = bool(state.available)
        previous_index = self._solution_combo.currentIndex() if self._latest_query == state.latest_query else 0
        # An unavailable current mechanism has no result.  Do not let a
        # stale or malformed view state make a previous source look current.
        self._latest_query = state.latest_query if available else ""
        self._latest_is_error = state.latest_is_error if available else False
        self._query_edit.setEnabled(available and not state.busy)
        self._evaluate_button.setEnabled(available and not state.busy)
        self._cancel_button.setEnabled(state.can_cancel)
        self._copy_result_button.setEnabled(available and state.can_copy_result)
        self._copy_context_button.setEnabled(available and state.can_copy_context)
        self._solution_combo.clear()
        self._solution_combo.addItems([label for label, _ in state.reuse_queries])
        self._solution_combo.setCurrentIndex(max(0, min(previous_index, len(state.reuse_queries)-1)))
        self._solution_combo.setVisible(len(state.reuse_queries) > 1)
        self._use_button.setVisible(bool(state.reuse_queries))
        self._use_button.setEnabled(available and not state.busy and bool(state.reuse_queries))
        self._use_button.setText("Use in query" if len(state.reuse_queries) > 1 else "Use result")
        self._use_button.setToolTip(
            "Put a calculation for the selected answer in the input box so you can build on it. "
            "Click Evaluate when you’re ready."
            if len(state.reuse_queries) > 1 else
            "Put this calculation back in the input box so you can edit it. "
            "Click Evaluate when you’re ready."
        )
        reason = state.reason
        self._status_label.setText(state.status if available else reason)
        if state.latest_result_html:
            prefix = "Error" if self._latest_is_error else "Result"
            query = self._latest_query
            self._result_view.setHtml(
                "<pre>" + html.escape(f"{query}\n{prefix}:") + "</pre>"
                + state.latest_result_html
            )
        else:
            self._result_view.clear()

    @QtCore.Slot()
    def use_result(self) -> None:
        query = self._on_use_result(self._latest_query, max(0, self._solution_combo.currentIndex()))
        if query:
            self._query_edit.setText(query)
            self._query_edit.setFocus()

    @QtCore.Slot()
    def show_help(self) -> None:
        if self._help_dialog is None:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Symbolic Calculator Help")
            layout = QtWidgets.QVBoxLayout(dialog)
            browser = QtWidgets.QTextBrowser(dialog)
            browser.setObjectName("symbolicCalculatorHelpText")
            browser.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard)
            browser.document().setDocumentMargin(16)
            browser.document().setDefaultStyleSheet("h2 {margin-top: 24px;} p, pre {margin-top: 10px; margin-bottom: 10px;} pre {white-space: pre-wrap;}")
            browser.setHtml(calculator_help_html())
            layout.addWidget(browser)
            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close, parent=dialog)
            buttons.rejected.connect(dialog.hide)
            layout.addWidget(buttons)
            clamp_initial_size(dialog, 850, 700)
            self._help_dialog = dialog
        self._help_dialog.show()
        self._help_dialog.raise_()
        self._help_dialog.activateWindow()

    @QtCore.Slot()
    def evaluate_current_query(self) -> None:
        if not self.inputs_enabled():
            return
        query = str(self._query_edit.text() or "").strip()
        if not query:
            return
        self.render_state(self._on_evaluate(query))

    @QtCore.Slot()
    def cancel_current_query(self) -> None:
        self.render_state(self._on_cancel())

    @QtCore.Slot()
    def copy_compact_result(self) -> None:
        text = str(self._on_copy_result() or "")
        if not text:
            return
        QtWidgets.QApplication.clipboard().setText(text)

    @QtCore.Slot()
    def copy_full_context(self) -> None:
        text = str(self._on_copy_context() or "")
        if not text:
            return
        QtWidgets.QApplication.clipboard().setText(text)
