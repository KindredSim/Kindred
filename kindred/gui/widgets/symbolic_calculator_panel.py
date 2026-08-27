from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.symbolic_calculator_owner import SymbolicCalculatorViewState


class SymbolicCalculatorPanel(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        on_evaluate: Callable[[str], SymbolicCalculatorViewState],
        on_copy_result: Callable[[], str],
        on_copy_context: Callable[[], str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("symbolicCalculatorPanel")
        self._on_evaluate = on_evaluate
        self._on_copy_result = on_copy_result
        self._on_copy_context = on_copy_context
        self._latest_query = ""
        self._latest_result_text = ""
        self._latest_is_error = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QtWidgets.QLabel("Symbolic Calculator", self)
        header_font = header.font()
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

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
        self._evaluate_button.clicked.connect(self.evaluate_current_query)
        query_row.addWidget(self._evaluate_button)
        layout.addLayout(query_row)

        self._result_view = QtWidgets.QPlainTextEdit(self)
        self._result_view.setObjectName("symbolicCalculatorResultView")
        self._result_view.setReadOnly(True)
        self._result_view.setMinimumHeight(120)
        result_font = QtGui.QFont("Courier New", 10)
        self._result_view.setFont(result_font)
        layout.addWidget(self._result_view, stretch=1)

        copy_row = QtWidgets.QHBoxLayout()
        copy_row.setSpacing(6)
        self._copy_result_button = QtWidgets.QPushButton("Copy Result", self)
        self._copy_result_button.setObjectName("symbolicCalculatorCopyResultButton")
        self._copy_result_button.clicked.connect(self.copy_compact_result)
        copy_row.addWidget(self._copy_result_button)
        self._copy_context_button = QtWidgets.QPushButton("Copy Context", self)
        self._copy_context_button.setObjectName("symbolicCalculatorCopyContextButton")
        self._copy_context_button.clicked.connect(self.copy_full_context)
        copy_row.addWidget(self._copy_context_button)
        copy_row.addStretch(1)
        layout.addLayout(copy_row)

        examples = QtWidgets.QLabel(
            "Supported: dA/dt, dA/dB, d(dA/dt)/dB, odes(), jacobian(), "
            "simplify, factor, expand, cancel, collect, expand_params",
            self,
        )
        examples.setObjectName("symbolicCalculatorExamplesLabel")
        examples.setWordWrap(True)
        layout.addWidget(examples)

        self.render_state(
            SymbolicCalculatorViewState(
                available=False,
                reason="Canonical mechanism unavailable.",
            )
        )

    def inputs_enabled(self) -> bool:
        return bool(self._query_edit.isEnabled() and self._evaluate_button.isEnabled())

    def render_state(self, state: SymbolicCalculatorViewState) -> None:
        if not isinstance(state, SymbolicCalculatorViewState):
            self._render_unavailable_after_invalid_state()
            raise TypeError("state must be SymbolicCalculatorViewState.")
        available = bool(state.available)
        # An unavailable current mechanism has no result.  Do not let a
        # stale or malformed view state make a previous source look current.
        self._latest_query = state.latest_query if available else ""
        self._latest_result_text = state.latest_result_text if available else ""
        self._latest_is_error = state.latest_is_error if available else False
        self._query_edit.setEnabled(available)
        self._evaluate_button.setEnabled(available)
        self._copy_result_button.setEnabled(available and state.can_copy_result)
        self._copy_context_button.setEnabled(available and state.can_copy_context)
        reason = state.reason
        self._status_label.setText("" if available else reason)
        if self._latest_result_text:
            prefix = "Error" if self._latest_is_error else "Result"
            query = self._latest_query
            self._result_view.setPlainText(f"{query}\n{prefix}: {self._latest_result_text}")
        else:
            self._result_view.clear()

    def _render_unavailable_after_invalid_state(self) -> None:
        self._latest_query = ""
        self._latest_result_text = ""
        self._latest_is_error = False
        self._query_edit.setEnabled(False)
        self._evaluate_button.setEnabled(False)
        self._copy_result_button.setEnabled(False)
        self._copy_context_button.setEnabled(False)
        self._status_label.setText("Symbolic calculator unavailable.")
        self._result_view.clear()

    @QtCore.Slot()
    def evaluate_current_query(self) -> None:
        if not self.inputs_enabled():
            return
        query = str(self._query_edit.text() or "").strip()
        if not query:
            return
        self.render_state(self._on_evaluate(query))

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
