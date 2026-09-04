"""Constrained dependent-parameter chooser for exact source correction."""

from __future__ import annotations

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    ExactCorrectionAssessment,
    select_exact_correction_basis,
)
from kindred.core.simulator.source_correction import (
    CorrectionPlanner,
    CorrectionTarget,
    build_correction_expressions,
)


class ExactCorrectionBasisDialog(QtWidgets.QDialog):
    """Keep every reachable checkbox state extendible to one exact basis."""

    def __init__(
        self,
        *,
        assessment: ExactCorrectionAssessment,
        visible_source: MechanismAuthoringSource,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose dependent parameters")
        self.setObjectName("exact_correction_basis_dialog")
        self._assessment = assessment
        self._visible_source = visible_source
        self._planner = CorrectionPlanner(assessment.composition, assessment.targets)
        self._targets = {target.name: target for target in assessment.targets}
        self._selected = set(assessment.plan.basis)
        self._selected_assessment = assessment

        layout = QtWidgets.QVBoxLayout(self)
        required = self._planner.required_basis_size
        instruction = QtWidgets.QLabel(
            f"Select {required} dependent parameter"
            + ("." if required == 1 else "s.")
            + " The remaining parameters stay independent.",
            self,
        )
        instruction.setObjectName("exact_correction_required_count")
        layout.addWidget(instruction)

        self.target_list = QtWidgets.QListWidget(self)
        self.target_list.setObjectName("exact_correction_target_list")
        self.target_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        for target in assessment.targets:
            item = QtWidgets.QListWidgetItem(f"{target.name} ({target.family})")
            item.setData(Qt.ItemDataRole.UserRole, target.name)
            item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if target.name in self._selected
                else Qt.CheckState.Unchecked
            )
            self.target_list.addItem(item)
        layout.addWidget(self.target_list)

        layout.addWidget(QtWidgets.QLabel("Generated parameter equations:", self))
        self.equations = QtWidgets.QPlainTextEdit(self)
        self.equations.setObjectName("exact_correction_generated_equations")
        self.equations.setReadOnly(True)
        self.equations.setMaximumBlockCount(max(1, required))
        layout.addWidget(self.equations)

        self.buttons = QtWidgets.QDialogButtonBox(self)
        self.buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = self.buttons.addButton(
            "Apply selection",
            QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self.apply_button.setObjectName("exact_correction_apply_selection")
        layout.addWidget(self.buttons)
        self.buttons.rejected.connect(self.reject)
        self.buttons.accepted.connect(self.accept)
        self.target_list.itemChanged.connect(self._on_item_changed)
        self._refresh_state()

    @property
    def selected_assessment(self) -> ExactCorrectionAssessment:
        return self._selected_assessment

    def _selection_in_target_order(self) -> tuple[str, ...]:
        return tuple(
            target.name
            for target in self._assessment.targets
            if target.name in self._selected
        )

    @staticmethod
    def _intrinsic_unavailability(target: CorrectionTarget) -> str:
        reasons: list[str] = []
        if not target.source_backed:
            reasons.append("it has no editable source coordinate")
        elif target.locked:
            reasons.append("it is locked")
        elif not target.editable:
            reasons.append("its authored coordinate is not editable")
        if target.derived:
            reasons.append("it is already derived")
        detail = ", ".join(reasons) or "it is not an admitted correction target"
        return f"Cannot be selected because {detail}."

    @staticmethod
    def _set_description(item: QtWidgets.QListWidgetItem, description: str) -> None:
        item.setToolTip(description)
        item.setData(Qt.ItemDataRole.AccessibleDescriptionRole, description)

    def _refresh_state(self) -> None:
        selected = self._selection_in_target_order()
        viable = set(self._planner.viable_next_targets(selected))
        self.target_list.blockSignals(True)
        try:
            for row in range(self.target_list.count()):
                item = self.target_list.item(row)
                name = item.data(Qt.ItemDataRole.UserRole)
                target = self._targets[name]
                checked = name in self._selected
                enabled = checked or name in viable
                flags = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable
                if enabled:
                    flags |= Qt.ItemFlag.ItemIsEnabled
                item.setFlags(flags)
                item.setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
                if checked:
                    description = "Selected as a dependent parameter; uncheck to change the basis."
                elif not target.eligible:
                    description = self._intrinsic_unavailability(target)
                elif enabled:
                    description = (
                        "Can be selected while retaining a valid full-rank completion."
                    )
                else:
                    description = (
                        "Cannot be selected with the current choices because no valid "
                        "full-rank basis would remain."
                    )
                self._set_description(item, description)
        finally:
            self.target_list.blockSignals(False)

        plan = self._planner.plan_selected_basis(selected)
        if plan.available:
            self._selected_assessment = select_exact_correction_basis(
                self._assessment,
                self._visible_source,
                selected,
            )
            expressions = build_correction_expressions(
                self._assessment.composition,
                plan,
            )
            self.equations.setPlainText(
                "\n".join(
                    f"param {name} = {expression}"
                    for name, expression in expressions.items()
                )
            )
        else:
            remaining = self._planner.required_basis_size - len(selected)
            self.equations.setPlainText(
                f"Select {remaining} more dependent parameter"
                + ("." if remaining == 1 else "s.")
            )
        self.apply_button.setEnabled(plan.available)

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        name = item.data(Qt.ItemDataRole.UserRole)
        selected = set(self._selected)
        if item.checkState() == Qt.CheckState.Checked:
            selected.add(name)
        else:
            selected.discard(name)
        ordered = tuple(
            target.name
            for target in self._assessment.targets
            if target.name in selected
        )
        if not self._planner.partial_selection_can_complete(ordered):
            self.target_list.blockSignals(True)
            try:
                item.setCheckState(
                    Qt.CheckState.Checked
                    if name in self._selected
                    else Qt.CheckState.Unchecked
                )
            finally:
                self.target_list.blockSignals(False)
            return
        self._selected = selected
        self._refresh_state()
