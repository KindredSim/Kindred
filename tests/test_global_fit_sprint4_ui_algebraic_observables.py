import pytest

from kindred.gui.fitting.parameters_ics_tab import _AddFittableParameterDialog


pytestmark = [pytest.mark.gui]

M16_ALGEBRA_SNIPPET = "\n".join(
    [
        "let total_PBMP = [PBMPBPIN] + [PBMP]",
        "let selectivity = [PBMP] / max([PBMP] + [pinBOH], 1e-18)",
    ]
)


def test_add_parameter_dialog_has_algebraic_observables_tab(qt_app):
    dialog = _AddFittableParameterDialog(
        available_rates=["k1"],
        available_scalars=["a"],
        available_species=["A"],
        dataset_ids=["ds1", "ds2"],
        available_observables={"total_PBMP": "[PBMPBPIN] + [PBMP]"},
        parent=None,
    )
    try:
        titles = [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())]
        assert "Algebraic Observables" in titles
    finally:
        dialog.close()


def test_add_parameter_dialog_observable_dropdown_detects_existing(qt_app):
    from kindred.core.algebra.observable_introspection import extract_observables_from_algebra_text

    dialog = _AddFittableParameterDialog(
        available_rates=["k1"],
        available_scalars=["a"],
        available_species=["A"],
        dataset_ids=["ds1"],
        available_observables=extract_observables_from_algebra_text(M16_ALGEBRA_SNIPPET),
        parent=None,
    )
    try:
        titles = [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())]
        obs_idx = titles.index("Algebraic Observables")
        dialog._tabs.setCurrentIndex(obs_idx)
        items = [dialog._observable_combo.itemText(i) for i in range(dialog._observable_combo.count())]
        assert "total_PBMP" in items
        assert "selectivity" in items
        dialog._observable_combo.setCurrentText("selectivity")
        assert "max(" in dialog._observable_expr_preview.toPlainText()
        dialog._observable_shared_radio.setChecked(True)
        dialog.accept()
        selection = dialog.selection()
        assert selection is not None
        assert selection["type"] == "observable_existing"
        assert selection["name"] == "selectivity"
        assert "max(" in selection["expr"]
        assert selection["scalar_scope"] == "shared"
    finally:
        dialog.close()


def test_add_parameter_dialog_define_new_observable_flow_payload(qt_app):
    dialog = _AddFittableParameterDialog(
        available_rates=["k1"],
        available_scalars=["a"],
        available_species=["A"],
        dataset_ids=["ds1"],
        available_observables={"total_PBMP": "[PBMPBPIN] + [PBMP]"},
        parent=None,
    )
    try:
        titles = [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())]
        obs_idx = titles.index("Algebraic Observables")
        dialog._tabs.setCurrentIndex(obs_idx)
        dialog._define_new_button.click()
        qt_app.processEvents()
        dialog._new_observable_name_edit.setText("signal")
        dialog._new_observable_expr_edit.setText("scale * [A]")
        dialog.accept()
        selection = dialog.selection()
        assert selection is not None
        assert selection["type"] == "observable_new"
        assert selection["name"] == "signal"
        assert "scale * [A]" in selection["expr"]
    finally:
        dialog.close()
