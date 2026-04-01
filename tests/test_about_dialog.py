from __future__ import annotations

import pytest
from PySide6 import QtWidgets


pytestmark = pytest.mark.gui


def test_build_about_dialog_uses_light_brand_asset(main_window):
    main_window._theme_manager._dark_mode = False

    dialog = main_window._build_about_dialog()
    try:
        assert dialog.property("brand_asset_path") == "assets/kindred-full-mark.png"
        brand_label = dialog.findChild(QtWidgets.QLabel, "aboutBrandImageLabel")
        assert brand_label is not None
        assert brand_label.property("brand_asset_path") == "assets/kindred-full-mark.png"
        pixmap = brand_label.pixmap()
        assert pixmap is not None
        assert not pixmap.isNull()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_build_about_dialog_uses_dark_brand_asset(main_window):
    main_window._theme_manager._dark_mode = True

    dialog = main_window._build_about_dialog()
    try:
        assert dialog.property("brand_asset_path") == "assets/kindred-full-mark-dark.png"
        brand_label = dialog.findChild(QtWidgets.QLabel, "aboutBrandImageLabel")
        assert brand_label is not None
        assert brand_label.property("brand_asset_path") == "assets/kindred-full-mark-dark.png"
        pixmap = brand_label.pixmap()
        assert pixmap is not None
        assert not pixmap.isNull()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_build_about_dialog_shows_updated_body_copy(main_window):
    dialog = main_window._build_about_dialog()
    try:
        body_label = dialog.findChild(QtWidgets.QLabel, "aboutBodyLabel")
        assert body_label is not None
        assert body_label.wordWrap() is True
        assert body_label.text() == (
            "Desktop GUI for reaction mechanism investigation, kinetic modeling, and data fitting.\n\n"
            "The initial development of Kindred was led primarily by Pedro Helou de Oliveira.\n"
            "Kindred is now developed and maintained jointly by Pedro Helou de Oliveira and Annabel Flook.\n\n"
            "Certain GUI design elements were informed by ideas seen in Velocity, a kinetics software package developed and maintained by Chen Li.\n\n"
            "Kindred source code is licensed under the MIT License.\n"
            "Includes Qt for Python (PySide6/shiboken6) under LGPLv3/GPL terms.\n"
            "See NOTICE.txt, COPYING, COPYING.LESSER, and THIRD_PARTY_LICENSES.txt."
        )
    finally:
        dialog.close()
        dialog.deleteLater()


def test_show_about_executes_built_dialog(main_window, monkeypatch):
    class _FakeDialog:
        called = 0

        def exec(self):
            self.called += 1
            return QtWidgets.QDialog.DialogCode.Accepted

    fake_dialog = _FakeDialog()
    monkeypatch.setattr(main_window, "_build_about_dialog", lambda: fake_dialog)

    main_window._show_about()

    assert fake_dialog.called == 1
