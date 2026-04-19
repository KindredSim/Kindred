"""GUI project save/load robustness tests."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
from PySide6 import QtWidgets

from kindred.gui.main_window import MainWindow

pytestmark = [pytest.mark.gui, pytest.mark.slow]


class TestProjectSaveLoad:
    """Test project save/load robustness."""

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a project file that doesn't exist."""
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        window = MainWindow()
        try:
            nonexistent = str(tmp_path / "this_file_does_not_exist_12345.kin")
            assert not os.path.exists(nonexistent)

            with patch.object(QtWidgets.QMessageBox, "warning") as mock_warning:
                window._load_recent_project(nonexistent)
                assert mock_warning.called
        finally:
            window.close()

    def test_load_corrupted_json(self):
        """Test loading a project file with corrupted JSON."""
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        window = MainWindow()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".kin", delete=False) as f:
            f.write("{ this is not valid JSON }")
            temp_path = f.name

        try:
            with patch.object(QtWidgets.QMessageBox, "critical") as mock_critical:
                window._load_recent_project(temp_path)
                assert mock_critical.called
        finally:
            window.close()
            os.unlink(temp_path)
