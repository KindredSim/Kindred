from __future__ import annotations

import pytest
from PySide6 import QtWidgets

from kindred.gui.widgets.species_statistics_table import SpeciesStatisticsTable

pytestmark = [pytest.mark.gui]


def test_species_statistics_table_expands_and_has_no_hardcoded_max_height(qt_app):
    table = SpeciesStatisticsTable()
    try:
        policy = table.sizePolicy()
        assert policy.horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding
        assert policy.verticalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding

        # Qt's unconstrained maximum height (QWIDGETSIZE_MAX) is 16,777,215.
        assert table.maximumHeight() == 16_777_215
    finally:
        table.close()

