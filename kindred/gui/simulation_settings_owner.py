from __future__ import annotations

from PySide6.QtCore import QSettings


class SimulationSettingsOwner:
    """Owns GUI QSettings lifecycle and persistent settings writes."""

    def __init__(self) -> None:
        self._settings = QSettings("Kindred", "KindredGUI")

    @property
    def qsettings(self) -> QSettings:
        return self._settings

    def settings_set_value(self, key: str, value: object) -> None:
        self._settings.setValue(str(key), value)

    def settings_remove(self, key: str) -> None:
        self._settings.remove(str(key))

    def settings_sync(self) -> None:
        self._settings.sync()
