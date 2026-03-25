"""Unified Global Fit configuration dialog (master-detail)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtWidgets

from kindred.gui.controllers.dataset_manager import DatasetFitSettings

__all__ = ["GlobalFitConfigDialog"]


@dataclass(frozen=True)
class _DatasetUiInputs:
    dataset_name: str
    observed_species: List[str]
    mechanism_species: List[str]
    defaults: Dict[str, float]
    initial_settings: DatasetFitSettings


def _clone_fit_settings(settings: DatasetFitSettings) -> DatasetFitSettings:
    cloned = DatasetFitSettings(weight=float(getattr(settings, "weight", 1.0)))
    cloned.initial_conditions = dict(getattr(settings, "initial_conditions", {}) or {})
    cloned.fit_flags = dict(getattr(settings, "fit_flags", {}) or {})
    cloned.log10_flags = dict(getattr(settings, "log10_flags", {}) or {})
    cloned.bounds = dict(getattr(settings, "bounds", {}) or {})
    cloned.batch_set = getattr(settings, "batch_set", None)
    cloned.batch_set_id = getattr(settings, "batch_set_id", None)
    return cloned


class _DatasetDetailWidget(QtWidgets.QWidget):
    def __init__(self, inputs: _DatasetUiInputs, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset_name = str(inputs.dataset_name)
        self._observed_species = list(inputs.observed_species or [])
        self._mechanism_species = list(inputs.mechanism_species or [])
        self._defaults = dict(inputs.defaults or {})
        self._settings = _clone_fit_settings(inputs.initial_settings)
        self._settings.ensure_species(self._mechanism_species, self._defaults)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(self._tabs, stretch=1)

        self._tabs.addTab(self._build_species_tab(), "Species")
        self._tabs.addTab(self._build_initials_tab(), "Initial Conditions")

    def _build_species_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)

        help_text = QtWidgets.QLabel(
            "Select the observed dataset species to include in the global fit for this dataset.",
            tab,
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        scroll = QtWidgets.QScrollArea(tab)
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget(scroll)
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(6)

        self._species_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        if not self._observed_species:
            placeholder = QtWidgets.QLabel("No species detected in this dataset.", container)
            placeholder.setEnabled(False)
            placeholder.setWordWrap(True)
            vbox.addWidget(placeholder)
        else:
            for species_name in self._observed_species:
                checkbox = QtWidgets.QCheckBox(str(species_name), container)
                checkbox.setChecked(True)
                self._species_checkboxes[str(species_name)] = checkbox
                vbox.addWidget(checkbox)

        vbox.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll, stretch=1)
        return tab

    def _build_initials_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)

        help_text = QtWidgets.QLabel(
            "Configure per-species initial concentrations and choose which to fit.",
            tab,
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        if not self._mechanism_species:
            placeholder = QtWidgets.QLabel("No mechanism species available.", tab)
            placeholder.setEnabled(False)
            placeholder.setWordWrap(True)
            layout.addWidget(placeholder)
            layout.addStretch(1)
            self._table = None
            self._weight_spin = None
            return tab

        self._table = QtWidgets.QTableWidget(len(self._mechanism_species), 6, tab)
        self._table.setHorizontalHeaderLabels(["Species", "Initial", "Fit?", "Log10", "Min", "Max"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, stretch=1)

        self._populate_initial_rows()

        weight_layout = QtWidgets.QHBoxLayout()
        weight_layout.addWidget(QtWidgets.QLabel("Dataset weight:", tab))
        self._weight_spin = QtWidgets.QDoubleSpinBox(tab)
        self._weight_spin.setMinimum(0.0001)
        self._weight_spin.setMaximum(1e6)
        self._weight_spin.setDecimals(4)
        self._weight_spin.setValue(max(0.0001, float(getattr(self._settings, "weight", 1.0))))
        weight_layout.addWidget(self._weight_spin)
        weight_layout.addStretch(1)
        layout.addLayout(weight_layout)

        return tab

    def _populate_initial_rows(self) -> None:
        assert self._table is not None
        for row, species in enumerate(self._mechanism_species):
            defaults_value = float(self._defaults.get(species, 0.0))
            init_value = self._settings.initial_conditions.get(species, defaults_value)
            fit_flag = bool(self._settings.fit_flags.get(species, False))
            log10_flag = bool(self._settings.log10_flags.get(species, False))
            bounds = self._settings.bounds.get(species, (0.0, max(10.0, defaults_value * 10 or 10.0)))

            species_item = QtWidgets.QTableWidgetItem(species)
            species_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self._table.setItem(row, 0, species_item)

            init_item = QtWidgets.QTableWidgetItem(f"{float(init_value):.6g}")
            self._table.setItem(row, 1, init_item)

            fit_item = QtWidgets.QTableWidgetItem()
            fit_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            fit_item.setCheckState(QtCore.Qt.Checked if fit_flag else QtCore.Qt.Unchecked)
            self._table.setItem(row, 2, fit_item)

            log_item = QtWidgets.QTableWidgetItem()
            log_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            log_item.setCheckState(QtCore.Qt.Checked if log10_flag else QtCore.Qt.Unchecked)
            self._table.setItem(row, 3, log_item)

            min_item = QtWidgets.QTableWidgetItem(f"{float(bounds[0]):.6g}")
            max_item = QtWidgets.QTableWidgetItem(f"{float(bounds[1]):.6g}")
            self._table.setItem(row, 4, min_item)
            self._table.setItem(row, 5, max_item)

    def activate_species_tab(self) -> None:
        self._tabs.setCurrentIndex(0)

    def activate_initials_tab(self) -> None:
        self._tabs.setCurrentIndex(1)

    def selected_species(self) -> List[str]:
        selection: List[str] = []
        for species in self._observed_species:
            checkbox = self._species_checkboxes.get(str(species))
            if checkbox is not None and checkbox.isChecked():
                selection.append(str(species))
        return selection

    def collect_fit_settings(self) -> Tuple[Optional[DatasetFitSettings], Optional[str]]:
        if self._table is None or self._weight_spin is None:
            return None, "No mechanism species are available to configure initial conditions."

        updated = DatasetFitSettings(weight=float(self._weight_spin.value()))
        updated.batch_set = getattr(self._settings, "batch_set", None)
        updated.batch_set_id = getattr(self._settings, "batch_set_id", None)

        for row, species in enumerate(self._mechanism_species):
            init_item = self._table.item(row, 1)
            fit_item = self._table.item(row, 2)
            log_item = self._table.item(row, 3)
            min_item = self._table.item(row, 4)
            max_item = self._table.item(row, 5)

            try:
                init_val = float(init_item.text())
            except (ValueError, AttributeError):
                return None, f"Species '{species}' requires a numeric initial concentration."

            fit_flag = fit_item.checkState() == QtCore.Qt.Checked if fit_item else False
            log10_flag = log_item.checkState() == QtCore.Qt.Checked if log_item else False

            try:
                min_val = float(min_item.text())
                max_val = float(max_item.text())
            except (ValueError, AttributeError):
                return None, f"Species '{species}' requires numeric bounds."

            if fit_flag and not (min_val < max_val):
                return None, f"Species '{species}' bounds must satisfy min < max."

            if fit_flag and log10_flag:
                if not (init_val > 0.0 and min_val > 0.0 and max_val > 0.0):
                    return None, f"Species '{species}' requires initial/min/max > 0 when Log10 is enabled."

            updated.initial_conditions[species] = float(init_val)
            updated.fit_flags[species] = bool(fit_flag)
            updated.log10_flags[species] = bool(log10_flag)
            updated.bounds[species] = (float(min_val), float(max_val))

        return updated, None


class GlobalFitConfigDialog(QtWidgets.QDialog):
    """
    Unified configuration dialog for Global Fit (master-detail).

    Returns a config dict via get_config() after acceptance.
    """

    def __init__(
        self,
        *,
        datasets: Dict[str, Dict],
        dataset_species_map: Dict[str, Sequence[str]],
        mechanism_species: Sequence[str],
        defaults_by_dataset: Optional[Dict[str, Dict[str, float]]] = None,
        dataset_manager=None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Global Fit Configuration")
        self.setModal(True)
        self.resize(980, 620)

        self._datasets = dict(datasets or {})
        self._dataset_species_map = {str(k): list(v) for k, v in (dataset_species_map or {}).items()}
        self._mechanism_species = [str(s) for s in (mechanism_species or []) if str(s)]
        self._defaults_by_dataset = defaults_by_dataset or {}
        self._dataset_manager = dataset_manager

        self._config: Optional[Dict[str, object]] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        layout.addWidget(splitter, stretch=1)

        # Master: datasets list (checkable, multi-select supported).
        master = QtWidgets.QWidget(splitter)
        master_layout = QtWidgets.QVBoxLayout(master)
        master_layout.setContentsMargins(0, 0, 0, 0)

        master_layout.addWidget(QtWidgets.QLabel("Datasets", master))
        self._dataset_list = QtWidgets.QListWidget(master)
        self._dataset_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._dataset_list.setMinimumWidth(260)
        master_layout.addWidget(self._dataset_list, stretch=1)
        splitter.addWidget(master)

        # Detail: per-dataset tabs in a stack, driven by current selection.
        self._detail_stack = QtWidgets.QStackedWidget(splitter)
        splitter.addWidget(self._detail_stack)
        splitter.setStretchFactor(1, 1)

        self._items_by_name: Dict[str, QtWidgets.QListWidgetItem] = {}
        self._details_by_name: Dict[str, _DatasetDetailWidget] = {}

        dataset_names = sorted(map(str, self._datasets.keys()))
        for name in dataset_names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            item.setCheckState(QtCore.Qt.Checked)
            self._dataset_list.addItem(item)
            self._items_by_name[name] = item

            observed_species = list(map(str, self._dataset_species_map.get(name, []) or []))
            defaults = dict(self._defaults_by_dataset.get(name, {}) or {})
            if self._dataset_manager is not None:
                settings = _clone_fit_settings(self._dataset_manager.get_fit_settings(name))
            else:
                settings = DatasetFitSettings()
            inputs = _DatasetUiInputs(
                dataset_name=name,
                observed_species=observed_species,
                mechanism_species=list(self._mechanism_species),
                defaults=defaults,
                initial_settings=settings,
            )
            detail = _DatasetDetailWidget(inputs, parent=self._detail_stack)
            self._details_by_name[name] = detail
            self._detail_stack.addWidget(detail)

        self._dataset_list.currentRowChanged.connect(self._detail_stack.setCurrentIndex)
        if self._dataset_list.count() > 0:
            self._dataset_list.setCurrentRow(0)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_config(self) -> Dict[str, object]:
        return dict(self._config or {})

    def accept(self) -> None:  # noqa: D401 - Qt override
        included: List[str] = []
        config_datasets: Dict[str, Dict[str, object]] = {}

        for idx in range(self._dataset_list.count()):
            item = self._dataset_list.item(idx)
            name = str(item.text())
            include = item.checkState() == QtCore.Qt.Checked
            detail = self._details_by_name.get(name)
            if detail is None:
                continue

            selected_species = detail.selected_species()
            settings, error = detail.collect_fit_settings()
            if error:
                QtWidgets.QMessageBox.warning(self, "Invalid Initial Conditions", error)
                self._dataset_list.setCurrentRow(idx)
                detail.activate_initials_tab()
                return

            config_datasets[name] = {
                "include": bool(include),
                "selected_species": list(selected_species),
                "fit_settings": settings,
            }
            if include:
                included.append(name)

        if not included:
            QtWidgets.QMessageBox.warning(self, "Global Fit", "Select at least one dataset to include.")
            return

        missing = [name for name in included if not (config_datasets.get(name, {}) or {}).get("selected_species")]
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "Select species",
                "Select at least one species for each included dataset:\n"
                + "\n".join(f"  • {name}" for name in missing),
            )
            first = missing[0]
            item = self._items_by_name.get(first)
            if item is not None:
                idx = self._dataset_list.row(item)
                if idx >= 0:
                    self._dataset_list.setCurrentRow(int(idx))
                    self._details_by_name[first].activate_species_tab()
            return

        self._config = {
            "datasets": config_datasets,
            "included": list(included),
        }
        super().accept()
