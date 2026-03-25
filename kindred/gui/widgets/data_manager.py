# kindred/gui/widgets/data_manager.py
"""Data manager panel for loading and managing experimental datasets."""

from __future__ import annotations

from contextlib import suppress
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from kindred.core.datasets.csv_import import (
    CsvImportInterrupted,
    load_csv_dataset,
)

logger = logging.getLogger(__name__)

__all__ = ["DataManagerPanel"]

class CSVLoaderWorker(QtCore.QThread):
    """Background worker for loading CSV files without blocking UI."""

    # Signals
    finished = QtCore.Signal(str, dict)  # name, data
    cancelled = QtCore.Signal(str)  # dataset name
    error = QtCore.Signal(str)  # error message
    progress = QtCore.Signal(int)  # progress percentage

    def __init__(
        self,
        filepath: str,
        time_column: Optional[str] = None,
        species_columns: Optional[Sequence[str]] = None,
    ):
        """
        Initialize CSV loader worker.

        Parameters
        ----------
        filepath : str
            Path to CSV file to load
        """
        super().__init__()
        self.filepath = filepath
        self._time_column = (time_column or "").strip() or None
        self._species_columns = [col.strip() for col in (species_columns or []) if col and col.strip()]

    def run(self):
        """Load CSV file in background thread."""
        filename = self.filepath
        dataset_name = os.path.basename(filename)

        def _check_cancel(stage: str) -> bool:
            if self.isInterruptionRequested():
                logger.info(f"CSV import interrupted {stage}: {dataset_name}")
                self.cancelled.emit(dataset_name)
                return True
            return False

        # Log start
        logger.info(f"Starting CSV import: {dataset_name} from {filename}")

        try:
            self.progress.emit(10)

            # Check for interruption before file I/O
            if _check_cancel("before reading"):
                return

            self.progress.emit(30)

            # Stream CSV rows through the shared core importer so interruption can
            # be observed during file iteration rather than only after full read.
            _loaded_name, data = load_csv_dataset(
                filename,
                time_column=self._time_column,
                species_columns=self._species_columns,
                interruption_checker=self.isInterruptionRequested,
            )
            logger.debug(
                "Parsed CSV dataset: time column '%s', species columns: %s",
                data.get("metadata", {}).get("time_column"),
                list(data['species'].keys()),
            )

            self.progress.emit(80)

            if _check_cancel("after parsing"):
                return

            self.progress.emit(100)

            # Log success before emitting signal
            species = data.get('species', {})
            logger.info(
                "CSV import completed: %s (%d rows, %d species: %s)",
                dataset_name,
                len(data.get('t', [])),
                len(species),
                list(species.keys()),
            )

            self.finished.emit(dataset_name, data)

        except CsvImportInterrupted:
            self.cancelled.emit(dataset_name)
            return
        except Exception as e:
            logger.error(f"CSV import failed: {dataset_name} - {type(e).__name__}: {e}", exc_info=True)
            self.error.emit(f"{type(e).__name__}: {str(e)}")


class DataManagerPanel(QtWidgets.QWidget):
    """
    Data manager for loading/managing experimental datasets.

    Features:
    - Load CSV files with experimental data
    - Auto-detect time column (tries: time, time_s, t, Time, T, x)
    - Extract all numeric columns as species
    - Preview loaded datasets
    - Column mapping configuration
    - Multiple dataset support

    Signals:
        datasetLoaded(str, dict): Emitted when dataset is loaded
            - name: Dataset name (filename)
            - data: {'t': array, 'species': {name: array}}
        datasetRemoved(str): Emitted when a dataset is removed from the panel
        loadFinished(bool): Emitted once when a load cycle completes or is canceled
            - canceled: True if the operation was canceled
    """

    datasetLoaded = QtCore.Signal(str, dict)  # name, data dict
    datasetRemoved = QtCore.Signal(str)  # name
    loadFinished = QtCore.Signal(bool)  # canceled flag

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """
        Initialize data manager panel.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        # Header
        layout.addWidget(QtWidgets.QLabel("<b>Loaded Datasets</b>"))

        # Dataset list
        self._dataset_list = QtWidgets.QListWidget()
        self._dataset_list.currentItemChanged.connect(self._on_dataset_selected)
        layout.addWidget(self._dataset_list)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self._load_btn = QtWidgets.QPushButton("Load CSV")
        self._load_btn.setObjectName("loadDataButton")
        self._load_btn.clicked.connect(self._load_dataset)
        self._remove_btn = QtWidgets.QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_dataset)
        btn_layout.addWidget(self._load_btn)
        btn_layout.addWidget(self._remove_btn)
        layout.addLayout(btn_layout)

        # Preview area
        layout.addWidget(QtWidgets.QLabel("Preview:"))
        self._preview_label = QtWidgets.QLabel("No dataset selected")
        self._preview_label.setContentsMargins(10, 10, 10, 10)
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setWordWrap(True)
        layout.addWidget(self._preview_label)
        self._preview_label.hide()

        # Column mapping section
        layout.addWidget(QtWidgets.QLabel("Column Mapping:"))
        self._mapping_widget = QtWidgets.QWidget()
        mapping_layout = QtWidgets.QFormLayout(self._mapping_widget)
        self._time_col_edit = QtWidgets.QLineEdit()
        self._time_col_edit.setPlaceholderText("auto-detect (time, t, ...)")
        self._species_col_edit = QtWidgets.QLineEdit()
        self._species_col_edit.setPlaceholderText("comma-separated, leave blank for auto")
        mapping_layout.addRow("Time column:", self._time_col_edit)
        mapping_layout.addRow("Species columns:", self._species_col_edit)
        layout.addWidget(self._mapping_widget)
        self._mapping_widget.hide()

        # Store loaded datasets {name: {t: array, species: {name: array}}}
        self._datasets: Dict[str, Dict] = {}

        # Track multiple CSV workers for multi-file loading
        self._csv_workers: List[CSVLoaderWorker] = []
        self._pending_files_count = 0
        self._completed_files_count = 0
        self._progress_dialog: Optional[QtWidgets.QProgressDialog] = None
        self._cancel_requested = False
        self._load_finished_emitted = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def trigger_load_dialog(self) -> None:
        """Programmatically invoke the load dialog (mirrors Load button)."""
        self._load_dataset()

    def clear_datasets(self) -> None:
        """Clear loaded datasets and reset dataset-panel UI state."""
        self._datasets.clear()
        self._dataset_list.clear()
        self._preview_label.setText("No dataset selected")
        self._preview_label.hide()
        self._mapping_widget.hide()
        self._time_col_edit.clear()
        self._species_col_edit.clear()

    def _load_dataset(self):
        """Load CSV dataset(s) using background worker(s). Supports multi-select."""
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Load Dataset(s)",
            "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )

        if not filenames:
            return

        # Initialize counters for multi-file tracking
        self._pending_files_count = len(filenames)
        self._completed_files_count = 0
        self._cancel_requested = False
        self._load_finished_emitted = False

        file_count = len(filenames)
        file_list = ", ".join([os.path.basename(f) for f in filenames[:3]])
        if file_count > 3:
            file_list += f", ... ({file_count - 3} more)"

        logger.info(f"User initiated CSV import: {file_count} file(s) - {file_list}")

        # Create progress dialog
        self._progress_dialog = QtWidgets.QProgressDialog(
            f"Loading {file_count} file(s)...", "Cancel", 0, 100, self
        )
        self._progress_dialog.setWindowTitle("Loading Datasets")
        self._progress_dialog.setWindowModality(Qt.WindowModal)
        self._progress_dialog.setMinimumDuration(0)  # Show immediately
        self._progress_dialog.canceled.connect(self._on_load_canceled)
        self._progress_dialog.show()

        time_column = self._time_col_edit.text().strip() or None
        species_text = self._species_col_edit.text().strip()
        species_columns = [col.strip() for col in species_text.split(',') if col.strip()] if species_text else []

        # Create and start worker for each file
        for filename in filenames:
            worker = CSVLoaderWorker(
                filename,
                time_column=time_column,
                species_columns=species_columns or None,
            )
            worker.progress.connect(self._on_load_progress)
            worker.finished.connect(self._on_csv_loaded)
            worker.cancelled.connect(self._on_csv_cancelled)
            worker.error.connect(self._on_csv_error)
            worker.finished.connect(lambda *args, w=worker: self._cleanup_worker(w))
            worker.cancelled.connect(lambda *args, w=worker: self._cleanup_worker(w))
            worker.error.connect(lambda *args, w=worker: self._cleanup_worker(w))

            self._csv_workers.append(worker)
            worker.start()

    def _on_load_progress(self, percent: int):
        """Handle progress updates from CSV worker(s)."""
        if self._progress_dialog:
            # Calculate overall progress across all files
            if self._pending_files_count > 0:
                completed_fraction = self._completed_files_count / self._pending_files_count
                current_file_fraction = (1.0 / self._pending_files_count) * (percent / 100.0)
                overall_percent = int((completed_fraction + current_file_fraction) * 100)
                self._progress_dialog.setValue(min(overall_percent, 99))  # Reserve 100 for completion

    def _on_load_canceled(self):
        """Handle cancel request during CSV loading."""
        # Request interruption for all active workers
        for worker in self._csv_workers:
            if worker:
                worker.requestInterruption()

        logger.info("CSV import cancellation requested by user")
        self._cancel_requested = True

        # Close progress dialog immediately
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None

        self._maybe_finalize_load_cycle()

    def _on_csv_cancelled(self, name: str):
        """Handle CSV cancellation without surfacing an error dialog."""
        logger.info("CSV import canceled: %s", name)
        self._completed_files_count += 1
        self._maybe_finalize_load_cycle()

    def _emit_load_finished(self, canceled: bool) -> None:
        """Ensure overall completion/cancellation signal fires only once."""
        if self._load_finished_emitted:
            return
        self._load_finished_emitted = True
        self.loadFinished.emit(canceled)

    def _finalize_progress_dialog(self) -> None:
        """Close and clear the progress dialog if it exists."""
        if not self._progress_dialog:
            return
        with suppress(RuntimeError, TypeError):
            self._progress_dialog.setValue(100)
        self._progress_dialog.close()
        self._progress_dialog = None

    def _maybe_finalize_load_cycle(self):
        """Close progress UI and emit completion when all workers are gone."""
        if (
            self._pending_files_count > 0
            and self._completed_files_count >= self._pending_files_count
            and not self._csv_workers
        ):
            self._finalize_progress_dialog()
            self._pending_files_count = 0
            self._completed_files_count = 0
            self._emit_load_finished(self._cancel_requested)
            self._cancel_requested = False

    def _cleanup_worker(self, worker: CSVLoaderWorker):
        """Clean up worker thread after completion or error."""
        if not worker:
            return

        try:
            worker.requestInterruption()
        except RuntimeError:
            pass

        if worker.isRunning():
            worker.quit()
            worker.wait(2000)  # Wait up to 2 seconds
        else:
            worker.quit()
            worker.wait(2000)

        # Disconnect signals to prevent memory leaks
        for signal in (
            getattr(worker, "progress", None),
            getattr(worker, "finished", None),
            getattr(worker, "error", None),
            getattr(worker, "cancelled", None),
        ):
            if signal is None:
                continue
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                # Already disconnected or deleted
                pass

        # Remove from workers list
        if worker in self._csv_workers:
            self._csv_workers.remove(worker)

        # Delete worker
        worker.deleteLater()

        self._maybe_finalize_load_cycle()

    def _on_csv_loaded(self, name: str, data: dict):
        """Handle successful CSV load with automatic unique naming."""
        # Ensure unique dataset name
        unique_name = self._make_unique_dataset_name(name)

        # Store dataset with unique name
        self._datasets[unique_name] = data

        # Add to list
        self._dataset_list.addItem(unique_name)

        # Emit signal with unique name
        self.datasetLoaded.emit(unique_name, data)

        # Track completion for multi-file progress
        self._completed_files_count += 1

        self._maybe_finalize_load_cycle()

        # Note: Worker already logged completion, no need to log again here

    def _on_csv_error(self, error_msg: str):
        """Handle CSV load error."""
        # Track completion even for errors
        self._completed_files_count += 1

        # Show error message
        QtWidgets.QMessageBox.critical(
            self,
            "Load Error",
            f"Failed to load dataset:\n\n{error_msg}"
        )

        self._maybe_finalize_load_cycle()

        # Note: Worker already logged the error, no need to log again here

    def _remove_dataset(self):
        """Remove selected dataset."""
        current = self._dataset_list.currentItem()
        if not current:
            return

        name = current.text()
        self._dataset_list.takeItem(self._dataset_list.row(current))
        if name in self._datasets:
            del self._datasets[name]
            self.datasetRemoved.emit(name)

    def _on_dataset_selected(self, current, previous):
        """Update preview when dataset is selected."""
        if not current:
            self._preview_label.setText("No dataset selected")
            self._preview_label.hide()
            self._mapping_widget.hide()
            return

        name = current.text()
        if name not in self._datasets:
            self._preview_label.hide()
            self._mapping_widget.hide()
            return

        data = self._datasets[name]
        self._preview_label.show()
        self._mapping_widget.show()
        t = data['t']
        species = data['species']

        metadata = data.get('metadata', {})
        mapping_source = metadata.get('mapping_source', 'auto')
        preview = f"<b>{name}</b><br>"
        preview += f"{len(t)} time points<br>"
        preview += f"Time column: {metadata.get('time_column', 'unknown')} ({mapping_source})<br>"
        preview += f"{len(species)} species: {', '.join(list(species.keys())[:5])}"
        if len(species) > 5:
            preview += f", ... ({len(species)-5} more)"

        self._preview_label.setText(preview)

    def _make_unique_dataset_name(self, name: str) -> str:
        """
        Generate a unique dataset name by appending _1, _2, etc. if needed.

        Parameters
        ----------
        name : str
            Desired dataset name (usually base filename)

        Returns
        -------
        str
            Unique name not already in self._datasets
        """
        if name not in self._datasets:
            return name

        # Extract base name and extension
        base, ext = os.path.splitext(name)

        # Try name_1, name_2, etc.
        counter = 1
        while True:
            candidate = f"{base}_{counter}{ext}"
            if candidate not in self._datasets:
                return candidate
            counter += 1

    def get_datasets(self) -> Dict[str, Dict]:
        """
        Get all loaded datasets.

        Returns
        -------
        dict
            Dictionary of {name: {'t': array, 'species': {name: array}}}
        """
        return self._datasets

    def get_selected_dataset(self) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Get currently selected dataset.

        Returns
        -------
        tuple
            (name, data) or (None, None) if no selection
        """
        current = self._dataset_list.currentItem()
        if not current:
            return None, None

        name = current.text()
        return name, self._datasets.get(name)

    def get_dataset(self, name: str) -> Optional[Dict]:
        """
        Retrieve a dataset by name.

        Parameters
        ----------
        name : str
            Dataset identifier (usually filename)

        Returns
        -------
        dict or None
            Dataset payload or None if not loaded.
        """
        return self._datasets.get(name)
