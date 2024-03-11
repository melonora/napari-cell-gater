from pathlib import Path
from typing import Any

from napari import Viewer
from napari.utils.history import (
    get_open_history,
    update_open_history,
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from cell_gater.model.data_model import DataModel
from cell_gater.utils.csv_df import stack_csv_files
from cell_gater.utils.misc import napari_notification


class SampleWidget(QWidget):
    """Sample widget for loading required data."""

    def __init__(self, viewer: Viewer) -> None:
        super().__init__()
        self._viewer = viewer
        self._model = DataModel()
        self.setLayout(QGridLayout())

        # Open sample directory dialog
        self.load_samples_button = QPushButton("Load regionprops dir")
        self.load_samples_button.clicked.connect(self._open_sample_dialog)
        self.layout().addWidget(self.load_samples_button, 0, 0)

        # Open image directory dialog
        self.load_image_dir_button = QPushButton("Load image dir")
        self.load_image_dir_button.clicked.connect(self._open_image_dir_dialog)
        self.layout().addWidget(self.load_image_dir_button, 0, 1)

        # Open mask directory dialog
        self.load_mask_dir_button = QPushButton("Load mask dir")
        self.load_mask_dir_button.clicked.connect(self._open_mask_dir_dialog)
        self.layout().addWidget(self.load_mask_dir_button, 0, 2)

        lower_col = QLabel("Select lowerbound marker column:")
        self.lower_bound_marker_col = QComboBox()
        if len(self.model.regionprops_df) > 0:
            self.lower_bound_marker_col.addItems(self.model.regionprops_df.columns)
        self.lower_bound_marker_col.currentTextChanged.connect(self._update_marker_widget)
        self.layout().addWidget(lower_col, 2, 0)
        self.layout().addWidget(self.lower_bound_marker_col, 3, 0)

        # Dropdown of samples once directory is loaded
        selection_label = QLabel("Select sample:")
        self.sample_selection_dropdown = QComboBox()
        if len(self.model.samples) > 0:
            self.sample_selection_dropdown.addItems([None])
            self.sample_selection_dropdown.addItems(self.model.samples)
        self.sample_selection_dropdown.currentTextChanged.connect(self._on_sample_changed)

        self.layout().addWidget(selection_label, 4, 0, Qt.AlignCenter)
        self.layout().addWidget(self.sample_selection_dropdown, 4, 1)

        self.model.events.regionprops_df.connect(self._set_samples_dropdown)
        self.model.events.regionprops_df.connect(self._set_marker_lowerbound)

    @property
    def viewer(self) -> Viewer:
        """The napari viewer."""
        return self._viewer

    @property
    def model(self) -> DataModel:
        """Data model of the widget."""
        return self._model

    def _on_sample_changed(self):
        pass

    def _dir_dialog(self):
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)
        return dlg.getExistingDirectory(
            self,
            "select folder",
            hist[0],
            QFileDialog.Options(),
        )

    def _open_sample_dialog(self):
        """Open directory file dialog for regionprop directory."""
        folder = self._dir_dialog()

        if folder not in {"", None}:
            self._assign_regionprops_to_model(folder)
            update_open_history(folder)

    def _open_image_dir_dialog(self):
        """Open directory file dialog for the image directory."""
        folder = self._dir_dialog()

        if folder not in {"", None}:
            self._set_image_paths(folder)

    def _open_mask_dir_dialog(self):
        """Open directory file dialog for the mask directory."""
        folder = self._dir_dialog()

        if folder not in {"", None}:
            self._set_mask_paths(folder)

    def _set_image_paths(self, folder):
        self.model.image_paths = list(Path(folder).glob("*tif"))
        napari_notification(f"{len(self.model.mask_paths)} paths of images loaded.")

    def _set_mask_paths(self, folder):
        self.model.mask_paths = list(Path(folder).glob("*tif"))
        napari_notification(f"{len(self.model.mask_paths)} paths of masks loaded.")

    def _assign_regionprops_to_model(self, folder):
        self.model.regionprops_df = stack_csv_files(folder)

    def _set_samples_dropdown(self, event: Any):
        self.model.samples = list(self.model.regionprops_df["sample_id"].cat.categories)

        # New directory loaded so we reload the dropdown items
        self.sample_selection_dropdown.clear()
        if len(self.model.samples) > 0:
            self.sample_selection_dropdown.addItems([None])
            self.sample_selection_dropdown.addItems(self.model.samples)

    def _update_marker_widget(self):
        pass

    def _set_marker_lowerbound(self):
        self.lower_bound_marker_col.clear()
        if len(self.model.regionprops_df) > 0:
            self.lower_bound_marker_col.addItems(self.model.regionprops_df.columns)
