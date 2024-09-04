from __future__ import annotations

from pathlib import Path

from napari import Viewer
from napari.utils.history import (
    get_open_history,
    update_open_history,
)
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from cell_gater.model.data_model import DataModel
from cell_gater.utils.csv_df import stack_csv_files
from cell_gater.utils.misc import napari_notification
from cell_gater.widgets.scatter_widget import ScatterInputWidget

import skimage.io
import numpy
import pandas as pd
import sys
from loguru import logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")

class SampleWidget(QWidget):
    """Sample widget for loading required data."""

    def __init__(self, napari_viewer: Viewer, model: DataModel | None = None) -> None:
        """
        Create the QWidget visuals.

        This widget sets up the QtWidget elements used to determine what the input is that the user wants to use
        for cell gating. It also connects these elements to their appropriate callback functions.

        Parameters
        ----------
        viewer : napari.Viewer
            The napari Viewer instance.
        model : DataModel | None
            The data model dataclass. If provided, this means that the plugin is used by means of a CLI to be
            implemented.

        """
        super().__init__()
        self._scatter_widget = None
        self._viewer = napari_viewer
        self._model = DataModel() if model is None else model
        self.setLayout(QGridLayout())
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # object, int row, int column, int rowSpan = 1, int columnSpan = 1
        # load_label = QLabel("Load data:")
        # self.layout().addWidget(load_label, 0, 0, 1, 1)
        # Open sample directory dialog
        self.load_samples_button = QPushButton("Load quantifications dir")
        self.load_samples_button.clicked.connect(self._open_sample_dialog)
        self.layout().addWidget(self.load_samples_button, 0, 0, 1, 1)

        # Open image directory dialog
        self.load_image_dir_button = QPushButton("Load image dir")
        self.load_image_dir_button.clicked.connect(self._open_image_dir_dialog)
        self.layout().addWidget(self.load_image_dir_button, 0, 1, 1, 1)

        # Open mask directory dialog
        self.load_mask_dir_button = QPushButton("Load mask dir")
        self.load_mask_dir_button.clicked.connect(self._open_mask_dir_dialog)
        self.layout().addWidget(self.load_mask_dir_button, 0, 2, 1, 1)

        # Open manual channel mapping csv
        self.load_manual_channel_mapping_button = QPushButton("Opt: Load channel map")
        self.load_manual_channel_mapping_button.clicked.connect(self._open_manual_channel_mapping)
        self.layout().addWidget(self.load_manual_channel_mapping_button, 0, 3, 1, 1)

        # The lower bound marker column dropdown
        lower_col = QLabel("Select lowerbound marker column:")
        self.lower_bound_marker_col = QComboBox()
        if len(self.model.regionprops_df) > 0:
            self.lower_bound_marker_col.addItems([None] + self.model.regionprops_df.columns)
        self.lower_bound_marker_col.currentTextChanged.connect(self._update_model_lowerbound)
        self.layout().addWidget(lower_col, 1, 0, 1, 2)
        self.layout().addWidget(self.lower_bound_marker_col, 1, 2, 1, 2)

        # The upper bound marker column dropdown
        upper_col = QLabel("Select upperbound marker column:")
        self.upper_bound_marker_col = QComboBox()
        if len(self.model.regionprops_df) > 0:
            self.upper_bound_marker_col.addItems([None] + self.model.regionprops_df.columns)
        self.upper_bound_marker_col.currentTextChanged.connect(self._update_model_upperbound)
        self.layout().addWidget(upper_col, 2, 0, 1, 2)
        self.layout().addWidget(self.upper_bound_marker_col, 2, 2, 1, 2)

        # Filter field for user to pass on strings to filter markers out.
        filter_label = QLabel("Remove markers with prefix (default: DNA,DAPI)")
        self.filter_field = QLineEdit("DNA, DAPI", placeholderText="Prefixes separated by commas.")
        self.filter_field.editingFinished.connect(self._update_filter)
        self.layout().addWidget(filter_label, 3, 0, 1, 2)
        self.layout().addWidget(self.filter_field, 3, 2, 1, 2)

        # Button to start validating all the input
        self.validate_button = QPushButton("Validate input")
        self.validate_button.clicked.connect(self._validate)
        self.layout().addWidget(self.validate_button, 4, 0, 1, 4)

        self.model.events.regionprops_df.connect(self._set_dropdown_marker_lowerbound)
        self.model.events.regionprops_df.connect(self._set_dropdown_marker_upperbound)

    def update_ref_channel(self):
        """Update the reference channel in the data model upon change of text in the reference channel column widget."""
        self.model.ref_channel = self.ref_channel.currentText()

    @property
    def viewer(self) -> Viewer:
        """The napari viewer."""
        return self._viewer

    @property
    def model(self) -> DataModel:
        """Data model of the widget."""
        return self._model

    def _dir_dialog(self):
        """Open dialog for a user to pass on a directory."""
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)
        return dlg.getExistingDirectory(
            self,
            "select folder",
            hist[0],
            QFileDialog.Options(),
        )

    def _file_dialog(self):
        """Open dialog for a user to select a file."""
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)
        options = QFileDialog.Options()
        return dlg.getOpenFileName(
            self,
            "Select file",
            hist[0],
            "CSV Files (*.csv)",
            options=options,
        )

    def _open_sample_dialog(self, folder: str | None = None):
        """Open directory file dialog for regionprop directory."""
        if not folder:
            folder = self._dir_dialog()

        if isinstance(folder, str) and folder != "":
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

    def _open_manual_channel_mapping(self):
        """Open dialog for a user to pass on a csv file."""
        logger.info("Opening manual channel mapping file.")
        file_path, _ = self._file_dialog()
        if file_path:
            self.model.manual_channel_mapping = file_path
            logger.info(f"Manual channel mapping file loaded: {file_path}")

    def _set_image_paths(self, folder: str) -> None:
        """Set the image paths in the DataModel."""
        types = ("*.tif", "*.tiff")  # the tuple of file types
        self.model.image_paths = []
        for ext in types:
            self.model.image_paths.extend(list(Path(folder).glob(ext)))
        napari_notification(f"{len(self.model.image_paths)} paths of images loaded.")

    def _set_mask_paths(self, folder: str) -> None:
        """Set the paths of the masks in the DataModel."""
        types = ("*.tif", "*.tiff")  # the tuple of file types
        self.model.mask_paths = []
        for ext in types:
            paths = list(Path(folder).glob(ext))
            filtered_paths = [path for path in paths if not path.name.startswith(".")]
            self.model.mask_paths.extend(filtered_paths)

        napari_notification(f"{len(self.model.mask_paths)} paths of masks loaded.")

    def _assign_regionprops_to_model(self, folder: str) -> None:
        """Read the csv files in the directory and assign the resulting concatenated DataFrame in the DataModel."""
        self.model.regionprops_df = stack_csv_files(Path(folder))

    def _set_dropdown_marker_lowerbound(self):
        """Add items to dropdown menu for lowerbound marker.

        This menu is a dropdown menu that responds to events of self.model.regionprops_df. When this changes it loads
        all the columns of this dataframe as items in the listwidget if a dataframe iis currently loaded.
        """
        self.lower_bound_marker_col.clear()
        region_props = self.model.regionprops_df
        if region_props is not None and len(region_props) > 0:
            self.lower_bound_marker_col.addItems(region_props.columns)
        self.lower_bound_marker_col.setCurrentIndex(1)  # Skip the cell id column

    def _set_dropdown_marker_upperbound(self):
        """Add items to dropdown menu for upperbound marker.

        This menu is a dropdown menu that responds to events of self.model.regionprops_df. When this changes it loads
        all the columns of this dataframe as items in the listwidget if a dataframe iis currently loaded.
        """
        self.upper_bound_marker_col.clear()
        region_props = self.model.regionprops_df
        if region_props is not None and len(region_props) > 0:
            self.upper_bound_marker_col.addItems(region_props.columns)
            # set default to the last column before X_centroid
            if "X_centroid" in region_props.columns:
                default_index = self.model.regionprops_df.columns.tolist().index("X_centroid")
                if default_index != -1:
                    self.upper_bound_marker_col.setCurrentIndex(default_index - 1)
            else:
                self.upper_bound_marker_col.setCurrentIndex(len(region_props.columns) - 1)

    def _update_model_lowerbound(self):
        """Update the lowerbound marker in the data model upon change of text in the lowerbound marker column widget."""
        lower_bound_marker = self.lower_bound_marker_col.currentText()
        self.model.lower_bound_marker = lower_bound_marker

    def _update_model_upperbound(self):
        """Update the upperbound marker in the data model upon change of text in the upperbound marker column widget."""
        upper_bound_marker = self.upper_bound_marker_col.currentText()
        self.model.upper_bound_marker = upper_bound_marker

    def _update_filter(self):
        """Update marker filter upon text change of the filter field widget."""
        self.model.marker_filter = self.filter_field.text()

    def _validate(self):
        """
        Perform validation of the input data upon press of the validate button by the user.

        This function checks whether all required input is in the data model and also filters the markers based on
        the current value of the filter marker in the data model.
        """
        assert self.model.regionprops_df.shape != (0, 0), "No regionprops file or directory seems to be loaded."
        assert len(self.model.image_paths) != 0, "No image file or directory seems to be loaded."
        assert len(self.model.mask_paths) != 0, "No mask file or directory seems to be loaded."
        assert len(self.model.image_paths) == len(
            self.model.mask_paths
        ), "Number of images and segmentation masks do not match."

        # First check whether there is a difference between the file names without extension and then assign as samples
        image_paths_set = {i.stem if ".ome" not in i.stem else i.stem.rstrip(".ome") for i in self.model.image_paths}
        mask_paths_set = {i.stem if ".ome" not in i.stem else i.stem.rstrip(".ome") for i in self.model.mask_paths}
        if len(diff := image_paths_set.symmetric_difference(mask_paths_set)):
            raise ValueError(f"Images and masks do not seem to match. Found {','.join(diff)}")

        # This allows to use the dropdowns to directly map to the paths for opening.
        self.model.samples = image_paths_set
        self.model.sample_image_mapping = {i.stem.rstrip(".ome") if ".ome" in i.stem else i.stem: i for i in self.model.image_paths}
        self.model.sample_mask_mapping = {i.stem.rstrip(".ome") if ".ome" in i.stem else i.stem: i for i in self.model.mask_paths}

        # Selecting the markers of interest
        columns = list(self.model.regionprops_df.columns)[1:]
        lowerbound_index = columns.index(self.model.lower_bound_marker)
        upperbound_index = columns.index(self.model.upper_bound_marker)
        self.model.markers = columns[lowerbound_index : upperbound_index + 1]
        n_markers = len(self.model.markers)
        # filter DNA and DAPI channels by default
        for filter in self.model.marker_filter.split(","):
            for marker in self.model.markers.copy():
                if filter.lower() in marker.lower():
                    self.model.markers.remove(marker)
        napari_notification(f"Removed {n_markers - len(self.model.markers)} out of list of {n_markers}.")

        # Find num of channels in image, maybe more efficient way, WSI would take very long
        image_shape = skimage.io.imread(self.model.image_paths[0]).shape
        n_channels = image_shape[0]
        logger.info(f"Number of channels in image: {n_channels}")

        # Mapping df columns to image channels
        self.model.markers_image_indices = {}

        if self.model.manual_channel_mapping:
            logger.info("self.model.manual_channel_mapping is TRUE")
            # load manual csv
            df = pd.read_csv(self.model.manual_channel_mapping)
            df["channel_in_matrix"] = ~df["csv_column_name"].isnull()

            for index, row in df.iterrows():
                if row["channel_in_matrix"]:
                    self.model.markers_image_indices[row["csv_column_name"]] = row["channel_index"] -1
        else:
            assert n_channels < len(columns), "Number of channels in image is larger than number of columns in matrix."
            for metric_round in range(0, len(columns)//n_channels):
                metric_index = metric_round * n_channels
                for channel_index in range(1, n_channels+1):
                    self.model.markers_image_indices[columns[int(metric_index) + int(channel_index)-1]] = channel_index-1

        logger.info(f"markers_image_indices: {self.model.markers_image_indices}")

        self._scatter_widget = ScatterInputWidget(self.model, self.viewer)
        self.viewer.window.add_dock_widget(
            self._scatter_widget, name="cell_gater", area="right", menu=self._viewer.window.window_menu, tabify=True
        )

        self.model.validated = True