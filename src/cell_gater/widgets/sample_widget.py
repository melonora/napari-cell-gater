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

class SampleWidget(QWidget):
    """Sample widget for loading required data."""

    def __init__(self, viewer: Viewer, model: DataModel | None = None) -> None:
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
        self._viewer = viewer
        self._model = DataModel() if model is None else model
        self.setLayout(QGridLayout())
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        load_label = QLabel("Load data:")
        self.layout().addWidget(load_label, 0, 0)
        # Open sample directory dialog
        self.load_samples_button = QPushButton("Load regionprops dir")
        self.load_samples_button.clicked.connect(self._open_sample_dialog)
        self.layout().addWidget(self.load_samples_button, 1, 0)

        # Open image directory dialog
        self.load_image_dir_button = QPushButton("Load image dir")
        self.load_image_dir_button.clicked.connect(self._open_image_dir_dialog)
        self.layout().addWidget(self.load_image_dir_button, 2, 0)

        # Open mask directory dialog
        self.load_mask_dir_button = QPushButton("Load mask dir")
        self.load_mask_dir_button.clicked.connect(self._open_mask_dir_dialog)
        self.layout().addWidget(self.load_mask_dir_button, 3, 0)

        # The lower bound marker column dropdown
        lower_col = QLabel("Select lowerbound marker column:")
        self.lower_bound_marker_col = QComboBox()
        if len(self.model.regionprops_df) > 0:
            self.lower_bound_marker_col.addItems([None] + self.model.regionprops_df.columns)
        self.lower_bound_marker_col.currentTextChanged.connect(self._update_model_lowerbound)

        self.layout().addWidget(lower_col, 4, 0)
        self.layout().addWidget(self.lower_bound_marker_col, 5, 0)

        # The upper bound marker column dropdown
        upper_col = QLabel("Select upperbound marker column:")
        self.upper_bound_marker_col = QComboBox()
        if len(self.model.regionprops_df) > 0:
            self.upper_bound_marker_col.addItems([None] + self.model.regionprops_df.columns)
        self.upper_bound_marker_col.currentTextChanged.connect(self._update_model_upperbound)
        self.layout().addWidget(upper_col, 6, 0)
        self.layout().addWidget(self.upper_bound_marker_col, 7, 0)

        # Filter field for user to pass on strings to filter markers out.
        filter_label = QLabel("Remove markers with prefix (default: DNA,DAPI)")
        self.filter_field = QLineEdit(
            "DNA, DAPI",
            placeholderText="Prefixes separated by commas.",
        )
        self.filter_field.editingFinished.connect(self._update_filter)
        self.layout().addWidget(filter_label, 8, 0)
        self.layout().addWidget(self.filter_field, 9, 0)

        # Button to start validating all the input
        self.validate_button = QPushButton("Validate input")
        self.validate_button.clicked.connect(self._validate)
        self.layout().addWidget(self.validate_button, 10, 0)

        self.model.events.regionprops_df.connect(self._set_dropdown_marker_lowerbound)
        self.model.events.regionprops_df.connect(self._set_dropdown_marker_upperbound)

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
            self.model.mask_paths.extend(list(Path(folder).glob(ext)))

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

    def _set_dropdown_marker_upperbound(self):
        """Add items to dropdown menu for upperbound marker.

        This menu is a dropdown menu that responds to events of self.model.regionprops_df. When this changes it loads
        all the columns of this dataframe as items in the listwidget if a dataframe iis currently loaded.
        """
        self.upper_bound_marker_col.clear()
        region_props = self.model.regionprops_df
        if region_props is not None and len(region_props) > 0:
            self.upper_bound_marker_col.addItems(region_props.columns)

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
        self.model.sample_image_mapping = {
            i.stem.rstrip(".ome") if ".ome" in i.stem else i.stem: i for i in self.model.image_paths
        }
        self.model.sample_mask_mapping = {
            i.stem.rstrip(".ome") if ".ome" in i.stem else i.stem: i for i in self.model.mask_paths
        }

        # This is to retrieve the correct markers with their index in the images
        column_ls = list(self.model.regionprops_df.columns)
        lowerbound_index = column_ls.index(self.model.lower_bound_marker)
        upperbound_index = column_ls.index(self.model.upper_bound_marker)
        marker_columns = column_ls[lowerbound_index : upperbound_index + 1]
        self.model.markers = {marker: i for i, marker in enumerate(marker_columns)}
        n_markers = len(self.model.markers)

        for filter in self.model.marker_filter.split(","):
            # Do this because changing length would cause errors when deleting in loop.
            for marker in self.model.markers.copy():
                if filter.lower() in marker.lower():
                    del self.model.markers[marker]

        napari_notification(f"Removed {n_markers - len(self.model.markers)} out of list of {n_markers}.")

        self._scatter_widget = ScatterInputWidget(self.model, self.viewer)
        self.viewer.window.add_dock_widget(
            self._scatter_widget, name="cell_gater", area="right", menu=self._viewer.window.window_menu
        )

        self.model.validated = True
