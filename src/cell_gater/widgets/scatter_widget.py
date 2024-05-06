from __future__ import annotations

import sys
from itertools import product

import pandas as pd
from dask_image.imread import imread
from loguru import logger
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
)
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from napari import Viewer
from napari.layers import Image, Points
from napari.utils.history import (
    get_open_history,
)
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from cell_gater.model.data_model import DataModel
from cell_gater.utils.misc import napari_notification

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")

#Good to have features
#TODO Dynamic loading of markers, without reloading masks or DNA channel, so deprecate Load Sample and Marker button

#Ideas to maybe implement
#TODO log axis options for scatter plot
#TODO dynamic plotting of points on top of created polygons
#TODO save plots as images for QC, perhaps when saving gates run plotting function to go through all samples and markers and save plots

class ScatterInputWidget(QWidget):
    """Widget for a scatter plot with markers on the x axis and any dtype column on the y axis."""

    def __init__(self, model: DataModel, viewer: Viewer) -> None:
        super().__init__()

        self.setLayout(QGridLayout())
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

        self._model = model
        self._viewer = viewer

        logger.debug("ScatterInputWidget initialized")
        logger.debug(f"Model regionprops_df shape: {self.model.regionprops_df.shape}")
        logger.debug(f"Model regionprops_df columns: {self.model.regionprops_df.columns}")

        # Reason for setting current sample here as well is, so we can check whether we have to load a new mask.
        self._current_sample = None
        self._image = None
        self._mask = None

        # Dropdown of samples once directory is loaded
        selection_label = QLabel("Select sample:")
        self.sample_selection_dropdown = QComboBox()
        self.sample_selection_dropdown.addItems(sorted(self.model.samples, key=self.natural_sort_key) )
        self.sample_selection_dropdown.currentTextChanged.connect(self._on_sample_changed)

        marker_label = QLabel("Marker label:")
        self.marker_selection_dropdown = QComboBox()
        self.marker_selection_dropdown.addItems(self.model.markers)
        self.marker_selection_dropdown.currentTextChanged.connect(self._on_marker_changed)

        apply_button = QPushButton("Load Sample and Marker")
        apply_button.clicked.connect(self._load_images_and_scatter_plot)

        choose_y_axis_label = QLabel("Choose Y-axis")
        self.choose_y_axis_dropdown = QComboBox()
        self.choose_y_axis_dropdown.addItems(self.model.regionprops_df.columns)
        self.choose_y_axis_dropdown.setCurrentText("Area")
        self.choose_y_axis_dropdown.currentTextChanged.connect(self._on_y_axis_changed)

        # Reference channel
        DNA_to_show = QLabel("Select reference channel")
        self.ref_channel_dropdown = QComboBox()
        self.ref_channel_dropdown.addItems(self.model.markers_image_indices.keys())
        self.ref_channel_dropdown.currentTextChanged.connect(self.update_ref_channel)

        self.layout().addWidget(selection_label, 0, 0)
        self.layout().addWidget(self.sample_selection_dropdown, 0, 1)
        self.layout().addWidget(marker_label, 0, 2)
        self.layout().addWidget(self.marker_selection_dropdown, 0, 3)
        self.layout().addWidget(apply_button, 1, 0, 1, 4)
        self.layout().addWidget(choose_y_axis_label, 2, 0, 1, 1)
        self.layout().addWidget(self.choose_y_axis_dropdown, 2, 1, 1, 1)
        self.layout().addWidget(DNA_to_show, 2, 2, 1, 1)
        self.layout().addWidget(self.ref_channel_dropdown, 2, 3, 1, 1)

        # we have to do this because initially the dropdowns did not change texts yet so these variables are still None.
        self.model.active_sample = self.sample_selection_dropdown.currentText()
        self.model.active_marker = self.marker_selection_dropdown.currentText()
        self.model.active_y_axis = self.choose_y_axis_dropdown.currentText()
        self.model.active_ref_marker = self.ref_channel_dropdown.currentText()

        self._read_data(self.model.active_sample)
        self._load_layers(self.model.markers_image_indices[self.model.active_marker])

        # scatter plot
        self.scatter_canvas = PlotCanvas(self.model)
        self.layout().addWidget(self.scatter_canvas.fig, 3, 0, 1, 4)

        # slider
        self.slider_figure = Figure(figsize=(5, 1))
        self.slider_canvas = FigureCanvas(self.slider_figure)
        self.slider_ax = self.slider_figure.add_subplot(111)
        self.update_slider()
        self.layout().addWidget(self.slider_canvas, 4, 0, 1, 4)

        # plot points button
        plot_points_button = QPushButton("Plot Points")
        plot_points_button.clicked.connect(self.plot_points)
        self.layout().addWidget(plot_points_button, 5,0,1,1)

        # Initialize gates dataframe 
        sample_marker_combinations = list(product(
            self.model.regionprops_df["sample_id"].unique(),
            self.model.markers
        ))
        self.model.gates = pd.DataFrame(sample_marker_combinations, columns=["sample_id", "marker_id"])
        self.model.gates["gate_value"] = float(0)

        # gate buttons
        save_gate_button = QPushButton("Save Gate")
        save_gate_button.clicked.connect(self.save_gate)
        self.layout().addWidget(save_gate_button, 5, 1, 1, 1)

        load_gates_button = QPushButton("Load Gates Dataframe")
        load_gates_button.clicked.connect(self.load_gates_dataframe)
        self.layout().addWidget(load_gates_button, 5, 2, 1, 1)

        save_gates_dataframe_button = QPushButton("Save Gates Dataframe")
        save_gates_dataframe_button.clicked.connect(self.save_gates_dataframe)
        self.layout().addWidget(save_gates_dataframe_button, 5, 3, 1, 1)


    ########################### FUNCTIONS ###########################

    def update_ref_channel(self):
        self.model.active_ref_marker = self.ref_channel_dropdown.currentText()
        self._load_images_and_scatter_plot()

    ###################
    ### PLOT POINTS ###
    ###################

    #TODO dynamic plotting of points on top of created polygons

    def plot_points(self):
        """Plot positive cells in Napari."""
        assert self.model.active_sample is not None
        assert self.model.active_marker is not None

        df = self.model.regionprops_df
        df = df[df["sample_id"] == self.model.active_sample]

        for layer in self.viewer.layers:
            if isinstance(layer, Points):
                layer.visible = False

        self.viewer.add_points(
            df[df[self.model.active_marker] > self.model.current_gate][["Y_centroid", "X_centroid"]],
            name=f"Gate: {round(self.model.current_gate)}  {self.model.active_sample} {self.model.active_marker}",
            face_color="#ff00ff",
            edge_color="yellow",
            size=8,
            opacity=0.5,
        )

    ####################################
    ### GATES DATAFRAME INPUT OUTPUT ###
    ####################################

    def load_gates_dataframe(self):
        file_path, _ = self._file_dialog()
        if file_path:
            self.model.gates = pd.read_csv(file_path)
        self.model.gates["sample_id"] = self.model.gates["sample_id"].astype(str)
        # check if dataframe has samples and markers
        assert "sample_id" in self.model.gates.columns
        assert "marker_id" in self.model.gates.columns
        assert "gate_value" in self.model.gates.columns
        # check if dataframe has the same samples and markers as the regionprops_df
        assert set(self.model.gates["sample_id"].unique()) == set(self.model.regionprops_df["sample_id"].unique()), "Samples in gates dataframe do not match samples in regionprops dataframe."
        assert set(self.model.gates["marker_id"].unique()) == set(self.model.markers)

    def save_gates_dataframe(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Gates Dataframe", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            self.model.gates.to_csv(fileName, index=False)
            logger.info(f"File saved to: {fileName}")

    def save_gate(self):
        if self.model.current_gate == 0:
            napari_notification("Gate not saved, please select a gate value.")
        if self.access_gate() == self.model.current_gate:
            napari_notification("No changes detected.")
        if self.access_gate() != self.model.current_gate:
            napari_notification(f"Old gate {self.access_gate().round(2)} overwritten to {self.model.current_gate.round(2)}")
        self.model.gates.loc[
            (self.model.gates["sample_id"] == self.model.active_sample) & 
            (self.model.gates["marker_id"] == self.model.active_marker), 
            "gate_value"] = self.model.current_gate
        assert self.access_gate() == self.model.current_gate
        logger.debug(f"Gate saved: {self.model.current_gate}")

    def access_gate(self):
        assert self.model.active_sample is not None
        assert self.model.active_marker is not None
        try:
            gate_value = self.model.gates.loc[
                (self.model.gates["sample_id"] == self.model.active_sample) &
                (self.model.gates["marker_id"] == self.model.active_marker),
                "gate_value"].values[0]
        except IndexError:
            logger.debug(f"Gate not found for {self.model.active_sample} and {self.model.active_marker}.")
            logger.debug(self.model.gates)
        assert isinstance(gate_value, float)
        return gate_value

    ##########################
    #### SLIDER FUNCTIONS ####
    ##########################

    def get_min_max_median_step(self) -> tuple:
        df = self.model.regionprops_df
        df = df[df["sample_id"] == self.model.active_sample]
        min = df[self.model.active_marker].min() + 1
        max = df[self.model.active_marker].max()
        init = df[self.model.active_marker].median()
        step = min / 100
        return min, max, init, step

    def slider_changed(self, val):
        self.model._current_gate = val
        self.scatter_canvas.update_vertical_line(val)
        self.scatter_canvas.fig.draw() 

    def update_slider(self):
        min, max, init, step = self.get_min_max_median_step()
        self.slider_ax.clear()
        self.slider = Slider(self.slider_ax, "Gate", min, max, valinit=init, valstep=step, color="black")
        self.slider.on_changed(self.slider_changed)
        self.slider_canvas.draw()

    ##########################
    ###### LOADING DATA ######
    ##########################

    def update_plot(self):
        self.scatter_canvas.ax.clear()
        self.scatter_canvas.plot_scatter_plot(self.model)
        self.scatter_canvas.fig.draw()

    def _load_images_and_scatter_plot(self):
        self._clear_layers(clear_all=True)
        self._read_data(self.model.active_sample)
        # active marker is a string
        # markers is dict with marker_name_string:index (based on dropdowns)
        self._load_layers(self.model.markers_image_indices[self.model.active_marker])
        logger.debug(f"loading index {self.model.markers_image_indices[self.model.active_marker]}")
        self.update_plot()
        self.update_slider()

    def _read_data(self, sample: str | None) -> None:
        logger.info(f"Reading data for sample {sample}.")
        if sample is not None:
            logger.debug(f"Reading image from {self.model.sample_image_mapping[sample]}.")
            image_path = self.model.sample_image_mapping[sample]
            logger.debug(f"Reading mask from {self.model.sample_mask_mapping[sample]}.")
            mask_path = self.model.sample_mask_mapping[sample]

            self._image = imread(image_path)
            self._mask = imread(mask_path)

    def _load_layers(self, marker_index):

        # if self.model.active_sample != self._current_sample:
        #     self._current_sample = copy(self.model.active_sample)

        self.viewer.add_image(
            self._image[self.model.markers_image_indices[self.model.active_ref_marker]],
            name="Ref:"+ self.model.active_ref_marker + "_" + self.model.active_sample,
            blending="additive", visible=False
        )
        self.viewer.add_labels(
            self._mask,
            name="mask_" + self.model.active_sample,
            visible=False, opacity=0.4
        )
        self.viewer.add_image(
            self._image[marker_index],
            name=self.model.active_marker + "_" + self.model.active_sample,
            blending="additive",
        )

    def _on_sample_changed(self):
        self.model.active_sample = self.sample_selection_dropdown.currentText()
    def _on_marker_changed(self):
        self.model.active_marker = self.marker_selection_dropdown.currentText()

    def _clear_layers(self, clear_all: bool) -> None:
        """Remove all layers upon changing sample."""
        if clear_all is True:
            self.viewer.layers.select_all()
            self.viewer.layers.remove_selected()
        else:
            for layer in self.viewer.layers:
                if isinstance(layer, Image):
                    self.viewer.layers.remove(layer)

    # def _reinitiate_marker_selection_dropdown(self) -> None:
    #     """Reiniatiate the marker selection dropdown after sample has changed."""
    #     # This is preemptively added for clearing visual completed feedback once implemented.
    #     # We also block the outgoing signal in order not to update the layer when there is no current active marker.
    #     self.marker_selection_dropdown.blockSignals(True)
    #     self.marker_selection_dropdown.clear()
    #     self.marker_selection_dropdown.addItems(self.model.markers)
    #     self.marker_selection_dropdown.blockSignals(False)

    def _set_samples_dropdown(self) -> None:
        """Set the items for the samples dropdown QComboBox."""
        if (region_props := self.model.regionprops_df) is not None:
            self.model.samples = list(region_props["sample_id"].cat.categories)

            # New directory loaded so we reload the dropdown items
            self.sample_selection_dropdown.clear()
            if len(self.model.samples) > 0:
                self.sample_selection_dropdown.addItems([None])
                self.sample_selection_dropdown.addItems(self.model.samples)

    def _on_y_axis_changed(self):
        """Set active y-axis and update the scatter plot."""
        self.model.active_y_axis = self.choose_y_axis_dropdown.currentText()
        self.update_plot()

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

    def natural_sort_key(self, s):
        """Key function for natural sorting."""
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

    @property
    def model(self) -> DataModel:
        """The dataclass model that stores information required for cell_gating."""
        return self._model

    @model.setter
    def model(self, model: DataModel) -> None:
        self._model = model

    @property
    def viewer(self) -> Viewer:
        """The napari Viewer."""
        return self._viewer

    @viewer.setter
    def viewer(self, viewer: Viewer) -> None:
        self._viewer = viewer

class PlotCanvas():
    """The canvas class for the gating scatter plot."""

    def __init__(self, model: DataModel):

        self.model = DataModel() if model is None else model
        self.fig = FigureCanvas(Figure()) 
        self.fig.figure.subplots_adjust(left=0.1, bottom=0.1)
        self.ax = self.fig.figure.subplots()
        self.ax.set_title("Scatter plot")
        #run function to plot scatter plot
        self.plot_scatter_plot(self.model)

    @property
    def model(self) -> DataModel:
        """The dataclass model that stores information required for cell_gating."""
        return self._model

    @model.setter
    def model(self, model: DataModel) -> None:
        self._model = model

    def plot_scatter_plot(self, model: DataModel) -> None:
        """Plot the scatter plot."""
        assert self.model.active_marker is not None
        assert self.model.active_sample is not None

        df = self.model.regionprops_df
        df = df[df["sample_id"] == self.model.active_sample]

        logger.debug(f"Plotting scatter plot for {self.model.active_sample} and {self.model.active_marker}.")

        self.ax.scatter(
            x=df[self.model.active_marker],
            y=df[self.model.active_y_axis],
            color="steelblue",
            ec="white",
            lw=0.1,
            alpha=1.0,
            s=80000 / int(df.shape[0]),
        )
        # Set x-axis limits
        self.ax.set_xlim(df[self.model.active_marker].min(), df[self.model.active_marker].max())
        self.ax.set_ylabel(self.model.active_y_axis)
        self.ax.set_xlabel(f"{self.model.active_marker} intensity")

        logger.debug(f"The current gate is {self.model.current_gate}.")
        if self.model.current_gate > 0.0:
            self.ax.axvline(x=self.model.current_gate, color="red", linewidth=1.0, linestyle="--")
        else:
            self.ax.axvline(x=1, color="red", linewidth=1.0, linestyle="--")

    def update_vertical_line(self, x_position):
        """Update the position of the vertical line."""
        self.ax.lines[0].set_xdata(x_position)