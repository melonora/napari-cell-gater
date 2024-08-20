from __future__ import annotations

import sys
from itertools import product


import numpy as np
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
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from cell_gater.model.data_model import DataModel
from cell_gater.utils.misc import napari_notification

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")

# Ideas to maybe implement
# TODO dynamic plotting of points on top of created polygons
# TODO save plots as images for QC, perhaps when saving gates run plotting function to go through all samples and markers and save plots


class ScatterInputWidget(QWidget):
    """Widget for a scatter plot with markers on the x axis and any dtype column on the y axis."""

    def __init__(self, model: DataModel, viewer: Viewer) -> None:
        super().__init__()

        self.setLayout(QGridLayout())
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

        self._model = model
        self._viewer = viewer

        self._current_sample = None
        self._image = None
        self._mask = None

        selection_label = QLabel("Select sample:")
        self.sample_selection_dropdown = QComboBox()
        self.sample_selection_dropdown.addItems(sorted(self.model.samples, key=self.natural_sort_key))
        self.sample_selection_dropdown.currentTextChanged.connect(self._on_sample_changed)

        marker_label = QLabel("Marker label:")
        self.marker_selection_dropdown = QComboBox()
        self.marker_selection_dropdown.addItems(self.model.markers)
        self.marker_selection_dropdown.currentTextChanged.connect(self._on_marker_changed)

        choose_y_axis_label = QLabel("Choose Y-axis")
        self.choose_y_axis_dropdown = QComboBox()
        self.choose_y_axis_dropdown.addItems(self.model.regionprops_df.columns)
        self.choose_y_axis_dropdown.setCurrentText("Area")
        self.choose_y_axis_dropdown.currentTextChanged.connect(self._on_y_axis_changed)

        ref_channel = QLabel("Select reference channel")
        self.ref_channel_dropdown = QComboBox()
        self.ref_channel_dropdown.addItems(self.model.markers_image_indices.keys())
        self.ref_channel_dropdown.currentTextChanged.connect(self.update_ref_channel)

        log_label = QLabel("Logarithmic scale")
        self.log_scale_dropdown = QComboBox()
        self.log_scale_dropdown.addItems(["No", "Yes"])
        self.log_scale_dropdown.currentTextChanged.connect(self.update_log_scale)

        plot_type_label = QLabel("Plot type")
        self.plot_type_dropdown = QComboBox()
        self.plot_type_dropdown.addItems(["Scatter", "Hexbin"])
        self.plot_type_dropdown.currentTextChanged.connect(self.update_plot_type)

        manual_input_gate_label = QLabel("Manual gate input:")
        self.manual_gate_input_text = QLineEdit()
        self.manual_gate_input_text.setPlaceholderText("Gate value (linear scale)")
        self.manual_gate_input_QPushButton = QPushButton("Set gate manually")
        self.manual_gate_input_QPushButton.clicked.connect(self.manual_gate_input)

        # self.layout().addWidget(object, int row, int column, int rowSpan = 1, int columnSpan = 1)
        self.layout().addWidget(selection_label, 0, 0)
        self.layout().addWidget(self.sample_selection_dropdown, 0, 1)
        self.layout().addWidget(marker_label, 0, 2)
        self.layout().addWidget(self.marker_selection_dropdown, 0, 3)
        self.layout().addWidget(choose_y_axis_label, 1, 0, 1, 1)
        self.layout().addWidget(self.choose_y_axis_dropdown, 1, 1, 1, 1)
        self.layout().addWidget(ref_channel, 1, 2, 1, 1)
        self.layout().addWidget(self.ref_channel_dropdown, 1, 3, 1, 1)
        self.layout().addWidget(log_label, 2, 0, 1, 1)
        self.layout().addWidget(self.log_scale_dropdown, 2, 1, 1, 1)
        self.layout().addWidget(plot_type_label, 2, 2, 1, 1)
        self.layout().addWidget(self.plot_type_dropdown, 2, 3, 1, 1)

        # we have to do this because initially the dropdowns did not change texts yet so these variables are still None.
        self.model.active_sample = self.sample_selection_dropdown.currentText()
        self.model.active_marker = self.marker_selection_dropdown.currentText()
        self.model.active_y_axis = self.choose_y_axis_dropdown.currentText()
        self.model.active_ref_marker = self.ref_channel_dropdown.currentText()
        self._read_marker_image()
        self._read_mask_image()
        self._load_labels()
        self._load_image()
        self.load_ref_channel()

        # scatter plot
        self.scatter_canvas = PlotCanvas(self.model)
        self.layout().addWidget(self.scatter_canvas.fig, 3, 0, 1, 4)

        # Initialize gates dataframe
        sample_marker_combinations = list(product(self.model.regionprops_df["sample_id"].unique(), self.model.markers_image_indices.keys()))
        self.model.gates = pd.DataFrame(sample_marker_combinations, columns=["sample_id", "marker_id"])
        self.model.gates["gate_value"] = float(0)

        # slider
        self.slider_figure = Figure(figsize=(5, 1))
        self.slider_canvas = FigureCanvas(self.slider_figure)
        self.slider_ax = self.slider_figure.add_subplot(111)
        self.update_slider()
        self.layout().addWidget(self.slider_canvas, 4, 0, 1, 4)

        #manual input gate
        self.layout().addWidget(manual_input_gate_label, 5, 0, 1, 1)
        self.layout().addWidget(self.manual_gate_input_text, 5, 1, 1, 1)
        self.layout().addWidget(self.manual_gate_input_QPushButton, 5, 2, 1, 2)

        # plot points button
        plot_points_button = QPushButton("Plot Points")
        plot_points_button.clicked.connect(self.plot_points)
        self.layout().addWidget(plot_points_button, 6,0,1,1)

        # autosave path
        self.csv_path = None

        # gate buttons
        save_gate_button = QPushButton("Save Gate")
        save_gate_button.clicked.connect(self.save_gate)
        self.layout().addWidget(save_gate_button, 6, 1, 1, 1)
        load_gates_button = QPushButton("Load existing gates")
        load_gates_button.clicked.connect(self.load_gates_dataframe)
        self.layout().addWidget(load_gates_button, 6, 2, 1, 2)

    #################################################################
    ########################### FUNCTIONS ###########################
    #################################################################

    def update_ref_channel(self):
        """Update the reference channel for visualization."""
        self.model.active_ref_marker = self.ref_channel_dropdown.currentText()
        self.load_ref_channel()

    def update_log_scale(self):
        """Update the log scale for the scatter plot."""
        logger.debug(f"Log scale dropdown changed to {self.log_scale_dropdown.currentText()}.")
        if self.log_scale_dropdown.currentText() == "Yes":
            self.model.log_scale = True
        elif self.log_scale_dropdown.currentText() == "No":
            self.model.log_scale = False
        logger.debug(f"Log scale set to {self.model.log_scale}.")
        self.update_slider()
        self.update_plot()

    def update_plot_type(self):
        """Update the plot type for the plot."""
        logger.debug(f"Plot type dropdown changed to {self.plot_type_dropdown.currentText()}.")
        if self.plot_type_dropdown.currentText() == "Scatter":
            self.model.plot_type = "scatter"
        elif self.plot_type_dropdown.currentText() == "Hexbin":
            self.model.plot_type = "hexbin"
        logger.debug(f"Plot type set to {self.model.plot_type}.")
        self.update_plot()

    def manual_gate_input(self):
        """Manual gate input."""
        logger.info("Manual gate input initiated.")
        logger.debug(f"Manual gate input: {self.manual_gate_input_text.text()}")
        self.model.current_gate = float(self.manual_gate_input_text.text())
        self.update_slider()
        self.update_plot()

    def update_plot(self):
        """Update the scatter plot."""
        self.scatter_canvas.ax.clear()
        self.scatter_canvas.plot_scatter_plot()
        self.scatter_canvas.fig.draw()
        if self.access_gate() > 0.0:
            self.scatter_canvas.fixed_vertical_line()


    ###################
    ### PLOT POINTS ###
    ###################

    # TODO dynamic plotting of points on top of created polygons

    def plot_points(self, ref_gate=False):
        """Plot positive cells in Napari."""
        df = self.model.regionprops_df
        df = df[df["sample_id"] == self.model.active_sample]

        for layer in self.viewer.layers:
            if isinstance(layer, Points):
                layer.visible = False

        logger.info("Plotting points in Napari.")
        logger.debug(f"sample: {self.model.active_sample}")
        logger.debug(f"marker: {self.model.active_marker}")
        logger.debug(f"current_gate: {self.model.current_gate}")

        if ref_gate:
            ref_gate_value = self.access_gate()
            self.viewer.add_points(
                df[df[self.model.active_marker] > ref_gate_value][["Y_centroid", "X_centroid"]],
                name=f"Gate: {round(ref_gate_value)} | {self.model.active_sample}:{self.model.active_marker}",
                face_color="yellow", edge_color="black", size=12, opacity=0.6)
        else:
            self.viewer.add_points(
                df[df[self.model.active_marker] > self.model.current_gate][["Y_centroid", "X_centroid"]],
                name=f"Gate: {round(self.model.current_gate)} | {self.model.active_sample}:{self.model.active_marker}",
                face_color="yellow", edge_color="black", size=12, opacity=0.6)

    ####################################
    ### GATES DATAFRAME INPUT OUTPUT ###
    ####################################

    def load_gates_dataframe(self):
        """Load gates dataframe from csv."""
        file_path, _ = self._file_dialog()
        if file_path:
            self.model.gates = pd.read_csv(file_path)

        self.model.gates["sample_id"] = self.model.gates["sample_id"].astype(str)
        assert "sample_id" in self.model.gates.columns, "sample_id column not found in gates dataframe."
        assert "marker_id" in self.model.gates.columns, "marker_id column not found in gates dataframe."
        assert "gate_value" in self.model.gates.columns, "gate_value column not found in gates dataframe."
        assert set(self.model.gates["sample_id"].unique()) == set(self.model.regionprops_df["sample_id"].unique()), "Samples do not match."
        assert set(self.model.gates["marker_id"].unique()) == set(self.model.markers_image_indices.keys()), "Markers don't match, pick the same quantification files."

        self.csv_path = file_path
        logger.debug(f"Gates dataframe from {file_path} loaded and checked.")
        logger.debug(f"self.access_gate(): {self.access_gate()}")
        self.scatter_canvas.fixed_vertical_line()
        self.plot_points(ref_gate=True)
        napari_notification(f"Gates dataframe loaded from: {file_path}")

    def select_save_directory(self):
        """Select the directory where the gates CSV file will be saved."""
        if self.csv_path:
            logger.debug(f"Save directory already selected: {self.csv_path}")
            napari_notification(f"Save directory already selected: {self.csv_path}")
        else:
            fileName, _ = QFileDialog.getSaveFileName(self, "Save gates in csv", "", "CSV Files (*.csv);;All Files (*)", options=QFileDialog.Options())
            if fileName:
                self.csv_path = fileName
                logger.debug(f"Save directory selected: {self.csv_path}")
                napari_notification(f"Save directory selected: {self.csv_path}")

    def save_gates_dataframe(self):
        """Save gates dataframe to csv."""
        if not self.csv_path:
            self.select_save_directory()

        if self.csv_path:
            self.model.gates.to_csv(self.csv_path, index=False)
            logger.debug(f"Gates dataframe saved to {self.csv_path}")
            napari_notification(f"File saved to: {self.csv_path}")

    def save_gate(self):
        """Save the current gate value to the gates dataframe."""
        if self.model.current_gate == 0:
            napari_notification("Gate not saved, please select a gate value.")
            return
        if self.access_gate() == self.model.current_gate:
            napari_notification("No changes detected.")
            return
        if self.access_gate() != self.model.current_gate:
            napari_notification(f"Old gate {round(self.access_gate(), 2)} overwritten to {round(self.model.current_gate, 2)}")

        self.model.gates.loc[
            (self.model.gates["sample_id"] == self.model.active_sample) &
            (self.model.gates["marker_id"] == self.model.active_marker),
            "gate_value"] = self.model.current_gate

        assert self.access_gate() == self.model.current_gate
        self.save_gates_dataframe()
        self.scatter_canvas.fixed_vertical_line()
        logger.debug(f"Gate saved: {self.model.current_gate}")

    def access_gate(self):
        """Access the current gate value."""
        assert self.model.active_sample is not None
        assert self.model.active_marker is not None
        gate_value = self.model.gates.loc[
            (self.model.gates["sample_id"] == self.model.active_sample) &
            (self.model.gates["marker_id"] == self.model.active_marker),
            "gate_value"].values[0]
        assert isinstance(gate_value, float)
        return gate_value

    ##########################
    #### SLIDER FUNCTIONS ####
    ##########################

    def get_min_max_median_step(self) -> tuple:
        """Get the min, max, median and step for the slider."""
        df = self.model.regionprops_df
        df = df[df["sample_id"] == self.model.active_sample]
        if self.model.log_scale:
            marker_values = df[self.model.active_marker] + 1
            min = np.log10(marker_values.min())
            max = np.log10(marker_values.max())
            init = np.log10(marker_values.median())
            step = 0.0001
        elif not self.model.log_scale:
            min = df[self.model.active_marker].min() + 1
            max = df[self.model.active_marker].max()
            init = df[self.model.active_marker].median()
            step = min / 100
        logger.debug(f"min: {min}, max: {max}, init: {init}, step: {step}")
        return min, max, init, step

    def slider_changed(self, val):
        """Update the current gate value and the vertical line on the scatter plot."""
        logger.debug(f"Slider changed to {val}. log10? {self.model.log_scale}")
        if self.model.log_scale:
            self.model._current_gate = 10**val
        elif not self.model.log_scale:
            self.model._current_gate = val
        self.scatter_canvas.update_vertical_line(val)

    def update_slider(self):
        """Update the slider with the min, max, median and step values."""
        logger.debug("Updating slider.")
        min, max, init, step = self.get_min_max_median_step()
        self.slider_ax.clear()
        self.slider = Slider(self.slider_ax, "Gate", min, max, valinit=init, valstep=step, color="black")
        self.slider.on_changed(self.slider_changed)
        self.slider_canvas.draw()

    ##########################
    ###### LOADING DATA ######
    ##########################

    def _read_marker_image(self):
        """Read the marker image for the selected marker."""
        image_path = self.model.sample_image_mapping[self.model.active_sample]
        self._image = imread(image_path)

    def _read_mask_image(self):
        """Read the mask image for the selected sample."""
        mask_path = self.model.sample_mask_mapping[self.model.active_sample]
        self._mask = imread(mask_path)

    def load_ref_channel(self):
        """Load the reference channel."""
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and "REF:" in layer.name:
                self.viewer.layers.remove(layer)
        self.viewer.add_image(
            self._image[self.model.markers_image_indices[self.model.active_ref_marker]],
            name="REF:" + self.model.active_ref_marker + "_" + self.model.active_sample,
            blending="additive", visible=True, colormap="magenta")

    def _load_labels(self):
        """Load the labels into the napari viewer."""
        self.viewer.add_labels(self._mask, name="mask_" + self.model.active_sample, visible=False, opacity=0.4)

    def _load_image(self):
        """Load the image into the napari viewer."""
        marker_index = self.model.markers_image_indices[self.model.active_marker]
        self.viewer.add_image(self._image[marker_index], name=self.model.active_marker + "_" + self.model.active_sample, blending="additive", colormap="green")

    def _on_sample_changed(self):
        """Set active sample, load mask, image, and ref."""
        self.model.active_sample = self.sample_selection_dropdown.currentText()
        self._clear_layers()
        self._read_marker_image()
        self._read_mask_image()
        self.load_ref_channel()
        self._load_labels()
        self._load_image()
        self.update_plot()
        self.update_slider()
        if self.access_gate() > 0.0:
            self.scatter_canvas.fixed_vertical_line()
            self.plot_points(ref_gate=True)

    def _on_marker_changed(self):
        """Set active marker, load only new marker image."""
        self.model.active_marker = self.marker_selection_dropdown.currentText()
        for layer in list(self.viewer.layers):
            if isinstance(layer, Image) and "REF:" not in layer.name:  # noqa: SIM114
                self.viewer.layers.remove(layer)
            elif isinstance(layer, Points):
                self.viewer.layers.remove(layer)
        self._read_marker_image()
        self._load_image()
        self.update_plot()
        self.update_slider()
        if self.access_gate() > 0.0:
            self.scatter_canvas.fixed_vertical_line()
            self.plot_points(ref_gate=True)

    def _clear_layers(self) -> None:
        """Remove all layers upon changing sample."""
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()

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

class PlotCanvas:
    """The canvas class for the gating scatter plot."""

    def __init__(self, model: DataModel):
        self.model = DataModel() if model is None else model
        self.fig = FigureCanvas(Figure())
        self.fig.figure.subplots_adjust(left=0.1, bottom=0.1)
        self.ax = self.fig.figure.subplots()
        self.ax.set_title("Scatter plot")
        self.plot_scatter_plot()

    @property
    def model(self) -> DataModel:
        """The dataclass model that stores information required for cell_gating."""
        return self._model

    @model.setter
    def model(self, model: DataModel) -> None:
        self._model = model

    def plot_scatter_plot(self):
        """Plot the scatter plot."""
        df = self.model.regionprops_df
        df = df[df["sample_id"] == self.model.active_sample]

        x_data = df[self.model.active_marker]
        y_data = df[self.model.active_y_axis]

        if self.model.log_scale:
            x_data = x_data + 1
            self.ax.set_xscale("log")
            self.ax.set_yscale("log")

        if self.model.plot_type == "scatter":
            self.ax.scatter(x=x_data, y=y_data,
                color="steelblue", ec="white",
                lw=0.1, alpha=1.0, s=80000 / int(df.shape[0]))

        elif self.model.plot_type == "hexbin" and self.model.log_scale is True:
            self.ax.hexbin(x=x_data, y=y_data, gridsize=50, cmap="viridis",
                bins="log",xscale="log", yscale="log")

        elif self.model.plot_type == "hexbin" and self.model.log_scale is False:
            self.ax.hexbin(x=x_data, y=y_data, gridsize=50, cmap="viridis")

        # Set x-axis limits
        self.ax.set_xlim(x_data.min(), x_data.max())
        self.ax.set_ylabel(self.model.active_y_axis)
        self.ax.set_xlabel(f"{self.model.active_marker}")

        # Initate vertical line
        self.ax.axvline(x=1.0, color="red", linewidth=1.0, linestyle="--")

    def update_vertical_line(self, x_position):
        """Update the position of the vertical line."""
        logger.debug(f"Updating vertical line to {x_position}, self.ax.lines: {self.ax.lines}")
        x_position = 10**(x_position) if self.model.log_scale else x_position
        self.ax.lines[0].set_xdata(x=[x_position])
        self.fig.draw()

    def access_gate(self):
        """Access the current gate value."""
        assert self.model.active_sample is not None
        assert self.model.active_marker is not None
        gate_value = self.model.gates.loc[
            (self.model.gates["sample_id"] == self.model.active_sample) &
            (self.model.gates["marker_id"] == self.model.active_marker),
            "gate_value"].values[0]
        assert isinstance(gate_value, float)
        return gate_value

    def fixed_vertical_line(self):
        """Draw a vertical fixed line on the scatter plot."""
        if self.access_gate() > 0.0:

            if len(self.ax.lines) == 1:
                self.ax.axvline(x=self.access_gate(), color="blue", linewidth=1.0, linestyle="-.")
            if len(self.ax.lines) > 1:
                self.ax.lines[1].remove()
                self.ax.axvline(x=self.access_gate(), color="blue", linewidth=1.0, linestyle="-.")

            df = self.model.regionprops_df
            df = df[df["sample_id"] == self.model.active_sample]
            y_data = df[self.model.active_y_axis]
            if len(self.ax.texts) > 0:
                self.ax.texts[0].set_position((self.access_gate(), y_data.max()))
                self.ax.texts[0].set_text(f" Gate:{round(self.access_gate(),3)}")
            else:
                self.ax.text(x=self.access_gate(), y=y_data.max(), s=f" Gate:{round(self.access_gate(),3)}", color="blue")
            self.fig.draw()
