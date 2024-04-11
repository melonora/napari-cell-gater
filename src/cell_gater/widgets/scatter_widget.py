from __future__ import annotations

from copy import copy

from dask_image.imread import imread
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from napari import Viewer
from napari.layers import Image
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QGridLayout,
    QSlider
)

from matplotlib.widgets import Slider
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from cell_gater.model.data_model import DataModel
from  cell_gater.utils.misc import napari_notification  
import numpy as np
import sys
from loguru import logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")

#TODO Critical issue with getting data from self.model.regionprops_df[self.model.active_marker]
# The data selecting is weird, I think that the DNA columns are not being removed from the model.regionprops_df
# causing a shift.. perhaps we should create a filtered dataframe.. 
# still not sure why if we are using the column header string to access marker specific data

#tracing it back
# after some manual exploration, it seems that channel calling is bringing one channel after the desired one. 
# Channel 10 is calling channel 11.

# the index used to load images comes from csv columns
# index in images is 0-based
# Two things have to be done:
# Remove CellID and morphological features from the markers list, even for dropdown
# correct the index used to load images



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
        self.sample_selection_dropdown.addItems(self.model.samples)
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

        self.layout().addWidget(selection_label, 0, 0)
        self.layout().addWidget(self.sample_selection_dropdown, 1, 0)
        self.layout().addWidget(marker_label, 2, 0)
        self.layout().addWidget(self.marker_selection_dropdown, 3, 0)
        self.layout().addWidget(apply_button, 4, 0)
        self.layout().addWidget(choose_y_axis_label, 5, 0)
        self.layout().addWidget(self.choose_y_axis_dropdown, 6, 0)

        # we have to do this because initially the dropdowns did not change texts yet so these variables are still None.
        self.model.active_sample = self.sample_selection_dropdown.currentText()
        self.model.active_marker = self.marker_selection_dropdown.currentText()
        self.model.active_y_axis = self.choose_y_axis_dropdown.currentText()

        self._read_data(self.model.active_sample)
        self._load_layers(self.model.markers[self.model.active_marker])

        self.scatter_canvas = PlotCanvas(self.model)
        self.layout().addWidget(self.scatter_canvas.fig, 7, 0)
        # self.update_plot()

        #maybe should do the same for the slider as the plotcanvas 

        self.slider_figure = Figure(figsize=(5, 1))
        self.slider_canvas = FigureCanvas(self.slider_figure)
        self.slider_ax = self.slider_figure.add_subplot(111)
        min, max, init, step = self.get_min_max_median_step()
        self.slider = Slider(self.slider_ax, "Gate", min, max, valinit=init, valstep=step, color="black")
        self.slider.on_changed(self.slider_changed)
        self.layout().addWidget(self.slider_canvas, 9, 0)


    def get_min_max_median_step(self) -> tuple:
        df = self.model.regionprops_df
        df = df[df["sample_id"] == self.model.active_sample]
        min = df[self.model.active_marker].min()
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

    def update_plot(self):
        self.scatter_canvas.ax.clear()
        self.scatter_canvas.plot_scatter_plot(self.model)
        self.scatter_canvas.fig.draw()

    def _load_images_and_scatter_plot(self):
        self._clear_layers(clear_all=True)
        self._read_data(self.model.active_sample)
        self._load_layers(self.model.markers[self.model.active_marker])
        logger.debug(f"loading index {self.model.markers[self.model.active_marker]}")
        self.update_plot()
        self.update_slider()

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

    def _read_data(self, sample: str | None) -> None:
        logger.info(f"Reading data for sample {sample}.")
        if sample is not None:
            logger.debug(f"Reading image from {self.model.sample_image_mapping[sample]}.")
            image_path = self.model.sample_image_mapping[sample]
            logger.debug(f"Reading mask from {self.model.sample_mask_mapping[sample]}.")
            mask_path = self.model.sample_mask_mapping[sample]

            self._image = imread(image_path)
            self._mask = imread(mask_path)
            #TODO add DNA channel to the layers added

    def _load_layers(self, marker_index):

        # FOR NOW
        # if self.model.active_sample != self._current_sample:
        #     self._current_sample = copy(self.model.active_sample)
        logger.debug(f"_load_layers(self, {marker_index})")

        self.viewer.add_labels(
            self._mask, 
            name="mask_" + self.model.active_sample,
            visible=False, opacity=0.4
        )
        logger.debug(f"Added mask for {self.model.active_sample}.")

        self.viewer.add_image(
            self._image[marker_index],
            name=self.model.active_marker + "_" + self.model.active_sample,
            blending="additive",
        )
        logger.debug(f"Added image: marker_index {marker_index} of the image with shape {self._image.shape}. ")

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
            

    
    # # TODO add the plot to a widget and display it
    #
    # def plot_points(self, model: DataModel) -> None:
    #     """Plot positive cells in Napari."""
    #     if self.model.active_marker is not None:
    #         marker = self.model.active_marker
    #     if self.model.active_sample is not None:
    #         sample = self.model.active_sample
    #
    #     viewer = self.model.viewer
    #     df = self.model.regionprops_df
    #     df = df[df["sample_id"] == self.model.active_sample]
    #     gate = self.model.gates.loc[marker, sample]
    #
    #     viewer.add_points(
    #         df[df[marker] > gate][["X_centroid", "Y_centroid"]],
    #         name=f"{gate} and its positive cells",
    #         face_color="red",
    #         edge_color="black",
    #         size=15,
    #     )
    #
    # def plot_points_button(self):
    #     """Plot points button."""
    #     self.plot_points_button = QPushButton("Plot Points")
    #     self.plot_points_button.clicked.connect(self.plot_points)
    #     self.layout().addWidget(self.plot_points_button, 1, 2)  # not sure where to put this button
