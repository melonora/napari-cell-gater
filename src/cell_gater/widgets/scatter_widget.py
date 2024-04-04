from __future__ import annotations

from copy import copy

from dask_image.imread import imread
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from napari import Viewer
from napari.layers import Image
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QGridLayout,
)

from cell_gater.model.data_model import DataModel


class ScatterInputWidget(QWidget):
    """Widget for a scatter plot with markers on the x axis and any dtype column on the y axis."""

    def __init__(self, model: DataModel, viewer: Viewer) -> None:
        super().__init__()

        self.setLayout(QGridLayout())
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

        self._model = model
        self._viewer = viewer

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

        choose_y_axis_label = QLabel("Choose Y-axis")
        self.choose_y_axis_dropdown = QComboBox()
        self.choose_y_axis_dropdown.addItems([None] + self.model.regionprops_df.columns)
        self.choose_y_axis_dropdown.setCurrentText("Area")
        self.choose_y_axis_dropdown.currentTextChanged.connect(self._on_y_axis_changed)

        self.layout().addWidget(selection_label, 0, 0)
        self.layout().addWidget(self.sample_selection_dropdown, 1, 0)
        self.layout().addWidget(marker_label, 2, 0)
        self.layout().addWidget(self.marker_selection_dropdown, 3, 0)
        self.layout().addWidget(choose_y_axis_label, 4, 0)
        self.layout().addWidget(self.choose_y_axis_dropdown, 5, 0)
        
        # we have to do this because initially the dropdowns did not change texts yet so these variables are still None.
        self.model.active_sample = self.sample_selection_dropdown.currentText()
        self.model.active_marker = self.marker_selection_dropdown.currentText()
        self.model.active_y_axis = self.choose_y_axis_dropdown.currentText()

        self._read_data(self.model.active_sample)
        self._load_layers(self.model.markers[self.model.active_marker])

        #this has to go after active sample and marker are set
        self.scatter_canvas = PlotCanvas(self.model)
        # self.layout().addWidget(NavigationToolbar(self.gate_canvas, self))
        self.layout().addWidget(self.scatter_canvas.fig, 6, 0)
        # Update the plot initially
        self.update_plot()
        # the scatter plot is not updating when the gate is changed
        # unsure what is happening here

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
        if sample is not None:
            image_path = self.model.sample_image_mapping[sample]
            mask_path = self.model.sample_mask_mapping[sample]

            self._image = imread(image_path)
            self._mask = imread(mask_path)

    def _load_layers(self, marker_index):

        if self.model.active_sample != self._current_sample:
            self._current_sample = copy(self.model.active_sample)
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
        """Set the active sample.

        This changes the active sample and clears the layers and the marker selection dropdown.
        Subsequently, the new layers are loaded.
        """
        self.model.active_sample = self.sample_selection_dropdown.currentText()

        self._clear_layers(clear_all=True)
        self._reinitiate_marker_selection_dropdown()

        self._read_data(self.model.active_sample)
        self._load_layers(self.model.markers[self.model.active_marker])
        self.update_plot()

    def _clear_layers(self, clear_all: bool) -> None:
        """Remove all layers upon changing sample."""
        if clear_all is True:
            self.viewer.layers.select_all()
            self.viewer.layers.remove_selected()
        else:
            for layer in self.viewer.layers:
                if isinstance(layer, Image):
                    self.viewer.layers.remove(layer)

    def _reinitiate_marker_selection_dropdown(self) -> None:
        """Reiniatiate the marker selection dropdown after sample has changed."""
        # This is preemptively added for clearing visual completed feedback once implemented.
        # We also block the outgoing signal in order not to update the layer when there is no current active marker.
        self.marker_selection_dropdown.blockSignals(True)
        self.marker_selection_dropdown.clear()
        self.marker_selection_dropdown.addItems(self.model.markers)
        self.marker_selection_dropdown.blockSignals(False)

    def _on_marker_changed(self):
        """Set active marker and update the marker image layer."""
        self.model.active_marker = self.marker_selection_dropdown.currentText()
        self._clear_layers(clear_all=False)
        self._load_layers(self.model.markers[self.model.active_marker])
        self.update_plot()

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

    def update_plot(self):
        self.scatter_canvas.plot_scatter_plot(self.model)


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
        
        # check if sample and marker are selected
        assert self.model.active_marker is not None
        assert self.model.active_sample is not None
    
        # get the data for the scatter plot
        df = self.model.regionprops_df
        df = df[df["sample_id"] == self.model.active_sample]

        self.ax.scatter(
            x=df[self.model.active_marker],
            y=df[self.model.active_y_axis],  # later change to desired_y_axis
            color="steelblue",
            ec="white",
            lw=0.1,
            alpha=1.0,
            s=80000 / int(df.shape[0]),
        )
    #
        self.ax.set_ylabel("Area")  # later change to desired_y_axis
        self.ax.set_xlabel(f"{self.model.active_marker} intensity")
    #
    #     # add vertical line at current gate if it exists
    #     if self.model.current_gate is not None:
    #         ax.axvline(x=self.model.current_gate, color="red", linewidth=1.0, linestyle="--")
    #
    #     minimum = df[self.model.active_marker].min()
    #     maximum = df[self.model.active_marker].max()
    #     value_initial = df[self.model.active_marker].median()
    #     value_step = minimum / 100
    #
    #     # add slider as an axis, underneath the scatter plot
    #     axSlider = fig.add_axes([0.1, 0.01, 0.8, 0.03], facecolor="yellow")
    #     slider = Slider(axSlider, "Gate", minimum, maximum, valinit=value_initial, valstep=value_step, color="black")
    #
    #     def update_gate(val):
    #         self.model.current_gate = val
    #         ax.axvline(x=self.model.current_gate, color="red", linewidth=1.0, linestyle="--")
    #         napari_notification(f"Gate set to {val}")
    #
    #     slider.on_changed(update_gate)
    #
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
