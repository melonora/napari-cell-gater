from typing import Any

# for scatter plot
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cell_gater.model.data_model import DataModel
from cell_gater.utils.misc import napari_notification


class ScatterWidget(QWidget):
    """Widget for a scatter plot with markers on the x axis and any dtype column on the y axis."""

    def __init__(self, model: DataModel) -> None:
        super().__init__()

        self.setLayout(QVBoxLayout())
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)
        self._model = model

        # Dropdown of samples once directory is loaded
        selection_label = QLabel("Select sample:")
        self.sample_selection_dropdown = QComboBox()
        if len(self.model.samples) > 0:
            self.sample_selection_dropdown.addItems(self.model.samples)
        self.sample_selection_dropdown.currentTextChanged.connect(self._on_sample_changed)

        self.layout().addWidget(selection_label, 4, 0, Qt.AlignCenter)
        self.layout().addWidget(self.sample_selection_dropdown, 4, 1)

    def _on_sample_changed(self):
        """Set the active sample."""
        self.model.active_sample = self.sample_selection_dropdown.currentText()

    def _set_samples_dropdown(self, event: Any) -> None:
        """Set the items for the samples dropdown QComboBox."""
        if (region_props := self.model.regionprops_df) is not None:
            self.model.samples = list(region_props["sample_id"].cat.categories)

            # New directory loaded so we reload the dropdown items
            self.sample_selection_dropdown.clear()
            if len(self.model.samples) > 0:
                self.sample_selection_dropdown.addItems([None])
                self.sample_selection_dropdown.addItems(self.model.samples)

    def plot_scatter_plot(self, model: DataModel) -> None:
        """Plot the scatter plot."""
        # check if sample and marker are selected
        assert self.model.active_marker is not None
        assert self.model.active_sample is not None

        # get the data for the scatter plot
        df = self.model.regionprops_df
        df = df[df["sample_id"] == self.model.active_sample]

        # plot the scatter plot
        fig, ax = plt.subplots()
        # plot scatter on canvas axis

        # if desired_y_axis is None:
        #     desired_y_axis = "Area"
        # should this be through a dropdown widget? or datamodel attribute?

        ax.scatter(
            x=df[self.active_marker],
            y=df["Area"],  # later change to desired_y_axis
            color="steelblue",
            ec="white",
            lw=0.1,
            alpha=1.0,
            s=80000 / int(df.shape[0]),
        )

        ax.set_ylabel("Area")  # later change to desired_y_axis
        ax.set_xlabel(f"{self.active_marker} intensity")

        # add vertical line at current gate if it exists
        if self.model.current_gate is not None:
            ax.axvline(x=self.model.current_gate, color="red", linewidth=1.0, linestyle="--")

        minimum = df[self.model.active_marker].min()
        maximum = df[self.model.active_marker].max()
        value_initial = df[self.model.active_marker].median()
        value_step = minimum / 100

        # add slider as an axis, underneath the scatter plot
        axSlider = fig.add_axes([0.1, 0.01, 0.8, 0.03], facecolor="yellow")
        slider = Slider(axSlider, "Gate", minimum, maximum, valinit=value_initial, valstep=value_step, color="black")

        def update_gate(val):
            self.model.current_gate = val
            ax.axvline(x=self.model.current_gate, color="red", linewidth=1.0, linestyle="--")
            napari_notification(f"Gate set to {val}")

        slider.on_changed(update_gate)

    # TODO add the plot to a widget and display it

    def plot_points(self, model: DataModel) -> None:
        """Plot positive cells in Napari."""
        if self.model.active_marker is not None:
            marker = self.model.active_marker
        if self.model.active_sample is not None:
            sample = self.model.active_sample

        viewer = self.model.viewer
        df = self.model.regionprops_df
        df = df[df["sample_id"] == self.model.active_sample]
        gate = self.model.gates.loc[marker, sample]

        viewer.add_points(
            df[df[marker] > gate][["X_centroid", "Y_centroid"]],
            name=f"{gate} and its positive cells",
            face_color="red",
            edge_color="black",
            size=15,
        )

    def plot_points_button(self):
        """Plot points button."""
        self.plot_points_button = QPushButton("Plot Points")
        self.plot_points_button.clicked.connect(self.plot_points)
        self.layout().addWidget(self.plot_points_button, 1, 2)  # not sure where to put this button
