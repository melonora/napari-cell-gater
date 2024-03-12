from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cell_gater.model.data_model import DataModel


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
        pass

    def _set_samples_dropdown(self, event: Any) -> None:
        """Set the items for the samples dropdown QComboBox."""
        if (region_props := self.model.regionprops_df) is not None:
            self.model.samples = list(region_props["sample_id"].cat.categories)

            # New directory loaded so we reload the dropdown items
            self.sample_selection_dropdown.clear()
            if len(self.model.samples) > 0:
                self.sample_selection_dropdown.addItems([None])
                self.sample_selection_dropdown.addItems(self.model.samples)
