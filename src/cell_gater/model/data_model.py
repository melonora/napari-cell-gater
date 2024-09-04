from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from napari.utils.events import EmitterGroup, Event


@dataclass
class DataModel:
    """Model containing all necessary fields for gating."""

    events: EmitterGroup = field(init=False, default=None, repr=True)
    _samples: Sequence[str] = field(default_factory=list, init=False)
    _regionprops_df: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    _regionprops_columns: Sequence[str] = field(default_factory=list, init=False)
    _image_paths: Sequence[Path] = field(default_factory=list, init=False)
    _sample_image_mapping: dict[str, Path] = field(default_factory=dict, init=False)
    _sample_mask_mapping: dict[str, Path] = field(default_factory=dict, init=False)
    _mask_paths: Sequence[Path] = field(default_factory=list, init=False)
    _lower_bound_marker: str | None = field(default=None, init=False)
    _upper_bound_marker: str | None = field(default=None, init=False)
    _markers: Sequence[str] = field(default_factory=list, init=False)
    _markers_image_indices: Sequence[str] = field(default_factory=list, init=False)
    _marker_filter: str = field(default="dna,dapi", init=True)
    _validated: bool = field(default=False, init=True)

    _active_marker: str | None = field(default=None, init=False)
    _active_sample: str | None = field(default=None, init=False)
    _active_y_axis: str | None = field(default=None, init=False)
    _active_ref_marker: str | None = field(default=None, init=False)

    _log_scale: bool = field(default=False, init=False)
    _plot_type: str = field(default="scatter", init=False)

    _gates: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    _current_gate: float = field(default_factory=float, init=False)
    _manual_channel_mapping: str | None = field(default=None, init=False)

    @property
    def active_ref_marker(self):
        """The reference marker for the gates."""
        return self._active_ref_marker

    @active_ref_marker.setter
    def active_ref_marker(self, marker: str) -> None:
        self._active_ref_marker = marker

    @property
    def gates(self):
        """The gates dataframe."""
        return self._gates

    @gates.setter
    def gates(self, gates: pd.DataFrame) -> None:
        self._gates = gates

    @property
    def manual_channel_mapping(self):
        """The manual channel mapping dataframe."""
        return self._manual_channel_mapping

    @manual_channel_mapping.setter
    def manual_channel_mapping(self, manual_channel_mapping:str) -> None:
        self._manual_channel_mapping = manual_channel_mapping

    @property
    def current_gate(self) -> float:
        """The current gate value."""
        return self._current_gate

    @current_gate.setter
    def current_gate(self, value: float) -> None:
        self._current_gate = value

    def __post_init__(self) -> None:
        """Allow fields in the dataclass to emit events when changed."""
        self.events = EmitterGroup(source=self, samples=Event, regionprops_df=Event, validated=Event)

    @property
    def sample_image_mapping(self) -> Mapping[str, Path]:
        """Mapping sample names to image paths."""
        return self._sample_image_mapping

    @sample_image_mapping.setter
    def sample_image_mapping(self, mapping: dict[str, Path]) -> None:
        self._sample_image_mapping = mapping

    @property
    def sample_mask_mapping(self) -> Mapping[str, Path]:
        """Mapping of sample names to mask paths."""
        return self._sample_mask_mapping

    @sample_mask_mapping.setter
    def sample_mask_mapping(self, mapping: dict[str, Path]) -> None:
        self._sample_mask_mapping = mapping

    @property
    def samples(self):
        """Samples derived from the regionprops csv file names."""
        return self._samples

    @samples.setter
    def samples(self, samples: Sequence[str]) -> None:
        self._samples = samples

    @property
    def regionprops_df(self):
        """Regionprops dataframe derived from possibly multiple csv files."""
        return self._regionprops_df

    @regionprops_df.setter
    def regionprops_df(self, regionprops: pd.DataFrame) -> None:
        self._regionprops_df = regionprops
        self.events.regionprops_df()

    @property
    def image_paths(self):
        """The paths to the images."""
        return self._image_paths

    @image_paths.setter
    def image_paths(self, image_paths: Sequence[Path]) -> None:
        self._image_paths = image_paths

    @property
    def mask_paths(self):
        """The paths to the mask images."""
        return self._mask_paths

    @mask_paths.setter
    def mask_paths(self, mask_paths: Sequence[Path]) -> None:
        self._mask_paths = mask_paths

    @property
    def lower_bound_marker(self):
        """The lower bound column name of the marker columns to be included from regionprops_df."""
        return self._lower_bound_marker

    @lower_bound_marker.setter
    def lower_bound_marker(self, marker: str) -> None:
        self._lower_bound_marker = marker

    @property
    def upper_bound_marker(self):
        """The inclusive upper bound column name of the marker columns to be included from regionprops_df."""
        return self._upper_bound_marker

    @upper_bound_marker.setter
    def upper_bound_marker(self, marker: str) -> None:
        self._upper_bound_marker = marker

    @property
    def markers(self):
        """The markers included for gating."""
        return self._markers

    @markers.setter
    def markers(self, markers: Sequence[str]) -> None:
        self._markers = markers

    @property
    def markers_image_indices(self):
        """The markers included for gating."""
        return self._markers_image_indices

    @markers_image_indices.setter
    def markers_image_indices(self, markers_image_indices: Sequence[str]) -> None:
        self._markers_image_indices = markers_image_indices

    @property
    def active_marker(self):
        """The marker currently used on x-axis for gating."""
        return self._active_marker

    @active_marker.setter
    def active_marker(self, marker: str) -> None:
        self._active_marker = marker

    @property
    def active_sample(self) -> str | None:
        """The sample currently opened in the viewer."""
        return self._active_sample

    @active_sample.setter
    def active_sample(self, sample: str) -> None:
        self._active_sample = sample

    @property
    def active_y_axis(self) -> str | None:
        """The marker currently used on y-axis for gating."""
        return self._active_y_axis

    @active_y_axis.setter
    def active_y_axis(self, column: str) -> None:
        self._active_y_axis = column

    @property
    def log_scale(self) -> bool:
        """Whether the y-axis is in log scale."""
        return self._log_scale

    @log_scale.setter
    def log_scale(self, log_scale: bool) -> None:
        self._log_scale = log_scale

    @property
    def plot_type(self) -> str:
        """The plot type."""
        return self._plot_type

    @plot_type.setter
    def plot_type(self, plot_type: str) -> None:
        self._plot_type = plot_type

    @property
    def marker_filter(self):
        """The string filters separated by commas."""
        return self._marker_filter

    @marker_filter.setter
    def marker_filter(self, marker_filter: str) -> None:
        self._marker_filter = marker_filter

    @property
    def validated(self):
        """Whether the data is validated."""
        return self._validated

    @validated.setter
    def validated(self, validated: bool) -> None:
        self._validated = validated
        self.events.validated()
