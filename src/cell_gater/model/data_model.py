from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

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
    _mask_paths: Sequence[Path] = field(default_factory=list, init=False)
    _lower_bound_marker: str | None = field(default=None, init=False)
    _upper_bound_marker: str | None = field(default=None, init=False)
    _markers: Sequence[str] = field(default_factory=list, init=False)
    _active_marker: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Allow fields in the dataclass to emit events when changed."""
        self.events = EmitterGroup(
            source=self,
            samples=Event,
            regionprops_df=Event,
        )

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
    def active_marker(self):
        """The marker currently used on x-axis for gating."""
        return self._active_marker

    @active_marker.setter
    def active_marker(self, marker: str) -> None:
        self._active_marker = marker

    def validate(self):
        """Validate the input data from the user provided through the SampleWidget."""
        pass
