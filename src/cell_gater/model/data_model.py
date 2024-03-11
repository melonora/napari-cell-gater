from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd
from napari.utils.events import EmitterGroup, Event


@dataclass
class DataModel:
    events: EmitterGroup = field(init=False, default=None, repr=True)
    _samples: Sequence[str] = field(default_factory=list, init=False)
    _regionprops_df: pd.DataFrame = field(
        default_factory=pd.DataFrame, init=False
    )
    _image_paths: Sequence[Path] = field(default_factory=list, init=False)
    _mask_paths: Sequence[Path] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.events = EmitterGroup(
            source=self,
            samples=Event,
            regionprops_df=Event,
        )

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples: Sequence[str]):
        self._samples = samples

    @property
    def regionprops_df(self):
        return self._regionprops_df

    @regionprops_df.setter
    def regionprops_df(self, regionprops: pd.DataFrame):
        self._regionprops_df = regionprops
        self.events.regionprops_df()

    @property
    def image_paths(self):
        return self._image_paths

    @image_paths.setter
    def image_paths(self, image_paths: Sequence[Path]):
        self._image_paths = image_paths

    @property
    def mask_paths(self):
        return self._mask_paths

    @mask_paths.setter
    def mask_paths(self, mask_paths: Sequence[Path]):
        self._mask_paths = mask_paths
