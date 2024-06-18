import tempfile
from typing import Any

import numpy as np
import pandas as pd

from cell_gater.widgets.sample_widget import SampleWidget

rng = np.random.default_rng(42)
# if we get reports that with some column names there is a problem, we get it here to automatically test.
MARKER_DF = pd.DataFrame(
    {
        "CellID": list(range(5)),
        "DNA": rng.random(5),
        "Rabbit IgG": rng.random(5),
        "Goat IgG": rng.random(5),
        "DNA2": rng.random(5),
        "CD73": rng.random(5),
        "Some.name.with.dots": rng.random(5),
    }
)


def test_populate_markers_on_csv_load(make_napari_viewer: Any) -> None:
    viewer = make_napari_viewer()
    widget = SampleWidget(viewer)
    with tempfile.TemporaryDirectory() as tmpdir:
        MARKER_DF.to_csv(tmpdir + "/test1.csv", index=False)
        MARKER_DF.to_csv(tmpdir + "/test12.csv", index=False)

        widget._open_sample_dialog(folder=tmpdir)

        lower_marker_cols = [
            widget.lower_bound_marker_col.itemText(index) for index in range(widget.lower_bound_marker_col.count())
        ]
        upper_marker_cols = [
            widget.upper_bound_marker_col.itemText(index) for index in range(widget.upper_bound_marker_col.count())
        ]
        reference_cols = list(MARKER_DF.columns)
        reference_cols.append("sample_id")
        assert lower_marker_cols == upper_marker_cols == reference_cols
