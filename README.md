# napari-cell-gater

[![License BSD-3](https://img.shields.io/pypi/l/napari-cell-gater.svg?color=green)](https://github.com/CosciaLab/napari-cell-gater/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-cell-gater.svg?color=green)](https://pypi.org/project/napari-cell-gater)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cell-gater.svg?color=green)](https://python.org)
[![tests](https://github.com/CosciaLab/napari-cell-gater/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/CosciaLab/napari-cell-gater/actions/workflows/test_and_deploy.yml)
[![codecov](https://codecov.io/gh/CosciaLab/napari-cell-gater/branch/main/graph/badge.svg)](https://codecov.io/gh/CosciaLab/napari-cell-gater)

A napari plugin to perform cell marker gating for multiplexed immunofluorescent imaging.

`napari-cell-gater` is part of the [openDVP](https://github.com/CosciaLab/openDVP)
ecosystem ([documentation](https://coscialab.github.io/openDVP/)). Use it to pick, per
sample and per marker, the intensity threshold that separates positive from negative
cells — while looking at the image, the segmentation mask and the intensity
distribution side by side.

![Visual workflow of napari-cell-gater](https://raw.githubusercontent.com/CosciaLab/napari-cell-gater/main/docs/VisualWorkflow_highres.png)

----------------------------------

## Installation

`napari-cell-gater` needs Python 3.10–3.12 and a working [napari] installation
(see the [napari installation guide](https://napari.org/stable/tutorials/fundamentals/installation)).

We recommend a fresh environment:

```bash
conda create -n cell-gater python=3.11
conda activate cell-gater
pip install "napari[all]"
pip install napari-cell-gater
```

or, with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install "napari[all]" napari-cell-gater
```

To get the latest unreleased changes instead:

```bash
pip install git+https://github.com/CosciaLab/napari-cell-gater.git@main
```

Then launch `napari` and open the plugin from **Plugins → napari-cell-gater**.

## Inputs

You need three directories — images, segmentation masks and quantifications — with one
file per sample:

| Directory | File per sample | Notes |
|---|---|---|
| images | `1.ome.tif` or `1.tif` | channel order must match the marker columns of the quantification csv |
| masks | `1.tif` | label image; large masks can be slow to load |
| quantifications | `1.csv` | one row per cell, one column per marker, plus morphology columns |

Assumptions:

- Files are named after the sample (`1`, `2`, … or any shared stem), and the **same**
  stem is used in all three directories.
- Every sample must be present in all three directories.
- Any extra file in those directories can make the plugin fail — keep them clean.

A ready-to-run example dataset ships with the repository under
[`tests/test_data/`](https://github.com/CosciaLab/napari-cell-gater/blob/main/tests/test_data) — 5 samples as `imgs/{1..5}.ome.tif`,
`segs/{1..5}.tif` and `quants/{1..5}.csv`. Point the plugin at those three folders to
try it out without preparing your own data.

## Output

Gates are written to a csv with one row per sample/marker pair — see
[`gates_example.csv`](https://github.com/CosciaLab/napari-cell-gater/blob/main/gates_example.csv):

```csv
sample_id,marker_id,gate_value
68,Rabbit IgG,3478.531304347826
68,Goat IgG,0.0
```

Gate values are always saved in **linear** space, even when plotting in log10.
The same file can be read back with **Load existing gates**.

## How to use

1. Load your three directories with **Load quantifications dir**, **Load image dir**
   and **Load mask dir**. Optionally use **Opt: Load channel map** to supply an explicit
   channel-to-marker mapping.
2. Set **Select lowerbound marker column** and **Select upperbound marker column** to
   the first and last quantification columns you want to gate. Pick the same bounds
   every session if you plan to save and reload gates.
3. Optionally adjust **Remove markers with prefix** (default `DNA, DAPI`) to drop
   nuclear channels from the marker list.
4. Click **Validate input**. On success the gating controls appear; on failure fix the
   inputs and validate again.
5. Choose a **sample** and a **marker** from the dropdowns. Three layers load: the
   reference channel (**Select reference channel**), the segmentation mask, and the
   channel being gated. A plot of marker intensity (x) against **Choose Y-axis**
   (default `Area`) appears, with a slider underneath whose position is drawn as a
   vertical line.
6. Tune the view: switch **Plot type** to `Hexbin` for dense clusters, and
   **Logarithmic scale** to `Yes` to plot in log10 (often easier to read). Adjust image
   contrast from the napari layer controls at the top left.
7. Drag the slider to the threshold you think is correct, or type a value into
   **Manual gate input** and click **Set gate manually**.
8. Click **Plot Points** to overlay the cells the current gate calls positive.
9. Repeat steps 6–8 until you are satisfied, then click **Save Gate**. The first save
   asks where to write the gates csv; later saves reuse that path. Go back to step 5 for
   the next marker or sample.

## Contributing

Contributions are very welcome. To set up a development install:

```bash
git clone https://github.com/CosciaLab/napari-cell-gater.git
cd napari-cell-gater
pip install -e ".[dev]"
pytest
```

Please ensure test coverage at least stays the same before you submit a pull request.

## Citation

`napari-cell-gater` is developed as part of the [openDVP](https://github.com/CosciaLab/openDVP)
ecosystem. Until the paper is out, please cite this repository and the openDVP
documentation at <https://coscialab.github.io/openDVP/>.

<!-- TODO: add journal citation + DOI on publication -->

## License

Distributed under the terms of the [BSD-3] license,
"napari-cell-gater" is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[file an issue]: https://github.com/CosciaLab/napari-cell-gater/issues
