# napari-cell-gater

[![License BSD-3](https://img.shields.io/pypi/l/napari-cell-gater.svg?color=green)](https://github.com/melonora/napari-cell-gater/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-cell-gater.svg?color=green)](https://pypi.org/project/napari-cell-gater)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cell-gater.svg?color=green)](https://python.org)
[![tests](https://github.com/melonora/napari-cell-gater/workflows/tests/badge.svg)](https://github.com/melonora/napari-cell-gater/actions)
[![codecov](https://codecov.io/gh/melonora/napari-cell-gater/branch/main/graph/badge.svg)](https://codecov.io/gh/melonora/napari-cell-gater)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cell-gater)](https://napari-hub.org/plugins/napari-cell-gater)

A plugin to perform cell marker gating for multiplexed immunofluorescent imaging

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

This plugin is currently under heavy development. If you are interested in contributing, please create a fork. After
cloning, create an environment and install using `pip install -e .[test]`. Afterwards, please install the `pre-commit`
hook by running `pre-commit install` in the directory with the `.pre-commit-config.yaml`.

## Installation

You can install `napari-cell-gater` via [pip]:

    pip install napari-cell-gater

## Visual Workflow 

![Image Alt Text](/docs/VisualWorkflow_highres.png)

## How to use
Writing this helps me organize my thoughts on what to do next

1. Users will select the necesary directories: images, masks, and quantification directories.

Assumptions:
files inside these directories are named according to the samples name.
For example, the image for sample 1, should be named "1.ome.tif" or "1.tif"; the mask file should be named "1.tif"; and the quantification file "1.csv"

2. Select the lowerbound and upperbound channels to gate.

3. Select a sample, and a marker from dropdown menus. Then click "Load Sample and Marker", 3 layers will load:   
        (a.) the reference channel (default: first channel, changeable by dropdown menu)   
        (b.) the segmentation mask (for large images this might be a problem)  
        (c.) the channel_to_be_gated  
A scatter plot (default: x-axis=channel_to_be_gated intensity, y-axis=Area) (y-axis can be changed by dropdown)      
Underneath the scatterplot a slider will appear, the position of the slider will show up as a vertical line in the scatter plot  

4. Adjust the contrast with the Napari layer menu (top left)
5. Drag the slider to what they think is correct
6. Click "Plot Points" to plot points on top of positive cells.
7. Repeat steps 5 and 6 until satisfied.
8. Click "Save Gate" to save the gate for the current marker and sample. Go to step 3 and repeat.
9. Save the current gate values as a csv by clicking "Save Gates DataFrame".
10. This csv file can also be loaded midway through the gating process.  

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-cell-gater" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
