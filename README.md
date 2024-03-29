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
For example, the image for sample 1, should be named "1.ome.tif"; the mask file should be named "1.tif"; and the quantification file "1.csv"

2. Users will then select which channels to gate, default is all of them.

3. Users will then select a sample, and a marker from dropdown menus.
    Upon selecting sample and marker.
    3 layers will load:
        1. the nuclear stain of the image (we assume it is channel 0, perhaps we can allow users to pick which one to use with dropdown)
        2. the segmentation mask (as a labels layer) (for large images this might be a problem; Cylinter solves this by pyrimidazing a binary label)
        3. the channel_to_be_gated
    a Widget will appear showing a scatter plot (default: x-axis=channel_to_be_gated intensity, y-axis=Area) (y-axis could be change to another column)
    underneath the scatterplot a slider will appear, the position of the slider will show up as a vertical line in the scatter plot

4. Users will then adjust the contrast with the Napari menu

5. Users will drag the slider to what they think is correct

6. User will click a button, which will then plot a points layer on top of the image.
The points should be on the x,y coordenates of cells that have a channel intensity larger than the threshold picked by the slider.

7. User will repeat steps 5 and 6 until satisfied

8. User wil then click a button to save the gate for the current marker and sample.
    Here either the next marker will load, or if all markers for a sample are done, go to next sample.
    I prefer an automated switch, but we should allow a user to go back to sample_marker of interest for review.

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
