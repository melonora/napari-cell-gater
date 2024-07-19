# napari-cell-gater

[![License BSD-3](https://img.shields.io/pypi/l/napari-cell-gater.svg?color=green)](https://github.com/melonora/napari-cell-gater/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-cell-gater.svg?color=green)](https://pypi.org/project/napari-cell-gater)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cell-gater.svg?color=green)](https://python.org)
[![tests](https://github.com/melonora/napari-cell-gater/workflows/tests/badge.svg)](https://github.com/melonora/napari-cell-gater/actions)
[![codecov](https://codecov.io/gh/melonora/napari-cell-gater/branch/main/graph/badge.svg)](https://codecov.io/gh/melonora/napari-cell-gater)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cell-gater)](https://napari-hub.org/plugins/napari-cell-gater)

A plugin to perform cell marker gating for multiplexed immunofluorescent imaging

![Screenshot 2024-06-17 at 19 34 17](https://github.com/melonora/napari-cell-gater/assets/30318135/f469c380-ef53-42d6-a136-ebcae723e987)

----------------------------------

## Installation

Step 1. 
Install napari (see https://napari.org/stable/tutorials/fundamentals/installation)

Step 2.
Install `napari-cell-gater` via [pip]:

    pip install git+https://github.com/melonora/napari-cell-gater.git

## How to use

1. Users will select the necesary directories: images, masks, and quantification directories.

    Assumptions for inputs:  
        1.1 Files inside these directories are named according to the samples names.   
        1.2 The image for sample 1, should be "1.ome.tif" or "1.tif"; the mask file "1.tif"; and the quantification file "1.csv".  
        1.3 Each set of files should all be inside each of the three folders.  
        1.4 Any extra files in those folders can make code fail.  

3. Select the lowerbound and upperbound channels to gate. These are all the columns from the quantification csv file that you want to threshold. You must pick the same channels if you plan to save and reload the gates.  

4. Select a sample, and a marker from dropdown menus. 3 layers will load:   
        (a.) the reference channel (default: first channel, changeable by dropdown menu)   
        (b.) the segmentation mask (for large images this might be a problem)  
        (c.) the channel_to_be_gated  
A scatter plot (default: x-axis=channel_to_be_gated intensity, y-axis=Area) (y-axis can be changed by dropdown)      
Underneath the scatterplot a slider will appear, the position of the slider will show up as a vertical line in the scatter plot.
The scatter plot can also be changed to a hexbin plot, which really helps with dense clusters of cells.
Plotting the data in log10 space is also possible by dropdown. Most of the times it helps. Gates would still be saved in linear space.

6. Adjust the contrast with the Napari layer menu (top left)
7. Drag the slider to what they think is correct
8. Click "Plot Points" to plot points on top of positive cells.
9. Repeat steps 5 and 6 until satisfied.
10. Click "Save Gate" to save the gate for the current marker and sample. Go to step 4 and repeat.

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
