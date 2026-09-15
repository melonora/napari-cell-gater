# napari-cell-gater

`napari-cell-gater` turns marker gating for multiplexed immunofluorescence imaging into
an interactive, auditable step inside napari. Highly multiplexed imaging (CyCIF, t-CyCIF,
CODEX, IBEX, mIF) produces a cell-by-marker intensity table after segmentation, and
turning those intensities into positive/negative calls requires a threshold — a *gate* —
per marker and per sample. Picking those gates from the intensity distribution alone is
error prone: the number that looks right on a histogram is often visibly wrong once you
look at the image.

This plugin puts the three things you need in one place. Point it at a directory of
images, a directory of segmentation masks and a directory of quantification csv files,
and for each sample/marker pair it loads the channel being gated, a reference channel
and the segmentation mask into the napari viewer, next to a scatter or hexbin plot of
marker intensity against a morphology feature of your choice (cell area by default). A
slider under the plot moves a candidate gate; **Plot Points** overlays the cells that
gate calls positive directly on the image, so you can confirm the threshold against the
staining before committing to it. Plots can be drawn in log10 space for dense clusters,
and gates can also be typed in as exact values.

Gates are saved to a tidy csv (`sample_id,marker_id,gate_value`), always in linear
space, and can be reloaded to resume or revise a gating session — which makes the
thresholds behind a downstream analysis reviewable rather than lost in a notebook.
An example dataset of five samples ships with the repository so you can try the full
workflow immediately.

`napari-cell-gater` is part of the [openDVP](https://github.com/CosciaLab/openDVP)
ecosystem for deep visual proteomics. Source, documentation and issue tracker:
<https://github.com/CosciaLab/napari-cell-gater>.
