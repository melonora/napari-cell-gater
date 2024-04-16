from pathlib import Path

import pandas as pd

from cell_gater.utils.misc import napari_notification


def stack_csv_files(csv_dir: Path) -> pd.DataFrame | None:
    """
    Concatenate all csv files containing regionprops data into one dataframe.

    Parameters
    ----------
    csv_dir : str
        Path to the directory containing the regionprops csvs

    Returns
    -------
    pd.DataFrame
        Dataframe containing the data of all regionprops csvs in the directory.
    """
    csv_files = list(csv_dir.glob("*.csv"))
    if len(csv_files) == 0:
        napari_notification("No csv files found in the loaded regionprops directory. Please load the proper directory.")
        return None

    napari_notification(f"Loaded {len(csv_files)} regionprops csvs.")
    df = pd.DataFrame()
    for file in csv_files:
        if not file.name.startswith('.'):
            df_file = pd.read_csv(file)
            df_file["sample_id"] = file.stem
            df = pd.concat([df, df_file], ignore_index=True)
            df["sample_id"] = df.sample_id.astype("category")
        else:
            print(f"Skipping file {file.name} as it is a hidden file.")

    return df


def get_gates_from_regionprops_df(path_to_gate: Path, df: pd.DataFrame, markers: list[str]) -> pd.DataFrame:
    """
    Get gate dataframe.

    If path_to_gate is specified then the gate df is created by loading the csv into memory. Otherwise a new dataframe
    is created.

    Parameters
    ----------
    path_to_gate: Path
        Path to the csv containing the gates for markers.
    df: pd.DataFrame
        The stacked regionprops dataframe.
    markers: list[str]
        The list of markers
    """
    if path_to_gate is not None:
        assert path_to_gate.exists(), f"CSV path path_to_gate `{path_to_gate}` does not exist."
        gates = pd.read_csv(path_to_gate)
    else:
        gates = pd.DataFrame(index=markers, columns=df["sample_id"].unique())
    return gates


def get_markers_of_interest(df: pd.DataFrame, up_to: str, subset: tuple[int, int] | None = None) -> list[str]:
    """
    Get the marker columns from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe from which to get the marker columns.
    up_to : str
        The name of the column up to which to get the other dataframe columns. This parameter is exclusive, meaning
        that the name of this column is not includes in the column subset. The lower bound is always column 1 as
        column 1 is the cell_id. The resulting dataframe contains all the marker columns.
    subset : tuple[int, int] | None
        Tuple indicating the lowerbound and exclusive upperbound to get from the marker dataframe (dataframe after
        subsetting by up_to).

    Returns
    -------
    list[str]
        The marker names.

    Discussion
    ----------
    We might want to filter out some markers that are usually not gated:
    the nuclear stains (DAPI/DNA/Hoechst) should be removed from the list (not sure if at this step or before)
    In Cylinter, there was a congifg.yml parameter that denoted the list of markers to ignore.
    For now I think that just removing the nuclear stains is enough. I will do this on the get_markers_of_interest function.
    Also an option is to take the markers.csv file from MCMICRO and use it to filter the markers.

    """
    subset_slice = (
        slice(subset[0], subset[1]) if isinstance(subset, tuple) and all(isinstance(i, int) for i in subset) else subset
    )

    # find column index of the column specified by up_to
    x_centroid_col = df.columns.get_loc(up_to)
    markers = df.columns[1:x_centroid_col]
    if subset_slice is not None:
        markers = markers[subset_slice].tolist()
    # remove nuclear stains
    return [item for item in markers if not any(item.startswith(prefix) for prefix in ["DNA", "DAPI", "Hoechst"])]
