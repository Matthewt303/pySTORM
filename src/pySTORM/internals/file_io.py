import numpy as np
import pandas as pd
import tifffile as tiff
import os
from typing import Generator


def gather_im_stacks(folder_path: str) -> list[str]:
    """
    Collates file paths for image stacks in a folder. Accepts the following
    file formats: .tif, .tiff, .TIF, .TIFF.
    ---------------------------------------------------------------
    In:
    folder_path - folder where image stacks are stored.
    ---------------------------------------------------------------
    Out:
    im_files - list of image stack file paths
    """

    im_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".tif")
        or file.endswith(".tiff")
        or file.endswith(".TIF")
        or file.endswith(".TIFF")
    ]

    return im_files


def load_image(path: str):
    """
    Loads an image stack from a file path as a memory map array.
    ---------------------------------------------------------------
    In:
    path - image stack file path.
    ---------------------------------------------------------------
    Out:
    memmap_stack - image stack as a 32-bit floating point np array
    """

    with tiff.TiffFile(path) as tif:
        memmap_stack = tif.asarray(out="memmap")

    return memmap_stack.astype(np.float32)


def load_stacks(paths: tuple[str]) -> Generator["np.ndarray", None, None]:
    """
    Generator object that outputs image stacks from a tuple containing
    the file paths of all image stacks.
    ---------------------------------------------------------------
    In:
    paths - tuple with image stack file paths.
    ---------------------------------------------------------------
    Out:
    memmap_stack - image stack as a 32-bit floating point np array
    """

    for path in paths:
        stack = load_image(path)

        yield stack


def save_localisation_table_csv(loc_data: list["np.ndarray"], out_folder: str) -> None:
    """
    Aggregates localisations, filters for unrealistic uncertainties,
    converts to pd dataframe, and saves the localisation table in a
    user-specified output folder.
    ---------------------------------------------------------------
    In:
    loc_data - list of np arrays where each array is the localisation
    data for one frame.

    out_folder - where the localisation data will be saved.
    ---------------------------------------------------------------
    Out:
    None. .csv file is saved, titled 'reconstruction.csv'
    """

    localisation_data = np.vstack(loc_data).reshape(-1, 8)

    # Remove unrealistically large uncertainties.
    localisation_data = localisation_data[localisation_data[:, -1] < 500]

    headers = [
        "id",
        "frame",
        "x [nm]",
        "y [nm]",
        "sigma [nm]",
        "intensity [photon]",
        "offset [photon]",
        "uncertainty [nm]",
    ]

    dataframe = pd.DataFrame(data=localisation_data, columns=headers, dtype=np.float32)

    df_filt = dataframe[dataframe["uncertainty [nm]"].notnull()]

    df_filt.to_csv(os.path.join(out_folder, "reconstruction.csv"), sep=",", index=False)


def save_localisation_table_hdf5(loc_data: list["np.ndarray"], out_folder: str) -> None:
    """
    Aggregates localisations, filters for unrealistic uncertainties,
    converts to pd dataframe, and saves the localisation table in a
    user-specified output folder.
    ---------------------------------------------------------------
    In:
    loc_data - list of np arrays where each array is the localisation
    data for one frame.

    out_folder - where the localisation data will be saved.
    ---------------------------------------------------------------
    Out:
    None. .csv file is saved, titled 'reconstruction.csv'
    """

    localisation_data = np.vstack(loc_data).reshape(-1, 8)

    # Remove unrealistically large uncertainties.
    localisation_data = localisation_data[localisation_data[:, -1] < 500]

    headers = [
        "id",
        "frame",
        "x [nm]",
        "y [nm]",
        "sigma [nm]",
        "intensity [photon]",
        "offset [photon]",
        "uncertainty [nm]",
    ]

    dataframe = pd.DataFrame(data=localisation_data, columns=headers, dtype=np.float32)

    df_filt = dataframe[dataframe["uncertainty [nm]"].notnull()]

    df_filt.to_csv(os.path.join(out_folder, "reconstruction.csv"), sep=",", index=False)
