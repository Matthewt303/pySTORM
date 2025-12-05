import numpy as np
import os
import warnings
from pySTORM.internals.file_io import gather_im_stacks
from pySTORM.internals.file_io import (
    save_localisation_table_csv,
    save_localisation_table_hdf5,
)


def get_movies(movie_folder: str) -> tuple[str]:
    """
    This function generates a list of all image stacks to be processed.
    ----------------------------------------------------------
    In:
    movie_folder - path of folder with image stacks.
    ----------------------------------------------------------
    Out:
    im_stack_paths - absolute paths for image stacks (tuple).
    ----------------------------------------------------------

    """
    ## INPUT CHECKS ##
    # ----------------------#

    if not isinstance(movie_folder, str):
        raise TypeError("Folder must be a string.")

    if not os.path.isdir(movie_folder):
        raise OSError("Directory not found.")

    im_stack_paths = gather_im_stacks(movie_folder)

    return tuple(im_stack_paths)


def get_camera_params(
    pixel_size: float = None, adu: float = None, offset: int = None, gain: int = None
) -> tuple[float]:
    """
    This function is for users to enter their camera specifications.
    Specs undergo a sanity check and raise errors if they are physically
    impossible and warnings if they are possible but implausible.
    ----------------------------------------------------------
    In:
    pixel_size - the pixel size in nm at the sample plane.
    adu - analog-digital conversion factor.
    offset - camera baseline gray value.
    gain - EM amplification.
    ----------------------------------------------------------
    Out:
    specs - tuple of camera specifications.
    ----------------------------------------------------------

    """

    ## INPUT CHECKS ##
    # ----------------------#

    if pixel_size is None or adu is None or offset is None or gain is None:
        raise ValueError("Missing one or more camera specification value.")

    if (
        not isinstance(pixel_size, float)
        or not isinstance(offset, int)
        or not isinstance(adu, float)
        or not isinstance(gain, int)
    ):
        raise TypeError(
            "One or more camera specs are not of the correct type."
            " Pixel size and adu must be floats while gain and offset must be an integer."
        )

    if pixel_size <= 0 or adu <= 0 or offset <= 0 or gain <= 0:
        raise ValueError("Camera specs cannot be negative.")

    if pixel_size < 30:
        warnings.warn(
            "Pixel size is very small. Image is probably undersampled", Warning
        )

    elif pixel_size > 250:
        warnings.warn(
            "Pixel size is very large. Localization accuracy will likely be poor",
            Warning,
        )

    specs = (pixel_size, adu, offset, gain)

    return specs


def save_loc_table(
    loc_data: list["np.ndarray"], output_folder: str, format: str
) -> None:
    """
    This function saves the localization data either as a .csv file or a
    .hdf5 file, depending on user specifications. The file is used in the
    user-specified output folder.
    ----------------------------------------------------------
    In:
    loc_data - a list of N x 8 numpy arrays where N is the number of localizations
    detected in a frame.

    output_folder - user-specified output folder.

    format - either 'csv' or 'hdf5'
    ----------------------------------------------------------
    Out:
    None but a file is saved in the output folder.
    ----------------------------------------------------------

    """

    ## INPUT CHECKS ##
    # --------------------------#

    if not isinstance(output_folder, str):
        raise TypeError("Output folder must be a string")

    if not os.path.isdir(output_folder):
        raise OSError("Directory not found")

    if format not in ("csv", "hdf5"):
        warnings.warn(
            "Invalid file format. Defaulting to .csv. Use either csv or hdf5 next time",
            Warning,
        )

        format = "csv"

    ## SAVE FILES ##
    # --------------------------#

    if format == "csv":
        save_localisation_table_csv(loc_data, output_folder)

    else:
        save_localisation_table_hdf5(loc_data, output_folder)
