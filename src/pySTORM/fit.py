import numpy as np
import warnings
from pySTORM.internals.file_io import load_stacks
from pySTORM.internals.image_analysis import get_spots
from pySTORM.internals.single_mol_fitting import localise_frame


def check_threshold(threshold: float) -> None:
    """
    This function checks the user-supplied threshold to prevent
    invalid or negative inputs, as well as warn users for very low
    or very high thresholds.
    ----------------------------------------------------------
    In:
    threshold - the minimum intensity required for local maxima detection
    ----------------------------------------------------------
    Out:
    None
    ----------------------------------------------------------

    """

    if not isinstance(threshold, float) and not isinstance(threshold, int):
        raise TypeError("Threshold must be float or integer.")

    if threshold <= 0:
        raise ValueError("Threshold cannot be less than or equal to zero.")

    elif threshold > 0 and threshold < 0.5:
        warnings.warn(
            "Threshold is very low. Many false positives will be detected.", Warning
        )

    elif threshold > 8.0:
        warnings.warn(
            "Threshold is very high. Many localisations will be missed.", Warning
        )


def localize(
    stack_folder_paths: tuple[str], camera_specs: tuple, threshold: float
) -> list["np.ndarray"]:
    """
    This function carries out single-molecule localization for
    a given set of image stacks.
    ----------------------------------------------------------
    In:
    stack_folder_paths - file paths of image stacks
    camera_specs - the pixel size, adu, offset, and gain of the camera
    threshold - the minimum intensity in the filtered image for
    local maxima detection
    ----------------------------------------------------------
    Out:
    A list of N x 8 numpy arrays where N denotes the number of localizations detected
    in a frame.
    ----------------------------------------------------------
    """

    # Ensure only valid thresholds are input #

    check_threshold(threshold)

    ##--------INITIALIZATION--------##

    pix_size, adu, offset, gain = camera_specs

    frame_num, id = 1, 1

    localisations = []

    ##--------PROGRAM START--------##

    stacks = load_stacks(stack_folder_paths)

    for j, stack in enumerate(stacks):
        print("Processing stack " + str(j + 1) + " of " + str(len(stack_folder_paths)))

        for i, frame in enumerate(stack):
            image_spots, maxima_coords = get_spots(frame, pix_size, threshold)

            frame_locs = localise_frame(
                image_spots,
                maxima_coords,
                pix_res=pix_size,
                frame_num=frame_num,
                id_num=id,
                adu=adu,
                gain=gain,
                offset=offset,
            )

            frame_locs = frame_locs[frame_locs[:, -1] != np.inf]
            frame_locs = frame_locs[frame_locs[:, -1] != 0]

            localisations.append(frame_locs)

            id += frame_locs.shape[0]
            frame_num += 1

            if (i + 1) % 100 == 0:
                print("Processed " + str(i + 1) + "/" + str(stack.shape[0]) + " frames")

            del frame_locs

    return localisations
