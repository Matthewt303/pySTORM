from pySTORM.internals.file_io import load_image
import pySTORM.internals.preview as pview


def preview(path: str, threshold: float) -> None:
    """
    This function is for users to determine whether they have entered a
    satisfactory value for the threshold. The function displays three images
    from the first image stack file: the first frame, the middle frame, and
    the final frame, as well as the local maxima detection overlaid. Users
    have the option to continue or exit.
    ----------------------------------------------------------
    In:
    path - path of folder with image stacks.
    threshold - the minimum intensity in the filtered image for
    local maxima detection
    ----------------------------------------------------------
    Out:
    None but program should either exit or continue.
    ----------------------------------------------------------

    """

    first_stack_file = path[0]

    stack = load_image(first_stack_file)

    indices = (0, stack.shape[0] // 2, stack.shape[0] - 1)

    for i in indices:
        image = stack[i]

        maxima = pview.filt_fit(image, threshold)

        pview.show_image(image, maxima)

    pview.message()
