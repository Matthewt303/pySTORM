import os
import warnings
from internals.file_io import gather_im_stacks

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
    #----------------------#

    if not isinstance(movie_folder, str):

        raise TypeError("Folder must be a string.")
    
    if not os.path.isdir(movie_folder):

        raise OSError("Directory not found.")

    im_stack_paths = gather_im_stacks(movie_folder)
    
    return tuple(im_stack_paths)

def get_camera_params(pixel_size: float=None, adu: float=None,
                      gain: int=None) -> tuple[float]:
    
    """
    This function is for users to enter their camera specifications.
    Specs undergo a sanity check and raise errors if they are physically
    impossible and warnings if they are possible but implausible.
    ----------------------------------------------------------
    In:
    pixel_size - the pixel size in nm at the sample plane.
    adu - analog-digital conversion factor. 
    gain - EM amplification.
    ----------------------------------------------------------
    Out:
    specs - tuple of camera specifications.
    ----------------------------------------------------------

    """
    
    ## INPUT CHECKS ##
    #----------------------#
    
    if (pixel_size is None
        or adu is None
        or gain is None
        ):

        raise ValueError("Missing one or more camera specification value.")
    
    if (not isinstance(pixel_size, float)
        or not isinstance(adu, float)
        or not isinstance(gain, int)
        ):

        raise TypeError("One or more camera specs are not of the correct type." \
        " Pixel size and adu must be floats while gain must be an integer.")
    
    if (pixel_size <= 0
        or adu <= 0
        or gain <= 0
        ):

        raise ValueError("Camera specs cannot be negative.")
    
    if pixel_size < 30:
    
        warnings.warn("Pixel size is very small. Image is probably undersampled",
                      Warning)
        
    elif pixel_size > 250:

        warnings.warn("Pixel size is very large. Localization accuracy will likely be poor",
                      Warning)
    
    specs = (pixel_size, adu, gain)

    return specs

