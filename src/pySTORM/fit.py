import warnings
from internals.image_analysis import get_spots
from internals.single_mol_fitting import localise_frame

def check_threshold(threshold: float) -> None:

    if not isinstance(threshold, float) or not isinstance(threshold, int):

        raise TypeError("Threshold must be float or integer.")
    
    if threshold <= 0:

        raise ValueError("Threshold cannot be less than or equal to zero.")
    
    elif threshold > 0 and threshold < 0.5:

        warnings.warn("Threshold is very low. Many false positives will be detected.",
                      Warning)
    
    elif threshold > 8.0:

        warnings.warn("Threshold is very high. Many localisations will be missed.",
                      Warning)

def localize():

    pass