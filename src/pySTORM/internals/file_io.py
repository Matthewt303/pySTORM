import numpy as np
import pandas as pd
import tifffile as tiff
import os

def gather_im_stacks(folder_path: str) -> list[str]:

    pass

def load_image(path: str):

    with tiff.TiffFile(path) as tif:
        memmap_stack = tif.asarray(out='memmap')
    
    return memmap_stack.astype(np.float32)

def save_localisation_table(loc_data: list, out_folder: str,
                            denoised=None):

    localisation_data = np.vstack(loc_data).reshape(-1, 8)

    # Remove unrealistically large uncertainties.
    localsiation_data = localisation_data[localisation_data[:, -1] < 500]

    headers = ['id',
               'frame',
               'x [nm]',
               'y [nm]',
               'sigma [nm]',
               'intensity [photon]',
               'offset [photon]',
               #'bkgstd [photon]',
               'uncertainty [nm]']
    
    dataframe = pd.DataFrame(data=localisation_data,
                             columns=headers,
                             dtype=np.float32)
    
    df_filt = dataframe[dataframe['uncertainty [nm]'].notnull()]

    if denoised is not None:

        df_filt.to_csv(os.path.join(out_folder, 'dn_reconstruction_tstorm.csv'),
                     sep=',',
                     index=False)
    
    else:
    
        df_filt.to_csv(os.path.join(out_folder, 'reconstruction_tstorm.csv'),
                     sep=',',
                     index=False)