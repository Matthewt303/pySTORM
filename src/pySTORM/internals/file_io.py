import numpy as np
import pandas as pd
import tifffile as tiff
import os
from typing import Generator

def gather_im_stacks(folder_path: str) -> list[str]:
    
    im_files = [
        os.path.join(folder_path, file) for file in os.listdir(folder_path)
        if file.endswith('.tif')
    ]

    return im_files

def load_image(path: str):

    with tiff.TiffFile(path) as tif:
        memmap_stack = tif.asarray(out='memmap')
    
    return memmap_stack.astype(np.float32)

def load_stacks(paths: tuple[str]) -> Generator['np.ndarray', None, None]:

    for path in paths:

        stack = load_image(path)

        yield stack

def save_localisation_table_csv(loc_data: list, out_folder: str):

    localisation_data = np.vstack(loc_data).reshape(-1, 8)

    # Remove unrealistically large uncertainties.
    localisation_data = localisation_data[localisation_data[:, -1] < 500]

    headers = ['id',
               'frame',
               'x [nm]',
               'y [nm]',
               'sigma [nm]',
               'intensity [photon]',
               'offset [photon]',
               'uncertainty [nm]']
    
    dataframe = pd.DataFrame(data=localisation_data,
                             columns=headers,
                             dtype=np.float32)
    
    df_filt = dataframe[dataframe['uncertainty [nm]'].notnull()]
    
    df_filt.to_csv(os.path.join(out_folder, 'reconstruction_tstorm.csv'),
                     sep=',',
                     index=False)