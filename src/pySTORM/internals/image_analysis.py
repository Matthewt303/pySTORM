import numpy as np
from numba import jit, prange
from numba.typed import List
from skimage.feature import peak_local_max
from skimage.filters import difference_of_gaussians
import matplotlib.pyplot as plt

def extract_local_maxima(image, threshold):

    coordinates = peak_local_max(
    image,
    min_distance=2,          # Minimum number of pixels between peaks
    threshold_abs=threshold,      # Minimum intensity to be considered a spot
    num_peaks=np.inf,      # Maximum number of peaks to return
    )

    return coordinates.astype(np.int32)

def dog_filter(image):

    im = image.copy()

    filt_im = difference_of_gaussians(im, 1, 6)

    return filt_im

@jit(nopython=True, nogil=True, cache=False)
def convert_pix_to_um(data, pix_res: float):

    return data * pix_res

@jit(nopython=True, nogil=True, cache=False)
def get_spot_edges(x: int, y: int, width: int):

    # Width in pixels

    x_min, y_min = x - int(0.5 * width), y - int(0.5 * width),

    x_max, y_max = x + int(0.5 * width), y + int(0.5 * width)

    return np.array([x_min, x_max, y_min, y_max]).reshape(1, 4)

@jit(nopython=True, nogil=True, cache=False)
def extract_spot(image, edges):

    im = image.copy()

    horizontal_filt = im[:, edges[0, 0]:edges[0, 1]]

    vertical_filt = horizontal_filt[edges[0, 2]:edges[0, 3], :]

    return vertical_filt

@jit(nopython=True, nogil=True, cache=False)
def extract_spot_rois(image, spot_centers, pix_res):

    spots = List()

    spot_edges = List()

    for i in range(0, spot_centers.shape[0]):

        y, x = spot_centers[i, 0], spot_centers[i, 1]

        edge_coords = get_spot_edges(x, y, width=8)

        if np.any(edge_coords > image.shape[0] - 1) is np.True_:

            pass

        elif np.any(edge_coords < 0) is np.True_:

            pass

        else:
        
            spot = extract_spot(image, edge_coords)

            edge_coord_xy = np.array([edge_coords[0, 0], edge_coords[0, 2]]).reshape(1, 2)

            edge_coord_um = convert_pix_to_um(edge_coord_xy, pix_res=pix_res)

            spots.append(spot)

            spot_edges.append(edge_coord_um.astype(np.float32))
    
    return spots, spot_edges

@jit(nopython=True, nogil=True, cache=False)
def convert_pix_to_um(data, pix_res: float):

    return data * pix_res

def get_spots(image_frame: 'np.ndarray', pix_res: float) -> list:

    """
    Carries out wavelet filter, otsu segmentation, and local maxima
    identification for a single image frame to detect molecules. A 
    10 px x 10 px ROI is then extracted for each molecule.

    In: image

    Out: spots - list of 10 x 10 spots, each containing the image of a molecule
    spot_coords - list of coordinates specifying the top-left corner of each ROI
    """

    #image = np.array(image_frame.copy())

    image = image_frame.copy()

    smoothed_im = dog_filter(image)

    threshold = np.sqrt(np.mean(smoothed_im ** 2))

    local_maxima = extract_local_maxima(smoothed_im, 5.0 * threshold)

    spots, spot_coords = extract_spot_rois(image, local_maxima, pix_res)
    
    return spots, spot_coords

def get_spots_denoised(denoised_im: 'np.ndarray', noisy_im: 'np.ndarray', pix_res: float) -> 'np.ndarray':

    image = denoised_im.copy()
    noisy_image = noisy_im.copy()

    smoothed_im = dog_filter(noisy_image)

    threshold = np.sqrt(np.mean(smoothed_im ** 2))

    local_maxima = extract_local_maxima(smoothed_im, 2.0 * threshold)

    spots, spot_coords = extract_spot_rois(image, local_maxima, pix_res)

    return spots, spot_coords

