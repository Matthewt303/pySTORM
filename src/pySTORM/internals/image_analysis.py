import numpy as np
from numba import jit
from numba.typed import List
import cv2 as cv


def extract_local_maxima(img, threshold, neighborhood=8):
    dilated = cv.dilate(img, np.ones((neighborhood, neighborhood)))

    local_max_mask = img == dilated

    local_max_mask &= img > threshold

    ys, xs = np.where(local_max_mask)
    coords = np.array(list(zip(ys, xs))).reshape(-1, 2)

    return coords.astype(np.int32)


def dog_filter(image):
    less_filt = cv.GaussianBlur(image, (11, 11), 1, borderType=cv.BORDER_REPLICATE)
    more_filt = cv.GaussianBlur(image, (51, 51), 6, borderType=cv.BORDER_REPLICATE)

    filt_im = less_filt - more_filt

    return filt_im


@jit(nopython=True, nogil=True, cache=False)
def convert_pix_to_um(data, pix_res: float):
    return data * pix_res


@jit(nopython=True, nogil=True, cache=False)
def get_spot_edges(x: int, y: int, width: int):
    # Width in pixels

    x_min, y_min = (
        x - int(0.5 * width),
        y - int(0.5 * width),
    )

    x_max, y_max = x + int(0.5 * width), y + int(0.5 * width)

    return np.array([x_min, x_max, y_min, y_max]).reshape(1, 4)


@jit(nopython=True, nogil=True, cache=False)
def extract_spot(image, edges):
    im = image.copy()

    horizontal_filt = im[:, edges[0, 0] : edges[0, 1]]

    vertical_filt = horizontal_filt[edges[0, 2] : edges[0, 3], :]

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

            edge_coord_xy = np.array([edge_coords[0, 0], edge_coords[0, 2]]).reshape(
                1, 2
            )

            edge_coord_um = convert_pix_to_um(edge_coord_xy, pix_res=pix_res)

            spots.append(spot)

            spot_edges.append(edge_coord_um.astype(np.float32))

    return spots, spot_edges


def get_spots(image_frame: "np.ndarray", pix_res: float, threshold: float) -> list:
    """


    In: image

    Out: spots - list of 10 x 10 spots, each containing the image of a molecule
    spot_coords - list of coordinates specifying the top-left corner of each ROI
    """

    # image = np.array(image_frame.copy())

    image = image_frame.copy()

    smoothed_im = dog_filter(image)

    rms = np.sqrt(np.mean(smoothed_im**2))

    local_maxima = extract_local_maxima(smoothed_im, threshold * rms)

    spots, spot_coords = extract_spot_rois(image, local_maxima, pix_res)

    return spots, spot_coords
