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

    image = image_frame.view()

    smoothed_im = dog_filter(image)

    rms = np.sqrt(np.mean(smoothed_im**2))

    local_maxima = extract_local_maxima(smoothed_im, threshold * rms)

    spots, spot_coords = extract_spot_rois(image, local_maxima, pix_res)

    return spots, spot_coords


def bin_image(locs: "np.ndarray", size: int) -> "np.ndarray":
    """
    Bins localisation data into an image
    ---------------------------------------------------------------
    In:
    ---------------------------------------------------------------
    locs - xy localisation data
    size - Maximum dimensions (in pixels) of the image, determined by max xy position
    ---------------------------------------------------------------
    Output:
    image - super-resolution image
    """

    image = np.zeros((size, size), dtype=np.float32)

    for x, y in locs:
        im_x, im_y = np.int32(x), np.int32(y)

        if 0 <= im_x < size and 0 <= im_y < size:
            image[im_y, im_x] += 1

    return image


def bin_locs(locs: "np.ndarray", mag: float) -> "np.ndarray":
    """
    Summary:
    Converts xy localisation data into an image with a user-specified
    scaling factor
    ---------------------------------------------------------------
    In:
    locs - xy localisation data
    size - Maximum dimensions (in pixels) of the image, determined by max xy position
    mag - scaling of image. E.g. for locs in nm, mag = 0.1 = 0.1 pix/nm = 10 nm / pix
    ---------------------------------------------------------------
    Out:
    image - super-resolution image as a 2D array with dimensions of size x size.

    """

    xy_locs = mag * locs[:, 2:4].reshape(-1, 2)

    x_locs, y_locs = xy_locs[:, 0], xy_locs[:, 1]

    size = np.int64(np.ceil(np.max(xy_locs)))

    locs_to_keep = (x_locs >= 0) & (x_locs < size) & (y_locs >= 0) & (y_locs < size)

    new_locs = xy_locs[locs_to_keep]

    image = bin_image(new_locs, size)

    return image
