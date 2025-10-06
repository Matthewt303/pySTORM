import numpy as np
from numba import jit, prange
import math
from internals.image_analysis import convert_pix_to_um
import matplotlib.pyplot as plt

## PARAMETER INITIALISATION ##
#-------------------------------------------------#

@jit(nopython=True, nogil=True, cache=False)
def sum_and_center_of_mass(spot, size):
    x = 0.0
    y = 0.0
    _sum_ = 0.0
    for i in range(size):
        for j in range(size):
            x += spot[i, j] * i
            y += spot[i, j] * j
            _sum_ += spot[i, j]
    x /= _sum_
    y /= _sum_
    return y, x

@jit(nopython=True, nogil=True, cache=False)
def estimate_bg(spot):

    return np.min(spot)

@jit(nopython=True, nogil=True, cache=False)
def estimate_intensity(spot, bg):

    return np.max(spot)

@jit(nopython=True, nogil=True, cache=False)
def initial_parameters(spot, size):
   
    """
    This function has been changed from the original.
    """

    y, x = sum_and_center_of_mass(spot, size)
    bg = estimate_bg(spot)
    photons = estimate_intensity(spot, bg)
    photons_sane = np.maximum(1.0, photons)
    sy, sx = spot.shape[0] / 6, spot.shape[1] / 6
    return x, y, photons_sane, bg, sx, sy

@jit(nopython=True, nogil=True, cache=False)
def initial_theta_sigma(spot, size):
    theta = np.zeros(5, dtype=np.float32)
    theta[0], theta[1], theta[2], theta[3], sx, sy = initial_parameters(spot, size)
    theta[4] = (sx + sy) / 2
    return theta

## MODELS AND DERIVATIVES ##
#-------------------------------------------------#

@jit(nopython=True, nogil=True, cache=False)
def gaussian_integral(x, mu, sigma):
    sq_norm = 0.70710678118654757 / sigma  # sq_norm = sqrt(0.5/sigma**2)
    d = x - mu
    return 0.5 * (math.erf((d + 0.5) * sq_norm) - math.erf((d - 0.5) * sq_norm))

@jit(nopython=True, nogil=True, cache=False)
def derivative_gaussian_integral(x, mu, sigma, photons, PSFc):
    d = x - mu
    a = np.exp(-0.5 * ((d + 0.5) / sigma) ** 2)
    b = np.exp(-0.5 * ((d - 0.5) / sigma) ** 2)
    dudt = -photons * PSFc * (a - b) / (np.sqrt(2.0 * np.pi) * sigma)

    return dudt

@jit(nopython=True, nogil=True, cache=False)
def derivative_gaussian_integral_1d_sigma(x, mu, sigma, photons, PSFc):
    ax = np.exp(-0.5 * ((x + 0.5 - mu) / sigma) ** 2)
    bx = np.exp(-0.5 * ((x - 0.5 - mu) / sigma) ** 2)
    dudt = (
        -photons
        * (ax * (x + 0.5 - mu) - bx * (x - 0.5 - mu))
        * PSFc
        / (np.sqrt(2.0 * np.pi) * sigma**2)
    )

    return dudt

@jit(nopython=True, nogil=True)
def derivative_gaussian_integral_2d_sigma(x, y, mu, nu, sigma, photons, PSFx, PSFy):
    dSx = derivative_gaussian_integral_1d_sigma(x, mu, sigma, photons, PSFy)
    dSy = derivative_gaussian_integral_1d_sigma(y, nu, sigma, photons, PSFx)
    dudt = dSx + dSy

    return dudt

## OPTIMISATION ##
#-------------------------------------------------#

@jit(nopython=True, nogil=True, cache=False)
def neg_log_likelihood_sigma(theta, spot):

    x, y, A, bg, sigma = theta[:5]
    sigma_y = theta[5] if len(theta) > 5 else sigma

    height, width = spot.shape
    margin = 1.0
    if (
        x < -margin or x > width + margin or
        y < -margin or y > height + margin or
        sigma < 0.01 or sigma_y < 0.01
    ):
        return 1e10

    logL = 0.0
    for i in range(height):
        PSFy = gaussian_integral(i, y, sigma_y)
        for j in range(width):
            PSFx = gaussian_integral(j, x, sigma)
            model = A * PSFx * PSFy + bg
            model = max(model, 1e-9)
            data = spot[i, j]
            logL += data * np.log(model) - model

    return -logL

@jit(nopython=True, nogil=True, cache=False)
def norm_by_column(matrix):

    column_norms = np.zeros((1, matrix.shape[1]))

    for i in range(0, matrix.shape[1]):

        column_norms[0, i] = np.linalg.norm(matrix[:, i])
    
    return column_norms

@jit(nopython=True, nogil=True, cache=False)
def nelder_mead_sigma(obj_func, x0: 'np.ndarray[np.float32]',
                       spot: tuple, max_iter: int,
                       tol: float) -> 'np.ndarray[np.float32]':
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    n = len(x0)

    simplex = np.zeros((n + 1, n), dtype=np.float32)
    fvals = np.zeros(n + 1, dtype=np.float32)

    simplex[0] = x0

    for i in range(n):
        y = x0.copy()
        if y[i] != 0:
            y[i] = (1 + 0.05) * y[i]
        else:
            y[i] = 0.00025
        simplex[i + 1] = y

    for i in range(n + 1):
        fvals[i] = obj_func(simplex[i], *spot)

    it = 0
    while it < max_iter:
        it += 1
        indices = np.argsort(fvals)
        simplex = simplex[indices]
        fvals = fvals[indices]

        centroid = np.sum(simplex[:-1], axis=0) / (simplex.shape[0] - 1)
        xr = centroid + alpha * (centroid - simplex[-1])
        fxr = obj_func(xr, *spot)

        if fxr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fxe = obj_func(xe, *spot)
            if fxe < fxr:
                simplex[-1] = xe
                fvals[-1] = fxe
            else:
                simplex[-1] = xr
                fvals[-1] = fxr
        elif fxr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fxr
        else:
            if fxr < fvals[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
            fxc = obj_func(xc, *spot)
            if fxc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fxc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = obj_func(simplex[i], *spot)
        
        if np.max(norm_by_column(simplex[0] - simplex[1:])) < tol:
            break

    return simplex[0]

@jit(nopython=True, nogil=True, cache=False)
def mlefit_nm_sigma(spot, eps, max_it):

    size, _ = spot.shape

    # theta is [x, y, N, bg, S]
    theta0 = initial_theta_sigma(spot, size)
    theta_opt = nelder_mead_sigma(neg_log_likelihood_sigma, theta0, (spot,), max_it, eps)

    return theta_opt

## LOCALISATION PRECISION CALCULATIONS ##
#-------------------------------------------------#

@jit(nopython=True, nogil=True, cache=False)
def get_crlb(fisher_info: 'np.ndarray'):

    inv_fisher = np.linalg.pinv(fisher_info)

    crlb = np.diag(inv_fisher)

    return crlb

@jit(nopython=True, nogil=True, cache=False)
def calc_loc_prec(spot, theta_opt):
    
    width, height  = spot.shape
    n_params = theta_opt.shape[0]

    fisher = np.zeros((n_params, n_params), dtype=np.float32)
    dmodel_dt = np.zeros(n_params, dtype=np.float32)

    for i in range(height):
        PSFy = gaussian_integral(i, theta_opt[1], theta_opt[4])
        for j in range(width):
            PSFx = gaussian_integral(j, theta_opt[0], theta_opt[4])

            model = theta_opt[2] * PSFx * PSFy + theta_opt[3]

            dmodel_dt[0] = derivative_gaussian_integral(j, theta_opt[0],
                                                        theta_opt[4],
                                                        theta_opt[2],
                                                        PSFy)
            dmodel_dt[1] = derivative_gaussian_integral(i, theta_opt[1],
                                                        theta_opt[4],
                                                        theta_opt[2],
                                                        PSFx)
             
            dmodel_dt[4] = derivative_gaussian_integral_2d_sigma(
                 j, i, theta_opt[0], theta_opt[1], theta_opt[4], theta_opt[2],
                 PSFx, PSFy
             )

            dmodel_dt[2] = PSFx * PSFy
            dmodel_dt[3] = 1.0

            for k in range(n_params):
                for l in range(k, n_params):
                        fisher[k, l] += dmodel_dt[l] * dmodel_dt[k] / (model + 1e-9)
                        fisher[l, k] = fisher[k, l]
    
    crlb = get_crlb(fisher)
    return (np.sqrt(crlb[0]) + np.sqrt(crlb[1])) / 2

## PER SPOT SUBPIXEL LOCALISATION ##
#-------------------------------------------------#

@jit(nopython=True, nogil=True, cache=False, parallel=True)
def gaussmle(spots, eps, max_it):

    n = len(spots)
    thetas = np.zeros((n, 5), dtype=np.float32)

    loc_prec = np.zeros((n, 1), dtype=np.float32)

    func = mlefit_nm_sigma

    for i in prange(n):

        i = np.int64(i)
        
        spot = spots[i]

        thetas[i, :] = func(spot, eps, max_it)
        loc_prec[i, 0] = calc_loc_prec(spot, thetas[i, :])

    return thetas, loc_prec


## ID AND FRAME GENERATION AND CONVERSION FACTORS ##
#-------------------------------------------------#

def generate_frames(index, number_of_frames):

    return np.full((number_of_frames, 1), index)

def extract_xy(theta):

    xy = np.ascontiguousarray(theta[:, 0:2]).reshape((-1 ,2))

    return xy

def convert_theta_xy(theta_xy: 'np.ndarray', coords: list, roi_size: int, pix_res: float):

    theta_xy_as_frac = theta_xy / roi_size

    coords_array = np.array(coords).reshape(-1 ,2)

    # Ensure consistency in units between coords and theta then calc xy
    theta_xy_um = roi_size * pix_res * theta_xy_as_frac

    # xy localizations are now properly defined along x and y
    xy_locs = 1000 * (coords_array + theta_xy_um)

    return xy_locs

@jit(nopython=True, nogil=True, cache=False)
def convert_to_photons(data: 'np.ndarray', adu: float,
                        gain: int, offset: float) -> 'np.ndarray':

    baseline = np.minimum(offset, np.min(data))
    
    data_photons = (data - baseline) * adu * (1 / gain)

    return data_photons

## MAIN FUNCTION ##
#-------------------------------------------------#

def localise_frame(spots, coords, pix_res: float, frame_num: int,
                   id_num: int, adu: float,
                   gain: int, offset: int) -> 'np.ndarray':

    theta, loc_prec = gaussmle(spots, 0.001, 150)

    ids = np.linspace(id_num, id_num + theta.shape[0] - 1,
                              theta.shape[0]).reshape(-1, 1)

    frame_column = generate_frames(frame_num, theta.shape[0])

    xy_theta = extract_xy(theta)
    xy_locs = convert_theta_xy(xy_theta, coords, roi_size=8, pix_res=pix_res)

    intensity_ph = convert_to_photons(theta[:, 2],
                                      adu, gain, offset).reshape(-1, 1)
    
    bg_ph = convert_to_photons(theta[:, 3],
                                      adu, gain, offset).reshape(-1, 1)

    sigma_nm = 1000 * convert_pix_to_um(theta[:, 4], pix_res=pix_res)

    sigma_nm = sigma_nm.reshape(-1, 1)

    loc_prec_nm = 1000 * convert_pix_to_um(loc_prec,
                                           pix_res=pix_res).reshape(-1 ,1)
    
    loc_table = np.hstack((ids,
                           frame_column,
                           xy_locs,
                           sigma_nm,
                           intensity_ph,
                           bg_ph,
                           loc_prec_nm))
    
    return loc_table