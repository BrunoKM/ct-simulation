import numpy as np
import scipy
from scipy import interpolate
from ct_lib import *
from ct_detect import *


def ct_calibrate(photons, material, sinogram, scale, correct=True):
    """ ct_calibrate convert CT detections to linearised attenuation
    sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
    in phantom x [angles, samples] and returns a linear attenuation sinogram
    [angles, samples]. photons is the source energy distribution, material is the
    material structure containing names, linear attenuation coefficients and
    energies in mev, and scale is the size of each pixel in x, in cm."""

    # Get dimensions and work out detection for just air of twice the side
    # length (has to be the same as in ct_scan.m)
    n_samples = sinogram.shape[1]

    max_width = 2 * n_samples * scale  # Width of the scanned area
    calibration_scan = ct_detect(photons, material.coeff("Air"), max_width)

    # perform calibration
    total_attenuation = -np.log(sinogram / calibration_scan)

    return total_attenuation
