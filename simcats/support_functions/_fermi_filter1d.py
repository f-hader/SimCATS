"""This module contains an implementation of a 1D Fermi-Dirac filter.
It is used by the class OccupationTransitionBlurringFermiDirac in simcats.distortions.

@author: f.hader
"""

from typing import Union

import numpy as np
from scipy.ndimage import convolve1d


def fermi_dirac_derivative(x: np.ndarray, x0: float, sigma: float):
    """Evaluates the derivative of the Fermi–Dirac distribution.

    The distribution is defined by a center x0 and sigma, at given positions x. Allows creating a Fermi-Dirac filter
    kernel for the blurring of occupation transitions.

    Args:
        x (np.ndarray): The position of the points (in x-/voltage-space) for which the derivative will be evaluated
        x0 (float): The center of the function
        sigma (float): Gamma-value of the distribution

    Returns:
        np.ndarray: Y-values for the supplied x-values
    """
    return (0.5 * (1 / np.cosh((x - x0) / sigma)) ** 2) / sigma


def fermi_filter1d(
    input: np.ndarray,
    sigma: float,
    axis: int = -1,
    mode: str = "reflect",
    cval: float = 0.0,
    radius: Union[None, int] = None,
):
    """1-D Fermi filter.

    Args:
        input (np.ndarray): The data that will be convolved
        sigma (float): The sigma for the Fermi-Dirac distribution
        axis (int): The axis of input along which to calculate. Default is -1.
        mode (str): One of 'reflect', 'constant', 'nearest', 'mirror', or 'wrap'. The mode parameter determines how
            the input array is extended beyond its boundaries. Default is 'reflect'. \n
            Behavior for each valid value is as follows: \n
            - 'reflect' (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last
              pixel. This mode is also sometimes referred to as half-sample symmetric.\n
            - 'constant' (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with
              the same constant value, defined by the cval parameter.\n
            - 'nearest' (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.\n
            - 'mirror' (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last
              pixel. This mode is also sometimes referred to as whole-sample symmetric.\n
            - 'wrap' (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.\n
            For consistency with the interpolation functions, the following mode names can also be used:\n
            - 'grid-mirror': This is a synonym for 'reflect'.\n
            - 'grid-constant': This is a synonym for 'constant'.\n
            - 'grid-wrap': This is a synonym for 'wrap'.
        cval (float): Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        radius (Union[None, int]): The radius of the filter kernel. The kernel size will be 2*radius+1. If radius is
            None, a default radius = round(4.0 * sigma) will be used (similar to gaussian_kernel1d behavior).
            Default is None.

    Returns:
        np.ndarray: Input data smoothed by the 1-D Fermi filter.
    """
    # dynamically calculate kernel size (see gaussian_kernel1d documentation for reference)
    if radius is None:
        radius = round(4.0 * sigma)
    # Construct the tanh kernel centered at 0
    kernel = fermi_dirac_derivative(x=np.arange(-radius, radius + 1), x0=0, sigma=sigma)
    # Normalize the kernel so that it sums to 1
    kernel /= np.sum(kernel)
    return convolve1d(input=input, weights=kernel, axis=axis, mode=mode, cval=cval)
