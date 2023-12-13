"""
This module contains implementations of different cumulative distribution functions.
They are used by the get_electron_occupation function in simcats.ideal_csd.geometric.

@author: f.hader
"""

import numpy as np


def sigmoid_cdf(x: np.ndarray, x0: float, gam: float) -> np.ndarray:
    """Evaluates a Cumulative Distribution Function (CDF) based on a Sigmoid.

    The distribution is defined by x0 and gamma, at given positions x.

    Args:
        x (np.ndarray): The position of the points (in x-/voltage-space) for which the sigmoid CDF will be evaluated
        x0 (float): The center of the function
        gam (float): Gamma-value of the distribution

    Returns:
        np.ndarray: Y-values for the supplied x-values
    """
    if np.isclose(gam, 0.0):
        result = x - x0
        result[result > 0] = 1
        result[result < 0] = 0
        return result
    return 0.5 * (1 + np.tanh((x - x0) / gam))


def multi_sigmoid_cdf(x: np.ndarray, params: list) -> np.ndarray:
    """Evaluates a function based on multiple additive Sigmoid Cumulative Distribution Functions (CDFs).

    Each CDF is defined by a x0 and gamma, at given positions x.

    Args:
        x (np.ndarray): The position of the points (in x-/voltage-space) for which the sigmoid CDF will be evaluated
        params (list): List of parameters for the Sigmoid CDFs. For each CDF a floating point value defining the center
            and a floating point value defining gamma are expected. \n
            Example: \n
            [x0_0, gamma_0, x0_1, gamma_1, ...]

    Returns:
        np.ndarray: Y-values for the supplied x-values
    """
    assert not (len(params) % 2)
    return sum([sigmoid_cdf(x, *params[i : i + 2]) for i in range(0, len(params), 2)])


def cauchy_cdf(x: np.ndarray, x0: float, gam: float) -> np.ndarray:
    """Evaluates a Cauchy Cumulative Distribution Function (CDF).

    The distribution is defined by x0 and gamma, at given positions x.

    Args:
        x (np.ndarray): The position of the points (in x-/voltage-space) for which the Cauchy CDF will be evaluated
        x0 (float): The center of the function
        gam (float): Gamma-value of the distribution

    Returns:
        np.ndarray: Y-values for the supplied x-values
    """
    if np.isclose(gam, 0.0):
        result = x - x0
        result[result > 0] = 1
        result[result < 0] = 0
        return result
    return 1 / np.pi * np.arctan((x - x0) / gam) + 0.5


def multi_cauchy_cdf(x: np.ndarray, params: list) -> np.ndarray:
    """Evaluates a function based on multiple additive Cauchy Cumulative Distribution Functions (CDFs).

    Each CDF is defined by a x0 and gamma, at given positions x.

    Args:
        x (np.ndarray): The position of the points (in x-/voltage-space) for which the Cauchy CDF will be evaluated
        params (list): List of parameters for the Cauchy CDFs. For each CDF a floating point value defining the center
            and a floating point value defining gamma are expected. \n
            Example: \n
            [x0_0, gamma_0, x0_1, gamma_1, ...]

    Returns:
        np.ndarray: Y-values for the supplied x-values
    """
    assert not (len(params) % 2)
    return sum([cauchy_cdf(x, *params[i : i + 2]) for i in range(0, len(params), 2)])