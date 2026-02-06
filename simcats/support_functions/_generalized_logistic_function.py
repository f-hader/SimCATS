"""This module contains an implementation of the generalized logistic function (GLF) used for sensor barriers.

@author: b.papajewski
"""

from typing import Union

import numpy as np


def glf(potential: Union[float, np.ndarray],
        asymptote_left: float,
        asymptote_right: float,
        growth_rate: float,
        asymmetry: float,
        shape_factor: float,
        denominator_offset: float = 1,
        offset: float = 0
        ) -> Union[float, np.ndarray]:
    """Function implementing the generalized logistic function.
    For further information see: https://en.wikipedia.org/wiki/Generalised_logistic_function

    Args:
        potential (Union[float, np.ndarray]): Originally called t. The potential is the variable of the GLF for which
            the value of the function should be calculated.
        asymptote_left (float): Originally called A. This parameter is the left horizontal asymptote of the function.
            Any rational number can be used as the left asymptote. This parameter may take any rational number.
        asymptote_right (float): Originally called K. Specifies the right horizontal asymptote of the function
            when denominator_offset=1. If asymptote_left=0 and denominator_offset=1 then this parameter is also called
            the carrying capacity. This parameter may take any rational number.
        growth_rate (float): Originally called B. The growth rate of the function. The value must be a float and can be
            any rational number. Be careful with negative values, because the function is mirrored on a vertical
            straight line for these. This line passes through the point where the potential equals `offset`.
        asymmetry (float): Originally called nu. This parameter introduces skew and affects symmetry. It also affects
            near which asymptote maximum growth occurs. The value of asymmetry must be a rational number greater than
            zero. \n
            - `asymmetry > 1`: the curve rises more gradually before the midpoint and more sharply after. \n
            - `asymmetry < 1`: the curve rises quickly early on and levels off more slowly.
        shape_factor (float): Originally called Q. is related to the value Y(0) and adjusts the curve’s value at the
            y-intercept. Thereby it changes the shape of the function without changing the asymptotes. The shape factor
            can be any rational number.
        denominator_offset (float): Originally called C. A constant added to the denominator inside the power. Controls
            the initial level of the denominator.This parameter must be a rational number.
            It typically takes a value of 1. Otherwise, the upper asymptote is
            asymptote_left + (asymptote_right-asymptote_left)/(denominator_offset^(1/asymmetry)).
        offset (float): Parameter that shifts the function starting from the zero point. If the offset is positive, the
            function is shifted to the right and if it is negative, it is shifted to the left.

    Returns:
        Union[float, np.ndarray]: Value of the GLF at the given potential (originally time t) for a given set of
        parameters. The returned datatype is the same as the type of `potential`.
    """

    return asymptote_left + ((asymptote_right - asymptote_left) / (
        np.power(denominator_offset + shape_factor * np.exp(-growth_rate * (potential - offset)), (1 / asymmetry))))


def inverse_glf(
        value: Union[float, np.ndarray],
        asymptote_left: float,
        asymptote_right: float,
        growth_rate: float,
        asymmetry: float,
        shape_factor: float,
        denominator_offset: float = 1,
        offset: float = 0
) -> Union[float, np.ndarray]:
    """
    Inverse of the function `glf` which computes the inverse of a generalized logistic function for a set of parameters.

    Args:
        value: The input value(s) for which to compute the inverse GLF. Can be a single float or a numpy array.
        asymptote_left (float): Originally called A. This parameter is the left horizontal asymptote of the function.
            Any rational number can be used as the left asymptote. This parameter may take any rational number.
        asymptote_right (float): Originally called K. Specifies the right horizontal asymptote of the function
            when denominator_offset=1. If asymptote_left=0 and denominator_offset=1 then this parameter is also called
            the carrying capacity. This parameter may take any rational number.
        growth_rate (float): Originally called B. The growth rate of the function. The value must be a float and can be
            any rational number. Be careful with negative values, because the function is mirrored on a vertical
            straight line for these. This line passes through the point where the potential equals `offset`.
        asymmetry (float): Originally called nu. This parameter introduces skew and affects symmetry. It also affects
            near which asymptote maximum growth occurs. The value of asymmetry must be a rational number greater than
            zero. \n
            - `asymmetry > 1`: the curve rises more gradually before the midpoint and more sharply after. \n
            - `asymmetry < 1`: the curve rises quickly early on and levels off more slowly.
        shape_factor (float): Originally called Q. is related to the value Y(0) and adjusts the curve’s value at the
            y-intercept. Thereby it changes the shape of the function without changing the asymptotes. The shape factor
            can be any rational number.
        denominator_offset (float): Originally called C. A constant added to the denominator inside the power. Controls
            the initial level of the denominator.This parameter must be a rational number.
            It typically takes a value of 1. Otherwise, the upper asymptote is \n
            asymptote_left + (asymptote_right-asymptote_left)/(denominator_offset^(1/asymmetry)).
        offset (float): Parameter that shifts the function starting from the zero point. If the offset is positive, the
            function is shifted to the right and if it is negative, it is shifted to the left.

    Returns:
        Union[float, np.ndarray]: Potential (originally time t) of the inverse GLF at the given value for a given set of
        parameters. The returned datatype is the same as the type of `value`.
    """
    return (growth_rate * offset - np.log((((asymptote_left - asymptote_right) / (
            asymptote_left - value)) ** asymmetry - denominator_offset) / shape_factor)) / growth_rate


def multi_glf(potential: Union[float, np.ndarray], *params: float) -> Union[float, np.ndarray]:
    """Function that combines several GLFs as a sum into one function.
    For further information see: https://en.wikipedia.org/wiki/Generalised_logistic_function

    Each GLF has the following parameters: \n
    - asymptote_left (float): Originally called A. This parameter is the left horizontal asymptote of the function.
      Any rational number can be used as the left asymptote.
    - asymptote_right (float): Originally called K. Specifies the right horizontal asymptote of the function
      when denominator_offset=1. If asymptote_left=0 and denominator_offset=1 then this parameter is also called the
      carrying capacity. This parameter may take any rational number.
    - growth_rate (float): Originally called B. The growth rate of the function. The value must be a float and can be
      any rational number. Be careful with negative values, because the function is mirrored on a vertical straight
      line for these. This line passes through the point where the potential equals `offset`.
    - asymmetry (float): Originally called nu. This parameter introduces skew and affects symmetry. It also affects
      near which asymptote maximum growth occurs. The value of asymmetry must be a rational number greater than zero. \n
      - If `asymmetry > 1`: the curve rises more gradually before the midpoint and more sharply after. \n
      - If `asymmetry < 1`: the curve rises quickly early on and levels off more slowly.
    - shape_factor (float):  Originally called Q. is related to the value Y(0) and adjusts the curve’s value at the
      y-intercept. Thereby it changes the shape of the function without changing the asymptotes. The shape factor
      can be any rational number.
    - denominator_offset (float):  Originally called C. A constant added to the denominator inside the power. Controls
      the initial level of the denominator.This parameter must be a rational number.
      It typically takes a value of 1. Otherwise, the upper asymptote is \n
      asymptote_left + (asymptote_right - asymptote_left) / (denominator_offset^(1 / asymmetry)).
    - offset (float): Potential offset, that shifts the function starting from the zero point. If the offset is
      positive, the function is shifted to the right and if it is negative, it is shifted to the left.

    The number of GLFs is specified by the number of parameters. To do this, the parameter count must be divisible by
    seven and a GLF is added for every seven other parameters.

    Args:
        potential (Union[float, np.ndarray]): Originally called t. The potential is the variable of the GLF for which
            the value of the function should be calculated.
        *params: Additional positional arguments representing the GLF parameters. The number of additional parameters
            must be divisible by seven and determines the number of GLFs that are used for the Multi-GLF. All parameters
            consist of sequential groups of seven floats that each represent a single GLF. All individual parameters are
            described above and are in the same order as they are described.

    Returns:
        Union[float, np.ndarray]: Value of the GLF at the given potential (originally time t).
    """
    assert not (len(params) % 7)
    return sum([glf(potential, *params[i: i + 7]) for i in range(0, len(params), 7)])
