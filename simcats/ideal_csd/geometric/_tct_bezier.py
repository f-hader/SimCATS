"""This module contains the implementation of the total charge transition (TCT) representation.

The TCTs are used to generate the boundaries/structures for the geometrically simulated charge stability diagrams (CSDs).

@author: f.hader
"""

import math

# used to check if a single x-value was passed to x_eval
import numbers
from typing import Union

import bezier
import numpy as np
import sympy


def tct_bezier(
    tct_params: np.ndarray,
    x_eval: Union[np.ndarray, numbers.Number],
    lut_entries: Union[int, None] = None,
    max_peaks: Union[int, None] = None,
) -> Union[np.ndarray, numbers.Number]:
    """Evaluates a total charge transition (TCT) with bezier curves and linear parts.
    A TCT separates the region with n electrons and the region with n+1 electrons in the system. The TCTs (series of
    connected lead-to-dot transitions) are represented by linear parts and bezier curves for the round triple-points.

    Args:
        tct_params (np.ndarray): Numpy array containing all required parameters to describe the TCT form. \n
            The parameters for a TCT are: \n
            [0] = length left (in x-/voltage-space, not number of points) \n
            [1] = length right (in x-/voltage-space, not number of points) \n
            [2] = gradient left (in voltages) \n
            [3] = gradient right (in voltages) \n
            [4] = start position x (bezier curve leftmost point) (in x-/voltage-space, not number of points) \n
            [5] = start position y (bezier curve leftmost point) (in x-/voltage-space, not number of points) \n
            [6] = end position x (bezier curve rightmost point) (in x-/voltage-space, not number of points) \n
            [7] = end position y (bezier curve rightmost point) (in x-/voltage-space, not number of points)
        x_eval (Union[np.ndarray, numbers.Number]): X-values for which the function is evaluated or a single x-value.
        lut_entries (Union[int, None]): Number of samples for the lookup-table. If this is not None, a lookup-table will
            be used to evaluate the points on the bezier curve, else it is solved explicitly. Default is None.
        max_peaks (Union[int, None]): Limit for the number of peaks of the TCT. If multiple TCTs are supplied, max_peaks
            is increased by 1 for every further wavefront. If None, all TCTs have an unlimited number of peaks and no
            outer linear part. Default is None.

    Returns:
        Union[np.ndarray, numbers.Number]: Y-values for the supplied x-values (if an array of values was supplied) or a
        single y-value.
    """
    # if x_eval is a single x-value, put it into a numpy array for the further processing
    if isinstance(x_eval, numbers.Number):
        single_value = True
        x_eval = np.array([x_eval])
    else:
        single_value = False
    # copy x_eval so that the original supplied array is not modified
    x_eval = np.copy(x_eval)

    # setup array for resulting y-values
    y_res = np.zeros(x_eval.shape)

    # setup coordinates for bezier curve nodes
    x_vals = np.array([tct_params[4], 0, tct_params[6]])
    y_vals = np.array([tct_params[5], 0, tct_params[7]])
    if x_vals[0] == x_vals[2] and y_vals[0] == y_vals[2]:
        # no interdot coupling region specified -> center anchor is equal to outer anchors
        x_vals[1] = x_vals[0]
        y_vals[1] = y_vals[0]
    else:
        # update center point by finding the intersection between left & right linear part
        x_vals[1] = (y_vals[0] - y_vals[2] + x_vals[2] * tct_params[3] - x_vals[0] * tct_params[2]) / (
            tct_params[3] - tct_params[2]
        )
        y_vals[1] = y_vals[0] + tct_params[2] * (x_vals[1] - x_vals[0])
    # check if the central bezier anchor is between left & right anchor (in x-space). Else the wavefront
    # can't be evaluated, as the "spike" of the bezier curve passes across one of the outer anchors and
    # leads to multiple y-values for some x-values
    assert x_vals[1] >= x_vals[0] and x_vals[1] <= x_vals[2]
    # check if the lengths of the left & right part are at least twice as long as the left/right bezier curve part
    assert tct_params[0] >= 2 * (x_vals[1] - x_vals[0]) and tct_params[1] >= 2 * (x_vals[2] - x_vals[1])
    # nodes as fortran array
    nodes = np.asfortranarray([x_vals, y_vals])

    # initialize bezier curve
    bezier_curve = bezier.Curve.from_nodes(nodes)
    xb, yb = sympy.symbols("x y")
    bezier_implicit = bezier_curve.implicitize()
    # retrieve a lookup-table if the given lut_entries value is not None
    if lut_entries:
        t = np.linspace(0, 1, lut_entries)
        bezier_lut = bezier_curve.evaluate_multi(t)

    # retrieve length of bezier curve & period length. Both are required to calculate the region
    # where the x-values to be evaluated are located
    bezier_length = x_vals[-1] - x_vals[0]
    left_lin_length = tct_params[0] - 2 * (x_vals[1] - x_vals[0])
    right_lin_length = tct_params[1] - 2 * (x_vals[2] - x_vals[1])
    period_length = tct_params[0] + tct_params[1]

    # substract leftmost x-value of the peak bezier-curve, to ensure that the period starts with
    # a bezier-curve at x=0 (to simplify the handling of x_eval using modulo operations)
    offset_x = x_vals[0]
    x_eval -= offset_x

    # calculate offset of rotated bezier (every second curve is rotated 180 degrees)
    # substract first point of bezier from last point of previous linear part
    y_offset_inv_bezier = (y_vals[2] + right_lin_length * tct_params[3]) - (-1 * y_vals[2])

    # calculate offset per period (y first point - y last point),
    # which is observed if the left & right sides have different lengths or slopes
    y_offset_period = y_vals[0] - ((y_offset_inv_bezier - y_vals[0]) + left_lin_length * tct_params[2])

    if max_peaks is not None:
        # extract ids within max peak range (to be simulated with honey_wave_bezier)
        # all other ids are simulated as linear parts
        # lowest value = left bezier anchor
        # highest value = right bezier anchor + #additional_peaks * period length
        x_range_wave = np.array([tct_params[4], (tct_params[6] + (max_peaks - 1) * (tct_params[0] + tct_params[1]))])
        ids_wave = np.where((x_eval + offset_x >= x_range_wave[0]) & (x_eval + offset_x <= x_range_wave[1]))[0]
    else:
        x_range_wave = [x_eval[0], x_eval[-1]]
        ids_wave = np.arange(0, x_eval.size)

    # check if any points of the wavefront in the image space belong to the honey-wave-bezier
    # or if all points in the image space belong to outer single dot regions.
    # Then, if points belong to the wave part, iterate over those & calculate their y-values
    if ids_wave.size > 0:
        # retrieve y values for given x_eval
        for x_id, x in zip(ids_wave, x_eval[ids_wave]):
            # detect position (rising/falling flank or bezier/rotated bezier)
            x_mod_per = np.mod(x, period_length)
            period_no = math.floor(x / period_length)
            if x_mod_per < bezier_length:
                # bezier at peak/maxima
                if lut_entries:
                    # retrieve y-value of x-value closest to the desired x-value
                    y_res[x_id] = bezier_lut[1, np.argmin(np.abs(bezier_lut[0, :] - (x_mod_per + offset_x)))]
                else:
                    # use sympy to solve symbolic expression for x, because s is not linearly mapped to x
                    temp_y = sympy.solve(bezier_implicit.subs({xb: x_mod_per + offset_x}), yb)
                    # might get two solutions because implicit function might be quadratic. If so, the second solution
                    # is the positive one and selected therefore, as we assume to have only positive y-values after
                    # rotating a signal with only positive x- & y-values by 45 degree
                    if len(temp_y) == 2:
                        y_res[x_id] = np.max(temp_y)
                    elif len(temp_y) == 1:
                        y_res[x_id] = temp_y[0]
            elif x_mod_per < (bezier_length + right_lin_length):
                # linear part starting at the end of the maxima/peak bezier curve
                y_res[x_id] = y_vals[2] + (x_mod_per - bezier_length) * tct_params[3]
            elif x_mod_per < (bezier_length * 2 + right_lin_length):
                # inverted bezier (rotated 180 degrees) at minima
                # multiply by -1 to mirror at x-axis
                # invert interval for bezier evaluation to go from max_x to min_x, to mirror at y-axis
                # add offset calculated beforehand, so that first point of inverted bezier matches the last point of right linear part
                if lut_entries:
                    # retrieve y-value of x-value closest to the desired x-value
                    y_res[x_id] = (
                        -1
                        * bezier_lut[
                            1,
                            np.argmin(
                                np.abs(bezier_lut[0, :] - (2 * bezier_length + right_lin_length - x_mod_per + offset_x))
                            ),
                        ]
                        + y_offset_inv_bezier
                    )
                else:
                    # use sympy to solve symbolic expression for x, because s is not linearly mapped to x
                    temp_y = sympy.solve(
                        bezier_implicit.subs({xb: 2 * bezier_length + right_lin_length - x_mod_per + offset_x}), yb
                    )
                    # might get two solutions because implicit function might be quadratic. If so, the second solution
                    # is the positive one and selected therefore, as we assume to have only positive y-values after
                    # rotating a signal with only positive x- & y-values by 45 degree
                    if len(temp_y) == 2:
                        y_res[x_id] = -1 * np.max(temp_y) + y_offset_inv_bezier
                    elif len(temp_y) == 1:
                        y_res[x_id] = -1 * temp_y[0] + y_offset_inv_bezier
            else:
                # linear part ending at the beginning of the maxima/peak bezier curve
                y_res[x_id] = (y_offset_inv_bezier - y_vals[0]) + (
                    x_mod_per - (bezier_length * 2 + right_lin_length)
                ) * tct_params[2]
            # add an offset depending on the period number. This offset comes from the fact that the periodic part
            # does not start & end on the same y-value
            y_res[x_id] -= y_offset_period * period_no

    # handle coordinates belonging to outer / single dot region (if existing)
    if ids_wave.size < x_eval.size:
        # Add linear part left to first peak (using known slope and known start position of first peak)
        ids_left = np.where(x_eval + offset_x <= x_range_wave[0])[0]
        if ids_left.size > 0:
            y_res[ids_left] = tct_params[5] + (x_eval[ids_left] + offset_x - x_range_wave[0]) * tct_params[2]
        # Add linear part right to last peak (using known slope and evaluated end position of last peak)
        ids_right = np.where(x_eval + offset_x >= x_range_wave[1])[0]
        if ids_right.size > 0:
            last_wave_y = y_vals[0] - (max_peaks - 1) * y_offset_period + (y_vals[2] - y_vals[0])
            y_res[ids_right] = last_wave_y + (x_eval[ids_right] + offset_x - x_range_wave[1]) * tct_params[3]

    if single_value:
        y_res = y_res[0]

    return y_res
