"""This module contains the required functionalities to calculate all bezier anchors for given total charge transitions
(TCTs).

@author: f.hader
"""

from typing import Dict, List, Union

import numpy as np

from simcats.support_functions import rotate_points


def calculate_all_bezier_anchors(
    tct_params: Union[np.ndarray, List[np.ndarray]], max_peaks: int = 1, rotation: float = -np.pi / 4
) -> Dict[int, np.ndarray]:
    """Function used to calculate all bezier-curve anchors for a given total-charge-transition (TCT) or for multiple TCTs.

    Args:
        tct_params (Union[np.ndarray, List[np.ndarray]]): Array with required parameters to describe the TCT form, or a
            list of such arrays. \n
            The parameters for a TCT are: \n
            [0] = length left (in x-/voltage-space, not number of points) \n
            [1] = length right (in x-/voltage-space, not number of points) \n
            [2] = gradient left (in voltages) \n
            [3] = gradient right (in voltages) \n
            [4] = start position x (bezier curve leftmost point) (in x-/voltage-space, not number of points) \n
            [5] = start position y (bezier curve leftmost point) (in x-/voltage-space, not number of points) \n
            [6] = end position x (bezier curve rightmost point) (in x-/voltage-space, not number of points) \n
            [7] = end position y (bezier curve rightmost point) (in x-/voltage-space, not number of points)
        max_peaks (int): Limit for the number of peaks. If multiple TCTs are supplied, then max_peaks is increased for
            every next TCT. Default is 1.
        rotation (float): The rotation to be applied to the TCT(s) (which is/are usually represented rotated by 45
            degrees). Default is -np.pi/4

    Returns:
        Dict[int, np.ndarray]: Dictionary with keys representing the maximum number of peaks (TCT ID) and values
        containing a numpy array with the coordinates of the corresponding bezier-curve anchors for each TCT. The
        dimensions of the array are mapped as follows: \n
        - first dimension = peak / valley identifier, \n
        - second dimension =anchor position (0 = left, 1 = center, 2 = right), \n
        - third dimension = coordinate axis identifier (0 = x-value, 1 = y-value).\n
    """
    # if just one TCT is supplied (as array with only one dimension or as list), temporarily put it into a list, so that
    # the iteration works
    if not (
        (
            isinstance(tct_params, list)
            and (
                all(isinstance(item, list) for item in tct_params)
                or all(isinstance(item, np.ndarray) and item.ndim == 1 for item in tct_params)
            )
        )
        or (isinstance(tct_params, np.ndarray) and tct_params.ndim == 2)
    ):
        tct_params = [tct_params]

    # create a dict to store the exact coordinates of all bezier anchors, which are required to calculate
    # the occupation boundaries
    bezier_coords = {}

    for wf_id, params in enumerate(tct_params):
        # calculate max number of iterations for adding peaks/valleys
        max_iter = (max_peaks + wf_id) * 2 - 1
        # collect coordinates of bezier anchors
        # retrieve coordinates of first left anchor
        left_anchor_x = params[4]
        left_anchor_y = params[5]
        # calculate first center anchor coordinates by finding the intersection
        # between left & right linear part
        center_anchor_x = (params[5] - params[7] + params[6] * params[3] - params[4] * params[2]) / (
            params[3] - params[2]
        )
        center_anchor_y = params[5] + params[2] * (center_anchor_x - params[4])
        # retrieve coordinates of first right anchor
        right_anchor_x = params[6]
        right_anchor_y = params[7]
        # calculate x_distance of left & right anchor to center anchor
        x_dist_left = center_anchor_x - left_anchor_x
        x_dist_right = right_anchor_x - center_anchor_x

        # setup variable to store all bezier curve anchors (will be used to identify
        # region with round triple points / tunnel coupling)
        temp_bezier_coords = np.empty((max_iter, 3, 2))

        # iterate and collect bezier anchor coordinates
        for i in range(max_iter):
            # add the coordinates to the array
            temp_bezier_coords[i, 0, 0] = left_anchor_x
            temp_bezier_coords[i, 0, 1] = left_anchor_y
            temp_bezier_coords[i, 1, 0] = center_anchor_x
            temp_bezier_coords[i, 1, 1] = center_anchor_y
            temp_bezier_coords[i, 2, 0] = right_anchor_x
            temp_bezier_coords[i, 2, 1] = right_anchor_y
            # increase coordinates alternating with left/right length, as every second
            # bezier curve is rotated by 180 degrees (valleys/peaks)
            if i % 2:
                center_anchor_x += params[0]
                center_anchor_y += params[0] * params[2]
                left_anchor_x = center_anchor_x - x_dist_left
                left_anchor_y = center_anchor_y - (x_dist_left * params[2])
                right_anchor_x = center_anchor_x + x_dist_right
                right_anchor_y = center_anchor_y + (x_dist_right * params[3])
            else:
                center_anchor_x += params[1]
                center_anchor_y += params[1] * params[3]
                # every second bezier curve is rotated by 180 degrees, resulting in
                # the left anchor having the distance of the right anchor to the center
                # and vice versa, as the distance is defined for the curves which are
                # not rotated
                left_anchor_x = center_anchor_x - x_dist_right
                left_anchor_y = center_anchor_y - (x_dist_right * params[3])
                right_anchor_x = center_anchor_x + x_dist_left
                right_anchor_y = center_anchor_y + (x_dist_left * params[2])

        # rotate the peak coordinates into the original orientation
        bezier_coords_rot = np.empty(temp_bezier_coords.shape)
        for i in range(bezier_coords_rot.shape[1]):
            bezier_coords_rot[:, i, :] = rotate_points(points=temp_bezier_coords[:, i, :], angle=rotation)

        # add the rotated anchors of the current TCT to the dictionary
        bezier_coords[wf_id + max_peaks] = bezier_coords_rot

    return bezier_coords
