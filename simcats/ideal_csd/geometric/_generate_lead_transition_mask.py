"""This module contains the required functionalities to generate the lead transitions label mask for 2D and 1D
measurements.

@author: f.hader
"""

from typing import Callable, Dict, Optional

import numpy as np

from simcats.support_functions import rotate_points


def generate_lead_transition_mask_1d(csd_occ: np.ndarray) -> np.ndarray:
    """Generates the ground truth label mask for the lead transitions of 1D scans.

    The labels are numbers indicating which total charge number is separated by the corresponding
    total charge transition (TCT). Label with value n is the TCT (consisting of a series of connected lead transitions)
    between n-1 and n electrons in the system. The pixels with the labels are always part of the region
    with the higher number of electrons.

    Args:
        csd_occ (np.ndarray): The 1D-CSD occupation data.

    Returns:
        np.ndarray: The label mask for the lead transitions (aka total charge transitions).

    """
    # setup variable for the result
    csd_total_occ = np.sum(csd_occ, axis=1)
    csd_result = np.zeros(csd_total_occ.shape, dtype=np.uint8)

    # collect all ids where the total occupation of the dot system changes
    ids_total_occ_changed = np.where(csd_total_occ[:-1] != csd_total_occ[1:])[0]

    for _id in ids_total_occ_changed:
        if csd_total_occ[_id] < csd_total_occ[_id + 1]:
            csd_result[_id + 1] = csd_total_occ[_id + 1]
        else:
            csd_result[_id] = csd_total_occ[_id]

    return csd_result


def generate_lead_transition_mask_2d(
    csd: np.ndarray,
    volt_limits_g1: np.ndarray,
    volt_limits_g2: np.ndarray,
    tct_functions: Dict[int, Callable],
    rotation: float = -np.pi / 4,
    lut_entries: Optional[int] = None,
) -> np.ndarray:
    """Generates the ground truth label mask for the lead transitions of 2D scans.

    The labels are numbers indicating which total charge number is separated by the corresponding
    total charge transition (TCT). Label with value n is the TCT (consisting of a series of connected lead transitions)
    between n-1 and n electrons in the system. The pixels with the labels are always part of the region
    with the higher number of electrons. A TCT is described using a bezier-curve for the round triple-points and
    linear parts in between two bezier-curves.

    Args:
        csd (np.ndarray): Numpy array of the same shape as the corresponding 2D-CSD data.
        volt_limits_g1 (np.ndarray): Voltage limits of (plunger) gate 1 (second-/x-axis). \n
            Example: \n
            [min_V1, max_V1]
        volt_limits_g2 (np.ndarray): Voltage limits of (plunger) gate 2 (first-/y-axis). \n
            Example: \n
            [min_V2, max_V2]
        tct_functions (Dict[int, Callable]): Dictionary containing the functions of the TCTs in the CSD. The keys must
            specify the id of the TCT (starting with 1 for the TCT between 0 and 1 electrons in the system) and the
            items should be partially initialized versions of the function "tct_bezier".
        rotation (float): The rotation to be applied to the TCT(s) (which is/are usually represented rotated by 45
            degrees). Default is -np.pi/4.
        lut_entries (Optional[int]): Number of samples for the lookup-table. If this is not None, a lookup-table will be used to
            evaluate the points on the bezier curve. Default is None.

    Returns:
        np.ndarray: The label mask for the lead transitions (in our case: total charge transitions).

    """
    # setup variable for the result
    csd_result = np.copy(csd).astype(np.uint8)

    # resolution x/y
    x_res = csd.shape[1]
    y_res = csd.shape[0]
    # limits x/y
    x_lims = volt_limits_g1
    y_lims = volt_limits_g2
    # stepsize x/y
    x_step = (x_lims[-1] - x_lims[0]) / (x_res - 1)
    y_step = (y_lims[-1] - y_lims[0]) / (y_res - 1)

    # The required sampling range is calculated here.
    # For a 2D CSD , the sampling range is calculated based on the maximum
    # distance of the rotated corners of the CSD
    # rotate all corner points to calculate the maximal required x-range
    corner_points = np.array(
        [[x_lims[0], y_lims[0]], [x_lims[0], y_lims[1]], [x_lims[1], y_lims[0]], [x_lims[1], y_lims[1]]]
    )
    x_c_rot = rotate_points(points=corner_points, angle=-rotation)[:, 0]
    # generate enough x-values to cover the complete range of the CSD with a
    # higher resolution than required to have a precise result after discretization
    tct_points = np.empty(((x_res + y_res) * 4, 2))
    tct_points[:, 0] = np.linspace(np.min(x_c_rot) - x_step, np.max(x_c_rot) + x_step, (x_res + y_res) * 4)

    # Insert the transition lines into the CSD
    # If the CSD is 2D, the required TCT points are sampled and discretized.
    for tct_value, tct_func in tct_functions.items():
        # generate the y-values for all generated x-values
        tct_points[:, 1] = tct_func(x_eval=tct_points[:, 0], lut_entries=lut_entries)

        # rotate the TCT into the original orientation
        wf_points_rot = rotate_points(points=tct_points, angle=rotation)

        # select only TCT pixels that are in the csd-limits
        valid_ids = np.where(
            (wf_points_rot[:, 0] > (x_lims[0] - 0.5 * x_step))
            & (wf_points_rot[:, 0] < (x_lims[1] + 0.5 * x_step))
            & (wf_points_rot[:, 1] > (y_lims[0] - 0.5 * y_step))
            & (wf_points_rot[:, 1] < (y_lims[1] + 0.5 * y_step))
        )
        # x_h_rot = x_h_rot[valid_ids]
        # y_h_rot = y_h_rot[valid_ids]
        wf_points_rot = wf_points_rot[valid_ids[0], :]

        # insert TCT pixels into the csd
        # calculation of the ids for the values:
        # x = min(csd_x) + id * x_step
        # add half step size, so that the pixel id of the nearest pixel is obtained after the division
        # (round up if next higher value in range of 0.5 * step_size)
        x_id = np.floor_divide(wf_points_rot[:, 0] + 0.5 * x_step - x_lims[0], x_step).astype(int)
        y_id = np.floor_divide(wf_points_rot[:, 1] + 0.5 * y_step - y_lims[0], y_step).astype(int)
        csd_result[y_id, x_id] = tct_value

    return csd_result
