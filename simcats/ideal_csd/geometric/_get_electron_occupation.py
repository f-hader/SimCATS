"""This module contains the required functionalities to calculate the occupations numbers for 2D and 1D measurements. For
2D measurements the lead transitions label mask is used to find regions enclosed by total charge transitions (TCTs).
This speeds up the calculation, as it is only required to check for one pixel of the region, which total number of
electrons corresponds to the region. This can't be applied to 1D measurements, so that in this case the occupation
numbers are calculated without the need for a lead transition mask.

@author: f.hader
"""

from functools import partial
from typing import Callable, Dict, Tuple, Optional

# used to find all regions
import diplib as dip
import numpy as np

from simcats.support_functions import multi_cauchy_cdf, multi_sigmoid_cdf, rotate_points, signed_dist_points_line


def _occupation_func(
    points: np.ndarray, ref_line_points: np.ndarray, params: list, cdf_type: str = "sigmoid"
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the occupation for a measurement point based on the distance to the reference line defined by two points.

    Args:
        points (np.ndarray): The coordinates of the points for which the occupation will be calculated, shape = (n, 2).
        ref_line_points (np.ndarray): The coordinates of two points defining the reference line, shape = (2, 2)
            (interdot transition from occupation of dot 1 = 0 and occupation of dot 2 = max_occ to occupation of dot 1
            = 1 and occupation of dot 2 = max_occ - 1).
        params (list): List with parameters for the multi CDF representing the occupation in dot 1. \n
            Example: \n
            [0] = x0_1 \n
            [1] = gamma_1 \n
            [2] = x0_2 \n
            [3] = gamma_2 \n
            ...
        cdf_type (str): Name of the type of CDF to be used. Can be either "cauchy" or "sigmoid (default: "sigmoid").

    Returns:
        Tuple[np.ndarray, np.ndarray]: Occupation of dot 1, Occupation of dot 2
    """
    # calculate the maximal occupation in the system, based on the number of cauchy params
    max_occ = len(params) / 2
    # calculate signed distance from the reference line
    dist = signed_dist_points_line(points=points, line_points=ref_line_points)
    # calculate occupation in dot 1
    if cdf_type == "sigmoid":
        occ_1 = multi_sigmoid_cdf(dist, params)
    elif cdf_type == "cauchy":
        occ_1 = multi_cauchy_cdf(dist, params)
    else:
        raise ValueError(
            f"The cdf_type for the electron occupation calculation must be either 'sigmoid' or 'cauchy'."
            f"Got '{cdf_type}' instead."
        )
    return occ_1, max_occ - occ_1


def _get_occ_func(ref_line_points: np.ndarray, params: list, cdf_type: str = "sigmoid") -> Callable:
    """Generate a partially initialized function for the occupation.

    Args:
        ref_line_points (np.ndarray): The coordinates of two points defining the reference line, shape = (2, 2)
            (interdot transition from occupation of dot 1 = 0 and occupation of dot 2 = max_occ to occupation of dot 1
            = 1 and occupation of dot 2 = max_occ - 1).
        params (list): List with parameters for the multi CDF representing the occupation in dot 1. \n
            Example: \n
            [0] = x0_1 \n
            [1] = gamma_1 \n
            [2] = x0_2 \n
            [3] = gamma_2 \n
            ...
        cdf_type (str): Name of the type of CDF to be used. Can be either "cauchy" or "sigmoid (default: "sigmoid").

    Returns:
        Callable: Partially initialized function for the calculation of occupations in a region with fixed number of
        electrons in the system.
    """
    return partial(_occupation_func, ref_line_points=ref_line_points, params=params, cdf_type=cdf_type)


def get_electron_occupation(
    csd: np.ndarray,
    volt_limits_g1: np.ndarray,
    volt_limits_g2: np.ndarray,
    bezier_coords: Dict[int, np.ndarray],
    tct_functions: Dict[int, Callable],
    rotation: float = -np.pi / 4,
    lut_entries: Optional[int] = None,
    cdf_type: str = "sigmoid",
    cdf_gamma_factor: Optional[float] = None,
) -> np.ndarray:
    """Calculates the electron occupations for a given CSD represented by a numpy array.

    The total charge transitions (TCTs) in the CSD are expected to be labeled by their pixel values (all pixels
    with value 1 belong to TCT 1, etc.).

    Args:
        csd (np.ndarray): The CSD transition line mask (for 2D scans) or only zeros (for 1D scans).
        volt_limits_g1 (np.ndarray): Voltage limits of (plunger) gate 1 (second-/x-axis). \n
            Example: \n
            [min_V1, max_V1]
        volt_limits_g2 (np.ndarray): Voltage limits of (plunger) gate 2 (first-/y-axis). \n
            Example: \n
            [min_V2, max_V2]
        bezier_coords (Dict[int, np.ndarray]): Bezier anchor coordinates for all used total charge transitions (TCTs).
            The keys of the dictionary must specify the id of the TCT (starting with 1 for the TCT between 0 and 1
            electrons in the system). Each item is an array that corresponds to one TCT and has the shape (n, 3, 2). The
            first dimension refers to the id of the bezier curve, the second dimension to the position of the anchor
            point (0 = left, 1 = center, 2 = right), and the third dimension to the axis (0 = x-values, 1 = y-value).
        tct_functions (Dict[int, Callable]): Dictionary containing the functions of the TCTs in the CSD.
            The keys must specify the id of the TCT (starting with 1 for the TCT between 0 and 1 electrons in the
            system) and the items should be partially initialized versions of the function "tct_bezier".
        rotation (float): The rotation that has been applied to the TCT (which is usually represented with the
            tct_params rotated by 45 degrees). Default: is -np.pi/4.
        lut_entries (Optional[int]): Number of samples for the lookup-table for the bezier curves. If this is not
            None, a lookup-table will be used to evaluate the points on the bezier curve. Default is None.
        cdf_type (str): Name of the type of cumulative distribution function (CDF) to be used for interdot transitions.
            Can be either "cauchy" or "sigmoid. Default is "sigmoid".
        cdf_gamma_factor (Optional[float]): The factor used for the calculation of the gamma values of the CDF.
            If set to None, the default values for the selected cdf_type are used (2.2 for sigmoid, 6.15 for cauchy). \n
            Gamma is calculated as follows: \n
            gamma = width_bezier_curve / cdf_gamma_factor. \n
            Default is None.

    Returns:
        np.ndarray: The electron occupations for the CSD. The first two dimensions map to the axis of the CSD, while the
        third dimension indicates the dot of the corresponding occupation value.
    """
    # setup the correct gamma factor
    if cdf_gamma_factor is None:
        if cdf_type == "sigmoid":
            cdf_gamma_factor = 2.2
        elif cdf_type == "cauchy":
            cdf_gamma_factor = 6.15
        else:
            raise ValueError(
                f"The cdf_type for the electron occupation calculation must be either 'sigmoid' or 'cauchy'."
                f"Got '{cdf_type}' instead."
            )

    # ensure that TCT functions are sorted ascending by index/key
    tct_functions = dict(sorted(tct_functions.items()))

    # generate the voltage values for all ids
    row_values = np.linspace(volt_limits_g2[0], volt_limits_g2[1], csd.shape[0])
    if csd.ndim == 2:
        col_values = np.linspace(volt_limits_g1[0], volt_limits_g1[1], csd.shape[1])
    elif csd.ndim == 1:
        col_values = np.linspace(volt_limits_g1[0], volt_limits_g1[1], csd.shape[0])

    # create a copy of the csd to be used as mask
    csd_mask = np.copy(csd)

    # create numpy array to store the occupations
    csd_occupations = np.zeros(csd.shape + (2,))

    # find all regions with fixed electron numbers
    # for 2D, dipImage is used to find regions
    if csd.ndim == 2:
        a_bin = dip.Image()
        a_bin.Copy(csd)
        a_bin.Convert("BIN")
        # label all image regions
        im_labels = dip.Label(~a_bin, 1)
        regs = dip.GetObjectLabels(im_labels)
        im_labels = np.array(im_labels)
        # iterate over all regions to find assign them their region value
        for reg in regs:
            reg_val = 0
            indices = np.where(im_labels == reg)
            temp_point = np.array([col_values[indices[1][0]], row_values[indices[0][0]]])
            temp_point_rot = rotate_points(points=temp_point, angle=-rotation)
            for f_id, func in tct_functions.items():
                if func(x_eval=temp_point_rot[0], lut_entries=lut_entries) <= temp_point_rot[1]:
                    reg_val = f_id
                else:
                    break
            csd_mask[indices] = reg_val
    # It's not possible to find regions enclosed by transitions in 1D.
    # Therefore, every point must be evaluated. It can f.e. happen,
    # that a part of the 1D scan is enclosed by two lines of value n,
    # so that we don't know if we have n or n-1 electrons. This would happen
    # if the sweep is along the TCT, iterating between the total number
    # of n and n-1 electrons.
    elif csd.ndim == 1:
        csd_points = np.vstack((col_values, row_values)).T
        csd_points_rot = rotate_points(points=csd_points, angle=-rotation)
        # iterate TCTs
        for tct_value, tct_func in sorted(tct_functions.items(), reverse=True):
            y_vals_tct = tct_func(x_eval=csd_points_rot[:, 0], lut_entries=lut_entries)

            # check which CSD pixels are part of the region
            # if a pixel is below a transition line it is added to the current region
            region_pixels = np.where(csd_points_rot[:, 1] < y_vals_tct)
            csd_mask[region_pixels] = tct_value - 1

    # iterate over regions of the final mask & calculate occupations
    regs = np.unique(csd_mask)
    for reg in regs:
        # region 0 doesn't need to be calculated, as every pixel in this region keeps the default value (0., 0.)
        if reg == 0:
            continue
        # calculate reference line, given by the interdot transition where the first electron of dot 2
        # moves into the previously empty dot 1
        # the first interdot transition happens between peak 1 (first valley) of tct_bezier and
        # peak 0 (first peak) of tct_bezier-1
        try:
            ref_line = np.array([[bezier_coords[int(reg) + 1][1, 1, :]], [bezier_coords[int(reg)][0, 1, :]]])
        except KeyError as e:
            raise IndexError(
                f"bezier_coords dictionary does not contain an entry for total charge transition (TCT) {e}. "
                f"This TCT is required to calculate the interdot transitions in the regime with a total of {reg} "
                f"electrons in the double dot system, which has been found in the given CSD. Please make sure, that "
                f"the required bezier coordinates have been supplied. This error can occur, if the voltage limits of "
                f"the CSD are not properly set, so that regions above the last defined TCT are included."
            ) from e
        # calculate the cdf parameters for the occupation function, by iterating over peaks from
        # wf-1 (which belong to the interdot transitions of the current region) & calculating their distance
        # to the reference line
        occ_func_params = []
        for i in range(0, bezier_coords[int(reg)].shape[0], 2):
            # calculate the voltage range affected by interdot tunnel coupling
            volt_range_coupling = signed_dist_points_line(
                points=bezier_coords[int(reg)][i, 2], line_points=ref_line
            ) - signed_dist_points_line(points=bezier_coords[int(reg)][i, 0], line_points=ref_line)
            # add the next interdot transition at the distance from the current bezier peak to the first peak (ref_line)
            # with a gamma calculated from the bezier/transition widths and the given cdf_gama_factor
            occ_func_params += [
                signed_dist_points_line(points=bezier_coords[int(reg)][i, 1], line_points=ref_line),
                volt_range_coupling / cdf_gamma_factor,
            ]

        # setup occupation function for current region
        temp_occ_func = _get_occ_func(ref_line_points=ref_line, params=occ_func_params, cdf_type=cdf_type)

        region_ids = np.where(csd_mask == reg)
        index_temp = row_values[region_ids[0]]
        if csd.ndim == 2:
            column_temp = col_values[region_ids[1]]
        elif csd.ndim == 1:
            column_temp = col_values[region_ids[0]]
        points = np.array([column_temp, index_temp]).T
        occ = temp_occ_func(points)

        if csd.ndim == 2:
            csd_occupations[region_ids[0], region_ids[1], 0] = occ[0]
            csd_occupations[region_ids[0], region_ids[1], 1] = occ[1]
        elif csd.ndim == 1:
            csd_occupations[region_ids[0], 0] = occ[0]
            csd_occupations[region_ids[0], 1] = occ[1]

    return csd_occupations
