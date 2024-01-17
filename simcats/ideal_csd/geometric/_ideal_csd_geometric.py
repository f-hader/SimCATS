"""This module contains the required functionalities to generate ideal CSD data using supplied total charge
transitions (TCTs), voltage ranges and a desired resolution. It also implements the IdealCSDInterface.

@author: f.hader
"""

from typing import List, Tuple, Union

import numpy as np

from simcats.ideal_csd.geometric import (
    calculate_all_bezier_anchors,
    generate_lead_transition_mask_1d,
    generate_lead_transition_mask_2d,
)
from simcats.ideal_csd.geometric import get_electron_occupation as get_occ
from simcats.ideal_csd.geometric import initialize_tct_functions

__all__ = []


def ideal_csd_geometric(
    tct_params: List[np.ndarray],
    volt_limits_g1: np.ndarray,
    volt_limits_g2: np.ndarray,
    resolution: Union[int, np.ndarray] = np.array([100, 100]),
    rotation: float = -np.pi / 4,
    lut_entries: Union[int, None] = None,
    cdf_type: str = "sigmoid",
    cdf_gamma_factor: Union[float, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an ideal Charge Stability Diagram (CSD) based on the supplied parameters.

    The CSD is stored in a numpy array. The two axis refer to gate voltages, with axis 0 (y-axis) representing the
    voltage on gate 2 and axis 1 (x-axis) representing the voltage of gate 1. The total charge transitions (TCTs)
    (series of connected lead-to-dot transitions) separating regions with different numbers of electrons in the system
    are represented by linear parts and bezier curves.

    Args:
        tct_params (List[np.ndarray]): List containing a numpy array with parameters for every TCT in the CSD.
            Each array contains all required parameters to describe the TCT form. \n
            The parameters for a TCT are: \n
            [0] = length left (in x-/voltage-space, not number of points) \n
            [1] = length right (in x-/voltage-space, not number of points) \n
            [2] = gradient left (in voltages) \n
            [3] = gradient right (in voltages) \n
            [4] = start position x (bezier curve leftmost point) (in x-/voltage-space, not number of points) \n
            [5] = start position y (bezier curve leftmost point) (in x-/voltage-space, not number of points) \n
            [6] = end position x (bezier curve rightmost point) (in x-/voltage-space, not number of points) \n
            [7] = end position y (bezier curve rightmost point) (in x-/voltage-space, not number of points)
        volt_limits_g1 (np.ndarray): Voltage sweep range of (plunger) gate 1 (second-/x-axis). \n
            Example: \n
            [min_V1, max_V1]
        volt_limits_g2 (np.ndarray): Voltage sweep range of (plunger) gate 2 (first-/y-axis). \n
            Example: \n
            [min_V2, max_V2]
        resolution (np.ndarray): Desired resolution (in pixels) for the gates. If only one value is supplied, a 1D sweep
            is performed. Then, both gates are swept simultaneously. Default is np.array([100, 100]). \n
            Example: \n
            [res_g1, res_g2]
        rotation (float): Float value defining the rotation to be applied to the TCT (which is usually represented
            with the tct_params rotated by 45 degrees). Default is -np.pi/4.
        lut_entries (Union[int, None]): Number of samples for the lookup-table for bezier curves. If this is not None, a
            lookup-table will be used to evaluate the points on the bezier curves, else they are solved explicitly.
            Using a lookup-table speeds up the calculation at the possible cost of accuracy. Default is None.
        cdf_type (str): Name of the type of cumulative distribution function (CDF) to be used. Can be either
            "cauchy" or "sigmoid". Default is "sigmoid".
        cdf_gamma_factor (Union[float, None]): The factor used for the calculation of the gamma values of the CDF.
            If set to None (=default) the default values for the selected cdf_type are used. \n
            Gamma is calculated as follows: \n
            gamma = width_bezier_curve / cdf_gamma_factor \n
            Default is None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Occupation numbers and lead transition mask (in our case: total charge
        transitions). The occupation numbers are stored in a 3-dimensional numpy array. The first two dimensions map to
        the axis of the CSD, while the third dimension indicates the dot of the corresponding occupation value. The
        label mask for the lead-to-dot transitions is stored in a 2-dimensional numpy array with the axis mapping to
        the CSD axis.
    """
    # check if just one value is supplied for the resolution
    if isinstance(resolution, int):
        resolution = np.array([resolution])

    # Check if one gate is kept at a fixed voltage if a 2D scan is requested. This is only possible in 1D scans.
    if len(resolution) == 2 and (volt_limits_g1[0] == volt_limits_g1[1] or volt_limits_g2[0] == volt_limits_g2[1]):
        raise ValueError(
            "At least one of the voltage ranges 'volt_limits_g1' and 'volt_limits_g2' defines a fixed voltage. "
            "This is only supported for 1D sweeps (only one resolution), but two resolutions were specified."
        )

    # check if the gate voltage ranges have the correct number of entries
    if not (len(volt_limits_g1) == 2 and len(volt_limits_g2) == 2):
        raise ValueError(
            "The volt limits for the gates g1 and g2 must consist of exactly 2 values each. At least "
            "one volt range violates this."
        )

    # check if the resolution has at most two entries
    if len(resolution) > 2:
        raise ValueError(
            f"The specified resolution ({resolution}) has more than two entries. The resolution must either be a "
            f"single value (for 1D scans) or contain two entries (for 2D scans)."
        )

    # Check if one gate is kept at a fixed voltage if a 2D scan is requested. This is only possible in 1D scans.
    if (
        not type(resolution) == int
        and len(resolution) == 2
        and (volt_limits_g1[0] == volt_limits_g1[1] or volt_limits_g2[0] == volt_limits_g2[1])
    ):
        raise ValueError(
            "At least one of the voltage ranges 'volt_limits_g1' and 'volt_limits_g2' defines a fixed voltage. "
            "This is only supported for 1D sweeps (only one resolution), but two resolutions were specified."
        )

    # Check if one resolution is 1 (or smaller) for a 2D scan. A resolution of 1 indicates that the corresponding
    # gate is kept at a fixed voltage. This only makes sense for a 1D scan
    if not type(resolution) == int and len(resolution) == 2 and (resolution[0] <= 1 or resolution[1] <= 1):
        raise ValueError(
            f"The specified resolution ({resolution}) indicates that a 2D scan should be performed, but at least "
            f"one of the two entries is smaller or equal to 1. A resolution of 1 means that the corresponding gate "
            f"is not swept. Thus, it describes a 1D scan of a single gate. Please specify just one resolution and "
            f"a fixed voltage for the corresponding gate, to perform a single gate sweep."
        )

    # Initialize an empty numpy array which will be used to store the lead-dot transition line mask
    csd_transitions = np.zeros(resolution[::-1])

    # retrieve partially initialized wavefront functions (usually defined in a
    # voltage space that is rotated by 45 degrees)
    wf_functions = initialize_tct_functions(tct_params=tct_params, max_peaks=1)

    # retrieve all bezier anchors of the wavefronts (required for the occupation calculation)
    bezier_coords = calculate_all_bezier_anchors(tct_params=tct_params, max_peaks=1, rotation=rotation)

    if len(resolution) == 1:
        # calculate the occupations
        csd_occupations = get_occ(
            csd=csd_transitions,
            volt_limits_g1=volt_limits_g1,
            volt_limits_g2=volt_limits_g2,
            bezier_coords=bezier_coords,
            tct_functions=wf_functions,
            rotation=rotation,
            lut_entries=lut_entries,
            cdf_type=cdf_type,
            cdf_gamma_factor=cdf_gamma_factor,
        )
        csd_transitions = generate_lead_transition_mask_1d(csd_occ=csd_occupations)
    elif len(resolution) == 2:
        # our 2d simulation expects the voltages to be sorted from min to max. If this is not the case, the voltages are
        # sorted here and afterward the image is rearranged
        # add the lead-dot transitions defined by the provided parameters
        # for the 2D scans the transition line mask is used to calculate the occupations
        # in a more efficient way
        csd_transitions = generate_lead_transition_mask_2d(
            csd=csd_transitions,
            volt_limits_g1=np.sort(volt_limits_g1),
            volt_limits_g2=np.sort(volt_limits_g2),
            tct_functions=wf_functions,
            rotation=rotation,
            lut_entries=lut_entries,
        )

        # calculate the occupations
        csd_occupations = get_occ(
            csd=csd_transitions,
            volt_limits_g1=np.sort(volt_limits_g1),
            volt_limits_g2=np.sort(volt_limits_g2),
            bezier_coords=bezier_coords,
            tct_functions=wf_functions,
            rotation=rotation,
            lut_entries=lut_entries,
            cdf_type=cdf_type,
            cdf_gamma_factor=cdf_gamma_factor,
        )

        # rearrange the CSD data, if the supplied voltages were not sorted ascending
        if volt_limits_g2[0] > volt_limits_g2[1]:
            csd_transitions = csd_transitions[::-1, :]
            csd_occupations = csd_occupations[::-1, :, :]
        if volt_limits_g1[0] > volt_limits_g1[1]:
            csd_transitions = csd_transitions[:, ::-1]
            csd_occupations = csd_occupations[:, ::-1, :]

    return csd_occupations, csd_transitions
