"""This module contains the required functionalities to partially initialize total charge transition (TCT) functions.

These functions are used to generate the boundaries/structures for the geometrically simulated charge stability diagrams
(CSDs).

@author: f.hader
"""

from functools import partial
from typing import Callable, Dict, List, Union

import numpy as np

from simcats.ideal_csd.geometric import tct_bezier


def initialize_tct_functions(
    tct_params: Union[np.ndarray, List[np.ndarray]], max_peaks: int = 1
) -> Dict[int, Callable]:
    """Initializes total charge transition (TCT) functions.

    It is also possible to directly initialize multiple functions. The total charge transitions (TCTs)
    (series of connected lead-to-dot transitions), separating regions with different numbers of electrons in the system,
    are represented by linear parts and bezier curves for the round triple-points.

    Args:
        tct_params (Union[np.ndarray, List[np.ndarray]]): Numpy array containing all required parameters to describe the
            TCT form, or a list of such arrays. \n
            The parameters for a TCT are: \n
            [0] = length left (in x-/voltage-space, not number of points) \n
            [1] = length right (in x-/voltage-space, not number of points) \n
            [2] = gradient left (in voltages) \n
            [3] = gradient right (in voltages) \n
            [4] = start position x (bezier curve leftmost point) (in x-/voltage-space, not number of points) \n
            [5] = start position y (bezier curve leftmost point) (in x-/voltage-space, not number of points) \n
            [6] = end position x (bezier curve rightmost point) (in x-/voltage-space, not number of points) \n
            [7] = end position y (bezier curve rightmost point) (in x-/voltage-space, not number of points)
        max_peaks (int): Limit for the number of peaks of the TCT. If multiple TCTs are supplied, max_peaks is increased
            by 1 for every further wavefront. Default is 1.

    Returns:
        Dict[int, Callable]: Partially initialized function(s) of the TCT(s) (tct_bezier). The TCT(s) is/are not rotated
        into the actual voltage space. The keys of the dictionary refer to the corresponding TCT ID.
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

    # collect wavefront functions
    tct_functions = {}
    for i, params in enumerate(tct_params):
        # setup the partially initialized function of the wavefront
        tct_functions[max_peaks + i] = partial(tct_bezier, tct_params=params, max_peaks=max_peaks + i)

    return tct_functions
