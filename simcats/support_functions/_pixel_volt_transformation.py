"""This module contains functions for transforming pixel coordinates into voltages

@author: b.papajewski
"""

import numpy as np


def pixel_to_volt_1d(pixel: int, pixel_num: int, volt_limits: np.ndarray) -> np.ndarray:
    """Method that maps a pixel index to a voltage value within specified voltage limits.

    This function linearly maps a pixel position within a span of `pixel_num` pixels to a corresponding voltage value.
    The voltage values of the pixel span change uniformly between the two values in `volt_limits`.

    Args:
        pixel (int): The pixel index starting at 0.
        pixel_num (int): Total number of pixels in the span.
        volt_limits (np.ndarray): A 1D array of shape (2,) containing the start and end voltage values. The start value
            is the first of the two values and the end value is the last one.

    Returns:
        np.ndarray: The voltage value corresponding to the given pixel index.
    """
    return volt_limits[0] + (volt_limits[1] - volt_limits[0]) * (pixel / pixel_num)
