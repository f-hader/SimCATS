"""
This module contains a function for the calculation of the signed distance between points and a line.
It is used by the get_electron_occupation function in simcats.ideal_csd.geometric.

@author: f.hader
"""

import numpy as np


def signed_dist_points_line(points: np.ndarray, line_points: np.ndarray) -> np.ndarray:
    """
    Calculates the signed distance between points and a line defined by two points.

    Args:
        points (np.ndarray): The coordinates of the points for which the distance will be calculated, shape = (n, 2)
        line_points (np.ndarray): The coordinates of two points defining the line, shape = (2, 2)

    Returns:
        np.ndarray: The signed distances of the points to the line
    """
    return -np.cross(line_points[1, :] - line_points[0, :], points - line_points[0, :]) / np.linalg.norm(
        line_points[1, :] - line_points[0, :]
    )