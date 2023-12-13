"""
This module contains a function for rotating points by a given angle.

@author: f.hader
"""

import numpy as np


def rotate_points(points: np.ndarray, angle: float = np.pi / 4) -> np.ndarray:
    """
    Rotates a point (or multiple points) by the given angle.

    Args:
        points (np.ndarray): 2D-Numpy array with the coordinates of the points, shape = (n, 2), first column =
            x-coordinate, second column = y-coordinates.
        angle (float): The angle for the rotation. Default is np.pi/4.

    Returns:
        np.ndarray: Numpy array with rotated coordinates, shape = (n, 2).
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    return rotation_matrix.dot(points.T).T
