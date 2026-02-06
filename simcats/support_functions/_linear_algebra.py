"""This module provides functionality based on linear algebra for evaluating geometric relationships, such as between
two lines or between a point and a line.

@author: b.papajewski
"""

import numbers
from typing import Union, Sequence, Tuple, List

import numpy as np


def is_point_below_line(point: Tuple[float, float], line: Union[Tuple[float, float], float]) -> bool:
    """Method for evaluating whether a point is below a line.

    Args:
        point (Tuple[float, float]): Point to be evaluated.
        line (Union[Tuple[float, float], float]: Either a straight line (m*x+b) given as a tuple (m,b) or as a single
            value if it is a horizontal line. This then specifies the height b of the line.

    Returns:
        bool: True if the point is below the line, False otherwise.
    """
    x, y = point

    # If the straight line is specified as a single float, it is a horizontal line
    if isinstance(line, int):
        b = line
        return y < b
    # If the straight line is specified as a tuple (m, b)
    elif isinstance(line, tuple) and len(line) == 2:
        m, b = line
        y_line = m * x + b
        return y < y_line
    else:
        raise ValueError(
            "The straight line must be specified either as a tuple (m, b) or as a single integer for a horizontal line.")


def line_line_intersection(line1: Union[Sequence, np.ndarray, numbers.Real],
                           line2: Union[Sequence, np.ndarray, numbers.Real]):
    """Method for calculating the intersection of two lines.
    Both lines can be specified either by a straight line equation (m*x+b) or by a single number if it is a vertical
    line. The single number gives the x coordinate of the vertical line.

    Args:
        line1 (Union[Sequence, np.ndarray, numbers.Real]): A two element sequence when it's a straight line equation
            (m,b) or a single number when line1 is a vertical line. In this case, the number indicates the x coordinate
            of the vertical line.
        line2 (Union[Sequence, np.ndarray, numbers.Real]): A two element sequence when it's a straight line equation
            (m,b) or a single number when line1 is a vertical line. In this case, the number indicates the x coordinate
            of the vertical line.

    Returns:
        The x and y coordinates of the intersection of the two lines. If there is no intersection, an exception
        is thrown.
    """
    for i, line in enumerate((line1, line2), start=1):
        if not (isinstance(line, numbers.Real) or len(line) == 2):
            raise ValueError(f"Invalid parameter - line{i} must either be a single number or an sequence with two elements!")

    # Both lines are straight line equations
    if isinstance(line1, (Sequence, np.ndarray)) and isinstance(line2, (Sequence, np.ndarray)):
        m1, b1 = line1
        m2, b2 = line2

        # Check if lines are parallel
        if m1 == m2:
            if b1 == b2:
                raise ValueError("The lines are the same.")
            else:
                raise ValueError("The lines are parallel and do not intersect.")

        # Compute intersection point
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1

        return np.array([x, y])

    # Line1 is a straight line equation and line2 is a vertical line
    elif isinstance(line1, (Sequence, np.ndarray)):
        m1, b1 = line1
        return np.array([line2, m1 * line2 + b1])

    # Line1 is a vertical line and line2 is a straight line equation
    elif isinstance(line2, (Sequence, np.ndarray)):
        m2, b2 = line2
        return np.array([line1, m2 * line1 + b2])

    # Both lines are vertical lines und therefore also parallel
    else:
        if line1 == line2:
            raise ValueError("The lines are the same.")
        else:
            raise ValueError("The lines are parallel and do not intersect.")


def line_circle_intersection(line: Union[float, Tuple], circle_center: Tuple, radius: float):
    """Method for calculating the intersection of a line and a circle.

    Args:
        line (Union(float,Tuple)): The line with which the intersection points are to be calculated. This line can
            either be given as a tuple with the slope and y-intercept in the form (m,b) or as a float if it is a
            vertical line.
        circle_center (Tuple): The center of the circle. The center is specified as a tuple (x-center, y-center).
        radius (float): The radius of the circle. The radius is specified as float that can be any rational number
            greater than 0.

    Returns:
        List[Tuple]: This method returns a list of the intersection points of the line and a circle. The points are
        returned as a tuple with the form (x-coordinate,y-coordinate). The list of intersection points can contain zero
        to two tuples. Accordingly, an empty list is returned if the line and the circle have no intersection.
    """
    if radius <= 0:
        raise ValueError("The radius of the circle must be greater than 0.")

    center_x, center_y = circle_center

    # Check whether the line is a vertical line
    if isinstance(line, tuple):
        m, b = line

        # Calculation of the coefficients of the quadratic equation system
        A = 1 + m ** 2
        B = 2 * (m * b - m * center_y - center_x)
        C = center_x ** 2 + center_y ** 2 + b ** 2 - 2 * b * center_y - radius ** 2

        # Calculation of the discriminant
        discriminant = B ** 2 - 4 * A * C

        if discriminant < 0:
            # No intersection
            return []
        elif discriminant == 0:
            # Single intersection
            x = -B / (2 * A)
            y = m * x + b
            return [(x, y)]
        else:
            # Two intersections
            sqrt_discriminant = np.sqrt(discriminant)
            x1 = (-B + sqrt_discriminant) / (2 * A)
            y1 = m * x1 + b
            x2 = (-B - sqrt_discriminant) / (2 * A)
            y2 = m * x2 + b
            return [(x1, y1), (x2, y2)]
    else:
        # Vertical line x = c
        c = line

        # Calculation of the y-values
        A = 1
        B = -2 * center_y
        C = center_y ** 2 + (c - center_x) ** 2 - radius ** 2

        # Calculation of the discriminant
        discriminant = B ** 2 - 4 * A * C

        if discriminant < 0:
            # No intersection
            return []
        elif discriminant == 0:
            # Single intersection
            y = -B / (2 * A)
            return [(c, y)]
        else:
            # Two intersections
            sqrt_discriminant = np.sqrt(discriminant)
            y1 = (-B + sqrt_discriminant) / (2 * A)
            y2 = (-B - sqrt_discriminant) / (2 * A)
            return [(c, y1), (c, y2)]
