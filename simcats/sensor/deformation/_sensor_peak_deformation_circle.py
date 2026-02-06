"""This module contains the class for simulating a circular deformation of sensor peaks in a scan.

@author: b.papajewski
"""
from typing import Union

import numpy as np

from simcats.sensor.deformation import SensorPeakDeformationInterface

__all__ = []


class SensorPeakDeformationCircle(SensorPeakDeformationInterface):
    """Circular deformation implementation of the SensorPeakDeformationInterface."""

    def __init__(self, radius: float) -> None:
        """Initialization of an object of the class CircleSensorPeakDeformation.
        This kind of deformation uses a single circle for the deformation of a sensor peak.

        Args:
            radius (float): Radius of the circle that is used for the deformation. The radius can the positive or
                negative. If the radius is positive the center of the circle is below mu0 and if it is negative the
                center is above mu0.
        """
        super().__init__()

        self.__radius = radius

    @property
    def radius(self) -> float:
        """Returns the radius used for this deformation.
        Radius is specified in potential values.
        The radius can the positive or negative. If the radius is positive the center of the circle is below mu0 and if
        it is negative the center is above mu0.

        Returns:
            float: Radius of the circular deformation. The wavefront is deformed according to this circle.
        """
        return self.__radius

    @radius.setter
    def radius(self, radius: float) -> None:
        """Sets the radius of the circular deformation.
        Radius is specified in potential values.
        The radius can the positive or negative. If the radius is positive the center of the circle is below mu0 and if
        it is negative the center is above mu0.

        Args:
            radius (float): Radius of the circular deformation. The wavefront is deformed according to this circle.
        """
        self.__radius = radius

    def calc_mu(self, dist: Union[float, np.ndarray], mu0: float) -> Union[float, np.ndarray]:
        """
        Method that calculates the deformed potential value for mu0 of a sensor peak that is moved based on the
        distance (dist) to a middle line, from which the deformation originates, using a circle. The middle line follows
        the direction of the sensor peaks (i.e., the sensor dot potential) and is perpendicular to the wavefronts
        defined by the peaks.

        The center of the circle is either above or below the sensor peak and lies on the middle line. However, it is
        chosen so that the mu0 value lies on the edge of the circle. To determine the deformation, one of the points of
        the circle with the given distance to the middle line is used. The deformed mu0 value is the potential value of
        the point whose potential value is closest to the original mu0 value. If there is no point of the circle that
        has the given distance (dist) to the middle line, the potential of the point of the circle with the greatest
        distance to the middle straight line is used.

        Args:
            dist (Union[float, np.ndarray]): Distance from the middle line, from which the deformation originates. If a
                point is above the middle line the distance is positive and if it is below, the distance is negative.
                The parameter can either be a np.ndarray that contains multiple distances or a single distance as a
                float.
            mu0 (float): Potential offset for calculation of the deformed potential. Usually this should be the
                potential offset of the sensor peak, that should be deformed.

        Returns:
             Union[float, np.ndarray]: Deformed potential value mu0 of a sensor peak. The same type as the type of
             `dist` is returned.
        """
        # Here only positive distances are used as the deformation with a single circle is symmetrical
        dist = np.abs(dist)
        # Conversion of the raidus from potential values to voltage values
        radius_volt = self.__radius / np.linalg.norm(self.sensor.alpha_sensor_gate[0])

        radius_abs = np.abs(radius_volt)
        sign_radius = np.sign(self.__radius)

        # Prevents division by zero if radius_abs is zero
        with np.errstate(divide='ignore', invalid='ignore'):
            angle = np.arcsin(np.clip(dist / radius_abs, -1, 1))
            cos_angle = np.cos(angle)

        # Calculation of both possible points of the circle that have the distance dist to the middle line
        temp1 = mu0 - (radius_abs - cos_angle * radius_abs) * sign_radius
        temp2 = mu0 - radius_volt * np.sign(radius_volt)

        # Selection of the result that is closer to the output mu0
        result = np.where(dist <= radius_abs, temp1, temp2)

        # If the result is an array with only one element, you can convert it to a scalar
        if np.isscalar(dist):
            return result.item()
        else:
            return result

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(radius={self.radius})"
        )
