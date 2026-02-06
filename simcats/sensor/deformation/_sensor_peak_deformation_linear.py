"""This module contains the class for simulating a linear deformation of sensor peaks in a scan.

@author: b.papajewski
"""
import math
from typing import Union

import numpy as np

from simcats.sensor.deformation import SensorPeakDeformationInterface

__all__ = []


class SensorPeakDeformationLinear(SensorPeakDeformationInterface):
    """Linear deformation implementation of the SensorPeakDeformationInterface."""

    def __init__(self, angle: float) -> None:
        """Initialization of an object of the class LinearSensorPeakDeformation.
        This kind of deformation uses a single line for the deformation of a sensor peak (tilting the wavefront).

        Args:
            angle (float): Angle of the line that is used for the linear deformation of the sensor peak. This Angle
                is specified in radians. The angle between the two straight lines is specified, which lies above the
                deformation line and to the right of the middle line.
        """
        super().__init__()

        self.angle = angle

    @property
    def angle(self) -> float:
        """Returns the angle of the line used for deforming the sensor peak.

        Returns:
            float: Angle of the line that is used for the linear deformation of the sensor peak.
        """
        return self.__angle

    @angle.setter
    def angle(self, angle: float):
        self.__angle = angle

    def calc_mu(self, dist: Union[float, np.ndarray], mu0: float) -> Union[float, np.ndarray]:
        """
        Method that calculates the deformed potential value for mu0 of mu0 of a sensor peak that is moved based on the
        distance (`dist`) to a middle line, from which the deformation originates, using a line. The middle line follows
        the direction of the sensor peaks (i.e., the sensor dot potential) and is perpendicular to the wavefronts
        defined by the peaks.

        For the deformation, the potential value of the point of the line that has the given distance to the center line
        is used.

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

        # Calculation of the gradient of the deformation line
        m = 1 / math.tan(self.angle)

        dif = m * dist
        dif *= np.linalg.norm(self.sensor.alpha_sensor_gate[0])
        return mu0 + dif

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(angle={self.angle})"
        )
