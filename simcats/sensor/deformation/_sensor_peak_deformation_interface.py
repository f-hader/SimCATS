"""This module contains the interface for simulating deformations of sensor peaks in scans.

@author: b.papajewski
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

__all__ = []

from simcats.sensor import SensorInterface


class SensorPeakDeformationInterface(ABC):
    """General interface for sensor peak deformations.

    Classes that implement this abstract class realize different types of deformation of a single wavefront.
    """

    @abstractmethod
    def calc_mu(self, dist: Union[float, np.ndarray], mu0: float) -> float:
        """
        Method that calculates the deformed potential value of mu0 of a sensor peak that is moved based on
        the distance (`dist`) to a middle line from which the deformation originates. The middle line follows
        the direction of the sensor peaks (i.e., the sensor dot potential) and is perpendicular to the wavefronts
        defined by the peaks.

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
        raise NotImplementedError

    @property
    def sensor(self) -> SensorInterface:
        """
        Returns the sensor object to which the deformed sensor peak belongs.

        Returns:
            SensorInterface: Sensor object of the deformed sensor peak.
        """
        if self.__sensor:
            return self.__sensor
        else:
            return None

    @sensor.setter
    def sensor(self, sensor: SensorInterface) -> None:
        """
        Sets the sensor object to which the deformed sensor peak belongs.

        Args:
            sensor (SensorInterface): Sensor object of the deformed sensor peak.
        """
        self.__sensor = sensor
