"""This module contains functions for simulating the sensor response for a charge stability diagram (CSD) including the
cross coupling between the sensor and double dot (plunger) gates.

@author: f.hader
"""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

__all__ = []


class SensorPeakInterface(ABC):
    """Interface for all sensor peak functions, which are used to define the Coulomb Peaks of sensor dots for the simulation of CSDs.

    Implementations of the SensorInterface can consist of multiple peaks.
    """

    @abstractmethod
    def sensor_function(self, mu_sens: np.ndarray) -> np.ndarray:
        """This method has to be implemented in every object which should be used as sensor peak function during the simulation of CSDs.

        The electrochemical potential at the sensor should be given as one- or two-dimensional numpy
        array to this function. The parameters for this sensor peak function should be attributes of this class. This
        function should return the sensor (peak) values which result from the given electrochemical potential.

        Args:
            mu_sens (np.ndarray): The sensor potential, stored in a numpy array with the axis mapping to the CSD axis.

        Returns:
            np.ndarray: The sensor response (for the corresponding peak), calculated from the given potential. It is
            stored in a numpy array with the axis mapping to the CSD axis.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.__class__.__name__ + str(self.__dict__)


class SensorInterface(ABC):
    """Interface for all sensor functions, which are used in the simulation of CSDs."""

    @abstractmethod
    def __init__(
        self,
        sensor_peak_function: Union[SensorPeakInterface, List[SensorPeakInterface]],
        alpha_dot: np.ndarray,
        alpha_gate: np.ndarray,
        offset_mu_sens: float,
    ) -> None:
        """Initializes an object of the class for the simulation of the sensor response for a CSD.

        Args:
            sensor_peak_function (SensorPeakInterface): An implementation of the SensorPeakInterface. It is also
                possible to supply a list of such peaks, if a sensor with multiple peaks should be simulated.
            alpha_dot (np.ndarray): Lever-arm to the dots. The values should be negative. Influences the strengths of
                the edges in the CSD.
            alpha_gate (np.ndarray): Lever-arm to the double dot (plunger) gates. The values should be positive.
            offset_mu_sens (float): Electrochemical potential of the sensor dot for zero electrons in the dots and no
                applied voltage at the gates.
        """
        raise NotImplementedError

    @abstractmethod
    def sensor_response(self, mu_sens: np.ndarray) -> np.ndarray:
        """This method has to be implemented in every object which should be used as sensor function during the simulation of CSDs.

        The electrochemical potential at the sensor should be given as one- or two-dimensional numpy array to this
        function. The parameters for this sensor function should be attributes of this class. This function should
        return the sensor values which result from the given electrochemical potential.

        Args:
            mu_sens (np.ndarray): The sensor potential, stored in a 2-dimensional numpy array with the axis mapping to
                the CSD axis.

        Returns:
            np.ndarray: The sensor response, calculated from the given potential. It is stored in a numpy array with the
            axis mapping to the CSD axis.
        """
        raise NotImplementedError

    @abstractmethod
    def sensor_potential(
        self, occupations: np.ndarray, volt_limits_g1: np.ndarray, volt_limits_g2: np.ndarray
    ) -> np.ndarray:
        """Simulates the electrochemical potential at the sensing dot.

        This is done in dependency on the electron occupation in the double dot and the voltages applied at the double
        dot (plunger) gates.

        Args:
            occupations (np.ndarray): Occupation in left and right dot per applied voltage combination. The occupation
                numbers are stored in a 3-dimensional numpy array. The first two dimensions map to the axis of the CSD,
                while the third dimension indicates the dot of the corresponding occupation value.
            volt_limits_g1 (np.ndarray): Contains the beginning and ending of the swept range for the first (plunger)
                gate.
            volt_limits_g2 (np.ndarray): Contains the beginning and ending of the swept range for the second (plunger)
                gate.

        Returns:
            np.ndarray: The electrochemical potential at the sensing dot.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + str(self.__dict__)
