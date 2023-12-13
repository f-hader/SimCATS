"""This module contains functions and classes for Lorentzian sensor peak functions.

@author: f.hader
"""

import numpy as np

from simcats.sensor import SensorPeakInterface

__all__ = []


class SensorPeakLorentzian(SensorPeakInterface):
    """Lorentzian sensor peak function implementation of the SensorPeakInterface."""

    def __init__(self, mu0: float = 0, gamma: float = 1, height: float = 1, offset: float = 0):
        """Creates a new Lorentzian sensor peak object.

        Args:
            mu0 (float): Position of the maximum of the Lorentz curve. Default is 0.
            gamma (float): The gamma of the Lorentzian. Influences the width of the Lorentz curve. Default is 1.
            height (float): Scaling factor for the height of the Lorentz curve. Default is 1.
            offset (float): Lowest (y-)value of the Lorentz curve. Default is 0.
        """
        self.mu0 = mu0
        self.gamma = gamma
        self.height = height
        self.offset = offset

    @property
    def mu0(self) -> float:
        """Position of the maximum of the Lorentz curve."""
        return self.__mu0

    @mu0.setter
    def mu0(self, mu0: float):
        self.__mu0 = mu0

    @property
    def gamma(self) -> float:
        """The gamma of the Lorentzian. Influences the width of the Lorentz curve."""
        return self.__gamma

    @gamma.setter
    def gamma(self, gamma: float):
        if gamma <= 0:
            raise ValueError(f"gamma should be greater than 0, was {gamma}")
        self.__gamma = gamma

    @property
    def height(self) -> float:
        """Scaling factor for the height of the Lorentz curve."""
        return self.__height

    @height.setter
    def height(self, height):
        self.__height = height

    @property
    def offset(self) -> float:
        """Lowest (y-)value of the Lorentz curve."""
        return self.__offset

    @offset.setter
    def offset(self, offset: float):
        self.__offset = offset

    def sensor_function(self, mu_sens: np.ndarray):
        """Returns the sensor peak function values at the given electrochemical potentials.

        Args:
            mu_sens (np.ndarray): The sensor potential, stored in a numpy array with the axis mapping to the CSD axis.

        Returns:
            np.ndarray: The sensor response (for the corresponding peak), calculated from the given potential. It is
            stored in a numpy array with the axis mapping to the CSD axis.
        """
        return sensor_response_lorentz(
            mu_sens=mu_sens, mu0=self.mu0, gamma=self.gamma, height=self.height, offset=self.offset
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(mu0={self.mu0}, gamma={self.gamma}, height={self.height}, offset={self.offset})"
        )


def sensor_response_lorentz(
    mu_sens: np.ndarray, mu0: float = 0, gamma: float = 1, height: float = 1, offset: float = 0
) -> np.ndarray:
    """Calculates the sensor response out of the given sensing dot potential by simulating the sensor behavior with a Lorentz function.

    Args:
        mu_sens (np.ndarray): The sensor potential, stored in a numpy array with the axis mapping to the CSD axis.
        mu0 (float): Position of the maximum of the Lorentz curve. Default is 0.
        gamma (float): The gamma of the Lorentzian. Influences the width of the Lorentz curve. Default is 1.
        height (float): Scaling factor for the height of the Lorentz curve. Default is 1.
        offset (float): Lowest (y-)value of the Lorentz curve. Default is 0.

    Returns:
        np.ndarray: The sensor response (for the corresponding peak), calculated from the given potential. It is
        stored in a numpy array with the axis mapping to the CSD axis.
    """
    return offset + height * gamma**2 / (gamma**2 + (mu_sens - mu0) ** 2)
