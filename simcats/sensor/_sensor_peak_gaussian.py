"""This module contains functions and classes for Gaussian sensor peak functions.

@author: s.fleitmann
"""

import numpy as np

from simcats.sensor import SensorPeakInterface

__all__ = []


class SensorPeakGaussian(SensorPeakInterface):
    """This class contains the necessary parameters and methods for Gaussian sensor peak functions."""

    def __init__(self, mu0: float = 0, sigma: float = 1, height: float = 1, offset: float = 0):
        """Creates a new Gaussian sensor peak object.

        Args:
            mu0 (float): Position of the maximum of the Gaussian bell curve. Default is 0.
            sigma (float): Influences the width of the Gaussian bell curve. Should always be greater than 0.
                Default is 1.
            height (float): Scaling factor for the height of the Gaussian bell curve. Default is 1.
            offset (float): Lowest (y-)value of the Gaussian bell curve. Default is 0.
        """
        self.mu0 = mu0
        self.sigma = sigma
        self.height = height
        self.offset = offset

    @property
    def mu0(self) -> float:
        """Position of the maximum of the Gaussian bell curve."""
        return self.__mu0

    @mu0.setter
    def mu0(self, mu0: float):
        self.__mu0 = mu0

    @property
    def sigma(self) -> float:
        """Influences the width of the Gaussian bell curve. Should always be greater than 0."""
        return self.__sigma

    @sigma.setter
    def sigma(self, sigma: float):
        if sigma <= 0:
            raise ValueError(f"sigma should be greater than 0, was {sigma}")
        self.__sigma = sigma

    @property
    def height(self) -> float:
        """Scaling factor for the height of the Gaussian bell curve."""
        return self.__height

    @height.setter
    def height(self, height):
        self.__height = height

    @property
    def offset(self) -> float:
        """Lowest (y-)value of the Gaussian bell curve."""
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
        return sensor_response_gauss(
            mu_sens=mu_sens, mu0=self.mu0, sigma=self.sigma, height=self.height, offset=self.offset
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(mu0={self.mu0}, sigma={self.sigma}, height={self.height}, offset={self.offset})"
        )


def sensor_response_gauss(mu_sens: np.ndarray, mu0: float = 0, sigma: float = 1, height: float = 1, offset: float = 0):
    """Calculates the sensor response out of the given sensing dot potential by simulating the sensor behavior with a Gaussian function.

    Args:
        mu_sens (np.ndarray): The sensor potential, stored in a numpy array with the axis mapping to the CSD axis.
        mu0 (float): Position of the maximum of the Gaussian bell curve. Default is 0.
        sigma (float): Influences the width of the Gaussian bell curve. Should always be greater than 0. Default is 1.
        height (float): Scaling factor for the height of the Gaussian bell curve. Default is 1.
        offset (float): Lowest (y-)value of the Gaussian bell curve. Default is 0.

    Returns:
        np.ndarray: The sensor response (for the corresponding peak), calculated from the given potential. It is
        stored in a numpy array with the axis mapping to the CSD axis.
    """
    return offset + height * np.exp(-((mu_sens - mu0) / sigma) ** 2)
