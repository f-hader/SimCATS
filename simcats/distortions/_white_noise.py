"""This module contains functions for the simulation of white noise.

@author: s.fleitmann
"""

from typing import Union

import numpy as np

from simcats.distortions import SensorResponseDistortionInterface
from simcats.support_functions import ParameterSamplingInterface

__all__ = []


class SensorResponseWhiteNoise(SensorResponseDistortionInterface):
    """White noise implementation of the SensorResponseDistortionsInterface."""

    def __init__(
        self, sigma: Union[float, ParameterSamplingInterface], rng: np.random.Generator = np.random.default_rng()
    ):
        """Initializes an object of the class used to generate white noise.

        Args:
            sigma (Union[float, ParameterSamplingInterface]): Standard deviation for the Gaussian distribution of the
                noise, determines the strength of the noise. If sigma is of type ParameterSampling a new sigma is
                sampled accordingly per call of noise_function.
            rng (np.random.Generator): The random number generator used for the simulation of random numbers. Default is
                np.random.default_rng().
        """
        self.rng = rng
        self.sigma = sigma
        self.__latest_sigma = None

    @property
    def sigma(self) -> Union[float, ParameterSamplingInterface]:
        """Standard deviation for the Gaussian distribution of the noise, determines the strength of the noise."""
        return self.__sigma

    @sigma.setter
    def sigma(self, sigma: Union[float, ParameterSamplingInterface]):
        self.__sigma = sigma

    @property
    def rng(self) -> np.random.Generator:
        """The random number generator used for the simulation of random numbers."""
        return self.__rng

    @rng.setter
    def rng(self, rng):
        self.__rng = rng

    @property
    def latest_sigma(self) -> Union[float, None]:
        """The sigma that was used for the latest simulation.

        This is necessary because, depending on the setting, a sampler can be used instead of a fixed sigma.
        """
        return self.__latest_sigma

    def noise_function(
        self, sensor_response: np.ndarray, volt_limits_g1: np.ndarray, volt_limits_g2: np.ndarray
    ) -> np.ndarray:
        """Adds white noise to the sensor response.

        Args:
            sensor_response (np.ndarray): Contains the sensor response to which the noise are added. The sensor
                response is stored in a 2-dimensional numpy array with the axis mapping to the CSD axis. If there are
                other types of distortions which appear first, they should already be included in the response.
            volt_limits_g1 (np.ndarray): Voltage sweep range of (plunger) gate 1 (second-/x-axis). \n
                Example: \n
                [min_V1, max_V1]
            volt_limits_g2 (np.ndarray): Voltage sweep range of (plunger) gate 2 (first-/y-axis). \n
                Example: \n
                [min_V2, max_V2]

        Returns:
            np.ndarray: The sensor response with added noise. The sensor response is stored in a 2-dimensional numpy
            array with the axis mapping to the CSD axis.
        """
        try:
            sampled = self.sigma.sample_parameter()
        except AttributeError:
            sampled = self.sigma
        self.__latest_sigma = sampled
        return white_gaussian_noise(sensor_response, sampled, self.rng)

    def __repr__(self):
        return self.__class__.__name__ + f"(sigma={self.sigma}, rng={self.rng})"


def white_gaussian_noise(
    original: np.ndarray, sigma: float, rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
    """Adds white Gaussian noise to the original image.

    Args:
        original (np.ndarray): Original image where the noise should be added.
        sigma (float): Standard deviation for the Gaussian distribution of the noise, determines the strength of the
            noise. For an array, the entries determine the different noise strength in x and y direction.
        rng (np.random.Generator): The random number generator used for the simulation of random numbers. Default is
            np.random.default_rng().

    Returns:
        np.ndarray: Original image with added noise.
    """
    return original + rng.normal(0, sigma, size=original.shape)
