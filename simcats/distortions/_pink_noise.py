"""This module contains functions for the simulation of pink noise.

@author: s.fleitmann
"""

from typing import Union

import colorednoise as cn
import numpy as np

from simcats.distortions import SensorPotentialDistortionInterface
from simcats.support_functions import ParameterSamplingInterface

__all__ = []


class SensorPotentialPinkNoise(SensorPotentialDistortionInterface):
    """Pink noise implementation of the SensorPotentialDistortionsInterface."""

    def __init__(
        self,
        sigma: Union[float, ParameterSamplingInterface],
        fmin: float = 0,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """Initializes an object of the class used to generate pink noise.

        Args:
            sigma (Union[float, ParameterSamplingInterface]): The sigma for the pink noise distribution. If sigma is of
                type ParameterSampling a new sigma is sampled accordingly per call of noise_function.
            fmin (float): Minimal frequency which can be present in the generated noise. The maximal possible value is
                0.5 which generates white noise. The minimal value is 1/(number of pixels). The default is 0, which
                results in the same as 1/(number of pixels).
            rng (np.random.Generator): The random number generator used for the simulation of random numbers. Default is
                np.random.default_rng().
        """
        self.sigma = sigma
        self.fmin = fmin
        self.rng = rng
        self.__latest_sigma = None

    @property
    def sigma(self) -> Union[float, ParameterSamplingInterface]:
        """The sigma for the pink noise distribution."""
        return self.__sigma

    @sigma.setter
    def sigma(self, sigma):
        self.__sigma = sigma

    @property
    def fmin(self) -> float:
        """Minimal frequency which can be present in the generated noise."""
        return self.__fmin

    @fmin.setter
    def fmin(self, fmin):
        self.__fmin = fmin

    @property
    def rng(self) -> np.random.Generator:
        """The random number generator used for the simulation of random numbers."""
        return self.__rng

    @rng.setter
    def rng(self, rng):
        self.__rng = rng

    @property
    def latest_sigma(self) -> float:
        """The sigma that was used for the latest simulation.

        This is necessary because, depending on the setting, a sampler can be used instead of a fixed sigma.
        """
        return self.__latest_sigma

    def noise_function(self, mu_sens: np.ndarray, volt_limits_g1: np.ndarray, volt_limits_g2: np.ndarray) -> np.ndarray:
        """This function is used to add pink noise to the sensor potential of a CSD.

        Args:
            mu_sens (np.ndarray): Contains the sensor potential to which the distortions are added. The potential is
                stored in a 2-dimensional numpy array with the axis mapping to the CSD axis. If there are other types of
                distortions which appear first, they should already be included in the potential.
            volt_limits_g1 (np.ndarray): Contains the beginning and ending of the swept range for the first (plunger)
                gate.
            volt_limits_g2 (np.ndarray): Contains the beginning and ending of the swept range for the second (plunger)
                gate.

        Returns:
            np.ndarray: The sensor potential with added noise. The potential is stored in a 2-dimensional numpy array
            with the axis mapping to the CSD axis.
        """
        try:
            sampled = self.sigma.sample_parameter()
        except AttributeError:
            sampled = self.sigma
        self.__latest_sigma = sampled
        return pink_gaussian_noise(original=mu_sens, scale=sampled, fmin=self.fmin, rng=self.rng)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(sigma={self.sigma}{f'[latest={self.latest_sigma}]' if self.latest_sigma else '[latest=0]'}, \
fmin={self.fmin}, rng={self.rng})"
        )


def pink_gaussian_noise(
    original: np.ndarray, scale: float, fmin: float, rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
    """Adds pink Gaussian noise to the original image.

    Args:
        original (np.ndarray): Original image where the noise should be added.
        scale (float): Scaling factor with which the generated noise is multiplied. This is also the standard deviation
            (STD) for the noise.
        fmin (float): Minimal frequency which can be present in the generated noise. The maximal possible value is 0.5
            which generates white noise. The minimal value is 1/(number of pixels).
        rng (np.random.Generator): The random number generator used for the simulation of random numbers. Default is
                np.random.default_rng().

    Returns:
        np.ndarray: Original image distorted by pink noise.

    """
    noise = cn.powerlaw_psd_gaussian(1, np.prod(original.shape), fmin=fmin, random_state=rng) * scale
    noise = np.reshape(noise, original.shape)
    return original + noise
