"""This module contains functions for the simulation of random telegraph noise (RTN).

@author: s.fleitmann
"""

from typing import Union

import numpy as np

from simcats.distortions import (
    DistortionInterface,
    SensorPotentialDistortionInterface,
    SensorResponseDistortionInterface,
)
from simcats.support_functions import ParameterSamplingInterface

__all__ = []


class RandomTelegraphNoise(DistortionInterface):
    """Random telegraph noise (RTN) implementation of the (general) DistortionsInterface.

    This general implementation is independent of the distortion step/type (occupations, sensor potential, or sensor
    response). Specific implementations are derived from this generic one.
    """

    def __init__(
        self,
        scale: Union[float, ParameterSamplingInterface],
        std: Union[float, ParameterSamplingInterface],
        height: Union[float, ParameterSamplingInterface],
        ratio: float = 1,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """Initializes an object of the class used to simulate random telegraph noise.

        Args:
            scale (Union[float, ParameterSamplingInterface]): Specifies the length of the bursts/mathematical
                expectation for their length and also the length of regions without bursts. If the scale is of type
                ParameterSampling a new scale is sampled accordingly per call of noise_function.
            std (Union[float, ParameterSamplingInterface]): The standard deviation of the magnitude of the jumps. If std
                is of type ParameterSampling a new std is sampled accordingly per call of noise_function.
            height (Union[float, ParameterSamplingInterface]): The mean of the magnitude of the jumps. If the height is
                of type ParameterSampling a new height is sampled accordingly per call of noise_function.
            ratio (float): The ratio defining how often this type of noise should be active. For each simulation a
                random number decides if the noise type is active, based on the supplied ratio. Default is 1.
            rng (np.random.Generator): The random number generator used for the simulation of random numbers. Default is
                np.random.default_rng().
        """
        self.scale = scale
        self.std = std
        self.height = height
        self.ratio = ratio
        self.rng = rng
        self.__activated = None

    @property
    def scale(self) -> Union[float, ParameterSamplingInterface]:
        """Specifies the length of the bursts/mathematical expectation for their length and also the length of regions
        without bursts.
        """
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    @property
    def std(self) -> Union[float, ParameterSamplingInterface]:
        """The standard deviation of the magnitude of the jumps."""
        return self.__std

    @std.setter
    def std(self, std) -> None:
        self.__std = std

    @property
    def height(self) -> Union[float, ParameterSamplingInterface]:
        """The mean of the magnitude of the jumps."""
        return self.__height

    @height.setter
    def height(self, height) -> None:
        self.__height = height

    @property
    def ratio(self) -> float:
        """The ratio defining how often this type of noise should be active."""
        return self.__ratio

    @ratio.setter
    def ratio(self, ratio) -> None:
        self.__ratio = ratio

    @property
    def rng(self) -> np.random.Generator:
        """The random number generator used for the simulation of random numbers."""
        return self.__rng

    @rng.setter
    def rng(self, rng):
        self.__rng = rng

    @property
    def activated(self) -> Union[bool, None]:
        """This is true if the noise was activated during the last call of noise function."""
        return self.__activated

    def noise_function(
        self, original: np.ndarray, volt_limits_g1: np.ndarray, volt_limits_g2: np.ndarray
    ) -> np.ndarray:
        """Adds random telegraph noise to the original image.

        The distortion parameter `scale` is adjusted to the resolution, as it is specified in voltage space
        but must be applied in pixel space.

        Args:
            original (np.ndarray): Original image. Can be a sensor potential or a sensor response
            volt_limits_g1 (np.ndarray): Voltage sweep range of (plunger) gate 1 (second-/x-axis). \n
                Example: \n
                [min_V1, max_V1]
            volt_limits_g2 (np.ndarray): Voltage sweep range of (plunger) gate 2 (first-/y-axis). \n
                Example: \n
                [min_V2, max_V2]

        Returns:
            np.ndarray: Original image distorted by random telegraph noise.
        """
        if not bool(self.rng.binomial(1, self.ratio)):
            self.__activated = False
            return original
        else:
            self.__activated = True
        try:
            scale = self.scale.sample_parameter()
        except AttributeError:
            scale = self.scale
        # rescale scale according to the resolution per voltage
        if len(original.shape) == 2:
            resolution = original.shape
            if volt_limits_g1[0] == volt_limits_g1[1]:
                scale *= resolution[1] / (np.max(volt_limits_g2) - np.min(volt_limits_g2))
            elif volt_limits_g2[0] == volt_limits_g2[1]:
                scale *= resolution[0] / (np.max(volt_limits_g1) - np.min(volt_limits_g1))
            else:
                scale *= np.maximum(
                    resolution[0] / (np.max(volt_limits_g1) - np.min(volt_limits_g1)),
                    resolution[1] / (np.max(volt_limits_g2) - np.min(volt_limits_g2)),
                )
        else:
            resolution = (original.shape[0], original.shape[0])
            # sweep direction is always the x-axis direction
            scale *= resolution[0] / (np.max(volt_limits_g1) - np.min(volt_limits_g1))

        try:
            std = self.std.sample_parameter()
        except AttributeError:
            std = self.std

        try:
            height = self.height.sample_parameter()
        except AttributeError:
            height = self.height

        return random_telegraph_noise(original, scale, std, height, self.rng)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(scale={self.scale}, std={self.std}, height={self.height}, \
ratio={self.ratio}, rng={self.rng}){'[activated]' if self.activated else '[deactivated]'}"
        )


class SensorPotentialRTN(RandomTelegraphNoise, SensorPotentialDistortionInterface):
    """Random telegraph noise (RTN) implementation of the SensorPotentialDistortionInterface."""

    def __init__(
        self,
        scale: Union[float, ParameterSamplingInterface],
        std: Union[float, ParameterSamplingInterface],
        height: Union[float, ParameterSamplingInterface],
        ratio: float = 1,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """Initializes an object of the class used to simulate random telegraph noise affecting the sensor potential.

        Args:
            scale (Union[float, ParameterSamplingInterface]): Specifies the length of the bursts/mathematical
                expectation for their length and also the length of regions without bursts. If the scale is of type
                ParameterSampling a new scale is sampled accordingly per call of noise_function.
            std (Union[float, ParameterSamplingInterface]): The standard deviation of the magnitude of the jumps. If std
                is of type ParameterSampling a new std is sampled accordingly per call of noise_function.
            height (Union[float, ParameterSamplingInterface]): The mean of the magnitude of the jumps. If the height is
                of type ParameterSampling a new height is sampled accordingly per call of noise_function.
            ratio (float): The ratio defining how often this type of noise should be active. For each simulation a
                random number decides if the noise type is active, based on the supplied ratio. Default is 1.
            rng (np.random.Generator): The random number generator used for the simulation of random numbers. Default is
                np.random.default_rng().
        """
        RandomTelegraphNoise.__init__(self, scale, std, height, ratio, rng)

    def noise_function(self, mu_sens: np.ndarray, volt_limits_g1: np.ndarray, volt_limits_g2: np.ndarray) -> np.ndarray:
        """Adds random telegraph noise to the sensor potential.

        The distortion parameters are adjusted to the resolution, as they are specified in voltage space
        but must be applied in pixel space.

        Args:
            mu_sens (np.ndarray): Contains the sensor potential to which the noise is added. The potential is
                stored in a 2-dimensional numpy array with the axis mapping to the CSD axis. If there are other types of
                distortions which appear first, they should already be included in the potential.
            volt_limits_g1 (np.ndarray): Voltage sweep range of (plunger) gate 1 (second-/x-axis). \n
                Example: \n
                [min_V1, max_V1]
            volt_limits_g2 (np.ndarray): Voltage sweep range of (plunger) gate 2 (first-/y-axis). \n
                Example: \n
                [min_V2, max_V2]

        Returns:
            np.ndarray: The sensor potential with added noise. The potential is stored in a 2-dimensional numpy array
            with the axis mapping to the CSD axis.
        """
        return RandomTelegraphNoise.noise_function(self, mu_sens, volt_limits_g1, volt_limits_g2)


class SensorResponseRTN(RandomTelegraphNoise, SensorResponseDistortionInterface):
    """Random telegraph noise (RTN) implementation of the SensorResponseDistortionInterface."""

    def __init__(
        self,
        scale: Union[float, ParameterSamplingInterface],
        std: Union[float, ParameterSamplingInterface],
        height: Union[float, ParameterSamplingInterface],
        ratio: float = 1,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """Initializes an object of the class used to simulate random telegraph noise affecting the sensor response.

        Args:
            scale (Union[float, ParameterSamplingInterface]): Specifies the length of the bursts/mathematical
                expectation for their length and also the length of regions without bursts. If the scale is of type
                ParameterSampling a new scale is sampled accordingly per call of noise_function.
            std (Union[float, ParameterSamplingInterface]): The standard deviation of the magnitude of the jumps. If std
                is of type ParameterSampling a new std is sampled accordingly per call of noise_function.
            height (Union[float, ParameterSamplingInterface]): The mean of the magnitude of the jumps. If the height is
                of type ParameterSampling a new height is sampled accordingly per call of noise_function.
            ratio (float): The ratio defining how often this type of noise should be active. For each simulation a
                random number decides if the distortions type is active, based on the supplied ratio. Default is 1.
            rng (np.random.Generator): The random number generator used for the simulation of random numbers. Default is
                np.random.default_rng().
        """
        RandomTelegraphNoise.__init__(self, scale, std, height, ratio, rng)

    def noise_function(
        self, sensor_response: np.ndarray, volt_limits_g1: np.ndarray, volt_limits_g2: np.ndarray
    ) -> np.ndarray:
        """Adds random telegraph noise to the sensor response.

        The distortion parameters are adjusted to the resolution, as they are specified in voltage space
        but must be applied in pixel space.

        Args:
            sensor_response (np.ndarray): Contains the sensor response to which the noise is added. The sensor
                response is stored in a 2-dimensional numpy array with the axis mapping to the CSD axis. If there are
                other types of distortions which appear first, they should already be included in the response.
            volt_limits_g1 (np.ndarray): Voltage sweep range of plunger gate 1 (second-/x-axis). \n
                Example: \n
                [min_V1, max_V1]
            volt_limits_g2 (np.ndarray): Voltage sweep range of plunger gate 2 (first-/y-axis). \n
                Example: \n
                [min_V2, max_V2]

        Returns:
            np.ndarray: The sensor response with added noise. The sensor response is stored in a 2-dimensional numpy
            array with the axis mapping to the CSD axis.
        """
        return RandomTelegraphNoise.noise_function(self, sensor_response, volt_limits_g1, volt_limits_g2)


def random_telegraph_noise(
    original: np.ndarray, scale: float, std: float, height: float, rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
    """Adds random telegraph noise to the original image.

    Args:
        original (np.ndarray): Original image where the noise should be added.
        scale (float): Specifies the length of the bursts/mathematical expectation for their length and also the length
            of regions without bursts.
        std (float): The standard deviation of the magnitude of the jumps.
        height (float): The mean of the magnitude of the jumps.
        rng (np.random.Generator): The random number generator used for the simulation of random numbers. Default is
            np.random.default_rng().

    Returns:
        np.ndarray: Original image with noisy added to it.
    """
    if scale == 0:
        return original
    size = np.prod(original.shape)
    # length of a burst follows an geometric distribution
    length = rng.geometric(p=1 / scale, size=size)

    noise = np.array([])
    bi = 0  # set binary states, 0 means no burst, 1 means burst
    for i in range(size):
        if length[i] > 0:
            noise = np.concatenate((noise, np.array([bi] * int(length[i])) * rng.normal(height, std)))
            bi = 1 if bi == 0 else 0
        if len(noise) >= size:
            break

    noise = noise[0:size].reshape(original.shape)
    return original + noise


def _random_telegraph_noise_two_scales(
    original: np.ndarray,
    scale_burst: float,
    scale_no_burst: float,
    std: float,
    height: float,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """Adds random telegraph noise to the original image.

    This method has the possibility to create a random telegraph noise with different lengths for regions
    with and without bursts.

    Args:
        original (np.ndarray): Original image where the noise should be added.
        scale_burst (float): Specifies the length of the bursts/mathematical expectation for their length.
        scale_not_burst (float): Specifies the length of the regions without bursts.
        std (float): The standard deviation of the magnitude of the jumps.
        height (float): The mean of the magnitude of the jumps.
        rng (np.random.Generator): The random number generator used for the simulation of random numbers. Default is
            np.random.default_rng().

    Returns:
        np.ndarray: Original image with noisy added to it.
    """
    if scale_burst == 0:
        return original
    size = np.prod(original.shape)
    # length of a burst follows an geometric distribution

    noise = np.array([])
    bi = 0  # set binary states, 0 means no burst, 1 means burst
    for _i in range(size):
        if bi == 0:
            # length of no burst follows an geometric distribution
            length = rng.geometric(p=1 / scale_no_burst, size=1)
            noise = np.concatenate((noise, np.array([bi] * int(length))))
            bi = 1
        else:
            # length of a burst follows an geometric distribution
            length = rng.geometric(p=1 / scale_burst, size=1)
            noise = np.concatenate((noise, np.array([bi] * int(length)) * rng.normal(height, std)))
            bi = 0
        if len(noise) >= size:
            break

    noise = noise[0:size].reshape(original.shape)
    return original + noise
