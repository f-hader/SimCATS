"""This module contains functionality for sampling a specific parameter from a parameter range.

The contained functions can be used for example for the parameter sampling in the distortions module.

@author: s.fleitmann, f.fuchs
"""

import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np

__all__ = []


class ParameterSamplingInterface(ABC):
    """Interface for parameter sampling.

    This is used for example for the fluctuation of the strengths of distortions during the simulation of CSDs.
    """

    @abstractmethod
    def sample_parameter(self):
        """This method is used to sample a parameter for example from a given range with a given distribution.

        Returns:
            Sampled parameter
        """
        raise NotImplementedError()

    @abstractmethod
    def last_sample(self):
        """This method is used to get the last sampled parameter.

        Which can be used to check which parameter exactly was used to simulate the last CSD.

        Returns:
            Last sampled parameter
        """
        raise NotImplementedError()


class NormalSamplingRange(ParameterSamplingInterface):
    """Normal sampling range implementation of ParameterSamplingInterface."""
    def __init__(
        self,
        total_range: Tuple,
        std: float,
        sampling_range: Union[float, None] = None,
        rng: Union[np.random.Generator, None] = None,
    ) -> None:
        """This class can be used to generate randomly normal sampled parameters within a given range.

        For example, for the distortions used in the simulation of CSDs.

        Args:
            total_range (Tuple): The total range in which the parameters can be sampled. This can be narrowed down
                randomly with the help of sampling_range. If the normal distribution generates a sample outside this
                range, a new sample is drawn until a sample inside the sampling_range/total_range was generated,
                leading to a truncated normal distribution.
            std (float): The standard deviation of the sampled elements, which is used in the normal distribution.
            sampling_range (Union[float, None]): The maximum range in which the parameter is allowed to change during
                the simulation. The explicit range is set up during the initialization, narrowing down the
                supplied total_range. Default is None, which leads to no narrowing of the given total_range.
            rng (np.random.Generator): random number generator used for the sampling of random numbers. If None,  the
                default generator of numpy (np.random.default_rng()) is used. Default is None.
        """
        if rng:
            self.__rng = rng
        else:
            self.__rng = np.random.default_rng()
        if sampling_range is None:
            self.__range = total_range
        else:
            if np.greater_equal(sampling_range, total_range[1] - total_range[0]):
                warnings.warn(
                    "The given reduced sampling range is equal or larger than the given total range. As "
                    "default the given total sampling range is taken.",
                    stacklevel=2,
                )
                self.__range = total_range
            else:
                sampled = self.__rng.uniform(
                    total_range[0] + 0.5 * sampling_range, total_range[1] - 0.5 * sampling_range
                )
                self.__range = (sampled - 0.5 * sampling_range, sampled + 0.5 * sampling_range)
        self.__std = std
        self.__last_sample = None

    def sample_parameter(self):
        sampled = self.__rng.normal(loc=np.mean(self.__range), scale=self.__std)
        # repeat sampling until the sampled value is in self.__range
        while sampled < self.__range[0] or sampled > self.__range[1]:
            sampled = self.__rng.normal(loc=np.mean(self.__range), scale=self.__std)
        self.__last_sample = sampled
        return sampled

    def last_sample(self):
        return self.__last_sample

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(last_sample={self.last_sample()}, range={self.__range}, std={self.__std}, rng={self.__rng})"
        )


class UniformSamplingRange(ParameterSamplingInterface):
    """Uniform sampling range implementation of ParameterSamplingInterface."""
    def __init__(
        self,
        total_range: Tuple,
        sampling_range: Union[float, None] = None,
        rng: Union[np.random.Generator, None] = None,
    ) -> None:
        """This class can be used to generate randomly uniform sampled parameters within a given range.

        For example, for the distortions used in the simulation of CSDs.

        Args:
            total_range (Tuple): The total range in which the parameters can be sampled. This can be narrowed down
                randomly with the help of sampling_range.
            sampling_range (Union[float, None]): The maximum range in which the parameter is allowed to change during
                the simulation. The explicit range is set up during the initialization, narrowing down the supplied
                total_range. Default is None, which leads to no narrowing of the given total_range.
            rng (np.random.Generator): random number generator used for the sampling of random numbers. If None,  the
                default generator of numpy (np.random.default_rng()) is used. Default is None.
        """
        if rng:
            self.__rng = rng
        else:
            self.__rng = np.random.default_rng()
        if sampling_range is None:
            self.__range = total_range
        else:
            if np.greater_equal(sampling_range, total_range[1] - total_range[0]):
                warnings.warn(
                    "The given reduced sampling range is equal or larger than the given total range. As "
                    "default the given total sampling range is taken.",
                    stacklevel=2,
                )
                self.__range = total_range
            else:
                sampled = self.__rng.uniform(
                    total_range[0] + 0.5 * sampling_range, total_range[1] - 0.5 * sampling_range
                )
                self.__range = (sampled - 0.5 * sampling_range, sampled + 0.5 * sampling_range)
        self.__last_sample = None

    def sample_parameter(self):
        sampled = self.__rng.uniform(self.__range[0], self.__range[1])
        self.__last_sample = sampled
        return sampled

    def last_sample(self):
        return self.__last_sample

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(last_sample={self.last_sample()}, range={self.__range}, rng={self.__rng})"


class LogNormalSamplingRange(ParameterSamplingInterface):
    """Logarithmic normal sampling range implementation of ParameterSamplingInterface."""
    def __init__(
        self,
        total_range: Tuple,
        sampling_range: Union[float, None] = None,
        rng: Union[np.random.Generator, None] = None,
        mean: float = 0,
        sigma: float = 1,
    ) -> None:
        """This class can be used to generate randomly log-normal sampled parameters within a given range.

        For example, for the distortions used in the simulation of CSDs.

        Args:
            total_range (Tuple): The total range in which the parameters can be sampled. This can be narrowed down
                randomly with the help of sampling_range. If the log-normal distribution generates a sample outside this
                range, a new sample is drawn until a sample inside the sampling_range/total_range was generated, leading
                toa truncated log-normal distribution.
            sampling_range (Union[float, None]): The maximum range in which the parameter is allowed to change during
                the simulation. The explicit range is set up during the initialization, narrowing down the supplied
                total_range. Default is None, which leads to no narrowing of the given total_range.
            rng (np.random.Generator): random number generator used for the sampling of random numbers. If None,  the
                default generator of numpy (np.random.default_rng()) is used. Default is None.
            mean (float): Mean value of the underlying normal distribution. Default is 0.
            sigma (float): Standard deviation of the underlying normal distribution. Must be non-negative. Default is 1.
        """
        if rng:
            self.__rng = rng
        else:
            self.__rng = np.random.default_rng()
        self.__mean: float = mean
        self.__sigma: float = sigma
        if sampling_range is None:
            self.__range = total_range
        else:
            if np.greater_equal(sampling_range, total_range[1] - total_range[0]):
                warnings.warn(
                    "The given reduced sampling range is equal or larger than the given total range. As "
                    "default the given total sampling range is taken.",
                    stacklevel=2,
                )
                self.__range = total_range
            else:
                sampled = self.__rng.uniform(
                    total_range[0] + 0.5 * sampling_range,
                    total_range[1] - 0.5 * sampling_range,
                )
                self.__range = (
                    sampled - 0.5 * sampling_range,
                    sampled + 0.5 * sampling_range,
                )
        self.__last_sample = None

    def sample_parameter(self):
        sampled = self.__rng.lognormal(mean=self.__mean, sigma=self.__sigma)
        # repeat sampling until the sampled value is in self.__range
        while sampled < self.__range[0] or sampled > self.__range[1]:
            sampled = self.__rng.lognormal(mean=self.__mean, sigma=self.__sigma)
        self.__last_sample = sampled
        return sampled

    def last_sample(self):
        return self.__last_sample

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(last_sample={self.last_sample()}, range={self.__range}, mean={self.__mean}, sigma={self.__sigma}"
            + f", rng={self.__rng})"
        )
