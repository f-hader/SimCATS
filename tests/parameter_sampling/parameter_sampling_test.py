"""Class to test distribution and parameter edge cases of the parameter samplers."""
import unittest

import numpy as np
from scipy import stats

from simcats.support_functions import (
    NormalSamplingRange,
    UniformSamplingRange,
)


class ParameterSamplingTests(unittest.TestCase):
    """Class to test distribution and parameter edge cases of the parameter samplers."""

    confidence = 0.05

    def test_normal_sampler_distribution(self) -> None:
        """Method to test distribution of normal sampler."""
        sampler = NormalSamplingRange((1e-10, 0.003), std=0.0003)
        samples = np.zeros(100000)
        for i in range(len(samples)):
            samples[i] = sampler.sample_parameter()
        ksstat, p_value = stats.kstest(
            samples,
            "norm",
            args=(samples.mean(), samples.std()),
        )
        assert p_value > ParameterSamplingTests.confidence

    def test_uniform_sampler_distribution(self) -> None:
        """Method to test distribution of uniform sampler."""
        total_range = (0, 1)
        sampler = UniformSamplingRange(total_range)
        samples = np.zeros(1000000)
        for i in range(len(samples)):
            samples[i] = sampler.sample_parameter()
        ksstat, p_value = stats.kstest(samples, "uniform", args=(total_range))
        assert p_value > ParameterSamplingTests.confidence

    def test_normal_sampler_sampling_range(self) -> None:
        """Check that the reduced range works even if the reduced range has the total width of the actual range."""
        ratio = 1
        for _ in range(100000):
            left = np.random.default_rng().uniform(0, 0.49)
            right = np.random.default_rng().uniform(0.51, 1)
            total_range = (left, right)
            sampler = NormalSamplingRange(
                total_range,
                std=1,
                sampling_range=(right - left) * ratio,
            )
            actual_range = sampler._NormalSamplingRange__range  # noqa: SLF001
            assert total_range[0] <= actual_range[0]
            assert total_range[1] >= actual_range[1]

    def test_normal_sampler_sampling_range_small(self) -> None:
        """Check that every sample is in the reduced range."""
        total_range = (0, 1)
        sampler = NormalSamplingRange(
            total_range,
            std=1,
            sampling_range=(total_range[1] - total_range[0]) * 0.5,
        )
        actual_range = sampler._NormalSamplingRange__range  # noqa: SLF001
        assert total_range[0] <= actual_range[0]
        assert total_range[1] >= actual_range[1]
        for _i in range(100000):
            sample = sampler.sample_parameter()
            assert actual_range[0] <= sample
            assert actual_range[1] >= sample

    def test_uniform_sampler_sampling_range(self) -> None:
        """Check that the reduced range works even if the reduced range has the total width of the actual range."""
        ratio = 1
        for _ in range(100000):
            left = np.random.default_rng().uniform(0, 0.49)
            right = np.random.default_rng().uniform(0.51, 1)
            total_range = (left, right)
            sampler = UniformSamplingRange(
                total_range,
                sampling_range=(right - left) * ratio,
            )
            actual_range = sampler._UniformSamplingRange__range  # noqa: SLF001
            assert total_range[0] <= actual_range[0]
            assert total_range[1] >= actual_range[1]

    def test_uniform_sampler_sampling_range_small(self) -> None:
        """Check that every sample is in the reduced range."""
        total_range = (0, 1)
        sampler = UniformSamplingRange(
            total_range,
            sampling_range=(total_range[1] - total_range[0]) * 0.5,
        )
        actual_range = sampler._UniformSamplingRange__range  # noqa: SLF001
        assert total_range[0] <= actual_range[0]
        assert total_range[1] >= actual_range[1]
        for _i in range(100000):
            sample = sampler.sample_parameter()
            assert actual_range[0] <= sample
            assert actual_range[1] >= sample
