import unittest
from copy import deepcopy

import numpy as np

from simcats.distortions import OccupationDotJumps, SensorPotentialPinkNoise, SensorPotentialRTN, \
    SensorResponseWhiteNoise
from simcats.support_functions import (
    NormalSamplingRange,
    UniformSamplingRange,
)


class DistortionRatioTests(unittest.TestCase):
    def test_occupation_dot_jumps_ratio(self) -> None:
        """Test if the OccupationDotJumps ratio parameter works."""
        for ratio in np.linspace(0, 1, 3):
            occupationDotJumps = OccupationDotJumps(
                ratio=ratio,
                scale=0.001 * 0.03 / 100,
                lam=6 * 0.03 / 100,
                axis=0,
            )
            lead_transitions_noiseless = np.arange(0, 500).reshape((100, 5))
            occupation_noiseless = np.array(
                [
                    deepcopy(
                        lead_transitions_noiseless,
                    ).T,  # remove transpose if rows > columns
                    deepcopy(lead_transitions_noiseless).T,
                ],
            ).T
            volt_limits_g1 = np.array([-0.05, 0.058])
            volt_limits_g2 = np.array([-0.05, 0.06])
            num_iter = 10000
            num_noisy = num_iter
            for _ in range(num_iter):
                (
                    occupation_noisy,
                    lead_transitions_noisy,
                ) = occupationDotJumps.noise_function(
                    deepcopy(occupation_noiseless),
                    deepcopy(lead_transitions_noiseless),
                    volt_limits_g1,
                    volt_limits_g2,
                )
                if (occupation_noisy == occupation_noiseless).all() and (
                    lead_transitions_noisy == lead_transitions_noiseless
                ).all():
                    num_noisy -= 1
            np.testing.assert_allclose(ratio, num_noisy / num_iter, atol=0.1)

    def test_sensor_potential_rtn_ratio(self) -> None:
        """Test if the SensorPotentialRTN ratio parameter works."""
        for ratio in np.linspace(0, 1, 3):
            sensorPotentialRTN = SensorPotentialRTN(
                scale=0.1,
                std=0,
                height=1,
                ratio=ratio,
            )
            sensor_potential = np.zeros((10, 10), dtype=float)
            volt_limits_g1 = np.array([0.1, 0.2])
            volt_limits_g2 = np.array([0.1, 0.2])
            num_iter = 100000
            num_noisy = 0
            for _ in range(num_iter):
                sensor_potential_noisy = sensorPotentialRTN.noise_function(
                    sensor_potential,
                    volt_limits_g1,
                    volt_limits_g2,
                )
                if not (sensor_potential == sensor_potential_noisy).all():
                    num_noisy += 1
            np.testing.assert_allclose(ratio, num_noisy / num_iter, atol=0.1)

    def test_sensor_potential_pink_noise_sampling_interface(self) -> None:
        """Method to test if pink noise works with the parameter sampling interface as sigma value."""
        sensor_potential_pink_noise = SensorPotentialPinkNoise(
            sigma=UniformSamplingRange((1.8250268077765864e-12, 9.125134038882932e-05)),
        )
        sensor_potential = np.zeros((10, 10), dtype=float)
        volt_limits_g1 = np.array([0.1, 0.2])
        volt_limits_g2 = np.array([0.1, 0.2])
        _ = sensor_potential_pink_noise.noise_function(
            sensor_potential,
            volt_limits_g1,
            volt_limits_g2,
        )
        assert True

    def test_sensor_potential_pink_noise_fixed_sigma(self) -> None:
        """Method to test if pink noise works with a fixed sigma value."""
        sensor_potential_pink_noise = SensorPotentialPinkNoise(
            sigma=1.8250268077765864e-6
        )
        sensor_potential = np.zeros((10, 10), dtype=float)
        volt_limits_g1 = np.array([0.1, 0.2])
        volt_limits_g2 = np.array([0.1, 0.2])
        _ = sensor_potential_pink_noise.noise_function(
            sensor_potential,
            volt_limits_g1,
            volt_limits_g2,
        )
        assert True

    def test_sensor_response_white_noise_sampling_interface(self) -> None:
        """Method to test if white noise works with the parameter sampling interface as sigma value."""
        sensor_response_white_noise = SensorResponseWhiteNoise(
            sigma=NormalSamplingRange((1e-10, 0.003), sampling_range=0.001, std=0.0003),
        )
        sensor_potential = np.zeros((10, 10), dtype=float)
        volt_limits_g1 = np.array([0.1, 0.2])
        volt_limits_g2 = np.array([0.1, 0.2])
        _ = sensor_response_white_noise.noise_function(
            sensor_potential,
            volt_limits_g1,
            volt_limits_g2,
        )
        assert True

    def test_sensor_response_white_noise_fixed_sigma(self) -> None:
        """Method to test if white noise works with a fixed sigma value."""
        sensor_response_white_noise = SensorResponseWhiteNoise(sigma=0.0003)
        sensor_potential = np.zeros((10, 10), dtype=float)
        volt_limits_g1 = np.array([0.1, 0.2])
        volt_limits_g2 = np.array([0.1, 0.2])
        _ = sensor_response_white_noise.noise_function(
            sensor_potential,
            volt_limits_g1,
            volt_limits_g2,
        )
        assert True
