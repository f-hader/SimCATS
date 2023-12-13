"""Module to test the simulation with different cumulative distribution functions (cdfs) and gamma factors."""
from copy import deepcopy
from unittest import TestCase

import numpy as np
import pytest

from simcats.ideal_csd import IdealCSDGeometric

from simcats import Simulation
from tests._test_configs import test_configs


class IdealCSDTests(TestCase):
    """Class to test the simulation with different cumulative distribution functions (cdfs) and gamma factors."""

    def test_cdf_cauchy(self) -> None:
        """Test is the simulation runs with a cdf of type cauchy."""
        config = deepcopy(test_configs["GaAs_v1"])

        config["ideal_csd_config"] = IdealCSDGeometric(
            **{
                "tct_params": config["ideal_csd_config"].tct_params,
                "rotation": -np.pi / 4,
                "lut_entries": 1000,
                "cdf_type": "cauchy",
                "cdf_gamma_factor": None,
            },
        )

        double_dot_device = Simulation(**config)

        # turn off noise
        double_dot_device.occupation_distortions = None
        double_dot_device.sensor_potential_distortions = None
        double_dot_device.sensor_response_distortions = None

        sweep_range_g1 = np.array([-0.05, 0.058])
        sweep_range_g2 = np.array([-0.05, 0.06])
        resolution = np.array(
            [300, 300]
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        # run simulation
        _, _, _, _ = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1, sweep_range_g2=sweep_range_g2, resolution=resolution
        )

        # just checking that this test runs without errors
        assert True

    def test_cdf_cauchy_gamma_factors(self) -> None:
        """Test different gamma factors for a simulation with cauchy cdf."""
        config = test_configs["GaAs_v1"]

        for gamma_factor in np.linspace(-10, 10, 20):
            config["ideal_csd_config"] = IdealCSDGeometric(
                **{
                    "tct_params": config["ideal_csd_config"].tct_params,
                    "rotation": -np.pi / 4,
                    "lut_entries": 1000,
                    "cdf_type": "cauchy",
                    "cdf_gamma_factor": gamma_factor,
                },
            )

            double_dot_device = Simulation(**config)

            # turn off noise
            double_dot_device.occupation_distortions = None
            double_dot_device.sensor_potential_distortions = None
            double_dot_device.sensor_response_distortions = None

            sweep_range_g1 = np.array([-0.05, 0.058])
            sweep_range_g2 = np.array([-0.05, 0.06])
            resolution = np.array(
                [300, 300],
            )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

            # run simulation
            _, _, _, _ = double_dot_device.measure(
                sweep_range_g1=sweep_range_g1, sweep_range_g2=sweep_range_g2, resolution=resolution
            )

        # just checking that this test runs without errors
        assert True

    def test_cdf_sigmoid(self) -> None:
        """Test is the simulation runs with a cdf of type sigmoid."""
        config = test_configs["GaAs_v1"]

        config["ideal_csd_config"] = IdealCSDGeometric(
            **{
                "tct_params": config["ideal_csd_config"].tct_params,
                "rotation": -np.pi / 4,
                "lut_entries": 1000,
                "cdf_type": "sigmoid",
                "cdf_gamma_factor": None,
            },
        )

        double_dot_device = Simulation(**config)

        # turn off noise
        double_dot_device.occupation_distortions = None
        double_dot_device.sensor_potential_distortions = None
        double_dot_device.sensor_response_distortions = None

        sweep_range_g1 = np.array([-0.05, 0.058])
        sweep_range_g2 = np.array([-0.05, 0.06])
        resolution = np.array(
            [300, 300],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        # run simulation
        _, _, _, _ = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1, sweep_range_g2=sweep_range_g2, resolution=resolution
        )

        # just checking that this test runs without errors
        assert True

    def test_cdf_sigmoid_gamma_factors(self) -> None:
        """Test different gamma factors for a simulation with sigmoid cdf."""
        config = test_configs["GaAs_v1"]

        for gamma_factor in np.linspace(-10, 10, 20):
            config["ideal_csd_config"] = IdealCSDGeometric(
                **{
                    "tct_params": config["ideal_csd_config"].tct_params,
                    "rotation": -np.pi / 4,
                    "lut_entries": 1000,
                    "cdf_type": "sigmoid",
                    "cdf_gamma_factor": gamma_factor,
                },
            )

            double_dot_device = Simulation(**config)

            # turn off noise
            double_dot_device.occupation_distortions = None
            double_dot_device.sensor_potential_distortions = None
            double_dot_device.sensor_response_distortions = None

            sweep_range_g1 = np.array([-0.05, 0.058])
            sweep_range_g2 = np.array([-0.05, 0.06])
            resolution = np.array(
                [300, 300],
            )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

            # run simulation
            _, _, _, _ = double_dot_device.measure(
                sweep_range_g1=sweep_range_g1, sweep_range_g2=sweep_range_g2, resolution=resolution
            )

        # just checking that this test runs without errors
        assert True

    def test_cdf_gaussian(self) -> None:
        """Test simulation with unexpected cdf type."""
        config = test_configs["GaAs_v1"]

        config["ideal_csd_config"] = IdealCSDGeometric(
            **{
                "tct_params": config["ideal_csd_config"].tct_params,
                "rotation": -np.pi / 4,
                "lut_entries": 1000,
                "cdf_type": "gaussian",
                "cdf_gamma_factor": None,
            },
        )

        double_dot_device = Simulation(**config)

        # turn off noise
        double_dot_device.occupation_distortions = None
        double_dot_device.sensor_potential_distortions = None
        double_dot_device.sensor_response_distortions = None

        sweep_range_g1 = np.array([-0.05, 0.058])
        sweep_range_g2 = np.array([-0.05, 0.06])
        resolution = np.array(
            [300, 300],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        # run simulation
        with pytest.raises(
            ValueError,
            match=r"The cdf_type for the electron occupation calculation must be either 'sigmoid' or 'cauchy'.",
        ):
            _, _, _, _ = double_dot_device.measure(
                sweep_range_g1=sweep_range_g1,
                sweep_range_g2=sweep_range_g2,
                resolution=resolution,
            )

    def test_to_few_wavefronts(self) -> None:
        """Test simulation with less wavefronts than max occupation."""
        config = deepcopy(test_configs["GaAs_v1"])

        double_dot_device_all_wavefronts = Simulation(**config)

        # turn off noise
        double_dot_device_all_wavefronts.occupation_distortions = None
        double_dot_device_all_wavefronts.sensor_potential_distortions = None
        double_dot_device_all_wavefronts.sensor_response_distortions = None

        sweep_range_g1 = np.array([-0.05, 0.058])
        sweep_range_g2 = np.array([-0.05, 0.06])
        resolution = np.array(
            [300, 300],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        # run simulation to find out max occupation
        _, occupations, _, _ = double_dot_device_all_wavefronts.measure(
            sweep_range_g1=sweep_range_g1, sweep_range_g2=sweep_range_g2, resolution=resolution
        )

        # remove wavefronts till we have one wavefront less than max occupation
        while len(config["ideal_csd_config"].tct_params) > occupations.sum(axis=2).max():
            config["ideal_csd_config"].tct_params.pop()

        # new device with less wavefronts
        double_dot_device_less_wavefronts = Simulation(**config)

        with pytest.raises(
            IndexError,
            match=r"bezier_coords dictionary does not contain an entry for total charge transition \(TCT\) 11.",
        ):
            _, _, _, _ = double_dot_device_less_wavefronts.measure(
                sweep_range_g1=sweep_range_g1,
                sweep_range_g2=sweep_range_g2,
                resolution=resolution,
            )
