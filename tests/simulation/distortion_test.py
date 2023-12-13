"""Module to test the simulation with different noise types."""

try:
    from importlib.resources import files as imp_files
except ImportError:
    from importlib_resources import files as imp_files
import unittest


import numpy as np
import pytest

from simcats.distortions import (
    OccupationDistortionInterface,
    SensorPotentialDistortionInterface,
    SensorResponseDistortionInterface,
)
from simcats.distortions import OccupationDotJumps, SensorPotentialPinkNoise, SensorPotentialRTN, SensorResponseRTN, \
    OccupationTransitionBlurringGaussian, SensorResponseWhiteNoise
from simcats import Simulation
from simcats.support_functions import (
    NormalSamplingRange,
    UniformSamplingRange,
)
from tests._test_configs import test_configs

from typing import Union, List, Tuple


class DistortionTests(unittest.TestCase):
    """Class to test the simulation with different noise types."""

    @classmethod
    def setUpClass(cls) -> None:
        """Class method to set up simulation class for each test."""
        # Initialize the simulation with default configurations
        cls.sim = Simulation(**test_configs["GaAs_v1"])

    def simulate_noisy_and_noiseless_measurement(
        self,
        occupation_distortions: Union[List[OccupationDistortionInterface], None] = None,
        sensor_potential_distortions: Union[List[SensorPotentialDistortionInterface], None] = None,
        sensor_response_distortions: Union[List[SensorResponseDistortionInterface], None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Helper method to run simulation with defined noise types and return occupation and lead transitions."""  # noqa: D401
        # Define the voltage sweep ranges and resolution for the simulation
        sweep_range_g1 = self.sim.volt_limits_g1
        sweep_range_g2 = self.sim.volt_limits_g2
        resolution = np.array([100, 100])

        # set noise
        self.sim.occupation_distortions = occupation_distortions
        self.sim.sensor_potential_distortions = sensor_potential_distortions
        self.sim.sensor_response_distortions = sensor_response_distortions

        # Perform the simulation with noise and noise-free data
        (
            _,
            occupations_noisy,
            lead_transitions_noisy,
            _,
        ) = self.sim.measure(
            sweep_range_g1,
            sweep_range_g2,
            resolution,
        )

        # set noise to None
        self.sim.occupation_distortions = None
        self.sim.sensor_potential_distortions = None
        self.sim.sensor_response_distortions = None

        (
            _,
            occupations_noiseless,
            lead_transitions_noiseless,
            _,
        ) = self.sim.measure(
            sweep_range_g1,
            sweep_range_g2,
            resolution,
        )

        return (
            occupations_noisy,
            lead_transitions_noisy,
            occupations_noiseless,
            lead_transitions_noiseless,
        )

    def test_all_noise(self) -> None:
        """Method to test simulation with all noise types."""
        (
            occupations_noisy,
            lead_transitions_noisy,
            occupations_noiseless,
            lead_transitions_noiseless,
        ) = self.simulate_noisy_and_noiseless_measurement(
            occupation_distortions=[
                OccupationTransitionBlurringGaussian(0.75 * 0.03 / 100),
                # in y direction
                OccupationDotJumps(
                    ratio=1,
                    scale=100 * 0.03 / 100,
                    lam=6 * 0.03 / 100,
                    axis=0,
                ),
                # in x direction
                OccupationDotJumps(
                    ratio=1,
                    scale=100 * 0.03 / 100,
                    lam=6 * 0.03 / 100,
                    axis=1,
                ),
            ],
            sensor_potential_distortions=[
                SensorPotentialPinkNoise(
                    sigma=UniformSamplingRange(
                        (1.8250268077765864e-12, 9.125134038882932e-05),
                    ),
                    fmin=0,
                ),
                SensorPotentialRTN(
                    scale=74.56704 * 0.03 / 100,
                    std=3.491734e-05,
                    height=2.53855325e-05,
                    ratio=1,
                ),
            ],
            sensor_response_distortions=[
                SensorResponseRTN(
                    scale=10000 * 0.03 / 100,
                    std=0.047453767599999995,
                    height=0.0152373696,
                    ratio=0.03,
                ),
                SensorResponseWhiteNoise(
                    sigma=NormalSamplingRange(
                        (1e-10, 0.003),
                        sampling_range=0.001,
                        std=0.0003,
                    ),
                ),
            ],
        )
        # This is only supported for 1D sweeps (only one resolution), but two resolutions were specified.
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                lead_transitions_noisy,
                lead_transitions_noiseless,
                err_msg="Lead Transitions should be identical between noisy and noise-free simulations",
            )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                occupations_noisy,
                occupations_noiseless,
                err_msg="Occupations should be identical between noisy and noise-free simulations",
            )

    def test_occupation_transition_blurring(self) -> None:
        """Method to test simulation with occupation transition blurring."""
        (
            occupations_noisy,
            lead_transitions_noisy,
            occupations_noiseless,
            lead_transitions_noiseless,
        ) = self.simulate_noisy_and_noiseless_measurement(
            occupation_distortions=[OccupationTransitionBlurringGaussian(0.75 * 0.03 / 100)],
            sensor_potential_distortions=[],
            sensor_response_distortions=[],
        )
        np.testing.assert_allclose(
            lead_transitions_noisy,
            lead_transitions_noiseless,
            err_msg="Lead Transitions should be identical between noisy and noise-free simulations",
        )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                occupations_noisy,
                occupations_noiseless,
                err_msg="Occupations should be identical between noisy and noise-free simulations",
            )

    def test_occupation_dot_jumps(self) -> None:
        """Method to test simulation with occupation dot jumps."""
        (
            occupations_noisy,
            lead_transitions_noisy,
            occupations_noiseless,
            lead_transitions_noiseless,
        ) = self.simulate_noisy_and_noiseless_measurement(
            occupation_distortions=[
                OccupationDotJumps(
                    ratio=1,
                    scale=100 * 0.03 / 100,
                    lam=6 * 0.03 / 100,
                    axis=0,
                ),  # in x direction
                OccupationDotJumps(
                    ratio=1,
                    scale=100 * 0.03 / 100,
                    lam=6 * 0.03 / 100,
                    axis=1,
                ),  # in y direction
            ],
            sensor_potential_distortions=[],
            sensor_response_distortions=[],
        )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                lead_transitions_noisy,
                lead_transitions_noiseless,
                err_msg="Lead Transitions should be identical between noisy and noise-free simulations",
            )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                occupations_noisy,
                occupations_noiseless,
                err_msg="Occupations should be identical between noisy and noise-free simulations",
            )

    def test_pink_noise(self) -> None:
        """Method to test simulation with pink noise."""
        (
            occupations_noisy,
            lead_transitions_noisy,
            occupations_noiseless,
            lead_transitions_noiseless,
        ) = self.simulate_noisy_and_noiseless_measurement(
            occupation_distortions=[],
            sensor_potential_distortions=[
                SensorPotentialPinkNoise(
                    sigma=UniformSamplingRange(
                        (1.8250268077765864e-12, 9.125134038882932e-05),
                    ),
                    fmin=0,
                ),
            ],
            sensor_response_distortions=[],
        )
        np.testing.assert_allclose(
            lead_transitions_noisy,
            lead_transitions_noiseless,
            err_msg="Lead Transitions should be identical between noisy and noise-free simulations",
        )
        np.testing.assert_allclose(
            occupations_noisy,
            occupations_noiseless,
            err_msg="Occupations should be identical between noisy and noise-free simulations",
        )

    def test_potential_rtn(self) -> None:
        """Method to test simulation with potential random telegraph noise."""
        (
            occupations_noisy,
            lead_transitions_noisy,
            occupations_noiseless,
            lead_transitions_noiseless,
        ) = self.simulate_noisy_and_noiseless_measurement(
            occupation_distortions=[],
            sensor_potential_distortions=[
                SensorPotentialRTN(
                    scale=74.56704 * 0.03 / 100,
                    std=3.491734e-05,
                    height=2.53855325e-05,
                    ratio=1 / 6,
                ),
            ],
            sensor_response_distortions=[],
        )
        np.testing.assert_allclose(
            lead_transitions_noisy,
            lead_transitions_noiseless,
            err_msg="Lead Transitions should be identical between noisy and noise-free simulations",
        )
        np.testing.assert_allclose(
            occupations_noisy,
            occupations_noiseless,
            err_msg="Occupations should be identical between noisy and noise-free simulations",
        )

    def test_response_rtn(self) -> None:
        """Method to test simulation with response random telegraph noise."""
        (
            occupations_noisy,
            lead_transitions_noisy,
            occupations_noiseless,
            lead_transitions_noiseless,
        ) = self.simulate_noisy_and_noiseless_measurement(
            occupation_distortions=[],
            sensor_potential_distortions=[],
            sensor_response_distortions=[
                SensorResponseRTN(
                    scale=10000 * 0.03 / 100,
                    std=0.047453767599999995,
                    height=0.0152373696,
                    ratio=0.03,
                ),
            ],
        )
        np.testing.assert_allclose(
            lead_transitions_noisy,
            lead_transitions_noiseless,
            err_msg="Lead Transitions should be identical between noisy and noise-free simulations",
        )
        np.testing.assert_allclose(
            occupations_noisy,
            occupations_noiseless,
            err_msg="Occupations should be identical between noisy and noise-free simulations",
        )

    def test_white_noise(self) -> None:
        """Method to test simulation with white noise."""
        (
            occupations_noisy,
            lead_transitions_noisy,
            occupations_noiseless,
            lead_transitions_noiseless,
        ) = self.simulate_noisy_and_noiseless_measurement(
            occupation_distortions=[],
            sensor_potential_distortions=[],
            sensor_response_distortions=[
                SensorResponseWhiteNoise(
                    sigma=NormalSamplingRange(
                        (1e-10, 0.003),
                        sampling_range=0.001,
                        std=0.0003,
                    ),
                ),
            ],
        )
        np.testing.assert_allclose(occupations_noisy, occupations_noiseless)
        np.testing.assert_allclose(lead_transitions_noisy, lead_transitions_noiseless)

    def test_noiseless(self) -> None:
        """Method to test simulation without noise."""
        # Define the voltage sweep ranges and resolution for the simulation
        sweep_range_g1 = self.sim.volt_limits_g1
        sweep_range_g2 = self.sim.volt_limits_g2
        resolution = np.array([100, 100])

        # set noise to None
        self.sim.occupation_distortions = None
        self.sim.sensor_potential_distortions = None
        self.sim.sensor_response_distortions = None

        # Perform the simulation with noise and noise-free data
        (
            _,
            occupations,
            lead_transitions,
            _,
        ) = self.sim.measure(
            sweep_range_g1,
            sweep_range_g2,
            resolution,
        )

        file_path = str(
            imp_files("tests.comparison_files").joinpath(
                "simulation_test_noiseless.npz",
            ),
        )

        saved_arrays = np.load(file_path, allow_pickle=True)

        np.testing.assert_allclose(
            occupations,
            saved_arrays["occupations"],
            atol=1e-13,
            err_msg="Occupations are not matching prior saved version!",
        )
        np.testing.assert_allclose(
            lead_transitions,
            saved_arrays["lead_transitions"],
            atol=1e-13,
            err_msg="Lead transitions are not matching prior saved version!",
        )
