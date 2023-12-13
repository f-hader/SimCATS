"""Module to test behavior of the simulation with different measurement directions."""
from unittest import TestCase

import numpy as np

from simcats.distortions import (
    OccupationTransitionBlurringGaussian,
    SensorPotentialPinkNoise,
    SensorPotentialRTN,
    SensorResponseRTN,
    SensorResponseWhiteNoise,
)
from simcats import Simulation
from simcats.support_functions import (
    NormalSamplingRange,
    UniformSamplingRange,
)
from tests._test_configs import test_configs


class DifferentMeasurementDirections(TestCase):
    """Class to test behavior of the simulation with different measurement directions."""

    def test_simulate_standard_case(self) -> None:
        """Test standard orientation/direction (x_start < x_stop & y_start < y_stop) of measurement."""
        config = test_configs["GaAs_v1"]

        double_dot_device = Simulation(**config)
        double_dot_device.occupation_distortions = [
            OccupationTransitionBlurringGaussian(0.75 * 0.03 / 100),
        ]
        double_dot_device.sensor_potential_distortions = [
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
                ratio=1 / 6,
            ),
        ]

        double_dot_device.sensor_response_distortions = [
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
        ]

        # standard case: x_start < x_stop & y_start < y_stop
        sweep_range_g1 = np.array([-0.05, 0.058])  # x_start < x_stop
        sweep_range_g2 = np.array([-0.05, 0.06])  # y_start < y_stop
        resolution = np.array(
            [100, 100],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        # run simulation
        (
            csd_standard_case,
            occupations_standard_case,
            lead_transitions_standard_case,
            _,
        ) = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )

        # Check that this test runs without errors
        assert True

    def test_xaxis_reverse_case(self) -> None:
        """Test reversed x direction (x_start > x_stop & y_start < y_stop) of measurement."""
        config = test_configs["GaAs_v1"]

        double_dot_device = Simulation(**config)
        double_dot_device.occupation_distortions = [
            OccupationTransitionBlurringGaussian(0.75 * 0.03 / 100),
        ]
        double_dot_device.sensor_potential_distortions = [
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
                ratio=1 / 6,
            ),
        ]

        double_dot_device.sensor_response_distortions = [
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
        ]

        # standard case: x_start < x_stop & y_start < y_stop
        sweep_range_g1 = np.array([-0.05, 0.058])  # x_start < x_stop
        sweep_range_g2 = np.array([-0.05, 0.06])  # y_start < y_stop
        resolution = np.array(
            [100, 100],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        # run simulation
        (
            csd_standard_case,
            occupations_standard_case,
            lead_transitions_standard_case,
            _,
        ) = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )

        # x-axis reverse case: x_start > x_stop & y_start < y_stop
        sweep_range_g1 = np.array([0.058, -0.05])  # x_start > x_stop
        sweep_range_g2 = np.array([-0.05, 0.06])  # y_start < y_stop
        resolution = np.array(
            [100, 100],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        # run simulation
        (
            csd_xaxis_reverse_case,
            occupations_xaxis_reverse_case,
            lead_transitions_xaxis_reverse_case,
            _,
        ) = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )

        occupations_xaxis_reverse_case_flipped = np.flip(
            occupations_xaxis_reverse_case,
            axis=1,
        )
        lead_transitions_xaxis_reverse_case_flipped = np.flip(
            lead_transitions_xaxis_reverse_case,
            axis=1,
        )

        np.testing.assert_almost_equal(
            occupations_xaxis_reverse_case_flipped,
            occupations_standard_case,
        )
        np.testing.assert_almost_equal(
            lead_transitions_xaxis_reverse_case_flipped,
            lead_transitions_standard_case,
        )

    def test_yaxis_reverse_case(self) -> None:
        """Test reversed y direction (x_start < x_stop & y_start > y_stop) of measurement."""
        config = test_configs["GaAs_v1"]

        double_dot_device = Simulation(**config)
        double_dot_device.occupation_distortions = [
            OccupationTransitionBlurringGaussian(0.75 * 0.03 / 100),
        ]
        double_dot_device.sensor_potential_distortions = [
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
                ratio=1 / 6,
            ),
        ]

        double_dot_device.sensor_response_distortions = [
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
        ]

        # standard case: x_start < x_stop & y_start < y_stop
        sweep_range_g1 = np.array([-0.05, 0.058])  # x_start < x_stop
        sweep_range_g2 = np.array([-0.05, 0.06])  # y_start < y_stop
        resolution = np.array(
            [100, 100],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        # run simulation
        (
            csd_standard_case,
            occupations_standard_case,
            lead_transitions_standard_case,
            _,
        ) = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )

        # y-axis reverse case: x_start < x_stop & y_start > y_stop
        sweep_range_g1 = np.array([-0.05, 0.058])  # x_start < x_stop
        sweep_range_g2 = np.array([0.06, -0.05])  # y_start > y_stop
        resolution = np.array(
            [100, 100],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        # run simulation
        (
            csd_yaxis_reverse_case,
            occupations_reverse_case,
            lead_transitions_reverse_case,
            _,
        ) = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )

        occupations_reverse_case_flipped = np.flip(occupations_reverse_case, axis=0)
        lead_transitions_reverse_case_flipped = np.flip(
            lead_transitions_reverse_case,
            axis=0,
        )

        np.testing.assert_almost_equal(
            occupations_reverse_case_flipped,
            occupations_standard_case,
        )
        np.testing.assert_almost_equal(
            lead_transitions_reverse_case_flipped,
            lead_transitions_standard_case,
        )

    def test_both_axis_reverse_case(self) -> None:
        """Test reversed x and y direction (x_start > x_stop & y_start > y_stop) of measurement."""
        config = test_configs["GaAs_v1"]

        double_dot_device = Simulation(**config)
        double_dot_device.occupation_distortions = [
            OccupationTransitionBlurringGaussian(0.75 * 0.03 / 100),
        ]
        double_dot_device.sensor_potential_distortions = [
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
                ratio=1 / 6,
            ),
        ]

        double_dot_device.sensor_response_distortions = [
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
        ]

        # standard case: x_start < x_stop & y_start < y_stop
        sweep_range_g1 = np.array([-0.05, 0.058])  # x_start < x_stop
        sweep_range_g2 = np.array([-0.05, 0.06])  # y_start < y_stop
        resolution = np.array(
            [100, 100],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        # run simulation
        (
            csd_standard_case,
            occupations_standard_case,
            lead_transitions_standard_case,
            _,
        ) = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )

        # both axis reverse case: x_start > x_stop & y_start > y_stop
        sweep_range_g1 = np.array([0.058, -0.05])  # x_start > x_stop
        sweep_range_g2 = np.array([0.06, -0.05])  # y_start > y_stop
        resolution = np.array(
            [100, 100],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        # run simulation
        (
            csd_both_axis_reverse_case,
            occupations_reverse_case,
            lead_transitions_reverse_case,
            _,
        ) = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )

        occupations_both_axis_reverse_case_flipped_x = np.flip(
            occupations_reverse_case,
            axis=1,
        )
        occupations_both_axis_reverse_case_flipped_both = np.flip(
            occupations_both_axis_reverse_case_flipped_x,
            axis=0,
        )

        lead_transitions_both_axis_reverse_case_flipped_x = np.flip(
            lead_transitions_reverse_case,
            axis=1,
        )
        lead_transitions_both_axis_reverse_case_flipped_both = np.flip(
            lead_transitions_both_axis_reverse_case_flipped_x,
            axis=0,
        )

        np.testing.assert_almost_equal(
            occupations_both_axis_reverse_case_flipped_both,
            occupations_standard_case,
        )
        np.testing.assert_almost_equal(
            lead_transitions_both_axis_reverse_case_flipped_both,
            lead_transitions_standard_case,
        )
