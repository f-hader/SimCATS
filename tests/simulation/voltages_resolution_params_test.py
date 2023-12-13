"""Module to test the simulation with different and wrong values for the voltages and resolution parameters."""
from copy import deepcopy
from unittest import TestCase

import numpy as np
import pytest

from simcats import Simulation
from tests._test_configs import test_configs


class VoltagesResolutionParamsTests(TestCase):
    """Class to test the simulation with different and wrong values for the voltages and resolution parameters."""

    def test_fixed_one_voltage_two_resolutions(self) -> None:
        """Method to test the simulation with one of the voltages fixed but 2D resolution."""
        config = test_configs["GaAs_v1"]

        double_dot_device = Simulation(**config)

        sweep_range_g1 = np.array(
            [0.0, 0.0],
        )  # Ideally: x_start < x_stop (-0.05, 0.058)
        sweep_range_g2 = np.array(
            [-0.05, 0.06],
        )  # Ideally: y_start < y_stop (-0.05, 0.06)
        resolution = np.array(
            [100, 100],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        with pytest.raises(
            ValueError,
            match=r"At least one of the voltage ranges 'sweep_range_g1' and 'sweep_range_g2' defines a fixed voltage. .*",
        ):
            # run simulation
            csd, occupations, lead_transitions, metadata = double_dot_device.measure(
                sweep_range_g1=sweep_range_g1,
                sweep_range_g2=sweep_range_g2,
                resolution=resolution,
            )

    def test_sweep_range_not_inside_volt_limits(self) -> None:
        """Method to test the simulation with one of the sweep ranges outsides the voltage limits."""
        config = test_configs["GaAs_v1"]

        double_dot_device = Simulation(**config)

        sweep_range_g1 = np.array(
            [-0.05, 0.06],
        )  # Ideally: x_start < x_stop (-0.05, 0.058)
        sweep_range_g2 = np.array(
            [-0.05, 0.06],
        )  # Ideally: y_start < y_stop (-0.05, 0.06)
        resolution = np.array(
            [100, 100],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        with pytest.raises(
            ValueError,
            match=r"The voltages defined by sweep_range_g1 .* must be in the range of the limits .* volt_limits_g1 .*",
        ):
            # run simulation
            _, _, _, _ = double_dot_device.measure(
                sweep_range_g1=sweep_range_g1,
                sweep_range_g2=sweep_range_g2,
                resolution=resolution,
            )

    def test_fixed_one_voltage_two_resolutions_but_one_is_just_one(self) -> None:
        """Method to test the simulation with one of the voltages fixed and a 2D resolution where one value is one."""
        config = test_configs["GaAs_v1"]

        double_dot_device = Simulation(**config)

        sweep_range_g1 = np.array(
            [0.0, 0.0],
        )  # Ideally: x_start < x_stop (-0.05, 0.058)
        sweep_range_g2 = np.array(
            [-0.05, 0.06],
        )  # Ideally: y_start < y_stop (-0.05, 0.06)
        resolution = np.array(
            [1, 100],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        with pytest.raises(
            ValueError,
            match=r"At least one of the voltage ranges 'sweep_range_g1' and 'sweep_range_g2' defines a fixed voltage. .*",
        ):
            csd, occupations, lead_transitions, metadata = double_dot_device.measure(
                sweep_range_g1=sweep_range_g1,
                sweep_range_g2=sweep_range_g2,
                resolution=resolution,
            )

    def test_sweep_range_with_more_than_two_entries(self) -> None:
        """Method to test the simulation with more than two entries for one of the sweep ranges."""
        config = test_configs["GaAs_v1"]

        double_dot_device = Simulation(**config)

        sweep_range_g1 = np.array(
            [-0.05, 0.06, 0.08],
        )  # Ideally: x_start < x_stop (-0.05, 0.058)
        sweep_range_g2 = np.array(
            [-0.05, 0.06],
        )  # Ideally: y_start < y_stop (-0.05, 0.06)
        resolution = np.array(
            [100, 100],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        with pytest.raises(
            ValueError,
            match=r"The sweep ranges for the gates g1 and g2 must consist of exactly 2 values each. At least one sweep range violates this.",
        ):
            _, _, _, _ = double_dot_device.measure(
                sweep_range_g1=sweep_range_g1,
                sweep_range_g2=sweep_range_g2,
                resolution=resolution,
            )

    def test_resolution_with_more_than_two_entries(self) -> None:
        """Method to test the simulation with more than two entries for the resolution."""
        config = test_configs["GaAs_v1"]

        double_dot_device = Simulation(**config)

        sweep_range_g1 = np.array(
            [-0.05, 0.051],
        )  # Ideally: x_start < x_stop (-0.05, 0.058)
        sweep_range_g2 = np.array(
            [-0.05, 0.051],
        )  # Ideally: y_start < y_stop (-0.05, 0.06)
        resolution = np.array(
            [100, 100, 100],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        with pytest.raises(
            ValueError,
            match=r".* The resolution must either be a single value \(for 1D scans\) or contain two entries \(for 2D scans\).",
        ):
            _, _, _, _ = double_dot_device.measure(
                sweep_range_g1=sweep_range_g1,
                sweep_range_g2=sweep_range_g2,
                resolution=resolution,
            )

    def test_resolution_smaller_than_one(self) -> None:
        """Method to test the simulation with a resolution value less than one."""
        config = test_configs["GaAs_v1"]

        double_dot_device = Simulation(**config)

        sweep_range_g1 = np.array(
            [-0.05, 0.051],
        )  # Ideally: x_start < x_stop (-0.05, 0.058)
        sweep_range_g2 = np.array(
            [-0.05, 0.051],
        )  # Ideally: y_start < y_stop (-0.05, 0.06)
        resolution = np.array(
            [100, 0],
        )  # first entry is (plunger) gate 1 resolution and second entry (plunger) gate 2 resolution

        with pytest.raises(
            ValueError,
            match=r"The specified resolution \(\[100   0\]\) indicates that a 2D scan should be performed, but at least one of the two entries is smaller or equal to 1.",
        ):
            _, _, _, _ = double_dot_device.measure(
                sweep_range_g1=sweep_range_g1,
                sweep_range_g2=sweep_range_g2,
                resolution=resolution,
            )

    def test_missing_volt_limit_g1(self) -> None:
        """Test to check if the simulation runs even if volt_limits_g1 is missing."""
        config = deepcopy(test_configs["GaAs_v1"])

        del config["volt_limits_g1"]

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
        # with self.assertRaises(ValueError):
        _, _, _, _ = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )
        assert True

    def test_missing_volt_limit_g2(self) -> None:
        """Test to check if the simulation runs even if volt_limits_g2 is missing."""
        config = deepcopy(test_configs["GaAs_v1"])

        del config["volt_limits_g2"]

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
        # with self.assertRaises(ValueError):
        _, _, _, _ = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )
        assert True
