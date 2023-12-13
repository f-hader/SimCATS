"""Module to test the simulation without tunnel coupling (setting all the left and right bezier anchors to identical values)."""
from copy import deepcopy
from unittest import TestCase

import numpy as np
import pytest

from simcats import Simulation
from tests._test_configs import test_configs


class TunnelCouplingTests(TestCase):
    """Class to test the simulation without tunnel coupling (setting all the left and right bezier anchors to identical values)."""

    def test_tunnel_coupling(self) -> None:
        """Test that the occupation consists only of integers if tunnel coupling is off (left bezier anchor==right bezier anchor)."""
        config = deepcopy(test_configs["GaAs_v1"])

        # set right anchor identically to left anchor
        for i in range(len(config["ideal_csd_config"].tct_params)):
            config["ideal_csd_config"].tct_params[i][6] = config["ideal_csd_config"].tct_params[i][4]
            config["ideal_csd_config"].tct_params[i][7] = config["ideal_csd_config"].tct_params[i][5]

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
        _, occupations, _, _ = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )

        # check if occupations are only integers
        assert pytest.approx((occupations.astype(int) - occupations).sum()) == 0
