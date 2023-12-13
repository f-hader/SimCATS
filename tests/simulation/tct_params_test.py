"""Module to test the simulation with different types of tcp parameter inputs."""
from copy import deepcopy
from unittest import TestCase

import numpy as np

from simcats import Simulation
from tests._test_configs import test_configs


class TCTParameterTest(TestCase):
    """Class to test the simulation with different types of tcp parameter inputs."""

    def test_list_of_lists(self) -> None:
        """Test if the simulation runs with a list of list as tct parameter.

        Raises:
            ValueError: Raised when config['ideal_csd_config'].tct_params is not a list of 1d np arrays.
        """
        config = deepcopy(test_configs["GaAs_v1"])

        # cast each numpy array to list
        tct_params = list()
        for tct_param in config["ideal_csd_config"].tct_params:
            if isinstance(tct_param, np.ndarray):
                tct_params.append(tct_param.tolist())
            else:
                raise ValueError(
                    "config['ideal_csd_config'].tct_params should be a list of 1d np arrays"
                )
        config["ideal_csd_config"].tct_params = tct_params
        double_dot_device = Simulation(**config)

        sweep_range_g1 = np.array([-0.05, 0.058])
        sweep_range_g2 = np.array([-0.05, 0.06])
        resolution = np.array([300, 300])

        # run simulation
        _, _, _, _ = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )

        # just checking that this test runs without errors
        assert True

    def test_list_of_1D_np_arrays(self) -> None:
        """Test if the simulation runs with a list of 1D np arrays as tct parameter."""
        config = deepcopy(test_configs["GaAs_v1"])

        double_dot_device = Simulation(**config)

        sweep_range_g1 = np.array([-0.05, 0.058])
        sweep_range_g2 = np.array([-0.05, 0.06])
        resolution = np.array([300, 300])

        # run simulation
        _, _, _, _ = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )

        # just checking that this test runs without errors
        assert True

    def test_2D_np_array(self) -> None:
        """Test if the simulation runs with a 2D np array as tct parameter."""
        config = deepcopy(test_configs["GaAs_v1"])

        # cast outer list to np array
        config["ideal_csd_config"].tct_params = np.array(
            config["ideal_csd_config"].tct_params
        )
        double_dot_device = Simulation(**config)

        sweep_range_g1 = np.array([-0.05, 0.058])
        sweep_range_g2 = np.array([-0.05, 0.06])
        resolution = np.array([300, 300])

        # run simulation
        _, _, _, _ = double_dot_device.measure(
            sweep_range_g1=sweep_range_g1,
            sweep_range_g2=sweep_range_g2,
            resolution=resolution,
        )

        # just checking that this test runs without errors
        assert True
