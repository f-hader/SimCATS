"""Module to test the the functions calculate_all_bezier_anchors and initialize_tct_functions with different types of tcp parameter inputs."""
from copy import deepcopy
from unittest import TestCase

import numpy as np

from simcats.ideal_csd.geometric import (
    calculate_all_bezier_anchors,
    initialize_tct_functions,
)
from tests._test_configs import test_configs


class TCTParameterTest(TestCase):
    """Class to test the the functions calculate_all_bezier_anchors and initialize_tct_functions with different types of tcp parameter inputs."""

    def test_calculate_all_bezier_anchors_list_of_lists(self) -> None:
        """Test if calculate_all_bezier_anchors runs with a list of list as tct parameter.

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
        _ = calculate_all_bezier_anchors(tct_params)
        assert True

    def test_calculate_all_bezier_anchors_list_of_1D_np_arrays(self) -> None:
        """Test if calculate_all_bezier_anchors runs with a list of 1D np arrays as tct parameter."""
        config = deepcopy(test_configs["GaAs_v1"])

        tct_params = config["ideal_csd_config"].tct_params
        _ = calculate_all_bezier_anchors(tct_params)

        # just checking that this test runs without errors
        assert True

    def test_calculate_all_bezier_anchors_2D_np_array(self) -> None:
        """Test if calculate_all_bezier_anchors runs with a 2D np array as tct parameter."""
        config = deepcopy(test_configs["GaAs_v1"])

        # cast outer list to np array
        tct_params = np.array(config["ideal_csd_config"].tct_params)
        _ = calculate_all_bezier_anchors(tct_params)

        # just checking that this test runs without errors
        assert True

    def test_calculate_all_bezier_anchors_list(self) -> None:
        """Test if calculate_all_bezier_anchors runs with a list as tct parameter."""
        config = deepcopy(test_configs["GaAs_v1"])

        # cast first numpy array to list
        tct_param = config["ideal_csd_config"].tct_params[0].tolist()
        _ = calculate_all_bezier_anchors(tct_param)
        assert True

    def test_calculate_all_bezier_anchors_1D_np_array(self) -> None:
        """Test if calculate_all_bezier_anchors runs with a 1D np array as tct parameter."""
        config = deepcopy(test_configs["GaAs_v1"])

        # get first numpy
        tct_param = config["ideal_csd_config"].tct_params[0]
        _ = calculate_all_bezier_anchors(tct_param)

        # just checking that this test runs without errors
        assert True

    def test_calculate_all_bezier_anchors_2D_np_array_one_row(self) -> None:
        """Test if calculate_all_bezier_anchors runs with a 2D np array with just one row as tct parameter."""
        config = deepcopy(test_configs["GaAs_v1"])

        # get first numpy and add second dimension
        tct_param = np.array([config["ideal_csd_config"].tct_params[0]])
        _ = calculate_all_bezier_anchors(tct_param)

        # just checking that this test runs without errors
        assert True

    def test_initialize_tct_functions_list_of_lists(self) -> None:
        """Test if initialize_tct_functions runs with a list of list as tct parameter."""
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
        _ = initialize_tct_functions(tct_params)
        assert True

    def test_initialize_tct_functions_list_of_1D_np_arrays(self) -> None:
        """Test if initialize_tct_functions runs with a list of 1D np arrays as tct parameter."""
        config = deepcopy(test_configs["GaAs_v1"])

        tct_params = config["ideal_csd_config"].tct_params
        _ = initialize_tct_functions(tct_params)

        # just checking that this test runs without errors
        assert True

    def test_initialize_tct_functions_2D_np_array(self) -> None:
        """Test if initialize_tct_functions runs with a 2D np array as tct parameter."""
        config = deepcopy(test_configs["GaAs_v1"])

        # cast outer list to np array
        tct_params = np.array(config["ideal_csd_config"].tct_params)
        _ = initialize_tct_functions(tct_params)

        # just checking that this test runs without errors
        assert True

    def test_initialize_tct_functions_list(self) -> None:
        """Test if initialize_tct_functions runs with a list as tct parameter."""
        config = deepcopy(test_configs["GaAs_v1"])

        # cast first numpy array to list
        tct_param = config["ideal_csd_config"].tct_params[0].tolist()
        _ = initialize_tct_functions(tct_param)
        assert True

    def test_initialize_tct_functions_1D_np_array(self) -> None:
        """Test if initialize_tct_functions runs with a 1D np array as tct parameter."""
        config = deepcopy(test_configs["GaAs_v1"])

        # get first numpy
        tct_param = config["ideal_csd_config"].tct_params[0]
        _ = initialize_tct_functions(tct_param)

        # just checking that this test runs without errors
        assert True

    def test_initialize_tct_functions_2D_np_array_one_row(self) -> None:
        """Test if initialize_tct_functions runs with a 2D np array with just one row as tct parameter."""
        config = deepcopy(test_configs["GaAs_v1"])

        # get first numpy and add second dimension
        tct_param = np.array([config["ideal_csd_config"].tct_params[0]])
        _ = initialize_tct_functions(tct_param)

        # just checking that this test runs without errors
        assert True
