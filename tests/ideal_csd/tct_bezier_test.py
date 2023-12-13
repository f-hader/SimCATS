"""Module to test the tct_bezier function."""
try:
    from importlib.resources import files as imp_files
except ImportError:
    from importlib_resources import files as imp_files
from copy import deepcopy
from unittest import TestCase

import numpy as np
import pytest

from simcats.ideal_csd.geometric import tct_bezier
from tests._test_configs import test_configs


class TCTBezierTests(TestCase):
    """Class to test the tct_bezier function."""

    def test_left_linear_region(self) -> None:
        """Method to test if the left linear region of the tct_bezier method is actually linear."""
        # Specific tct_params for testing
        specific_tct_params = test_configs["GaAs_v1"]["ideal_csd_config"].tct_params[0]

        # calculate start point of the left linear region
        start_left = {
            "x": specific_tct_params[4] - specific_tct_params[0] * 2,
            "y": specific_tct_params[5]
            - specific_tct_params[2] * specific_tct_params[0] * 2,
        }
        end_left = {"x": specific_tct_params[4], "y": specific_tct_params[5]}

        # get x values of the left linear region
        num_points = 100
        x_eval_left = np.linspace(start_left["x"], end_left["x"], num_points)

        # get y_values of the left linear region by tct_bezier method
        y_values = tct_bezier(specific_tct_params, x_eval_left, max_peaks=1)

        # underlying linear function
        def linear_func(x: np.ndarray) -> np.ndarray:
            return start_left["y"] + specific_tct_params[2] * (x - start_left["x"])

        # get y_values of the left linear region by the underlying linear function
        y_values_opt = linear_func(x_eval_left)

        # compare y_values by taking the sum of the absolute difference and comparing the sum to 0
        np.testing.assert_allclose(
            np.abs(y_values - y_values_opt).sum(), 0, atol=0.000001
        )

    def test_right_linear_region(self) -> None:
        """Method to test if the right linear region of the tct_bezier method is actually linear."""
        # Specific tct_params for testing
        specific_tct_params = test_configs["GaAs_v1"]["ideal_csd_config"].tct_params[0]

        # calculate start point of the right linear region
        start_right = {"x": specific_tct_params[6], "y": specific_tct_params[7]}
        end_right = {
            "x": specific_tct_params[6] + specific_tct_params[1] * 2,
            "y": specific_tct_params[7]
            + specific_tct_params[3] * specific_tct_params[1] * 2,
        }

        # get x values of the right linear region
        num_points = 100
        x_eval_right = np.linspace(start_right["x"], end_right["x"], num_points)

        # get y_values of the right linear region by tct_bezier method
        y_values = tct_bezier(specific_tct_params, x_eval_right, max_peaks=1)

        # underlying linear function
        def linear_func(x: np.ndarray) -> np.ndarray:
            return start_right["y"] + specific_tct_params[3] * (x - start_right["x"])

        # get y_values of the right linear region by the underlying linear function
        y_values_opt = linear_func(x_eval_right)

        # compare y_values by taking the sum of the absolute difference and comparing the sum to 0
        np.testing.assert_allclose(
            np.abs(y_values - y_values_opt).sum(), 0, atol=0.000001
        )

    def test_whole_region(self) -> None:
        """Method to test the behavior of the tct_bezier method for the whole region."""
        # Specific tct_params for testing
        specific_tct_params = test_configs["GaAs_v1"]["ideal_csd_config"].tct_params[0]

        # calculate the whole region
        start_left_x = specific_tct_params[4] - specific_tct_params[0] * 2

        end_right_x = specific_tct_params[6] + specific_tct_params[1] * 2

        # Generate x values within the middle region
        num_points = 100
        x_eval = np.linspace(start_left_x, end_right_x, num_points)

        # Calculate expected y values using the tct_bezier function
        y_values = tct_bezier(specific_tct_params, x_eval, max_peaks=1)

        file_path = str(
            imp_files("tests.comparison_files").joinpath(
                "test_bezier_whole_region.npz",
            ),
        )
        expected_y_values = np.load(file_path, allow_pickle=True)

        np.testing.assert_allclose(y_values, expected_y_values["arr_0"])

    def test_max_peaks_none(self) -> None:
        """Method to test the behavior of the tct_bezier method if max peaks is equal to None. This is the only case curves should appear on the left side of our initial curve."""
        # Specific tct_params for testing
        specific_tct_params = test_configs["GaAs_v1"]["ideal_csd_config"].tct_params[0]

        # calculate the whole region
        start_left_x = specific_tct_params[4] - specific_tct_params[0] * 4
        end_right_x = specific_tct_params[6] + specific_tct_params[1] * 2

        # Generate x values within the middle region
        num_points = 100
        x_eval = np.linspace(start_left_x, end_right_x, num_points)

        # Calculate expected y values using the tct_bezier function
        y_values = tct_bezier(specific_tct_params, x_eval, max_peaks=None)

        file_path = str(
            imp_files("tests.comparison_files").joinpath(
                "test_max_peaks_none.npz",
            ),
        )
        expected_y_values = np.load(file_path, allow_pickle=True)

        np.testing.assert_allclose(y_values, expected_y_values["arr_0"])

    def test_main_bezier_not_x_eval(self) -> None:
        """Method to test if the main/first bezier is present in the x_eval or not."""
        # Specific tct_params for testing
        specific_tct_params = test_configs["GaAs_v1"]["ideal_csd_config"].tct_params[0]

        # Test x_eval values
        x_eval = np.linspace(
            specific_tct_params[6] + 0.01,
            specific_tct_params[6] + 0.1,
            100,
        )

        # Call the tct_bezier function
        y_values = tct_bezier(specific_tct_params, x_eval, max_peaks=5)

        file_path = str(
            imp_files("tests.comparison_files").joinpath(
                "test_main_bezier_not_x_eval.npz",
            ),
        )
        expected_y_values = np.load(file_path, allow_pickle=True)

        np.testing.assert_allclose(y_values, expected_y_values["arr_0"])

    def test_bezier_anchor_points_slightly_less_halfway_between_curves(self) -> None:
        """Method to test tct_bezier with anchors slightly less than halfway between curves."""
        # Specific tct_params for testing
        specific_tct_params = test_configs["GaAs_v1"]["ideal_csd_config"].tct_params[0]

        # setup coordinates for bezier curve nodes
        x_vals = np.array([specific_tct_params[4], 0, specific_tct_params[6]])
        y_vals = np.array([specific_tct_params[5], 0, specific_tct_params[7]])
        if x_vals[0] == x_vals[2] and y_vals[0] == y_vals[2]:
            # no interdot coupling region specified -> center anchor is equal to outer anchors
            x_vals[1] = x_vals[0]
            y_vals[1] = y_vals[0]
        else:
            # update center point by finding the intersection between left & right linear part
            x_vals[1] = (
                y_vals[0]
                - y_vals[2]
                + x_vals[2] * specific_tct_params[3]
                - x_vals[0] * specific_tct_params[2]
            ) / (specific_tct_params[3] - specific_tct_params[2])
            y_vals[1] = y_vals[0] + specific_tct_params[2] * (x_vals[1] - x_vals[0])

        center_anchor = {"x": x_vals[1], "y": y_vals[1]}
        new_right_anchor = {
            "x": center_anchor["x"] + specific_tct_params[1] * 0.49,
            "y": center_anchor["y"]
            + specific_tct_params[3] * specific_tct_params[1] * 0.49,
        }
        new_left_anchor = {
            "x": center_anchor["x"] - specific_tct_params[0] * 0.49,
            "y": center_anchor["y"]
            - specific_tct_params[2] * specific_tct_params[0] * 0.49,
        }

        new_tcp_params = deepcopy(specific_tct_params)
        new_tcp_params[4] = new_left_anchor["x"]
        new_tcp_params[5] = new_left_anchor["y"]
        new_tcp_params[6] = new_right_anchor["x"]
        new_tcp_params[7] = new_right_anchor["y"]

        x = np.linspace(specific_tct_params[4] - 0.1, specific_tct_params[6] + 0.1, 100)
        tct_bezier(new_tcp_params, x, max_peaks=None)
        assert True

    def test_bezier_anchor_points_slightly_over_halfway_between_curves(self) -> None:
        """Method to test tct_bezier with anchors slightly over halfway between curves."""
        # Specific tct_params for testing
        specific_tct_params = test_configs["GaAs_v1"]["ideal_csd_config"].tct_params[0]

        # setup coordinates for bezier curve nodes
        x_vals = np.array([specific_tct_params[4], 0, specific_tct_params[6]])
        y_vals = np.array([specific_tct_params[5], 0, specific_tct_params[7]])
        if x_vals[0] == x_vals[2] and y_vals[0] == y_vals[2]:
            # no interdot coupling region specified -> center anchor is equal to outer anchors
            x_vals[1] = x_vals[0]
            y_vals[1] = y_vals[0]
        else:
            # update center point by finding the intersection between left & right linear part
            x_vals[1] = (
                y_vals[0]
                - y_vals[2]
                + x_vals[2] * specific_tct_params[3]
                - x_vals[0] * specific_tct_params[2]
            ) / (specific_tct_params[3] - specific_tct_params[2])
            y_vals[1] = y_vals[0] + specific_tct_params[2] * (x_vals[1] - x_vals[0])

        center_anchor = {"x": x_vals[1], "y": y_vals[1]}
        new_right_anchor = {
            "x": center_anchor["x"] + specific_tct_params[1] * 0.51,
            "y": center_anchor["y"]
            + specific_tct_params[3] * specific_tct_params[1] * 0.51,
        }
        new_left_anchor = {
            "x": center_anchor["x"] - specific_tct_params[0] * 0.51,
            "y": center_anchor["y"]
            - specific_tct_params[2] * specific_tct_params[0] * 0.51,
        }

        new_tcp_params = deepcopy(specific_tct_params)
        new_tcp_params[4] = new_left_anchor["x"]
        new_tcp_params[5] = new_left_anchor["y"]
        new_tcp_params[6] = new_right_anchor["x"]
        new_tcp_params[7] = new_right_anchor["y"]

        x = np.linspace(specific_tct_params[4] - 0.1, specific_tct_params[6] + 0.1, 100)
        with pytest.raises(AssertionError):
            tct_bezier(new_tcp_params, x, max_peaks=None)
