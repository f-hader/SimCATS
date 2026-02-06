from copy import deepcopy
from unittest import TestCase

from tests._test_configs import test_configs

import numpy as np
from numpy import array

from simcats.distortions import (
    OccupationDotJumps,
    OccupationTransitionBlurringGaussian,
    SensorPotentialPinkNoise,
    SensorPotentialRTN,
    SensorResponseRTN,
    SensorResponseWhiteNoise,
)
from simcats.ideal_csd import IdealCSDGeometric
from simcats.sensor import SensorGeneric, SensorPeakLorentzian, SensorScanSensorGeneric, SensorRiseGLF
from simcats.sensor.barrier_function import BarrierFunctionGLF


def deep_equal(obj1, obj2):
    """
    Performs a deep equality check between two objects, which can handle nested data structures,
    custom objects, numpy arrays, and numeric types with tolerance for floating-point precision differences.
    This function ensures that data contained within complex and composite structures are correctly compared.

    Args:
        obj1: The first object to compare.
        obj2: The second object to compare.

    Returns:
        bool: True if both objects are deeply equal, including all nested data structures, False otherwise.
    """

    if obj1 is obj2:
        return True

    if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
        if obj1.shape != obj2.shape:
            return False

        return np.allclose(obj1, obj2)

    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if obj1.keys() != obj2.keys():
            return False
        return all(deep_equal(obj1[k], obj2[k]) for k in obj1)

    if isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        return all(deep_equal(x, y) for x, y in zip(obj1, obj2))

    if hasattr(obj1, "__dict__") and hasattr(obj2, "__dict__"):
        if type(obj1) is not type(obj2):
            return False
        return deep_equal(obj1.__dict__, obj2.__dict__)

    if isinstance(obj1, (int, float, np.number)) and isinstance(obj2, (int, float, np.number)):
        return np.isclose(obj1, obj2)

    return obj1 == obj2


class SensorSimulationTests(TestCase):
    def test_eval_sensor_generic(self) -> None:
        config = deepcopy(test_configs["GaAs_v1"])

        config["occupation_distortions"] = None
        config["sensor_potential_distortions"] = None
        config["sensor_response_distortions"] = None

        eval_config = eval(repr(config))

        assert deep_equal(config, eval_config)

    def test_eval_sensor_scan_generic(self) -> None:
        config = deepcopy(test_configs["GaAs_v2_extended_sensor"])

        config["occupation_distortions"] = None
        config["sensor_potential_distortions"] = None
        config["sensor_response_distortions"] = None

        eval_config = eval(repr(config))

        assert deep_equal(config, eval_config)


