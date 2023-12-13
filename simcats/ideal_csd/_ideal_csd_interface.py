"""This module defines an interface for functions for simulating the ideal data for a charge stability diagram (CSD)
in dependency of the double dot (plunger) gate voltages.

@author: f.hader
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np

__all__ = []


class IdealCSDInterface(ABC):
    """Interface for functions, which are used for the simulation of ideal CSD data."""

    @abstractmethod
    def get_csd_data(
        self, volt_limits_g1: np.ndarray, volt_limits_g2: np.ndarray, resolution: Union[int, np.ndarray] = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Method which has to be implemented in every object used as ideal data function during the simulation of CSDs.

        It is used to retrieve ideal data (occupation numbers and a lead transition mask) for given gate voltages.
        The parameters specifying the structure / system should be attributes of this class.

        Retrieve ideal data (occupation numbers and a lead transition mask) for given gate voltages.

        Args:
            volt_limits_g1 (np.ndarray): Voltage sweep range of (plunger) gate 1 (second-/x-axis). \n
                Example: \n
                [min_V1, max_V1]
            volt_limits_g2 (np.ndarray): Voltage sweep range of (plunger) gate 2 (first-/y-axis). \n
                Example: \n
                [min_V2, max_V2]
            resolution (np.ndarray): Desired resolution (in pixels) for the gates. If only one value is supplied, a 1D
                sweep is performed. Then, both gates are swept simultaneously. Default is np.array([100, 100]). \n
                Example: \n
                [res_g1, res_g2]

        Returns:
            Tuple[np.ndarray, np.ndarray]: Occupation numbers and lead transition mask. The occupation numbers are
            stored in a 3-dimensional numpy array. The first two dimensions map to the axis of the CSD, while the third
            dimension indicates the dot of the corresponding occupation value. The label mask for the lead-to-dot
            transitions is stored in a 2-dimensional numpy array with the axis mapping to the CSD axis.
        """
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__ + str(self.__dict__)
