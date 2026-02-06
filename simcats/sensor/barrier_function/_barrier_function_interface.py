"""This module contains an interface that models the conductivity of a sensor dot barrier.

@author: b.papajewski
"""

__all__ = []

from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class BarrierFunctionInterface(ABC):
    """
    Interface that models the conductivity (similar to a pinch-off measurement) of a barrier.
    """

    @abstractmethod
    def get_value(self, potential: Union[float, np.ndarray]) -> float:
        """
        Returns the value of the barrier function for a given potential.

        Args:
            potential (Union[float, np.ndarray]): Potential for which the conductance of the barrier function is to be
                calculated.

        Returns:
            Union[float, np.ndarray]: Value of the barrier function for a given potential.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def pinch_off(self) -> float:
        """Potential value of the pinch off."""
        raise NotImplementedError

    @property
    @abstractmethod
    def fully_conductive(self) -> float:
        """Potential value of the point from which on the barrier vanishes and becomes fully conductive."""
        raise NotImplementedError
