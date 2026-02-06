"""This module contains a class that models the conductivity of a barrier using multiple generalized logistic functions (GLFs).

@author: b.papajewski
"""

from typing import Union, Tuple

import numpy as np

from simcats.sensor.barrier_function import BarrierFunctionInterface
from simcats.support_functions import multi_glf

__all__ = []


class BarrierFunctionMultiGLF(BarrierFunctionInterface):
    """
    Implementation of the BarrierFunctionInterface that models a conductivity off of a barrier using a combination of
    multiple generalized logistic functions (GLFs).
    For further information see: https://en.wikipedia.org/wiki/Generalised_logistic_function
    """

    def __init__(self, pinch_off: float, fully_conductive: float, *params: float):
        """Initializes an object of the class to represent the barrier function using several generalized logistic functions.

        For further information see: https://en.wikipedia.org/wiki/Generalised_logistic_function

        The number of GLFs is specified by the number of parameters. To do this, the parameter count must be
        divisible by seven and a GLF is added for every seven other parameters.

        Each GLF has the following parameters: \n
        - asymptote_left (float): Originally called A. This parameter is the left horizontal asymptote of the function.
          Any rational number can be used as the left asymptote.
        - asymptote_right (float): Originally called K. Specifies the right horizontal asymptote of the function when
          denominator_offset=1. If asymptote_left=0 and denominator_offset=1 then this parameter is also called the
          carrying capacity. This parameter may take any rational number.
        - growth_rate (float): Originally called B. The growth rate of the function. The value must be a float and can
          be any rational number. Be careful with negative values, because the function is mirrored on a vertical
          straight line for these. This line passes through the point where the potential equals `offset`.
        - asymmetry (float): Originally called nu. This parameter introduces skew and affects symmetry. It also affects
          near which asymptote maximum growth occurs. The value of asymmetry must be a rational number greater than
          zero. \n
          - If `asymmetry > 1`: the curve rises more gradually before the midpoint and more sharply after. \n
          - If `asymmetry < 1`: the curve rises quickly early on and levels off more slowly.
        - shape_factor (float): Originally called Q. is related to the value Y(0) and adjusts the curve’s value at the
          y-intercept. Thereby it changes the shape of the function without changing the asymptotes. The shape
          factor can be any rational number.
        - denominator_offset (float): Originally called C. A constant added to the denominator inside the power.
          Controls the initial level of the denominator.This parameter must be a rational number. It typically takes a
          value of 1. Otherwise, the upper asymptote is \n
          asymptote_left + (asymptote_right - asymptote_left) / (denominator_offset^(1 / asymmetry)).
        - offset (float): Potential offset, that shifts the function starting from the zero point. If the offset is
          positive, the function is shifted to the right and if it is negative, it is shifted to the left.

        Args:
            pinch_off (float): Potential at which the pinch-off of the barrier is located.
            fully_conductive (float): Potential of the point from which on the barrier function is fully conductive.
            *params: Additional positional arguments representing the GLF parameters. The number of additional
                parameters must be divisible by seven and determines the number of GLFs that are used for the Multi-GLF.
                All parameters consist of sequential groups of seven floats that each represent a single GLF. All
                individual parameters are described above and are in the same order as they are described.
        """
        self.pinch_off = pinch_off
        self.fully_conductive = fully_conductive
        self.params = params

    def get_value(self, potential: Union[float, np.ndarray]) -> float:
        """Returns the value of the barrier function for a given potential.

        Args:
            potential (Union[float, np.ndarray]): Potential for which the conductance of the barrier function is to be
                calculated.

        Returns:
            Union[float, np.ndarray]: Value of the barrier function for a given potential.
        """
        return multi_glf(potential, *self.__params)

    @property
    def pinch_off(self) -> float:
        """Potential value of the pinch off."""
        return self.__pinch_off

    @pinch_off.setter
    def pinch_off(self, pinch_off: float) -> None:
        """Updates the potential value of the pinch off.

        Args:
            pinch_off (float): Potential value of the pinch off
        """
        self.__pinch_off = pinch_off

    @property
    def fully_conductive(self) -> float:
        """Potential value of the point from which on the barrier vanishes and becomes fully conductive."""
        return self.__fully_conductive

    @fully_conductive.setter
    def fully_conductive(self, fully_conductive: float) -> None:
        """Updates the fully conductive point, the potential value at which the barrier vanishes.

        Args:
            fully_conductive (float): Potential value of the fully conductive point.
        """
        self.__fully_conductive = fully_conductive

    @property
    def params(self) -> Tuple[float]:
        """Parameters of the multiple GLFs.
        The number of GLFs is specified by the number of parameters. To do this, the parameter count must be divisible
        by seven and a GLF is added for every seven other parameters.

        Each GLF has the following parameters: \n
        - asymptote_left (float): Originally called A. This parameter is the left horizontal asymptote of the function.
          Any rational number can be used as the left asymptote.
        - asymptote_right (float): Originally called K. Specifies the right horizontal asymptote of the function when
          denominator_offset=1. If asymptote_left=0 and denominator_offset=1 then this parameter is also called the
          carrying capacity. This parameter may take any rational number.
        - growth_rate (float): Originally called B. The growth rate of the function. The value must be a float and can
          be any rational number. Be careful with negative values, because the function is mirrored on a vertical
          straight line for these. This line passes through the point where the potential equals `offset`.
        - asymmetry (float): Originally called nu. This parameter introduces skew and affects symmetry. It also affects
          near which asymptote maximum growth occurs. The value of asymmetry must be a rational number greater than
          zero. \n
          - If `asymmetry > 1`: the curve rises more gradually before the midpoint and more sharply after. \n
          - If `asymmetry < 1`: the curve rises quickly early on and levels off more slowly.
        - shape_factor (float): Originally called Q. is related to the value Y(0) and adjusts the curve’s value at the
          y-intercept. Thereby it changes the shape of the function without changing the asymptotes. The shape factor can
          be any rational number.
        - denominator_offset (float): Originally called C. A constant added to the denominator inside the power.
          Controls the initial level of the denominator. This parameter must be a rational number. It typically takes a
          value of 1. Otherwise, the upper asymptote is \n
          asymptote_left + (asymptote_right - asymptote_left) / (denominator_offset^(1 / asymmetry)).
        - offset (float): Potential offset, that shifts the function starting from the zero point. If the offset is
          positive, the function is shifted to the right and if it is negative, it is shifted to the left.

        Returns:
            The parameters of the multiple GLFs as described above.
        """
        return self.__params

    @params.setter
    def params(self, params: Tuple[float]) -> None:
        """Updates the parameter list of a multi GLF. A more detailed description of every parameter can be found above.

        Args:
            params (Tuple[float]): Parameters of the multiple GLFs as described above.
        """
        if len(params) % 7 != 0:
            raise ValueError("The parameter list must be divisible by 7")
        self.__params = params

    def __repr__(self):
        return (
                self.__class__.__name__
                + f"(pinch_off={self.pinch_off}, fully_conductive={self.fully_conductive}, *params={self.__params}"
        )
