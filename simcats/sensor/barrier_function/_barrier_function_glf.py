"""This module contains a class that models the conductivity of a barrier using the generalized logistic function (GLF).

@author: b.papajewski
"""

from typing import Union

import numpy as np

from simcats.sensor.barrier_function import BarrierFunctionInterface
from simcats.support_functions import glf, inverse_glf

__all__ = []


class BarrierFunctionGLF(BarrierFunctionInterface):
    """Implementation of the BarrierFunctionInterface that models a conductivity off of a barrier using a generalized logistic function (GLF).

    For further information see: https://en.wikipedia.org/wiki/Generalised_logistic_function
    """

    def __init__(self, pinch_off_percentage: float, fully_conductive_percentage: float, asymptote_left: float,
                 asymptote_right: float, growth_rate: float, asymmetry: float, shape_factor: float,
                 denominator_offset: float = 1, offset: float = 0):
        """
        Initializes an object of the class to represent the barrier function using the generalized logistic function.
        For further information see: https://en.wikipedia.org/wiki/Generalised_logistic_function

        Args:
            pinch_off_percentage (float): Percentage of the GLF height before which the barrier response is
                considered non-conductive. The height is defined as the absolute difference between the two asymptotes.
            fully_conductive_percentage (float): Percentage of the GLF height after which the barrier response is
                considered fully conductive. The height is defined as the absolute difference between the two
                asymptotes.
            asymptote_left (float): Originally called A. This parameter is the left horizontal asymptote of the
                function. Any rational number can be used as the left asymptote. This parameter may take any rational
                number.
            asymptote_right (float): Originally called K. Specifies the right horizontal asymptote of the function
                when denominator_offset=1. If asymptote_left=0 and denominator_offset=1 then this parameter is also
                called the carrying capacity. This parameter may take any rational number.
            growth_rate (float): Originally called B. The growth rate of the function. The value must be a float and can
                be any rational number. Be careful with negative values, because the function is mirrored on a vertical
                straight line for these. This line passes through the point where the potential equals `offset`.
            asymmetry (float): Originally called nu. This parameter introduces skew and affects symmetry. It also
                affects near which asymptote maximum growth occurs. The value of asymmetry must be a rational number
                greater than zero. \n
                - `asymmetry > 1`: the curve rises more gradually before the midpoint and more sharply after. \n
                - `asymmetry < 1`: the curve rises quickly early on and levels off more slowly.
            shape_factor (float): Originally called Q. is related to the value Y(0) and adjusts the curve's value at the
                y-intercept. Thereby it changes the shape of the function without changing the asymptotes. The shape
                factor can be any rational number.
            denominator_offset (float): Originally called C. A constant added to the denominator inside the power.
                Controls the initial level of the denominator.This parameter must be a rational number. It typically
                takes a value of 1. Otherwise, the upper asymptote is \n
                asymptote_left + (asymptote_right-asymptote_left)/(denominator_offset^(1/asymmetry)).
            offset (float): Potential offset that shifts the GLF starting from the zero point. If the offset is
                positive, the function is shifted to the right and if it is negative, it is shifted to the left. Default
                is 0.
        """
        self.pinch_off_percentage = pinch_off_percentage
        self.fully_conductive_percentage = fully_conductive_percentage
        self.asymptote_left = asymptote_left
        self.asymptote_right = asymptote_right
        self.growth_rate = growth_rate
        self.asymmetry = asymmetry
        self.shape_factor = shape_factor
        self.denominator_offset = denominator_offset
        self.offset = offset

    def get_value(self, potential: Union[float, np.ndarray]) -> float:
        """Returns the value of the barrier function for a given potential.

        Args:
            potential (Union[float, np.ndarray]): Potential for which the conductance of the barrier function is to be
                calculated.

        Returns:
             Union[float, np.ndarray]: Value of the barrier function for a given potential.
        """
        return glf(potential=potential, asymptote_left=self.asymptote_left, asymptote_right=self.asymptote_right,
                   growth_rate=self.growth_rate, asymmetry=self.asymmetry, shape_factor=self.shape_factor,
                   denominator_offset=self.denominator_offset, offset=self.offset)

    @property
    def pinch_off(self) -> float:
        """Potential value of the pinch off."""
        return inverse_glf(
            value=np.abs(self.asymptote_right - self.asymptote_left) * self.pinch_off_percentage + np.min(
                [self.asymptote_left, self.asymptote_right]),
            asymptote_left=self.asymptote_left,
            asymptote_right=self.asymptote_right,
            growth_rate=self.growth_rate,
            asymmetry=self.asymmetry,
            shape_factor=self.shape_factor,
            denominator_offset=self.denominator_offset,
            offset=self.offset
        )

    @property
    def pinch_off_percentage(self) -> float:
        """Percentage of the GLF height before which the barrier response is considered non-conductive. The height is
            defined as the absolute difference between the two asymptotes."""
        return self.__pinch_off_percentage

    @pinch_off_percentage.setter
    def pinch_off_percentage(self, pinch_off_percentage: float) -> None:
        """Updates the pinch of percentage.

        Args:
            pinch_off_percentage (float): Potential value of the pinch off
        """
        self.__pinch_off_percentage = pinch_off_percentage

    @property
    def fully_conductive(self) -> float:
        """Potential value of the point from which on the barrier vanishes and becomes fully conductive."""
        return inverse_glf(
            value=np.abs(self.asymptote_right - self.asymptote_left) * self.fully_conductive_percentage + np.min(
                [self.asymptote_left, self.asymptote_right]),
            asymptote_left=self.asymptote_left,
            asymptote_right=self.asymptote_right,
            growth_rate=self.growth_rate,
            asymmetry=self.asymmetry,
            shape_factor=self.shape_factor,
            denominator_offset=self.denominator_offset,
            offset=self.offset
        )

    @property
    def fully_conductive_percentage(self) -> float:
        """Percentage of the GLF height after which the barrier response is considered fully conductive. The height is
            defined as the absolute difference between the two asymptotes."""
        return self.__fully_conductive_percentage

    @fully_conductive_percentage.setter
    def fully_conductive_percentage(self, fully_conductive_percentage: float) -> None:
        """Updates the fully conductive percentage.

        Args:
            fully_conductive_percentage (float): Potential value of the fully conductive point.
        """
        self.__fully_conductive_percentage = fully_conductive_percentage

    @property
    def asymptote_left(self):
        """Originally called A. This parameter is the left horizontal asymptote of the function. Any rational number can
        be used as the left asymptote. This parameter may take any rational number.
        """
        return self.__asymptote_left

    @asymptote_left.setter
    def asymptote_left(self, asymptote_left: float) -> None:
        """Updates the left horizontal asymptote of the GLF.

        Args:
            asymptote_left (float): Left horizontal asymptote of the GLF. The value must be a float and can be any
                rational number.
        """
        self.__asymptote_left = asymptote_left

    @property
    def asymptote_right(self):
        """Originally called K. Specifies the right horizontal asymptote of the function when denominator_offset=1. If
        asymptote_left=0 and denominator_offset=1 then this parameter is also called the carrying capacity. This
        parameter may take any rational number.
        """
        return self.__asymptote_right

    @asymptote_right.setter
    def asymptote_right(self, asymptote_right: float) -> None:
        """Updates the right horizontal asymptote of the GLF.

        Args:
            asymptote_right (float): Right horizontal asymptote of the GLF. The value must be a float and can be any
                rational number.
        """
        self.__asymptote_right = asymptote_right

    @property
    def growth_rate(self):
        """Originally called B. The growth rate of the function. The value must be a float and can´be any rational
        number. Be careful with negative values, because the function is mirrored on a vertical straight line for these.
        This line passes through the point where the potential equals `offset`.
        """
        return self.__growth_rate

    @growth_rate.setter
    def growth_rate(self, growth_rate: float) -> None:
        """Updates the growth rate of the GLF.

        Args:
            growth_rate (float): Growth rate of the GLF. The value must be a float and can be any rational number.
        """
        self.__growth_rate = growth_rate

    @property
    def asymmetry(self):
        """Originally called nu. This parameter introduces skew and affects symmetry. It also affects near which
        asymptote maximum growth occurs. The value of asymmetry must be a rational number greater than zero. \n
        - If `asymmetry > 1`: the curve rises more gradually before the midpoint and more sharply after. \n
        - If `asymmetry < 1`: the curve rises quickly early on and levels off more slowly.
        """
        return self.__asymmetry

    @asymmetry.setter
    def asymmetry(self, asymmetry: float) -> None:
        """Updates asymmetry, which affects near which asymptote maximum growth occurs. The value of asymmetry must be
        a rational number greater than zero.

        Args:
            asymmetry (float): Asymmetry of the GLF. The value must be a float and can be any rational number greater
                than zero.
        """
        if asymmetry <= 0:
            raise ValueError(f"asymmetry should be positive, but was {asymmetry}")
        self.__asymmetry = asymmetry

    @property
    def shape_factor(self):
        """Originally called Q. is related to the value Y(0) and adjusts the curve’s value at the y-intercept. Thereby
        it changes the shape of the function without changing the asymptotes. The shape factor can be any rational
        number."""
        return self.__shape_factor

    @shape_factor.setter
    def shape_factor(self, shape_factor: float) -> None:
        """Updates shape_factor, which is related the value Y(0). The value can be any rational number.

        Args:
            shape_factor (float): Shape factor of the GLF. The value must be a float and can be any rational number.
        """
        self.__shape_factor = shape_factor

    @property
    def denominator_offset(self) -> float:
        """Originally called C. A constant added to the denominator inside the power. Controls the initial level of the
        denominator.This parameter must be a rational number. It typically takes a value of 1. Otherwise, the upper
        asymptote is \n
        asymptote_left + (asymptote_right-asymptote_left)/(denominator_offset^(1/asymmetry)).
        """
        return self.__denominator_offset

    @denominator_offset.setter
    def denominator_offset(self, denominator_offset: float) -> None:
        """Originally called C. A constant added to the denominator inside the power. Controls the initial level of the
        denominator.This parameter must be a rational number. It typically takes a value of 1. Otherwise, the upper
        asymptote is \n
        asymptote_left + (asymptote_right-asymptote_left)/(denominator_offset^(1/asymmetry)).

        Args:
            denominator_offset (float): Denominator offset of the GLF. The value must be a float and can be any rational
                number.
       """
        self.__denominator_offset = denominator_offset

    @property
    def offset(self) -> float:
        """Attribute that shifts the GLF starting from the zero point. If the offset is positive, the function is
        shifted to the right and if it is negative, it is shifted to the left. Default is 0.
        """
        return self.__offset

    @offset.setter
    def offset(self, offset: float) -> None:
        """Updates the potential offset of the GLF. This value shifts the GLF in relation to the zero point.

        Args:
            offset (float): Potential offset of the GLF. The value must be a float and can be any rational number.
        """
        self.__offset = offset

    def __repr__(self):
        return (
                self.__class__.__name__
                + f"(pinch_off_percentage={self.pinch_off_percentage}, "
                  f"fully_conductive_percentage={self.fully_conductive_percentage}, "
                  f"asymptote_left={self.asymptote_left}, asymptote_right={self.asymptote_right}, "
                  f"growth_rate={self.growth_rate}, asymmetry={self.asymmetry}, shape_factor={self.shape_factor}, "
                  f"denominator_offset={self.denominator_offset}, offset={self.offset})"
        )
