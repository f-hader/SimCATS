"""This module contains functions and a class for GLF based sensor rise functions.

@author: b.papajewski
"""

import numpy as np

from simcats.sensor import SensorRiseInterface
from simcats.support_functions._generalized_logistic_function import glf, inverse_glf

__all__ = []


class SensorRiseGLF(SensorRiseInterface):
    """This class realizes the SensorRiseInterface using a GLF.

    For further information on the GLF see: https://en.wikipedia.org/wiki/Generalised_logistic_function.
    """

    def __init__(self,mu0: float, asymptote_left: float, asymptote_right: float, growth_rate: float, asymmetry: float,
                 shape_factor: float, denominator_offset: float = 1, fully_conductive_percentage: float = 0.99):
        """Creates a new GLF sensor rise function.

        Args:
            mu0 (float): Potential shift of the GLF from the zero point. If the offset is positive, the function is
                shifted to the right and if it is negative, it is shifted to the left.
            asymptote_left (float): Originally called A. This parameter is the left horizontal asymptote of the
                function. Any rational number can be used as the left asymptote.
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
            shape_factor (float): Originally called Q. is related to the value Y(0) and adjusts the curve’s value at the
                y-intercept. Thereby it changes the shape of the function without changing the asymptotes. The shape
                factor can be any rational number.
            denominator_offset (float): Originally called C. A constant added to the denominator inside the power.
                Controls the initial level of the denominator.This parameter must be a rational number. It typically
                takes a value of 1. Otherwise, the upper asymptote is \n
                asymptote_left + (asymptote_right-asymptote_left)/(denominator_offset^(1/asymmetry)).
        """
        self.mu0 = mu0
        self.asymptote_left = asymptote_left
        self.asymptote_right = asymptote_right
        self.growth_rate = growth_rate
        self.asymmetry = asymmetry
        self.shape_factor = shape_factor
        self.denominator_offset = denominator_offset
        self.fully_conductive_percentage = fully_conductive_percentage

    @property
    def mu0(self):
        """ Potential shift of the GLF from the zero point. If mu0 is positive, the function is shifted to the right
        and if it is negative, it is shifted to the left.
        """
        return self.__mu0

    @mu0.setter
    def mu0(self, mu0: float) -> None:
        """Updates mu0, which specifies the shift of the GLF from the zero point. The value of mu0 can be any rational
        number.

        Args:
            mu0 (float): The shift of the GLF from the zero point.
        """
        self.__mu0 = mu0

    @property
    def asymptote_left(self):
        """Originally called A. This parameter is the left horizontal asymptote of the function. Any rational number can
        be used as the left asymptote.
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
        """Originally called B. The growth rate of the function. The value must be a float and can be any rational
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
        - `asymmetry > 1`: the curve rises more gradually before the midpoint and more sharply after. \n
        - `asymmetry < 1`: the curve rises quickly early on and levels off more slowly.
        """
        return self.__asymmetry

    @asymmetry.setter
    def asymmetry(self, asymmetry: float) -> None:
        """Updates asymmetry, which affects near which asymptote maximum growth occurs. The value of asymmetry must be
        a rational number greater than zero.

        Args:
            asymmetry (float): The asymmetry of the GLF, which affects near which asymptote maximum growth occurs. The
                value of asymmetry must be a rational number greater than zero.
        """
        if asymmetry <= 0:
            raise ValueError(f"asymmetry should be positive, but was {asymmetry}")
        self.__asymmetry = asymmetry

    @property
    def shape_factor(self):
        """Originally called Q. is related to the value Y(0) and adjusts the curve’s value at the y-intercept. Thereby
        it changes the shape of the function without changing the asymptotes. The shape factor can be any rational
        number.
        """
        return self.__shape_factor

    @shape_factor.setter
    def shape_factor(self, shape_factor: float) -> None:
        """Updates shape_factor, which is related the value Y(0). The value can be any rational number.

        Args:
            shape_factor (float): The shape_factor of the GLF, which is related the value Y(0). The value can be any
                rational number.
        """
        self.__shape_factor = shape_factor

    @property
    def denominator_offset(self) -> float:
        """Originally called C. A constant added to the denominator inside the power. Controls the initial level of the
        denominator. This parameter must be a rational number. It typically takes a value of 1. Otherwise, the upper
        asymptote is asymptote_left + (asymptote_right-asymptote_left)/(denominator_offset^(1/asymmetry)).
        """
        return self.__denominator_offset

    @denominator_offset.setter
    def denominator_offset(self, denominator_offset: float) -> None:
        """Originally called C. A constant added to the denominator inside the power. Controls the initial level of the
        denominator. This parameter must be a rational number. It typically takes a value of 1. Otherwise, the upper
        asymptote is asymptote_left + (asymptote_right-asymptote_left)/(denominator_offset^(1/asymmetry)).

        Args:
            denominator_offset (float): A constant added to the denominator inside the power. Controls the initial level
                of the denominator. This parameter must be a rational number.
        """
        self.__denominator_offset = denominator_offset

    @property
    def fully_conductive(self) -> float:
        """Potential value of the point at which the sensor rise reaches its maximum."""

        return inverse_glf(
            value=(self.asymptote_right - self.asymptote_left) * self.fully_conductive_percentage + self.asymptote_left,
            asymptote_left=self.asymptote_left,
            asymptote_right=self.asymptote_right,
            growth_rate=self.growth_rate,
            asymmetry=self.asymmetry,
            shape_factor=self.shape_factor,
            denominator_offset=self.denominator_offset,
            offset=self.mu0
        )

    def sensor_function(self, mu_sens: np.ndarray, offset: float) -> np.ndarray:
        """Returns the sensor rise function values at the given electrochemical potentials.

        Args:
            mu_sens (np.ndarray): The sensor potential, stored in a numpy array with the axis mapping to the scan axis.
            offset (float): Potential value to which the fully conductive point is shifted.

        Returns:
            np.ndarray: The sensor response (for the corresponding peak), calculated from the given potential. It is
            stored in a numpy array with the axis mapping to the scan axis.
        """
        return glf(potential=mu_sens,
                   asymptote_left=self.asymptote_left,
                   asymptote_right=self.asymptote_right,
                   growth_rate=self.growth_rate,
                   asymmetry=self.asymmetry,
                   shape_factor=self.shape_factor,
                   denominator_offset=self.denominator_offset,
                   offset= -self.fully_conductive + offset)

    def __repr__(self):
        return (
                self.__class__.__name__
                + f"(mu0={self.mu0}, asymptote_left={self.asymptote_left}, asymptote_right={self.asymptote_right}, "
                  f"growth_rate={self.growth_rate}, asymmetry={self.asymmetry}, shape_factor={self.shape_factor}, "
                  f"denominator_offset={self.denominator_offset}, "
                  f"fully_conductive_percentage={self.fully_conductive_percentage})"
        )

