"""This module contains a generic implementation of the SensorInterface, used for simulating the sensor response for a
charge stability diagram (CSD) including the cross coupling between the sensor and double dot (plunger) gates.

@author: f.hader
"""

from copy import deepcopy
from typing import List, Union

import numpy as np

from simcats.sensor import SensorInterface, SensorPeakInterface

__all__ = []


class SensorGeneric(SensorInterface):
    """Generic implementation of the SensorInterface."""

    def __init__(
        self,
        sensor_peak_function: Union[SensorPeakInterface, List[SensorPeakInterface], None] = None,
        alpha_dot: np.ndarray = np.array([-1, -1]),
        alpha_gate: np.ndarray = np.array([0, 0]),
        offset_mu_sens: float = 0,
    ):
        """Initializes an object of the class for the simulation of the sensor response for a CSD.

        Args:
            sensor_peak_function (Union[SensorPeakInterface, List[SensorPeakInterface], None]): An implementation of the
                SensorPeakInterface. It is also possible to supply a list of such peaks, if a sensor with multiple peaks
                should be simulated. Default is None.
            alpha_dot (np.ndarray): Lever-arm to the dots. The values should be negative. Influences the strengths of
                the edges in the CSD. Default is np.array([-1, -1]).
            alpha_gate (np.ndarray): Lever-arm to the double dot (plunger) gates. The values should be positive. Default
                is np.array([0, 0]).
            offset_mu_sens (float): Electrochemical potential of the sensor dot for zero electrons in the dots and no
                applied voltage at the gates. Default is 0.
        """
        self.sensor_peak_function = sensor_peak_function
        self.alpha_dot = alpha_dot
        self.alpha_gate = alpha_gate
        self.offset_mu_sens = offset_mu_sens

    @property
    def sensor_peak_function(self) -> Union[SensorPeakInterface, List[SensorPeakInterface], None]:
        """Returns the current sensor peak function configuration of the sensor.

        This configuration can then be adjusted and is directly used as new configuration, as the object is
        returned as call by reference.

        Returns:
            list[SensorPeakInterface]: A list of SensorPeakInterface implementations.
        """
        return self.__sensor_peak_function

    @sensor_peak_function.setter
    def sensor_peak_function(self, sensor_peak_function: Union[SensorPeakInterface, List[SensorPeakInterface], None]):
        """Updates the sensor peak function configuration of the sensor according to the supplied values.

        Args:
            sensor_peak_function (Union[SensorPeakInterface, List[SensorPeakInterface], None]): An implementation of the
                SensorPeakInterface. It is also possible to supply a list of such peaks, if a sensor with multiple
                peaks should be simulated.
        """
        # check datatype of sensor_peak_function
        if isinstance(sensor_peak_function, SensorPeakInterface):
            self.__sensor_peak_function = [deepcopy(sensor_peak_function)]
        elif isinstance(sensor_peak_function, list) and all(
            isinstance(x, SensorPeakInterface) for x in sensor_peak_function
        ):
            self.__sensor_peak_function = deepcopy(sensor_peak_function)
        elif sensor_peak_function is None:
            self.__sensor_peak_function = None
        else:
            raise ValueError(
                "The provided sensor_peak_function configuration is not supported. Must be either an "
                "implementation of the SensorPeakInterface, a list of such, or None."
            )

    @property
    def alpha_dot(self) -> np.ndarray:
        """Returns the current alpha dot (dot lever-arm) configuration of the sensor.

        This configuration can then be adjusted and is directly used as new configuration, as the object is
        returned as call by reference.

        Returns:
            np.ndarray: Lever-arm to the dots. Influences the strengths of the edges in the CSD.
        """
        return self.__alpha_dot

    @alpha_dot.setter
    def alpha_dot(self, alpha_dot: np.ndarray):
        """Updates the alpha dot (dot lever-arm) configuration of the sensor according to the supplied values.

        Args:
            alpha_dot (np.ndarray): Lever-arm to the dots. The values should be negative. Influences the strengths of
                the edges in the CSD.
        """
        # check datatype of alpha_dot
        if not (isinstance(alpha_dot, np.ndarray) and alpha_dot.size == 2):
            raise ValueError("The provided alpha_dot configuration is not supported. Must be a numpy array of size 2.")
        self.__alpha_dot = deepcopy(alpha_dot)

    @property
    def alpha_gate(self) -> np.ndarray:
        """Returns the current alpha gate (gate lever-arm) configuration of the sensor.

        This configuration can then be adjusted and is directly used as new configuration, as the object is
        returned as call by reference.

        Returns:
            np.ndarray: Lever-arm to the double dot (plunger) gates.
        """
        return self.__alpha_gate

    @alpha_gate.setter
    def alpha_gate(self, alpha_gate: np.ndarray):
        """Updates the alpha gate (gate lever-arm) configuration of the sensor according to the supplied values.

        Args:
            alpha_gate (np.ndarray): Lever-arm to the double dot (plunger) gates. The values should be positive.
        """
        # check datatype of alpha_gate
        if not (isinstance(alpha_gate, np.ndarray) and alpha_gate.size == 2):
            raise ValueError("The provided alpha_gate configuration is not supported. Must be a numpy array of size 2.")
        self.__alpha_gate = deepcopy(alpha_gate)

    @property
    def offset_mu_sens(self) -> float:
        """Returns the current offset_mu_sens configuration of the sensor.

        This configuration can then be adjusted and set as new configuration.

        Returns:
            float: Electrochemical potential of the sensor dot for zero electrons in the dots and no applied voltage at
            the gates.
        """
        return self.__offset_mu_sens

    @offset_mu_sens.setter
    def offset_mu_sens(self, offset_mu_sens: float):
        """Updates the offset_mu_sens configuration of the sensor according to the supplied value.

        Args:
            offset_mu_sens (float): Electrochemical potential of the sensor dot for zero electrons in the dots and no
                applied voltage at the gates.
        """
        # check datatype of offset_mu_sens
        if not isinstance(offset_mu_sens, (float, int)):
            raise ValueError("The provided offset_mu_sens configuration is not supported. Must be a float value.")
        self.__offset_mu_sens = offset_mu_sens

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(sensor_peak_function={self.__sensor_peak_function}, alpha_dot={repr(self.__alpha_dot)}, alpha_gate={repr(self.__alpha_gate)}, offset_mu_sens={self.__offset_mu_sens})"
        )

    def sensor_response(self, mu_sens: np.ndarray) -> np.ndarray:
        """This function returns the sensor values which result from the given electrochemical potential.

        If no sensor peak is defined, the potential is returned (acting like a "linear sensor").

        Args:
            mu_sens (np.ndarray): The sensor potential, stored in a 2-dimensional numpy array with the axis mapping to
                the CSD axis.

        Returns:
            np.ndarray: The sensor response, calculated from the given potential. It is stored in a numpy array with the
            axis mapping to the CSD axis.
        """
        # if no sensor peak function is defined, the sensor potential is returned instead (acting like a "linear"
        # sensor)
        if self.__sensor_peak_function is None:
            return mu_sens
        else:
            return np.sum([p.sensor_function(mu_sens) for p in self.__sensor_peak_function], axis=0)

    def sensor_potential(
        self, occupations: np.ndarray, volt_limits_g1: np.ndarray, volt_limits_g2: np.ndarray
    ) -> np.ndarray:
        """Simulates the electrochemical potential at the sensing dot.

        This is done in dependency on the electron occupation in the double dot and the voltages applied at the double
        dot (plunger) gates.

        Args:
            occupations (np.ndarray): Occupation in left and right dot per applied voltage combination. The occupation
                numbers are stored in a 3-dimensional numpy array. The first two dimensions map to the axis of the CSD,
                while the third dimension indicates the dot of the corresponding occupation value.
            volt_limits_g1 (np.ndarray): Contains the beginning and ending of the swept range for the first (plunger)
                gate.
            volt_limits_g2 (np.ndarray): Contains the beginning and ending of the swept range for the second (plunger)
                gate.

        Returns:
            np.ndarray: The electrochemical potential at the sensing dot
        """
        # Voltage matrix for 2D scans
        if occupations.ndim == 3:
            voltages_g1 = np.linspace(volt_limits_g1[0], volt_limits_g1[1], num=occupations.shape[1])
            voltages_g2 = np.linspace(volt_limits_g2[0], volt_limits_g2[1], num=occupations.shape[0])
            voltages = [
                [[voltages_g1[j], voltages_g2[i]] for j in range(len(voltages_g1))] for i in range(len(voltages_g2))
            ]
            voltages = np.array(voltages)
        # Voltage matrix for 1D scans
        elif occupations.ndim == 2:
            voltages_g1 = np.linspace(volt_limits_g1[0], volt_limits_g1[1], num=occupations.shape[0])
            voltages_g2 = np.linspace(volt_limits_g2[0], volt_limits_g2[1], num=occupations.shape[0])
            voltages = [[voltages_g1[i], voltages_g2[i]] for i in range(len(voltages_g1))]
            voltages = np.array(voltages)
        mu_sens = occupations.dot(self.__alpha_dot) + voltages.dot(self.__alpha_gate) + self.__offset_mu_sens
        return mu_sens
