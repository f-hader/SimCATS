"""This module contains an implementation of the SensorScanSensorInterface for simulating sensor scans.
This is a sensor that also enables sensor scans to be measured. The sensor function of this sensor is able to use
generic sensor peaks.

@author: b.papajewski
"""

import copy
import numbers
import sys
import warnings
from copy import deepcopy
from typing import List, Union, Dict, Tuple, Optional

import numpy as np

from simcats.sensor import SensorPeakInterface, SensorScanSensorInterface, SensorRiseInterface
from simcats.sensor.barrier_function._barrier_function_interface import BarrierFunctionInterface
from simcats.sensor.deformation import SensorPeakDeformationInterface
from simcats.support_functions import signed_dist_points_line, pixel_to_volt_1d

__all__ = []


class SensorScanSensorGeneric(SensorScanSensorInterface):
    """Generic implementation of the SensorScanSensorInterface."""

    def __init__(
            self,
            barrier_functions: Union[
                BarrierFunctionInterface, Tuple[BarrierFunctionInterface, BarrierFunctionInterface]],
            sensor_peak_function: Union[SensorPeakInterface, List[SensorPeakInterface]],
            alpha_sensor_gate: np.ndarray,
            alpha_gate: np.ndarray = np.array([[0, 0], [0, 0], [0, 0]]),
            alpha_dot: np.ndarray = np.array([[0, 0], [0, 0], [0, 0]]),
            offset_mu_sens: np.ndarray = np.array([0, 0, 0]),
            final_rise: Optional[SensorRiseInterface] = None,
            sensor_peak_deformations: Dict[int, SensorPeakDeformationInterface] = {}
    ):
        """Initializes an object of the class for the simulation of the sensor response for a sensor scan.

        Args:
            barrier_functions (Union[BarrierFunctionInterface, Tuple[BarrierFunctionInterface, BarrierFunctionInterface]]):
                Barrier functions model the responses of the barriers.
            sensor_peak_function (Union[SensorPeakInterface, List[SensorPeakInterface], None]): An implementation of the
                SensorPeakInterface. It is also possible to supply a list of such peaks, if a sensor with multiple peaks
                should be simulated.
            alpha_dot (np.ndarray): Lever-arm of the dots on the sensor potential. With these, the influence of the
                occupation of the two dots on the three potentials (sensor dot, barrier1, and barrier2) is specified.
                The lever-arms are specified as follows np.array([lever-arms sensor dot, lever-arms barrier1,
                lever-arms barrier2]). The individual three lever-arms are each specified as a numpy array. The first
                value describes the influence of the electron occupation of the first dot on the corresponding potential
                and the second the influence of the electron occupation of the second dot. The values should be
                negative. The default value is np.array([[0, 0], [0, 0], [0, 0]]). The lever-arms can also be passed as
                a numpy array with two elements. The lever-arms are then used for the influence on all three potentials.
                If None is passed instead of a numpy array the default value of np.array([[0, 0], [0, 0], [0, 0]]) is
                used.
            alpha_gate (np.ndarray): Lever-arms of the double dot (plunger) gates to the three potentials (sensor dot,
                barrier1, and barrier2). The values should be positive. The lever-arms are specified as follows
                np.array([lever-arms sensor dot, lever-arms barrier1, lever-arms barrier2]). The individual three
                lever-arms are each specified as a numpy array with two elements. The first value describes the
                influence of the voltage of the (plunger) gate of the first dot on the potential, with the second value
                defined analogously for the (plunger) gate of the second dot. Default is
                np.array([[0, 0], [0, 0], [0, 0]]). The lever-arms can also be passed as a numpy array with two
                elements. The lever-arms are then used for the influence on all three potentials. If None is passed
                instead of a numpy array the default value of np.array([[0, 0], [0, 0], [0, 0]]) is used.
            alpha_sensor_gate (np.ndarray): Lever-arms of the sensor dot gates to the three potentials (sensor dot,
                barrier1, and barrier2). The values should be positive. The lever-arms are specified as follows
                np.array([lever-arms sensor dot, lever-arms barrier1, lever-arms barrier2]). The individual three
                lever-arms are each specified as a numpy array with two elements. The first value describes the
                influence of the voltage of the first gate of the sensor dot to the potential and analogously for the
                second.
            offset_mu_sens (np.ndarray): Offset of the electrochemical potential of the three potentials (sensor dot,
                barrier1, and barrier2). This offset can have various causes.
            final_rise (Optional[SensorRiseInterface]): An implementation of the SensorRiseInterface interface. This
                interface represents the rise of the sensor function to the maximum sensor response when both barriers
                are open. This rise is also shifted so that it fits appropriately with the open barriers. The default
                value is None. If None is used, no rise is added to the sensor function.
            sensor_peak_deformations Dict[int, SensorPeakDeformationInterface]:
                Dictionary that specifies the deformations of the wavefronts.
                The key of the dictionary is an integer that specifies the wavefront to which the deformation belongs.
                This value refers to the wavefront number as listed in sensor_peak_function. The value of the dictionary
                is the actual deformation object used for a wavefront.
            """
        self.sensor_peak_function = sensor_peak_function
        self.sensor_peak_deformations = sensor_peak_deformations
        self.final_rise = final_rise

        if self.sensor_peak_function is not None:
            for idx, peak_func in enumerate(self.sensor_peak_function):
                if sensor_peak_deformations is not None and idx in sensor_peak_deformations:
                    deformation = sensor_peak_deformations[idx]
                    deformation.sensor = self

        self.alpha_dot = alpha_dot
        self.alpha_gate = alpha_gate
        self.alpha_sensor_gate = alpha_sensor_gate
        self.offset_mu_sens = offset_mu_sens

        self.barrier_functions = barrier_functions

    @property
    def sensor_peak_function(self) -> Union[SensorPeakInterface, List[SensorPeakInterface], None]:
        """Returns the current sensor peak function configuration of the sensor.

        This configuration can then be adjusted and is directly used as a new configuration, as the object is
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
    def final_rise(self) -> Optional[SensorRiseInterface]:
        """Returns the current sensor rise configuration of the sensor.

        This configuration can then be adjusted and is directly used as a new configuration, as the object is
        returned as call by reference.

        Returns:
            Optional[SensorRiseInterface]: An implementation of the SensorRiseInterface interface. This interface
            represents the rise of the sensor function to the maximum sensor response when both barriers are open. Can
            also return None, if no final rise is used.
        """
        return self.__final_rise

    @final_rise.setter
    def final_rise(self, final_rise: Optional[SensorRiseInterface]):
        """Updates the sensor rise of the sensor.

        Args:
            final_rise (Optional[SensorRiseInterface]): An implementation of the SensorRiseInterface interface. This
                interface represents the rise of the sensor function to the maximum sensor response when both barriers
                are open. This rise is also shifted so that it fits appropriately with the open barriers. The default
                value is None. If None is used, no rise is added to the sensor function.
        """
        if isinstance(final_rise, SensorRiseInterface) or final_rise is None:
            self.__final_rise = final_rise
        else:
            raise ValueError(
                "The provided final_rise is not supported. Must be an implementation of the SensorPeakInterface or None."
            )

    @property
    def barrier_functions(self) -> Tuple[BarrierFunctionInterface, BarrierFunctionInterface]:
        """This function returns the barrier functions.

        Returns:
            Tuple[BarrierFunctionInterface, BarrierFunctionInterface]: Current barrier functions.
        """
        return self.__barrier_functions

    @barrier_functions.setter
    def barrier_functions(self, barrier_functions: Union[
        BarrierFunctionInterface, Tuple[BarrierFunctionInterface, BarrierFunctionInterface]]):
        """This function updates the current barrier functions.
        Either one or two barrier function must be passed. If two functions are passed the first is the barrier function
        for the barrier 1 and the second for barrier 2. When only one function is passed the barrier function is used
        for both barriers.

        Args:
            barrier_functions (Union[BarrierFunctionInterface, Tuple[BarrierFunctionInterface, BarrierFunctionInterface]]):
                New barrier functions. This can either be a single barrier function that is then used for both barriers,
                or a tuple of two barrier functions. When a tuple is passed, the first element is the barrier function
                for the first gate and so on for the second element.
        """
        if isinstance(barrier_functions, BarrierFunctionInterface):
            self.__barrier_functions = [deepcopy(barrier_functions), deepcopy(barrier_functions)]
        elif isinstance(barrier_functions, (list, tuple)) and all(
                isinstance(x, BarrierFunctionInterface) for x in barrier_functions) and len(barrier_functions) == 2:
            self.__barrier_functions = barrier_functions
        else:
            raise ValueError(
                "The provided barrier_functions are invalid. The barrier functions must be a tuple with two "
                "BarrierFunctionInterface objects or a single BarrierFunctionInterface object."
            )

    @property
    def alpha_dot(self) -> np.ndarray:
        """Returns the current alpha dot (dot lever-arms) configuration of the sensor.
        With these, the influence of the occupation of the two dots on the three potentials (sensor dot, barrier1, and
        barrier2) is specified. The lever-arms are specified as follows np.array([lever-arms sensor dot, lever-arms
        barrier1, lever-arms barrier2]). The individual three lever-arms are each specified as a numpy array with two
        elements. The first value describes the influence of the electron occupation of the first dot on the
        corresponding potential and the second the influence of the electron occupation of the second dot. The values
        should be negative.

        Returns:
            np.ndarray: The three pairs of alpha dot lever-arms as a numpy array. The numpy array has the shape (3,2).
        """
        return self.__alpha_dot

    @alpha_dot.setter
    def alpha_dot(self, alpha_dot: np.ndarray):
        """Updates the alpha dot (dot lever-arms) configuration of the sensor.

        With these the influence of the occupations to the three potentials (sensor dot, barrier1, and barrier2)
        is specified.

        Args:
            alpha_dot (np.ndarray): Lever-arms to the dots. With these, the influence of the occupation of the two dots
                on the three potentials (sensor dot, barrier1, and barrier2) is specified. This should be a numpy array
                with shape (3,2), (2,) or None. The values should be negative.
                For an array with the shape (3,2) the array is specified as follows: np.array([lever-arms sensor dot,
                lever-arms barrier1, lever-arms barrier2]). The sub arrays contain the influence of both dots on one of
                the potentials. For an array with the shape (2,) the same lever-arms are used for all three potentials.
                If None is passed its assumed that all alpha dot lever-arms are zero.

        Raises:
            ValueError: A ValueError is raised if the passed alpha_dot is invalid. The alpha_dot is invalid when a numpy
                array with a shape other than (2,) or (3, 2) or None is provided, or when any datatype other than a
                numpy array or None is passed.
        """
        # check if alpha_dot is passed in an acceptable format
        if isinstance(alpha_dot, np.ndarray):
            if alpha_dot.shape == (2,):
                self.__alpha_dot = np.array([alpha_dot, alpha_dot, alpha_dot])
                return
            elif alpha_dot.shape == (3, 2):
                self.__alpha_dot = alpha_dot
                return
        elif alpha_dot is None:
            self.__alpha_dot = np.array([[0, 0], [0, 0], [0, 0]])
            return

        # Raise error if none of the cases for valid alpha_dot formats from before is true
        raise ValueError("The provided alpha_dot configuration is not supported. Must be None or a numpy array of "
                         "shape (2,) or (3,2).")

    @property
    def alpha_gate(self) -> np.ndarray:
        """Returns the alpha gate (gate lever-arms) configuration of the sensor.
        Lever-arms of the double dot (plunger) gates to the three potentials (sensor dot,
        barrier1, and barrier2). The values should be positive. The lever-arms are specified as follows
        np.array([lever-arms sensor dot, lever-arms barrier1, lever-arms barrier2]). The individual three
        lever-arms are each specified as a numpy array with two elements. The first value describes the
        influence of the voltage of the (plunger) gate of the first dot on the potential, with the second value
        defined analogously for the (plunger) gate of the second dot.

        Returns:
            Returns the three pairs of alpha gate lever-arms as a numpy array. The numpy array has the shape (3,2).
        """
        return self.__alpha_gate

    @alpha_gate.setter
    def alpha_gate(self, alpha_gate: np.ndarray):
        """Updates the alpha gate (gate lever-arms) configuration of the sensor.

        With these the influence of the plunger gates to the three potentials (sensor dot, barrier1, and barrier2)
        is specified.

        Args:
            alpha_gate (np.ndarray): Lever-arms of the double dot (plunger) gates to the three potentials (sensor dot,
                barrier1, and barrier2). All alpha gate values should be positive. This should be a numpy array with
                shape (3,2) or (2,) or None. For an array with the shape (3,2) the array is specified as follows:
                np.array([lever-arms sensor dot, lever-arms barrier1, lever-arms barrier2]). The sub arrays contain the
                influence of both gates on one of the potentials. For an array with the shape (2,) the same lever-arms
                are used for all three potentials. If None is passed its assumed that all alpha dot lever-arms are zero.

        Raises:
            ValueError: A ValueError is raised if the passed alpha_gate is invalid. The alpha_gate is invalid when a
                numpy array with a shape other than (2,) or (3, 2) or None is provided, or when any datatype other than
                a numpy array or None is passed.
        """
        # check if alpha_gate is passed in an acceptable format
        if isinstance(alpha_gate, np.ndarray):
            if alpha_gate.shape == (2,):
                self.__alpha_gate = np.array([alpha_gate, alpha_gate, alpha_gate])
                return
            elif alpha_gate.shape == (3, 2):
                self.__alpha_gate = alpha_gate
                return
        elif alpha_gate is None:
            self.__alpha_gate = np.array([[0, 0], [0, 0], [0, 0]])
            return

        # Raise error if none of the cases for valid alpha_gate formats from before is true
        raise ValueError("The provided alpha_gate configuration is not supported. Must be None or a numpy array of "
                         "shape (2,) or (3,2).")

    @property
    def alpha_sensor_gate(self) -> np.ndarray:
        """Returns the alpha sensor gate (sensor gate lever-arms) configuration of the sensor.
        With theses the influence of the sensor dot gates to the three potentials (sensor dot, barrier1, and barrier2)
        is specified. The values should be positive. The lever-arms are specified as follows np.array([lever-arms sensor
        dot, lever-arms barrier1,lever-arms barrier2]). The individual three lever-arms are each specified as a numpy
        array with two elements. The first value describes the influence of the voltage of the first gate of the sensor
        dot to the potential and analogously for the second.

        Returns:
            Returns the three pairs of alpha sensor gate lever-arms as a numpy array. The numpy array has the shape
            (3,2).
        """
        return self.__alpha_sensor_gate

    @alpha_sensor_gate.setter
    def alpha_sensor_gate(self, alpha_sensor_gate: np.ndarray):
        """Method to update the current alpha sensor gate (sensor gate lever-arms) configuration.

        With these the influence of the sensor dot gates to the three potentials (sensor dot, barrier1, and barrier2)
        is specified.

        Args:
            alpha_sensor_gate (np.ndarray): Lever-arms of the sensor dot (barrier) gates to the three potentials
                (sensor dot, barrier1, and barrier2). All alpha sensor gate values should be positive. This should be a
                numpy array with shape (3,2). The lever-arms are specified as follows:
                np.array([lever-arms sensor dot, lever-arms barrier1, lever-arms barrier2]). The sub arrays contain the
                influence of gates on one of the potentials.

        Raises:
            ValueError: A ValueError is raised if the passed alpha_sensor_gate is invalid. The alpha_sensor_gate is
                invalid when a numpy array with a shape other than (3, 2)  is provided or when any datatype other than
                a numpy array is passed.
        """
        # check if alpha_sensor_gate is passed in an acceptable format
        if isinstance(alpha_sensor_gate, np.ndarray) and alpha_sensor_gate.shape == (3, 2):
            self.__alpha_sensor_gate = alpha_sensor_gate
            return

        # Raise error if none of the cases for valid alpha_sensor_gate formats from before is true
        raise ValueError(
            "The provided alpha_sensor_gate configuration is not supported. Must be None or a numpy array of shape "
            "(3,2).")

    @property
    def offset_mu_sens(self) -> np.ndarray:
        """Returns the current offset_mu_sens configuration of the sensor.

        This configuration can then be adjusted and set as new configuration.

        Returns:
            np.ndarray: Electrochemical potential offset of the sensor dot and both barriers for zero electrons in the
            dots and no applied voltage at the gates. The first element contains the sensor dot offset, the second the
            offset of the potential of barrier 1 and last the offset of barrier 2
        """
        return self.__offset_mu_sens

    @offset_mu_sens.setter
    def offset_mu_sens(self, offset_mu_sens: Union[float, np.ndarray]):
        """Updates the offset_mu_sens configuration of the sensor according to the supplied value.

        Args:
            offset_mu_sens (Union[float, np.ndarray]): Electrochemical potential offset of the sensor dot and both
                barriers for zero electrons in the dots and no applied voltage at the gates. If a float is passed the
                same offset is used for all three potentials. Otherwise, if a numpy array is passed, the first element
                contains the sensor dot offset, the second the offset of the potential of barrie 1 and the last that of
                barrier 2.
        """
        # check datatype of offset_mu_sens
        if isinstance(offset_mu_sens, np.ndarray) and offset_mu_sens.shape == (3,):
            self.__offset_mu_sens = offset_mu_sens
        elif isinstance(offset_mu_sens, (float, int)):
            self.__offset_mu_sens = np.array([offset_mu_sens, offset_mu_sens, offset_mu_sens])
        else:
            raise ValueError("The provided offset_mu_sens configuration is not supported. Must either be a float value "
                             "or a numpy array of shape (3,).")

    def __repr__(self):
        return (
                self.__class__.__name__
                + f"(barrier_functions={self.barrier_functions},sensor_peak_function={self.__sensor_peak_function}, "
                  f"alpha_sensor_gate={repr(self.alpha_sensor_gate)}, alpha_gate={repr(self.alpha_gate)}, "
                  f"alpha_dot={repr(self.alpha_dot)}, offset_mu_sens={repr(self.__offset_mu_sens)}, "
                  f"final_rise={self.__final_rise}, sensor_peak_deformations={self.sensor_peak_deformations})"
        )

    def _peak_func(self,
                   mu_sens: Union[float, np.ndarray],
                   volt_g1: Union[float, np.ndarray],
                   volt_g2: Union[float, np.ndarray],
                   volt_sensor_g1: Union[float, np.ndarray],
                   volt_sensor_g2: Union[float, np.ndarray],
                   middle_line_point: Tuple[float, float]
                   ) -> np.ndarray:
        """Help method to calculate the sensor response itself if deformations are applied.

        This method calculates the distance between the point for which the response is to be calculated and the middle
        line. This distance is then used to calculate a deformed potential for each peak. This is then used to calculate
        the sensor response for each peak, which is added up. The middle line follows the direction of the sensor peaks
        (i.e., the sensor dot potential) and is perpendicular to the wavefronts defined by the peaks.

        The arrays `mu_sens`, `volt_x`, `volt_y`, `volt_x_sensor` and `volt_x_sensor` must have the same data type and
        if they are numpy arrays they all must have the same dimension.

        Args:
            mu_sens (Union[float, np.ndarray]): The sensor potential, passed either as a 2-dimensional numpy array
                with the axis mapping to the scan axis or as a float.
            volt_g1 (Union[float, np.ndarray]): Voltages for each pixel applied to the gate g1. The voltages are either
                passed  as a 2-dimensional numpy array or as a float.
            volt_g2 (Union[float, np.ndarray]): Voltages for each pixel applied to the gate g2. The voltages are either
                passed  as a 2-dimensional numpy array or as a float.
            volt_sensor_g1 (Union[float, np.ndarray]): Voltages for each pixel applied to the gate sensor g1. The
                voltages are either passed  as a 2-dimensional numpy array or as a float.
            volt_sensor_g2 (Union[float, np.ndarray]): Voltages for each pixel applied to the gate sensor g2. The
                voltages are either passed  as a 2-dimensional numpy array or as a float.
            middle_line_point (Tuple[float, float]): The base point of the middle line. This point is specified as a
                tuple with two floats.

        Returns:
            Union[float, np.ndarray]: The response of the sensor itself.
        """
        if self.__sensor_peak_function is None:
            return mu_sens
        else:
            m = self.alpha_sensor_gate[0, 1] / self.alpha_sensor_gate[0, 0]

            resp_sum = 0

            dist = signed_dist_points_line(points=np.array([[volt_sensor_g1, volt_sensor_g2]]),
                                           line_points=np.array(
                                               [middle_line_point, middle_line_point + np.array([1, m])]))[0]

            for idx, p in enumerate(self.__sensor_peak_function):
                old_mu0 = p.mu0
                if idx in self.sensor_peak_deformations.keys():
                    p.mu0 = self.sensor_peak_deformations[idx].calc_mu(dist=dist, mu0=p.mu0)
                    resp_sum += p.sensor_function(mu_sens)
                    p.mu0 = old_mu0
                else:
                    resp_sum += p.sensor_function(mu_sens)

            return resp_sum

    def sensor_response(self, mu_sens: np.ndarray) -> np.ndarray:
        """This function returns the sensor response for a given electrochemical potential.

        Args:
            mu_sens (np.ndarray): The given sensor potential.

        Returns:
            np.ndarray: The response, calculated from the given potential. It is stored in a numpy array with the axis
                mapping to the CSD axis. For a two-dimensional scan the response is a two-dimensional numpy array and
                for a one-dimensional scan it is a one-dimensional numpy array.
        """
        sensor_potential, barrier1_potential, barrier2_potential = deepcopy(mu_sens)

        # check if it is needed to apply peak deformations
        apply_deformations = len(self.sensor_peak_deformations) != 0

        sweep_only_sensor = self._volt_limits_g1[0] == self._volt_limits_g1[-1] and \
                            self._volt_limits_g2[0] == self._volt_limits_g2[-1]
        sensor_swept = self._volt_limits_g1[0] == self._volt_limits_g1[-1] or \
                       self._volt_limits_g2[0] == self._volt_limits_g2[-1]

        if apply_deformations and sweep_only_sensor:
            # Calculate sigma when deformations have to be applied
            sens_resp = np.zeros_like(sensor_potential)

            # Calculation of the middle point for deformations
            middle_point = self._calc_point_from_barrier_potentials(
                pot_bar1=self.barrier_functions[0].pinch_off,
                pot_bar2=self.barrier_functions[1].pinch_off,
                sensor_swept=sensor_swept
            )

            if sensor_potential.ndim == 1:
                for pixel_idx in range(sensor_potential.shape[0]):
                    voltage_y = pixel_to_volt_1d(pixel=pixel_idx, pixel_num=sensor_potential.shape[0] - 1,
                                                 volt_limits=self._volt_limits_g2)
                    voltage_y_sensor = pixel_to_volt_1d(pixel=pixel_idx, pixel_num=sensor_potential.shape[0] - 1,
                                                        volt_limits=self._volt_limits_sensor_g2)
                    voltage_x = pixel_to_volt_1d(pixel=pixel_idx, pixel_num=sensor_potential.shape[0] - 1,
                                                 volt_limits=self._volt_limits_g1)
                    voltage_x_sensor = pixel_to_volt_1d(pixel=pixel_idx, pixel_num=sensor_potential.shape[0] - 1,
                                                        volt_limits=self._volt_limits_sensor_g1)
                    sens_resp[pixel_idx] = self._peak_func(mu_sens=sensor_potential[pixel_idx], volt_g1=voltage_x,
                                                           volt_g2=voltage_y, volt_sensor_g1=voltage_x_sensor,
                                                           volt_sensor_g2=voltage_y_sensor,
                                                           middle_line_point=middle_point)

            elif sensor_potential.ndim == 2:
                for y in range(sensor_potential.shape[0]):
                    voltage_y = pixel_to_volt_1d(pixel=y, pixel_num=sensor_potential.shape[0] - 1,
                                                 volt_limits=self._volt_limits_g2)
                    voltage_y_sensor = pixel_to_volt_1d(pixel=y, pixel_num=sensor_potential.shape[0] - 1,
                                                        volt_limits=self._volt_limits_sensor_g2)
                    for x in range(sensor_potential.shape[1]):
                        voltage_x = pixel_to_volt_1d(pixel=x, pixel_num=sensor_potential.shape[1] - 1,
                                                     volt_limits=self._volt_limits_g1)
                        voltage_x_sensor = pixel_to_volt_1d(pixel=x, pixel_num=sensor_potential.shape[1] - 1,
                                                            volt_limits=self._volt_limits_sensor_g1)
                        sens_resp[y, x] = self._peak_func(mu_sens=sensor_potential[y, x],
                                                          volt_g1=voltage_x,
                                                          volt_g2=voltage_y,
                                                          volt_sensor_g1=voltage_x_sensor,
                                                          volt_sensor_g2=voltage_y_sensor,
                                                          middle_line_point=middle_point)

            else:
                raise ValueError("Invalid dimension of the sensor potential")

        else:
            if apply_deformations and not sweep_only_sensor:
                warnings.warn(
                    "Deformations can only swept when only the sensor gates are swept! Currently at least one "
                    "different gate is swept.")

            # Quicker way to calculate sens_resp when no deformations have to be applied
            if self.__sensor_peak_function is None:
                sens_resp = sensor_potential
            else:
                sens_resp = np.sum([p.sensor_function(sensor_potential) for p in self.__sensor_peak_function], axis=0)

            if self.final_rise is not None:
                # Calculation of the middle point
                g1_voltage = self._volt_limits_g1[0]
                g2_voltage = self._volt_limits_g2[0]

                if self._occupations.ndim == 2:
                    occ = (self._occupations[0, 0], self._occupations[0, 1])
                elif self._occupations.ndim == 3:
                    occ = (self._occupations[0, 0, 0], self._occupations[0, 0, 1])
                else:
                    raise ValueError(
                        "Invalid dimension of occupations! The dimension of the occupations must be one larger than "
                        "the scan dimension")

                fully_conductive_point = self._calc_point_from_barrier_potentials(
                    pot_bar1=self.barrier_functions[0].fully_conductive,
                    pot_bar2=self.barrier_functions[1].fully_conductive,
                    sensor_swept=sensor_swept
                )

                fully_conductive_potential = self.sensor_potential(
                    occupations=np.array((occ,)),
                    volt_limits_g1=g1_voltage,
                    volt_limits_g2=g2_voltage,
                    volt_limits_sensor_g1=fully_conductive_point[0],
                    volt_limits_sensor_g2=fully_conductive_point[1],
                )

                final_rise_resp = self.final_rise.sensor_function(sensor_potential,
                                                                  offset=fully_conductive_potential[0])

                sens_resp = np.maximum(sens_resp, final_rise_resp)

        barrier1_resp = self.barrier_functions[0].get_value(barrier1_potential)
        barrier2_resp = self.barrier_functions[1].get_value(barrier2_potential)

        if np.min(sens_resp) < 0 or np.min(barrier1_resp) < 0 or np.min(barrier2_resp) < 0:
            warnings.warn("At least one of the three responses is negative. This will probably not produce a realistic "
                          "scan! The minima of the barrier and sensor peak functions should be 0.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # combine the individual response values -> they are combined like resistors in a series circuit
            numerator = sens_resp * barrier1_resp * barrier2_resp
            denominator = barrier1_resp * barrier2_resp + sens_resp * barrier1_resp + sens_resp * barrier2_resp
            return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    def _calc_point_from_barrier_potentials(
            self,
            pot_bar1: float,
            pot_bar2: float,
            sensor_swept: bool = True
    ) -> Tuple[float, float]:
        """Calculate a point in voltage space from given barrier potentials.

        This function calculates the corresponding point in gate voltage space for given barrier potentials, taking into
        account voltage limits, gate influences, and occupations. The method applies potential offsets and converts
        barrier potentials to gate voltages.

        Args:
            pot_bar1 (float): Target potential for barrier 1.
            pot_bar2 (float): Target potential for barrier 2.
            sensor_swept (bool): Boolean that indicates whether the sensor gates are swept and a sensor scan is
                simulated or gates g1 and g2 are swept and a CSD is simulated. The boolean is true when a sensor scan is
                simulated and false otherwise.

        Returns:
            Tuple[float, float]: The calculated point coordinates (g1, g2) in gate voltage space.
        """

        if self._occupations.ndim == 2:
            occ = (self._occupations[0, 0], self._occupations[0, 1])
        elif self._occupations.ndim == 3:
            occ = (self._occupations[0, 0, 0], self._occupations[0, 0, 1])
        else:
            raise ValueError(
                "Invalid dimension of occupations! The dimension of the occupations must be one larger than "
                "the scan dimension")

        # Create copies to avoid modifying original values
        pot_bar1 = float(pot_bar1)
        pot_bar2 = float(pot_bar2)

        if sensor_swept:
            voltages = np.array([self._volt_limits_g1[0], self._volt_limits_g2[0]])
            gate_offset = np.dot(self.alpha_gate[1:], voltages)
        else:
            voltages = np.array([self._volt_limits_sensor_g1[0], self._volt_limits_sensor_g1[0]])
            gate_offset = np.dot(self.alpha_sensor_gate[1:], voltages)

        # Calculate potential offsets for both barriers
        offset1 = (gate_offset[0] +
                   self.offset_mu_sens[1] +
                   self.alpha_dot[1, 0] * occ[0] +
                   self.alpha_dot[1, 1] * occ[0])

        offset2 = (gate_offset[1] +
                   self.offset_mu_sens[2] +
                   self.alpha_dot[2, 0] * occ[0] +
                   self.alpha_dot[2, 1] * occ[1])

        # Apply offsets to barrier potentials
        pot_bar1 -= offset1
        pot_bar2 -= offset2

        # Calculate point from potential (inline implementation)
        if np.abs(self.alpha_sensor_gate[1, 1]) < sys.float_info.epsilon:  # case: slope of gate 1 line is infinity
            v1_pinch_off = pot_bar1 / self.alpha_sensor_gate[1, 0]
            v2_pinch_off_left = ((-self.alpha_sensor_gate[2, 0] * v1_pinch_off + pot_bar2) /
                                 self.alpha_sensor_gate[2, 1])
            point = (v1_pinch_off, v2_pinch_off_left)
        else:
            v1_pinch_off = (((pot_bar2) * self.alpha_sensor_gate[1, 1] -
                             (pot_bar1) * self.alpha_sensor_gate[2, 1]) /
                            (self.alpha_sensor_gate[2, 0] * self.alpha_sensor_gate[1, 1] -
                             self.alpha_sensor_gate[1, 0] * self.alpha_sensor_gate[2, 1]))
            v2_pinch_off_left = ((-self.alpha_sensor_gate[1, 0] * v1_pinch_off + pot_bar1) /
                                 self.alpha_sensor_gate[1, 1])
            point = (v1_pinch_off, v2_pinch_off_left.item())

        return point

    def sensor_potential(self,
                         occupations: np.ndarray,
                         volt_limits_g1: Union[np.ndarray, float, None] = None,
                         volt_limits_g2: Union[np.ndarray, float, None] = None,
                         volt_limits_sensor_g1: Union[np.ndarray, float, None] = None,
                         volt_limits_sensor_g2: Union[np.ndarray, float, None] = None
                         ) -> np.ndarray:
        """Calculates the electrochemical potential at the sensor dot and both barriers.

        This is done in dependency on the electron occupation in the double dot, the voltages applied at the double
        dot (plunger) gates, and voltages applied at the gates of sensor.
        Either the double dot gates or the sensor gates can be swept.

        Args:
            occupations (np.ndarray): Occupation in left and right dot per applied voltage combination. The occupation
                numbers are stored in a 3-dimensional numpy array. The first two dimensions map to the axis of the CSD,
                while the third dimension indicates the dot of the corresponding occupation value.
            volt_limits_g1 (Union[np.ndarray, float, None]): Voltages applied to the first (plunger) gate of the double
                dot. When a fixed voltage is applied this is a float and when this gate should be swept it is a numpy
                array with the minimum and maximum of the sweep.
            volt_limits_g2 (Union[np.ndarray, float, None]): Voltages applied to the second (plunger) gate of the double
                dot. When a fixed voltage is applied this is a float and when this gate should be swept it is a numpy
                array with the minimum and maximum of the sweep.
            volt_limits_sensor_g1 (Union[np.ndarray, float, None]): Voltages applied to the first sensor gate.
                When a fixed voltage is applied this is a float and when this gate should be swept it is a numpy array
                with the minimum and maximum of the sweep.
            volt_limits_sensor_g2 (Union[np.ndarray, float, None]): voltages applied to the second sensor gate.
                When a fixed voltage is applied this is a float and when this gate should be swept it is a numpy array
                with the minimum and maximum of the sweep.

        Returns:
            np.ndarray: The electrochemical potential of the sensor dot, and both barrier. It is returned as a
            three-dimensional array with the shape (3, occupations.shape[0], occupations.shape[1]). The first dimension
            corresponds to the three potentials involved. It is structured as follows:
            [sensor dot potential, barrier 1 potential, barrier 2 potential]
            The remaining two dimensions correspond to the swept voltages.
        """
        if not occupations.ndim in [2, 3]:
            raise ValueError("The occupations matrix has to have either two or three dimensions.")

        if len(self.sensor_peak_deformations) > 0:
            if isinstance(volt_limits_g1, np.ndarray) or isinstance(volt_limits_g2, np.ndarray):
                warnings.warn("Deformations should only be used with sweeps of sensor gates! But at least one of "
                              "plunger gates is sweept!")

        if (isinstance(volt_limits_g1, np.ndarray) or isinstance(volt_limits_g2, np.ndarray)) and (
                (isinstance(volt_limits_sensor_g1, np.ndarray) or isinstance(volt_limits_sensor_g2, np.ndarray))):
            raise ValueError("Either only the plunger gates of the double quantum dots or the gates of the sensor dots "
                             "can be swept. Both types of gates cannot be swept simultaneously.")

        volt_limits_list = [volt_limits_g1, volt_limits_g2, volt_limits_sensor_g1, volt_limits_sensor_g2]
        volt_limits_list = copy.deepcopy(volt_limits_list)
        for i in range(len(volt_limits_list)):
            if volt_limits_list[i] is None:
                volt_limits_list[i] = np.array([0, 0])
            elif isinstance(volt_limits_list[i], numbers.Real):
                volt_limits_list[i] = np.array([volt_limits_list[i], volt_limits_list[i]])
        volt_limits_g1, volt_limits_g2, volt_limits_sensor_g1, volt_limits_sensor_g2 = volt_limits_list

        self._volt_limits_g1 = volt_limits_g1
        self._volt_limits_g2 = volt_limits_g2
        self._volt_limits_sensor_g1 = volt_limits_sensor_g1
        self._volt_limits_sensor_g2 = volt_limits_sensor_g2
        self._occupations = occupations

        # Calculation of the potentials
        # Voltage matrix for 2D scans
        if occupations.ndim == 3:
            voltages_g1 = np.linspace(volt_limits_g1[0], volt_limits_g1[1], num=occupations.shape[1])
            voltages_g2 = np.linspace(volt_limits_g2[0], volt_limits_g2[1], num=occupations.shape[0])
            voltages = [
                [[voltages_g1[j], voltages_g2[i]] for j in range(len(voltages_g1))] for i in range(len(voltages_g2))
            ]
            voltages = np.array(voltages)

            voltages_sensor_g1 = np.linspace(volt_limits_sensor_g1[0], volt_limits_sensor_g1[1],
                                             num=occupations.shape[1])
            voltages_sensor_g2 = np.linspace(volt_limits_sensor_g2[0], volt_limits_sensor_g2[1],
                                             num=occupations.shape[0])
            voltages_sensor = [
                [[voltages_sensor_g1[j], voltages_sensor_g2[i]] for j in range(len(voltages_sensor_g1))] for i in
                range(len(voltages_sensor_g2))
            ]
            voltages_sensor = np.array(voltages_sensor)
        # Voltage matrix for 1D scans
        elif occupations.ndim == 2:
            voltages_g1 = np.linspace(volt_limits_g1[0], volt_limits_g1[1], num=occupations.shape[0])
            voltages_g2 = np.linspace(volt_limits_g2[0], volt_limits_g2[1], num=occupations.shape[0])
            voltages = [[voltages_g1[i], voltages_g2[i]] for i in range(len(voltages_g1))]
            voltages = np.array(voltages)

            voltages_sensor_g1 = np.linspace(volt_limits_sensor_g1[0], volt_limits_sensor_g1[1],
                                             num=occupations.shape[0])
            voltages_sensor_g2 = np.linspace(volt_limits_sensor_g2[0], volt_limits_sensor_g2[1],
                                             num=occupations.shape[0])
            voltages_sensor = [[voltages_sensor_g1[i], voltages_sensor_g2[i]] for i in range(len(voltages_sensor_g1))]
            voltages_sensor = np.array(voltages_sensor)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu_sens = (occupations.dot(self.__alpha_dot[0]) + voltages.dot(self.__alpha_gate[0]) +
                       voltages_sensor.dot(self.alpha_sensor_gate[0]) + self.offset_mu_sens[0])
            barrier1_pot = (occupations.dot(self.__alpha_dot[1]) + voltages.dot(self.__alpha_gate[1]) +
                            voltages_sensor.dot(self.alpha_sensor_gate[1]) + self.offset_mu_sens[1])
            barrier2_pot = (occupations.dot(self.__alpha_dot[2]) + voltages.dot(self.__alpha_gate[2]) +
                            voltages_sensor.dot(self.alpha_sensor_gate[2]) + self.offset_mu_sens[2])

        return np.array([mu_sens, barrier1_pot, barrier2_pot])

    def get_sensor_scan_labels(self,
                               volt_limits_g1: Union[np.ndarray, float, None],
                               volt_limits_g2: Union[np.ndarray, float, None],
                               volt_limits_sensor_g1: Union[np.ndarray, float, None],
                               volt_limits_sensor_g2: Union[np.ndarray, float, None],
                               potential: np.ndarray) -> (
            np.ndarray, np.ndarray):
        """This method returns the labels of the sensor scans.

        There are two labels for sensor scans: the conductive area mask and the Coulomb peak mask. Both masks are numpy
        arrays that consist of integers.
        The conductive area mask marks the non-conductive area, sensor oscillation regime, and fully conductive area.
        The non-conductive area is marked with 0. This is the area in which no electron can tunnel or flow through the
        sensor dot. The sensor oscillation regime is the area in which the barriers are open enough for oscillations to
        occur, as electrons tunnel periodically. This area is marked with 1. In the third area, the conductive area,
        both barriers are fully open and transport occurs as a continuous current rather than through a well-defined
        sensor dot. The fully conductive area is marked with 2.
        The Coulomb peak mask marks the peaks of the Coulomb peak as integers. The wave fronts are marked with values
        higher or equal to one. All maxima belonging to the same Coulomb peak are marked with the same integer.
        Depending on the potential value, the various Coulomb peaks are marked with ascending values.

        Args:
            volt_limits_g1 (Union[np.ndarray, float, None]): Voltages applied to the first double dot (plunger) gate.
                When a fixed voltage is applied this is a float and when this gate should be swept it is a numpy array
                with the minimum and maximum of the sweep. None can also be passed if no voltage is applied.
                Currently supports only scalar input (`float`) or `None`.
                Array input is reserved for future sweep functionality.
            volt_limits_g2 (Union[np.ndarray, float, None]): Voltages applied to the second double dot(plunger) gate.
                When a fixed voltage is applied this is a float and when this gate should be swept it is a numpy array
                with the minimum and maximum of the sweep. None can also be passed if no voltage is applied.
                Currently supports only scalar input (`float`) or `None`.
                Array input is reserved for future sweep functionality.
            volt_limits_sensor_g1 (Union[np.ndarray, float, None]): Voltages applied to the first sensor gate.
                When a fixed voltage is applied this is a float and when this gate should be swept it is a numpy array
                with the minimum and maximum of the sweep. None can also be passed if no voltage is applied.
            volt_limits_sensor_g2 (Union[np.ndarray, float, None]): Voltages applied to the second sensor gate.
                When a fixed voltage is applied this is a float and when this gate should be swept it is a numpy array
                with the minimum and maximum of the sweep. None can also be passed if no voltage is applied.
            potential (np.ndarray): Numpy array that contains the potential of the area for which labels should be
                generated. This potential must correspond to the voltages specified for volt_limits_g1,
                volt_limits_g2, volt_limits_sensor_g1 and volt_limits_sensor_g2.

        Returns:
            (np.ndarray, np.ndarray): Tuple with the two numpy arrays of the two labels of sensor scan.
            The returned tuple looks like: (conductive area mask, coulomb peak mask). Both arrays have the same shape
            as the provided potential.
        """

        sensor_pot, barrier1_pot, barrier2_pot = potential

        if self._occupations.ndim == 2:
            occ = (self._occupations[0, 0], self._occupations[0, 1])
        elif self._occupations.ndim == 3:
            occ = (self._occupations[0, 0, 0], self._occupations[0, 0, 1])
        else:
            raise ValueError(
                "Invalid dimension of occupations! The dimension of the occupations must be one larger than "
                "the scan dimension")

        apply_deformations = len(self.sensor_peak_deformations) != 0

        if not ((volt_limits_g1 is None or isinstance(volt_limits_g1, float)) and
                (volt_limits_g2 is None or isinstance(volt_limits_g2, float))):
            raise ValueError("The volt limits have to be fixed values or None when sensor scan labels are generated!")

        fully_conductive_point = self._calc_point_from_barrier_potentials(
            pot_bar1=self.barrier_functions[0].fully_conductive,
            pot_bar2=self.barrier_functions[1].fully_conductive
        )

        fully_conductive_potential = self.sensor_potential(
            occupations=np.array((occ,)),
            volt_limits_g1=volt_limits_g1,
            volt_limits_g2=volt_limits_g1,
            volt_limits_sensor_g1=fully_conductive_point[0],
            volt_limits_sensor_g2=fully_conductive_point[1],
        )

        coulomb_peak_mask = np.zeros_like(sensor_pot, dtype=np.uint8)
        conductive_mask = np.zeros_like(sensor_pot, dtype=np.uint8)

        m = self.alpha_sensor_gate[0, 1] / self.alpha_sensor_gate[0, 0]

        # Create array with same voltages tuples with the same resolution as the sensor potential
        if potential.ndim == 3:
            x_values = np.linspace(volt_limits_sensor_g1[0], volt_limits_sensor_g1[-1], sensor_pot.shape[1])
            y_values = np.linspace(volt_limits_sensor_g2[0], volt_limits_sensor_g2[-1], sensor_pot.shape[0])
            x_grid, y_grid = np.meshgrid(x_values, y_values)
            xy_array = np.dstack((x_grid, y_grid))
        if potential.ndim == 2:
            x_values = np.linspace(volt_limits_sensor_g1[0], volt_limits_sensor_g1[1], num=sensor_pot.shape[0])
            y_values = np.linspace(volt_limits_sensor_g2[0], volt_limits_sensor_g2[1], num=sensor_pot.shape[0])
            xy_array = [[x_values[i], y_values[i]] for i in range(len(x_values))]
            xy_array = np.array(xy_array)

        # Reshape all points to (n,2) to use it with distance calculation
        xy_array = xy_array.reshape(-1, 2)  # Shape: (num_x_values * num_y_values, 2)

        if apply_deformations:
            middle_point = self._calc_point_from_barrier_potentials(
                pot_bar1=self.barrier_functions[0].pinch_off,
                pot_bar2=self.barrier_functions[1].pinch_off
            )

            # Calculate distances from the middle line
            dist_array = signed_dist_points_line(points=xy_array,
                                                 line_points=np.array(
                                                     [middle_point, middle_point + np.array([1, m])]))

            # Reshape dist_array to the initial shape (shape of the potential)
            dist_array = dist_array.reshape(sensor_pot.shape)

        all_peaks = copy.deepcopy(self.sensor_peak_function)
        if isinstance(all_peaks, SensorPeakInterface):
            all_peaks = [all_peaks]

        # Sensor peaks are sorted according to their mu0 so that the increases in the masks function correctly
        # Peaks with a smaller mu0 overwrite the range of the previous peaks. The peaks, relating to the mu0, must
        # therefore be run through from large to small
        peaks_sorted_with_id = sorted(zip(all_peaks, range(len(all_peaks))), key=lambda x: x[0].mu0)
        peaks_sorted_with_id = list((idx, peak, id) for idx, (peak, id) in enumerate(peaks_sorted_with_id))

        for idx, peak, peak_id in reversed(list(peaks_sorted_with_id)):
            distorted_pot: np.ndarray
            if idx in self.sensor_peak_deformations.keys():
                deformation = self.sensor_peak_deformations[idx]
                distorted_pot = sensor_pot + (
                        peak.mu0 - deformation.calc_mu(dist=dist_array, mu0=self.sensor_peak_function[idx].mu0))
            else:
                distorted_pot = sensor_pot

            if potential.ndim == 3:
                # 2d scan
                diffs = np.diff(sensor_pot, axis=1)
                average_diff = np.mean(diffs)
                threshold = np.abs(average_diff)

                # Caculation of the distance of to the middle of a peak
                min_indices = np.argmin(np.abs(distorted_pot - peak.mu0), axis=1)

                is_boundary = (min_indices == 0) | (min_indices == conductive_mask.shape[1] - 1)
                row_indices = np.arange(conductive_mask.shape[0])
                boundary_distances = np.abs(distorted_pot[row_indices, min_indices] - peak.mu0)

                valid_mask = ~is_boundary | (boundary_distances < threshold)
                coulomb_peak_mask[row_indices[valid_mask], min_indices[valid_mask]] = peak_id + 1

            else:
                # 1d scan
                diffs = np.diff(sensor_pot)
                average_diff = np.mean(diffs)
                threshold = np.abs(average_diff)

                # Caculation of the distance of to the middle of a peak
                min_index = np.argmin(np.abs(distorted_pot - peak.mu0))

                if min_index in [0, conductive_mask.shape[0] - 1]:
                    if np.abs(distorted_pot[min_index] - peak.mu0) < threshold:
                        coulomb_peak_mask[min_index] = peak_id + 1
                else:
                    coulomb_peak_mask[min_index] = peak_id + 1

        conductive_mask[((barrier1_pot > self.barrier_functions[0].pinch_off) & (
                barrier2_pot > self.barrier_functions[1].pinch_off))] = 1

        conductive_mask[(sensor_pot > fully_conductive_potential[0]) & conductive_mask == 1] = 2

        coulomb_peak_mask[conductive_mask != 1] = 0

        return conductive_mask, coulomb_peak_mask
