"""This module contains functions for simulating the sensor response for a charge stability diagram (CSD) including the
cross coupling between the sensor and double dot (plunger) gates.

@author: f.hader, b.papajewski
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional

import numpy as np

__all__ = []


class SensorPeakInterface(ABC):
    """Interface for all sensor peak functions, which are used to define the Coulomb Peaks of sensor dots for the simulation of CSDs.

    Implementations of the SensorInterface can consist of multiple peaks.
    """

    @property
    @abstractmethod
    def mu0(self) -> Optional[float]:
        """Mu0 of the sensor peak.
        This property represents the potential value at which the sensor peak reaches its maximum.

        The method is needed for the generation of labels of scans. An example for that is the generation of labels for
        sensor scans.
        """
        raise NotImplementedError()

    @abstractmethod
    def sensor_function(self, mu_sens: np.ndarray) -> np.ndarray:
        """This method has to be implemented in every object which should be used as sensor peak function during the simulation of CSDs.

        The electrochemical potential at the sensor should be given as one- or two-dimensional numpy
        array to this function. The parameters for this sensor peak function should be attributes of this class. This
        function should return the sensor (peak) values which result from the given electrochemical potential.

        Args:
            mu_sens (np.ndarray): The sensor potential, stored in a numpy array with the axis mapping to the CSD axis.

        Returns:
            np.ndarray: The sensor response (for the corresponding peak), calculated from the given potential. It is
            stored in a numpy array with the axis mapping to the CSD axis.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.__class__.__name__ + str(self.__dict__)


class SensorRiseInterface(ABC):
    """Interface for all sensor rise functions that model a final rise of the sensor response.

    This type of rise is needed to simulate the rise at the end of a sensor function, that can be observed in sensor
    scans.
    """

    @property
    @abstractmethod
    def fully_conductive(self) -> float:
        """Potential value of the point at which the sensor rise reaches its maximum."""
        raise NotImplementedError

    @abstractmethod
    def sensor_function(self, mu_sens: np.ndarray, offset: float) -> np.ndarray:
        """This method has to be implemented in every object which should be used as sensor rise function during the simulation of CSDs.

        This function should return the sensor (rise) values which result from the given electrochemical potential.

        Args:
            mu_sens (np.ndarray): The sensor potential, stored in a numpy array with the axis mapping to the scan axis.
            offset (float): Potential value to which the fully conductive point is shifted.

        Returns:
            np.ndarray: The sensor response (for the corresponding rise), calculated from the given potential. It is
            stored in a numpy array with the axis mapping to the CSD axis.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.__class__.__name__ + str(self.__dict__)


class SensorInterface(ABC):
    """Interface for all sensor functions, which are used in the simulation of CSDs."""

    @abstractmethod
    def __init__(self,
                 sensor_peak_function: Union[SensorPeakInterface, List[SensorPeakInterface]],
                 alpha_dot: np.ndarray,
                 alpha_gate: np.ndarray,
                 offset_mu_sens: float,
    ) -> None:
        """Initializes an object of the class for the simulation of the sensor response for a CSD.

        Args:
            sensor_peak_function (SensorPeakInterface): An implementation of the SensorPeakInterface. It is also
                possible to supply a list of such peaks, if a sensor with multiple peaks should be simulated.
            alpha_dot (np.ndarray): Lever-arm to the dots. The values should be negative. Influences the strengths of
                the edges in the CSD.
            alpha_gate (np.ndarray): Lever-arm to the double dot (plunger) gates. The values should be positive.
            offset_mu_sens (float): Electrochemical potential of the sensor dot for zero electrons in the dots and no
                applied voltage at the gates.
        """
        raise NotImplementedError

    @abstractmethod
    def sensor_response(self, mu_sens: np.ndarray) -> np.ndarray:
        """This method has to be implemented in every object which should be used as sensor function during the simulation of CSDs.

        The electrochemical potential at the sensor should be given as one- or two-dimensional numpy array to this
        function. The parameters for this sensor function should be attributes of this class. This function should
        return the sensor response which result from the given electrochemical potential.

        Args:
            mu_sens (np.ndarray): The sensor potential, stored in a 2-dimensional numpy array with the axis mapping to
                the CSD axis.

        Returns:
            np.ndarray: The sensor response, calculated from the given potential. It is stored in a numpy array with the
            axis mapping to the CSD axis.
        """
        raise NotImplementedError

    @abstractmethod
    def sensor_potential(self,
                         occupations: np.ndarray,
                         volt_limits_g1: np.ndarray,
                         volt_limits_g2: np.ndarray
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
            np.ndarray: The electrochemical potential at the sensing dot.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + str(self.__dict__)


class SensorScanSensorInterface(SensorInterface):
    def __init__(self,
                 sensor_peak_function: Union[SensorPeakInterface, List[SensorPeakInterface], None],
                 alpha_dot: np.ndarray,
                 alpha_gate: np.ndarray,
                 alpha_sensor_gate: np.ndarray,
                 offset_mu_sens: np.ndarray,
    ) -> None:
        """This method initializes an object of the class SensorScanSensorInterface.

        Args:
            sensor_peak_function (Union[SensorPeakInterface, List[SensorPeakInterface], None]): An implementation of the
                SensorPeakInterface. It is also possible to supply a list of such peaks, if a sensor with multiple peaks
                should be simulated.
            alpha_dot (np.ndarray): Lever-arm of the dots for the sensor potential(s). The values should be negative.
            alpha_gate (np.ndarray): Lever-arm of the gates for the sensor potential(s). The values should be positive.
            alpha_sensor_gate (np.ndarray): Lever-arm of the sensor gates for the sensor potential(s). The values should
                be positive.
            offset_mu_sens (np.ndarray): Electrochemical potential of the sensor dot for zero electrons in the dots and
                no applied voltage at the gates.
        """
        raise NotImplementedError

    @abstractmethod
    def sensor_potential(self,
                         occupations: np.ndarray,
                         volt_limits_g1: Union[np.ndarray, float, None],
                         volt_limits_g2: Union[np.ndarray, float, None],
                         volt_limits_sensor_g1: Union[np.ndarray, float, None],
                         volt_limits_sensor_g2: Union[np.ndarray, float, None]
                         ) -> np.ndarray:
        """Simulates the electrochemical potential at the sensor dot and both barriers.

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
        raise NotImplementedError

    def get_sensor_scan_labels(self,
                               volt_limits_g1: Union[np.ndarray, float, None],
                               volt_limits_g2: Union[np.ndarray, float, None],
                               volt_limits_sensor_g1: Union[np.ndarray, float, None],
                               volt_limits_sensor_g2: Union[np.ndarray, float, None],
                               potential: np.ndarray) -> np.ndarray:
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
            volt_limits_g2 (Union[np.ndarray, float, None]): Voltages applied to the second double dot(plunger) gate.
                When a fixed voltage is applied this is a float and when this gate should be swept it is a numpy array
                with the minimum and maximum of the sweep. None can also be passed if no voltage is applied.
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
        raise NotImplementedError
