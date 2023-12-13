"""
This module contains the simulation class, that can be used to perform charge stability diagram (CSD) simulations.
Additionally, it provides default configurations for this class.

@author: f.hader
"""

from copy import deepcopy
from functools import partial
from typing import Union, Tuple, Dict, List

import numpy as np

from simcats.distortions import SensorResponseDistortionInterface, OccupationDistortionInterface, \
    SensorPotentialDistortionInterface
from simcats.ideal_csd import IdealCSDInterface
from simcats.sensor import SensorInterface, SensorGeneric

__all__ = []


class Simulation:
    """
    Class for simulating a double dot tuning run using charge stability diagrams (CSDs).
    The objects of the class are initialized with an ideal CSD configuration, a sensor configuration and the desired
    types of distortions. It is then possible to record individual CSDs in the selected voltage range of the (plunger)
    gates g1 and g2. g1 and g2 represent the (plunger) gates of the double dot, used to mainly adjust the dot
    potentials. After the analysis of a CSD by means of a tuning algorithm to be tested, this algorithm should then
    suggest voltages for the next measurement until the desired configuration has been reached.
    """

    def __init__(
        self,
        volt_limits_g1: Union[np.ndarray, None] = None,
        volt_limits_g2: Union[np.ndarray, None] = None,
        ideal_csd_config: Union[IdealCSDInterface, None] = None,
        sensor: SensorInterface = SensorGeneric(),
        occupation_distortions: Union[List[OccupationDistortionInterface], None] = None,
        sensor_potential_distortions: Union[List[SensorPotentialDistortionInterface], None] = None,
        sensor_response_distortions: Union[List[SensorResponseDistortionInterface], None] = None,
    ):
        """
        Initializes an object of the class to perform the simulation of charge stability diagrams (CSDs) for a double
        dot experiment. It is assumed, that the sweeps are performed linewise. For 2D scans this means, that (plunger)
        gate 1 is swept while (plunger) gate 2 is kept at a fixed voltage (and stepped between g1 sweeps). For 1D scans
        this means, that both gates are swept simultaneously. Noise is always applied along the measurement direction
        (to take into account time dependencies and correlations). If f.e. in a 2D scan g1 is swept from larger to
        smaller voltages, the distortions is also applied in this direction.

        Args:
            volt_limits_g1 (Union[np.ndarray, None]): Voltage limits of (plunger) gate 1 (second-/x-axis). Defines the 
                range in which data can be queried during the simulation. If set to None, all voltage values are 
                allowed. This can potentially lead to problems, if the structures are not available for some regions 
                and no restriction is applied. Default is None. \n
                Example: \n
                [min_V1, max_V1]
            volt_limits_g2 (Union[np.ndarray, None]): Voltage limits of (plunger) gate 2 (first-/y-axis). Defines the 
                range in which data can be queried during the simulation. If set to None, all voltage values are 
                allowed. This can potentially lead to problems, if the structures are not available for some regions 
                and no restriction is applied. Default is None. \n
                Example: \n
                [min_V2, max_V2]
            ideal_csd_config (Union[IdealCSDInterface, None]): Implementation of the IdealCSDInterface (from the
                ideal_csd module). Used to generate the ideal CSD data during the simulation. Default is None.
            sensor (SensorInterface): Implementation of the SensorInterface (from the sensor module). Used to calculate
                the sensor potential & response based on ideal CSD data. Default is SensorGeneric().
            occupation_distortions (Union[List[OccupationDistortionInterface], None]): List of implementations of the 
                OccupationDistortionInterface. This distortion type affects the occupations and lead transitions of the 
                CSD (the "structure"). The supplied implementations are applied in the order they appear in the list. 
                Default is None.
            sensor_potential_distortions (Union[List[SensorPotentialDistortionInterface], None]): List of 
                implementations of the SensorPotentialDistortionInterface. This distortion type affects the sensor 
                potential, which is calculated based on the occupations and (plunger) gate voltages. The supplied 
                implementations are applied in the order they appear in the list. Default is None.
            sensor_response_distortions (Union[List[SensorResponseDistortionInterface], None]): List of implementations 
                of the SensorResponseDistortionInterface. This distortions type affects the sensor response, which is 
                calculated based on the sensor potential. The supplied implementations are applied in the order they 
                appear in the list. Default is None.
        """
        self.volt_limits_g1 = volt_limits_g1
        self.volt_limits_g2 = volt_limits_g2
        self.ideal_csd_config = ideal_csd_config
        self.sensor = sensor
        self.occupation_distortions = occupation_distortions
        self.sensor_potential_distortions = sensor_potential_distortions
        self.sensor_response_distortions = sensor_response_distortions

    def measure(
        self,
        sweep_range_g1: np.ndarray,
        sweep_range_g2: np.ndarray,
        resolution: Union[int, np.ndarray] = np.array([100, 100]),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Simulates the measurement of the specified voltage sweep in the desired resolution. If two resolutions are
        supplied, a 2D scan is performed, and if only one resolution is supplied, a 1D scan is performed.

        Args:
            sweep_range_g1 (np.ndarray): Voltage sweep range of (plunger) gate 1 (second-/x-axis). \n
                Example: \n
                [min_V1, max_V1]
            sweep_range_g2 (np.ndarray): Voltage sweep range of (plunger) gate 2 (first-/y-axis). \n
                Example: \n
                [min_V2, max_V2]
            resolution (Union[int, np.ndarray]): The desired resolution (in pixels) for the two gates. If only one value
                is supplied, a 1D sweep is performed. Then, both gates are swept simultaneously. Default is
                np.array([100, 100]). \n
                Example: \n
                [res_g1, res_g2]

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]: Numpy array of the measured sensor signal (aka the CSD),
            3-dimensional numpy array containing the electron occupations. The first two dimensions map to the axis
            of the CSD, while the third dimension indicates the dot of the corresponding occupation value. Numpy
            array containing a label mask for the lead-to-dot transitions. Dictionary with metadata (all parameters
            of the system & the measurement). The axes of the three arrays match to the supplied sweep range. If the
            sweep is f.e. performed from high to low voltage, the values in the array are also arranged in that way
            (lowest index would then map to the highest voltage). In general: axis 0 = y-axis = g2 = V2 and axis 1 =
            x-axis = g1 = V1.
        """
        # Check if charge transitions have been configured for the system
        if self.__ideal_csd_config is None:
            raise ValueError("Clean CSD data configuration missing.")

        # check if the gate voltage ranges have the correct number of entries
        if not (len(sweep_range_g1) == 2 and len(sweep_range_g2) == 2):
            raise ValueError(
                "The sweep ranges for the gates g1 and g2 must consist of exactly 2 values each. At least "
                f"one sweep range violates this."
            )

        # Check if voltage range is restricted. If it is restricted, check if the start & stop voltage
        # of the scan are in the allowed range.
        if (
            self.__volt_limits_g1 is not None
            and (
                np.min(sweep_range_g1) < self.__volt_limits_g1[0]
                or np.min(sweep_range_g1) > self.__volt_limits_g1[1]
                or np.max(sweep_range_g1) < self.__volt_limits_g1[0]
                or np.max(sweep_range_g1) > self.__volt_limits_g1[1]
            )
            or (
                self.__volt_limits_g2 is not None
                and (
                    np.min(sweep_range_g2) < self.__volt_limits_g2[0]
                    or np.min(sweep_range_g2) > self.__volt_limits_g2[1]
                    or np.max(sweep_range_g2) < self.__volt_limits_g2[0]
                    or np.max(sweep_range_g2) > self.__volt_limits_g2[1]
                )
            )
        ):
            raise ValueError(
                f"The voltages defined by sweep_range_g1 ({sweep_range_g1}) must be in the range of the limits defined"
                f"by volt_limits_g1 ({self.__volt_limits_g1}) and the voltages defined by sweep_range_g2 "
                f"({sweep_range_g2}) must be in the range of the limits defined by volt_limits_g2 "
                f"({self.__volt_limits_g2})."
            )

        # check if the resolution has at most two entries
        if not type(resolution) == int and len(resolution) > 2:
            raise ValueError(
                f"The specified resolution ({resolution}) has more than two entries. The resolution must either be a "
                f"single value (for 1D scans) or contain two entries (for 2D scans)."
            )

        # Check if one gate is kept at a fixed voltage if a 2D scan is requested. This is only possible in 1D scans.
        if (
            not type(resolution) == int
            and len(resolution) == 2
            and (sweep_range_g1[0] == sweep_range_g1[1] or sweep_range_g2[0] == sweep_range_g2[1])
        ):
            raise ValueError(
                f"At least one of the voltage ranges 'sweep_range_g1' and 'sweep_range_g2' defines a fixed voltage. "
                f"This is only supported for 1D sweeps (only one resolution), but two resolutions were specified."
            )

        # Check if one resolution is 1 (or smaller) for a 2D scan. A resolution of 1 indicates that the corresponding
        # gate is kept at a fixed voltage. This only makes sense for a 1D scan
        if not type(resolution) == int and len(resolution) == 2 and (resolution[0] <= 1 or resolution[1] <= 1):
            raise ValueError(
                f"The specified resolution ({resolution}) indicates that a 2D scan should be performed, but at least "
                f"one of the two entries is smaller or equal to 1. A resolution of 1 means that the corresponding gate "
                f"is not swept. Thus, it describes a 1D scan of a single gate. Please specify just one resolution and "
                f"a fixed voltage for the corresponding gate, to perform a single gate sweep."
            )

        # setup clean data function pointer
        generate_csd = deepcopy(self.__ideal_csd_config.get_csd_data)

        # Perform simulation
        occupations, lead_transitions = generate_csd(
            volt_limits_g1=sweep_range_g1, volt_limits_g2=sweep_range_g2, resolution=resolution
        )

        # add distortions to the occupations
        if self.__occupation_distortions is not None:
            for i in self.__occupation_distortions:
                i.noise_function(occupations, lead_transitions, sweep_range_g1, sweep_range_g2, generate_csd)

                def generate_csd_with_noise(volt_limits_g1, volt_limits_g2, resolution, generate_csd, noise_function):
                    """
                    This helper function is used to add the previous distortions to the generate_csd function for the
                    next distortions.
                    """
                    occupations, lead_transitions = generate_csd(volt_limits_g1, volt_limits_g2, resolution)
                    noise_function(occupations, lead_transitions, volt_limits_g1, volt_limits_g2, generate_csd, freeze=True)
                    return occupations, lead_transitions
                generate_csd = partial(generate_csd_with_noise, generate_csd=generate_csd, noise_function=i.noise_function)

        # calculate the sensor potential from the distorted occupations
        potential = self.__sensor.sensor_potential(
            occupations=occupations, volt_limits_g1=sweep_range_g1, volt_limits_g2=sweep_range_g2
        )

        # Add distortions to the sensor potential
        if self.__sensor_potential_distortions is not None:
            for i in self.__sensor_potential_distortions:
                potential = i.noise_function(
                    mu_sens=potential, volt_limits_g1=sweep_range_g1, volt_limits_g2=sweep_range_g2
                )

        # Add the sensor function
        csd = self.__sensor.sensor_response(potential)

        # add distortions to the sensor signal
        if self.__sensor_response_distortions is not None:
            for i in self.__sensor_response_distortions:
                csd = i.noise_function(
                    sensor_response=csd, volt_limits_g1=sweep_range_g1, volt_limits_g2=sweep_range_g2
                )

        # create a dictionary containing all the metadata
        metadata = {
            "sweep_range_g1": deepcopy(sweep_range_g1),
            "sweep_range_g2": deepcopy(sweep_range_g2),
            "resolution": deepcopy(resolution),
            "ideal_csd_config": deepcopy(self.ideal_csd_config),
            "sensor": deepcopy(self.sensor),
            "occupation_distortions": deepcopy(self.occupation_distortions),
            "sensor_potential_distortions": deepcopy(self.sensor_potential_distortions),
            "sensor_response_distortions": deepcopy(self.sensor_response_distortions),
        }

        return csd, occupations, lead_transitions, metadata

    @property
    def volt_limits_g1(self) -> Union[np.ndarray, None]:
        """
        Returns the current plunger 1 voltage limit configuration of the system. This configuration can then be adjusted
        and is directly used as new configuration, as the object is returned as call by reference

        Returns:
            np.ndarray: The allowed voltage range for (plunger) gate 1.
        """
        return self.__volt_limits_g1

    @volt_limits_g1.setter
    def volt_limits_g1(self, volt_limits_g1: Union[np.ndarray, None]) -> None:
        """
        Updates the plunger 1 voltage limit configuration of the system according to the supplied values.

        Args:
            volt_limits_g1 (np.ndarray): The allowed voltage range for (plunger) gate 1.
        """
        # check datatype of voltage limits
        if not (
            volt_limits_g1 is None or (isinstance(volt_limits_g1, (np.ndarray, list)) and volt_limits_g1.size == 2)
        ):
            raise ValueError(
                f"The provided volt_limits_g1 configuration is not supported. Must be either None or "
                "a numpy array of size 2."
            )
        self.__volt_limits_g1 = deepcopy(volt_limits_g1)

    @property
    def volt_limits_g2(self) -> Union[np.ndarray, None]:
        """
        Returns the current (plunger) gate 2 voltage limit configuration of the system. This configuration can then be
        adjusted and is directly used as new configuration, as the object is returned as call by reference

        Returns:
            np.ndarray: The allowed voltage range for (plunger) gate 2.
        """
        return self.__volt_limits_g2

    @volt_limits_g2.setter
    def volt_limits_g2(self, volt_limits_g2: Union[np.ndarray, None]) -> None:
        """
        Updates the plunger 2 voltage limit configuration of the system according to the supplied values.

        Args:
            volt_limits_g2 (np.ndarray): The allowed voltage range plunger 2.
        """
        # check datatype of voltage limits
        if not (
            volt_limits_g2 is None or (isinstance(volt_limits_g2, (np.ndarray, list)) and volt_limits_g2.size == 2)
        ):
            raise ValueError(
                f"The provided volt_limits_g2 configuration is not supported. Must be either None or "
                "a numpy array of size 2."
            )
        self.__volt_limits_g2 = deepcopy(volt_limits_g2)

    @property
    def ideal_csd_config(self) -> Union[IdealCSDInterface, None]:
        """
        Returns the current ideal CSD configuration of the system. This configuration can then be adjusted and
        is directly used as new configuration, as the object is returned as call by reference

        Returns:
            IdealCSDInterface: Implementation of the IdealCSDInterface (part of module ideal_csd)
        """
        return self.__ideal_csd_config

    @ideal_csd_config.setter
    def ideal_csd_config(self, ideal_csd_config: Union[IdealCSDInterface, None]) -> None:
        """
        Updates the ideal CSD configuration of the system according to the supplied implementation.

        Args:
            ideal_csd_config (IdealCSDInterface): Implementation of the IdealCSDInterface (part of module
                ideal_csd)
        """
        # check ideal csd config for implementations of the correct interface
        if not (ideal_csd_config is None or isinstance(ideal_csd_config, IdealCSDInterface)):
            raise ValueError(
                f"The provided ideal CSD configuration is not supported, as it doesn't implement the "
                "interface 'IdealCSDInterface'."
            )
        self.__ideal_csd_config = deepcopy(ideal_csd_config)

    @property
    def sensor(self) -> SensorInterface:
        """
        Returns the current sensor configuration of the system. This configuration can then be adjusted and is directly
        used as new configuration, as the object is returned as call by reference

        Returns:
            SensorInterface: Implementation of the SensorInterface (from the sensor module). Used to calculate the
            sensor potential & response based on ideal CSD data.
        """
        return self.__sensor

    @sensor.setter
    def sensor(self, sensor: SensorInterface):
        """
        Updates the sensor configuration of the system according to the supplied values.

        Args:
            sensor (SensorInterface): Implementation of the SensorInterface (from the sensor module). Used to calculate
                the sensor potential & response based on ideal CSD data. If the sensor is set to None, the
                GenericSensor (default) is used.
        """
        if sensor is None:
            self.__sensor = SensorGeneric()
        # check sensor config for implementations of the correct interface
        elif isinstance(sensor, SensorInterface):
            self.__sensor = deepcopy(sensor)
        # raise an error if an invalid configuration has been supplied
        else:
            raise ValueError(
                f"The provided sensor configuration is not supported, as it doesn't implement the "
                "interface 'SensorInterface'."
            )

    @property
    def occupation_distortions(self) -> Union[List[OccupationDistortionInterface], None]:
        """
        Returns the current occupation distortion configuration of the system. This configuration can then be adjusted
        and is directly used as new configuration, as the object is returned as call by reference

        Returns:
            list: List with all occupation distortion implementations.
        """
        return self.__occupation_distortions

    @occupation_distortions.setter
    def occupation_distortions(self, occupation_distortions: Union[List[OccupationDistortionInterface], None]):
        """
        Updates the occupation distortion configuration of the system according to the supplied values.

        Args:
            occupation_distortions (list): List with all occupation distortion implementations.
        """
        # check occupation distortions config for implementations of the correct interface
        if not (
            occupation_distortions is None
            or all(isinstance(x, OccupationDistortionInterface) for x in occupation_distortions)
        ):
            raise ValueError(
                f"The provided occupation distortion configuration is not supported, as not all list "
                "objects implement the interface 'OccupationDistortionInterface'."
            )
        self.__occupation_distortions = deepcopy(occupation_distortions)

    @property
    def sensor_potential_distortions(self) -> Union[List[SensorPotentialDistortionInterface], None]:
        """
        Returns the current sensor potential distortion configuration of the system. This configuration can then be
        adjusted and is directly used as new configuration, as the object is returned as call by reference

        Returns:
            list: List with all sensor potential distortion implementations.
        """
        return self.__sensor_potential_distortions

    @sensor_potential_distortions.setter
    def sensor_potential_distortions(self, sensor_potential_distortions: Union[List[SensorPotentialDistortionInterface], None]):
        """
        Updates the sensor potential distortion configuration of the system according to the supplied values.

        Args:
            sensor_potential_distortions (list): List with all sensor potential distortion implementations.
        """
        # check sensor potential distortions config for implementations of the correct interface
        if not (
            sensor_potential_distortions is None
            or all(isinstance(x, SensorPotentialDistortionInterface) for x in sensor_potential_distortions)
        ):
            raise ValueError(
                f"The provided sensor potential distortion configuration is not supported, as not all "
                "list objects implement the interface 'SensorPotentialDistortionInterface'."
            )
        self.__sensor_potential_distortions = deepcopy(sensor_potential_distortions)

    @property
    def sensor_response_distortions(self) -> Union[List[SensorResponseDistortionInterface], None]:
        """
        Returns the current sensor response distortion configuration of the system. This configuration can then be
        adjusted and is directly used as new configuration, as the object is returned as call by reference

        Returns:
            list: List with all sensor response distortion implementations.
        """
        return self.__sensor_response_distortions

    @sensor_response_distortions.setter
    def sensor_response_distortions(self, sensor_response_distortions: Union[List[SensorResponseDistortionInterface], None]) -> None:
        """
        Updates the sensor response distortion configuration of the system according to the supplied values.

        Args:
            sensor_response_distortions (list): List with all sensor response distortion implementations.
        """
        # check sensor response distortions config for implementations of the correct interface
        if not (
            sensor_response_distortions is None
            or all(isinstance(x, SensorResponseDistortionInterface) for x in sensor_response_distortions)
        ):
            raise ValueError(
                f"The provided sensor response distortion configuration is not supported, as not all "
                "list objects implement the interface 'SensorResponseDistortionInterface'."
            )
        self.__sensor_response_distortions = deepcopy(sensor_response_distortions)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(volt_limits_g1={self.volt_limits_g1}, volt_limits_g2={self.volt_limits_g2}, ideal_csd_config={self.ideal_csd_config}, sensor={self.sensor}, occupation_distortions={self.occupation_distortions}, sensor_potential_distortions={self.sensor_potential_distortions}, sensor_response_distortions={self.sensor_response_distortions})"
        )
