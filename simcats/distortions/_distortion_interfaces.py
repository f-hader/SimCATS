"""This module contains interfaces for simulating the distortions in charge stability diagrams (CSDs) with the SimCATS
class Simulation.

@author: f.hader
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union

import numpy as np

__all__ = []


class DistortionInterface(ABC):
    """General interface for distortions.

    Explicit distortion types are derived from this. The simulation class expects to
    receive objects of the explicit types, which are assigned to different stages.
    """

    @abstractmethod
    def noise_function(self, *args, **kwargs):
        """This function has to be implemented for adding distortions in any stage of the simulation.

        Args:
            *args: Specified in subclasses
            **kwargs: Specified in subclasses

        Returns:
            distorted data
        """
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__ + str(self.__dict__)


class OccupationDistortionInterface(DistortionInterface):
    """Interface for distortions affecting the occupations and lead transitions ("the structure") of a simulated CSD."""

    @abstractmethod
    def noise_function(
        self,
        occupations: np.ndarray,
        lead_transitions: np.ndarray,
        volt_limits_g1: np.ndarray,
        volt_limits_g2: np.ndarray,
        generate_csd: Union[
            Callable[[np.ndarray, np.ndarray, Union[int, np.ndarray]], Tuple[np.ndarray, np.ndarray]], None
        ] = None,
        freeze: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """This method has to be implemented for adding distortions to the occupations and lead transitions.

        Args:
            occupations (np.ndarray): Contains the original occupations to which the distortions are added.
                If there are other types of distortions which appear first, they should already be included in the
                occupations.
            lead_transitions (np.ndarray): Contains the original lead transition mask to which the distortions are
                added. If there are other types of distortions which appear first, they should already be included in
                the lead transitions.
            volt_limits_g1 (np.ndarray): Contains the beginning and ending of the swept range for the first (plunger)
                gate.
            volt_limits_g2 (np.ndarray): Contains the beginning and ending of the swept range for the second (plunger)
                gate.
            generate_csd (Optional[Callable]): Function which generates data points outside the swept gate range.
                This is especially required for distortions, which shift the CSD structure. The generated data points
                also have to contain the distortions, which have already been added to the occupation and
                lead_transitions before. Default is None.
            freeze (bool): Indicates if the last used noise should be reused. This is important if there are noise
                types which need to generate data from outside the current CSD (for example if a part of the structure
                is shifted). This newly generated data also has to contain the noise which already has been applied to
                the CSD before.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Occupation numbers and lead transition mask (in our case: total charge
            transitions) with added distortions. The occupation numbers are stored in a 3-dimensional numpy array. The
            first two dimensions map to the axis of the CSD, while the third dimension indicates the dot of the
            corresponding occupation value. The label mask for the lead-to-dot transitions is stored in a 2-dimensional
            numpy array with the axis mapping to the CSD axis.
        """
        raise NotImplementedError()


class SensorPotentialDistortionInterface(DistortionInterface):
    """Interface for distortions affecting the potential of the sensor used to measure a simulated CSD."""

    @abstractmethod
    def noise_function(self, mu_sens: np.ndarray, volt_limits_g1: np.ndarray, volt_limits_g2: np.ndarray) -> np.ndarray:
        """This method has to be implemented in every object which should be used to distort the sensor potential of a CSD.

        Args:
            mu_sens (np.ndarray): Contains the sensor potential to which the distortions are added. The potential is
                stored in a 2-dimensional numpy array with the axis mapping to the CSD axis. If there are other types of
                distortions which appear first, they should already be included in the potential.
            volt_limits_g1 (np.ndarray): Contains the beginning and ending of the swept range for the first (plunger)
                gate.
            volt_limits_g2 (np.ndarray): Contains the beginning and ending of the swept range for the second (plunger)
                gate.

        Returns:
            np.ndarray: The sensor potential with added distortions. The potential is stored in a 2-dimensional numpy
            array with the axis mapping to the CSD axis.
        """
        raise NotImplementedError()


class SensorResponseDistortionInterface(DistortionInterface):
    """Interface for distortions affecting the sensor response of a simulated CSD.

    This includes every effect which happens after the sensor.
    """

    @abstractmethod
    def noise_function(
        self, sensor_response: np.ndarray, volt_limits_g1: np.ndarray, volt_limits_g2: np.ndarray
    ) -> np.ndarray:
        """Method which has to be implemented by every object used to distort the sensor response of a CSD.

        Args:
            sensor_response (np.ndarray): Contains the sensor response to which the distortions are added. The sensor
                response is stored in a 2-dimensional numpy array with the axis mapping to the CSD axis. If there are
                other types of distortions which appear first, they should already be included in the response.
            volt_limits_g1 (np.ndarray): Contains the beginning and ending of the swept range for the first (plunger)
                gate.
            volt_limits_g2 (np.ndarray): Contains the beginning and ending of the swept range for the second (plunger)
                gate.

        Returns:
            np.ndarray: The sensor response with added distortions. The sensor response is stored in a 2-dimensional
            numpy array with the axis mapping to the CSD axis.
        """
        raise NotImplementedError()
