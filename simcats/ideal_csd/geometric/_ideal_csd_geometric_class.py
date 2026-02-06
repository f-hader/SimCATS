"""This module contains the required functionalities to generate ideal CSD data using supplied total charge
transitions (TCTs), voltage ranges and a desired resolution. It also implements the IdealCSDInterface.

@author: f.hader
"""

from typing import List, Tuple, Union, Optional

import numpy as np

from simcats.ideal_csd import IdealCSDInterface
from simcats.ideal_csd.geometric import ideal_csd_geometric

__all__ = []


class IdealCSDGeometric(IdealCSDInterface):
    """Geometric simulation approach implementation of the IdealCSDInterface."""

    def __init__(
        self,
        tct_params: List[np.ndarray],
        rotation: float = -np.pi / 4,
        lut_entries: Optional[int] = None,
        cdf_type: str = "sigmoid",
        cdf_gamma_factor: Optional[float] = None,
    ) -> None:
        """Initializes an object of the class for the geometric simulation approach which is based on total charge
        transitions (TCTs).

        Args:
            tct_params (List[np.ndarray]): List containing a numpy array with parameters for every TCT in the CSD.
                Each array contains all required parameters to describe the TCT form. \n
                The parameters for a TCT are: \n
                [0] = length left (in x-/voltage-space, not number of points) \n
                [1] = length right (in x-/voltage-space, not number of points) \n
                [2] = gradient left (in voltages) \n
                [3] = gradient right (in voltages) \n
                [4] = start position x (bezier curve leftmost point) (in x-/voltage-space, not number of points) \n
                [5] = start position y (bezier curve leftmost point) (in x-/voltage-space, not number of points) \n
                [6] = end position x (bezier curve rightmost point) (in x-/voltage-space, not number of points) \n
                [7] = end position y (bezier curve rightmost point) (in x-/voltage-space, not number of points)
            rotation (float): Float value defining the rotation to be applied to the TCT (which is usually represented
                with the tct_params rotated by 45 degrees). Default is -np.pi/4.
            lut_entries (Optional[int]): Number of samples for the lookup-table for bezier curves. If this is not None, a
                lookup-table will be used to evaluate the points on the bezier curves, else they are solved explicitly.
                Using a lookup-table speeds up the calculation at the possible cost of accuracy. Default is None.
            cdf_type (str): Name of the type of cumulative distribution function (CDF) to be used. Can be either
                "cauchy" or "sigmoid". Default is "sigmoid".
            cdf_gamma_factor (Optional[float]): The factor used for the calculation of the gamma values of the CDF. If set to None
                (=default) the default values for the selected cdf_type are used (2.2 for sigmoid, 6.15 for cauchy).
                Default is None. \n
                Gamma is calculated as follows: \n
                gamma = width_bezier_curve / cdf_gamma_factor
        """
        self.tct_params = tct_params
        self.rotation = rotation
        self.lut_entries = lut_entries
        self.cdf_type = cdf_type
        self.cdf_gamma_factor = cdf_gamma_factor

    @property
    def tct_params(self) -> List[np.ndarray]:
        """List containing a numpy array with parameters for every TCT in the CSD. Each array contains all required
        parameters to describe the TCT form.
        """
        return self.__tct_params

    @tct_params.setter
    def tct_params(self, tct_params: list):
        self.__tct_params = tct_params

    @property
    def rotation(self) -> float:
        """Float value defining the rotation to be applied to the TCT (which is usually represented with the tct_params
        rotated by 45 degrees).
        """
        return self.__rotation

    @rotation.setter
    def rotation(self, rotation: float) -> None:
        self.__rotation = rotation

    @property
    def lut_entries(self) -> Optional[int]:
        """Number of samples for the lookup-table for bezier curves. If this is not None, a lookup-table will be used to
        evaluate the points on the bezier curves, else they are solved explicitly.
        """
        return self.__lut_entries

    @lut_entries.setter
    def lut_entries(self, lut_entries: Optional[int]):
        self.__lut_entries = lut_entries

    @property
    def cdf_type(self) -> str:
        """Name of the type of cumulative distribution function (CDF) to be used."""
        return self.__cdf_type

    @cdf_type.setter
    def cdf_type(self, cdf_type: str) -> None:
        self.__cdf_type = cdf_type

    @property
    def cdf_gamma_factor(self) -> Optional[float]:
        """The factor used for the calculation of the gamma values of the CDF. If set to None (=default) the default values
        for the selected cdf_type are used (2.2 for sigmoid, 6.15 for cauchy).
        """
        return self.__cdf_gamma_factor

    @cdf_gamma_factor.setter
    def cdf_gamma_factor(self, cdf_gamma_factor: Optional[float]):
        self.__cdf_gamma_factor = cdf_gamma_factor

    def get_csd_data(
        self,
        volt_limits_g1: np.ndarray,
        volt_limits_g2: np.ndarray,
        resolution: Union[int, np.ndarray] = np.array([100, 100]),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve ideal data (occupation numbers and a lead transition mask) for given gate voltages.

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
            Tuple[np.ndarray, np.ndarray]: Occupation numbers and lead transition mask (in our case: total charge
            transitions). The occupation numbers are stored in a 3-dimensional numpy array. The first two dimensions map
            to the axis of the CSD, while the third dimension indicates the dot of the corresponding occupation value.
            The label mask for the lead-to-dot transitions is stored in a 2-dimensional numpy array with the axis
            mapping to the CSD axis.

        """
        return ideal_csd_geometric(
            tct_params=self.__tct_params,
            volt_limits_g1=volt_limits_g1,
            volt_limits_g2=volt_limits_g2,
            resolution=resolution,
            rotation=self.__rotation,
            lut_entries=self.__lut_entries,
            cdf_type=self.__cdf_type,
            cdf_gamma_factor=self.__cdf_gamma_factor,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(tct_params={self.__tct_params}, rotation={self.__rotation}, lut_entries={self.__lut_entries}, cdf_type='{self.__cdf_type}', cdf_gamma_factor={self.__cdf_gamma_factor})"
        )
