"""This module contains functions for adding blurring to the transition lines in a simulated charge stability diagram (CSD).

@author: s.fleitmann
"""

from typing import Callable, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d

from simcats.distortions import OccupationDistortionInterface
from simcats.support_functions import ParameterSamplingInterface, fermi_filter1d

__all__ = []


class OccupationTransitionBlurringFermiDirac(OccupationDistortionInterface):
    """Fermi-Dirac filter blurring implementation of the OccupationDistortionsInterface."""

    def __init__(self, sigma: Union[float, ParameterSamplingInterface]):
        """Initializes an object of the class used to blur lead transitions and occupations changes.

        Args:
            sigma (Union[float, ParameterSamplingInterface]): The sigma of the Fermi-Dirac blur, defining the strength
                of the blurring. If sigma is of type ParameterSampling a new sigma is sampled accordingly per call of
                noise_function.
        """
        self.__sigma = sigma
        # save the last sampled sigma
        self.__latest_sigma = None

    @property
    def sigma(self) -> Union[float, ParameterSamplingInterface]:
        """The sigma of the Fermi-Dirac blur, defining the strength of the blurring."""
        return self.__sigma

    @sigma.setter
    def sigma(self, sigma):
        self.__sigma = sigma

    @property
    def latest_sigma(self) -> Union[float, None]:
        """The sigma that was used for the latest simulation.

        This is necessary because, depending on the setting, a sampler can be used instead of a fixed sigma.
        """
        return self.__latest_sigma

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
        """This function is used to add blurring to the supplied occupations and lead transitions.

        The distortion parameter is adjusted to the resolution, as it is specified in voltage space but must be
        applied in pixel space.
        Warning: This function changes the supplied occupation and lead_transitions arrays. If you want to keep the
        original ones, please give a copy of them into this function.

        Args:
            occupations (np.ndarray): Contains the original occupation to which the distortions are added.
                If there are other types of distortions which appear first, they should already be included in the
                occupations. Warning: this argument is changed during the execution of the function.
            lead_transitions (np.ndarray): Contains the original lead transition mask to which the distortions are
                added. If there are other types of distortions which appear first, they should already be included in
                the lead transitions. Warning: this argument is changed during the execution of the function.
            volt_limits_g1 (np.ndarray): Contains the beginning and ending of the swept range for the first (plunger)
                gate.
            volt_limits_g2 (np.ndarray): Contains the beginning and ending of the swept range for the second (plunger)
                gate.
            generate_csd (Union[Callable, None]): Function which generates data points outside the swept gate range.
                This is especially required for distortions, which shift the CSD structure. The generated data points
                also have to contain the distortions, which have already been added to the occupation and
                lead_transitions before.
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
        if freeze:
            sigma = self.latest_sigma
        else:
            try:
                sigma = self.sigma.sample_parameter()
            except AttributeError:
                sigma = self.sigma
        self.__latest_sigma = sigma
        if sigma > 0:
            if len(occupations.shape) == 3:  # 2d scan
                resolution = occupations.shape[0:2]
                # rescale the sigma according to the resolution (as it's defined per volt but must be applied per pixel)
                # sweeping always happens in x-direction
                sigma *= resolution[0] / (np.max(volt_limits_g1) - np.min(volt_limits_g1))
                occupations[:, :, 0] = fermi_filter1d(occupations[:, :, 0], sigma=sigma)
                occupations[:, :, 1] = fermi_filter1d(occupations[:, :, 1], sigma=sigma)
            if len(occupations.shape) == 2:  # 1d scan
                resolution = (occupations.shape[0], occupations.shape[0])
                # rescale the sigma according to the resolution (as it's defined per volt but must be applied per pixel)
                if volt_limits_g1[0] == volt_limits_g1[1]:
                    sigma *= resolution[1] / (np.max(volt_limits_g2) - np.min(volt_limits_g2))
                elif volt_limits_g2[0] == volt_limits_g2[1]:
                    sigma *= resolution[0] / (np.max(volt_limits_g1) - np.min(volt_limits_g1))
                else:
                    sigma *= np.maximum(
                        resolution[0] / (np.max(volt_limits_g1) - np.min(volt_limits_g1)),
                        resolution[1] / (np.max(volt_limits_g2) - np.min(volt_limits_g2)),
                    )
                occupations[:, 0] = fermi_filter1d(occupations[:, 0], sigma=sigma)
                occupations[:, 1] = fermi_filter1d(occupations[:, 1], sigma=sigma)
        return occupations, lead_transitions

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(sigma={self.sigma}{f'[latest={self.latest_sigma}]' if self.latest_sigma else '[latest=0]'})"
        )


class OccupationTransitionBlurringGaussian(OccupationDistortionInterface):
    """Gaussian filter blurring implementation of the OccupationDistortionsInterface."""

    def __init__(self, sigma: Union[float, ParameterSamplingInterface]):
        """Initializes an object of the class used to blur lead transitions and occupations changes.

        Args:
            sigma (Union[float, ParameterSamplingInterface]): The sigma of the gaussian blur, defining the strength of
                the blurring. If sigma is of type ParameterSampling a new sigma is sampled accordingly per call of
                noise_function.
        """
        self.__sigma = sigma
        # save the last sampled sigma
        self.__latest_sigma = None

    @property
    def sigma(self) -> Union[float, ParameterSamplingInterface]:
        """The sigma of the gaussian blur, defining the strength of the blurring."""
        return self.__sigma

    @sigma.setter
    def sigma(self, sigma):
        self.__sigma = sigma

    @property
    def latest_sigma(self) -> Union[float, None]:
        """The sigma that was used for the latest simulation.

        This is necessary because, depending on the setting, a sampler can be used instead of a fixed sigma.
        """
        return self.__latest_sigma

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
        """This function is used to add blurring to the supplied occupations and lead transitions.

        The distortion parameter is adjusted to the resolution, as it is specified in voltage space but must be
        applied in pixel space.
        Warning: This function changes the supplied occupation and lead_transitions arrays. If you want to keep the
        original ones, please give a copy of them into this function.

        Args:
            occupations (np.ndarray): Contains the original occupation to which the distortions are added.
                If there are other types of distortions which appear first, they should already be included in the
                occupations. Warning: this argument is changed during the execution of the function.
            lead_transitions (np.ndarray): Contains the original lead transition mask to which the distortions are
                added. If there are other types of distortions which appear first, they should already be included in
                the lead transitions. Warning: this argument is changed during the execution of the function.
            volt_limits_g1 (np.ndarray): Contains the beginning and ending of the swept range for the first (plunger)
                gate.
            volt_limits_g2 (np.ndarray): Contains the beginning and ending of the swept range for the second (plunger)
                gate.
            generate_csd (Union[Callable, None]): Function which generates data points outside the swept gate range.
                This is especially required for distortions, which shift the CSD structure. The generated data points
                also have to contain the distortions, which have already been added to the occupation and
                lead_transitions before.
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
        if freeze:
            sigma = self.latest_sigma
        else:
            try:
                sigma = self.sigma.sample_parameter()
            except AttributeError:
                sigma = self.sigma
        self.__latest_sigma = sigma
        if sigma > 0:
            if len(occupations.shape) == 3:  # 2d scan
                resolution = occupations.shape[0:2]
                # rescale the sigma according to the resolution (as it's defined per volt but must be applied per pixel)
                # sweeping always happens in x-direction
                sigma *= resolution[0] / (np.max(volt_limits_g1) - np.min(volt_limits_g1))
                occupations[:, :, 0] = gaussian_filter1d(occupations[:, :, 0], sigma=sigma)
                occupations[:, :, 1] = gaussian_filter1d(occupations[:, :, 1], sigma=sigma)
            if len(occupations.shape) == 2:  # 1d scan
                resolution = (occupations.shape[0], occupations.shape[0])
                # rescale the sigma according to the resolution (as it's defined per volt but must be applied per pixel)
                if volt_limits_g1[0] == volt_limits_g1[1]:
                    sigma *= resolution[1] / (np.max(volt_limits_g2) - np.min(volt_limits_g2))
                elif volt_limits_g2[0] == volt_limits_g2[1]:
                    sigma *= resolution[0] / (np.max(volt_limits_g1) - np.min(volt_limits_g1))
                else:
                    sigma *= np.maximum(
                        resolution[0] / (np.max(volt_limits_g1) - np.min(volt_limits_g1)),
                        resolution[1] / (np.max(volt_limits_g2) - np.min(volt_limits_g2)),
                    )
                occupations[:, 0] = gaussian_filter1d(occupations[:, 0], sigma=sigma)
                occupations[:, 1] = gaussian_filter1d(occupations[:, 1], sigma=sigma)
        return occupations, lead_transitions

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(sigma={self.sigma}{f'[latest={self.latest_sigma}]' if self.latest_sigma else '[latest=0]'})"
        )
