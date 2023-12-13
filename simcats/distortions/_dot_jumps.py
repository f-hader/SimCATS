"""This module contains functions for the simulation of dot jumps.

@author: s.fleitmann
"""

import warnings
from typing import Callable, Tuple, Union

import numpy as np

from simcats.distortions import OccupationDistortionInterface
from simcats.support_functions import ParameterSamplingInterface

__all__ = []


class OccupationDotJumps(OccupationDistortionInterface):
    """Dot jumps implementation of the OccupationDistortionsInterface.

    Dot jumps are simulated by adding shifts in the occupations.
    """

    def __init__(
        self,
        scale: Union[float, ParameterSamplingInterface],
        lam: Union[float, ParameterSamplingInterface],
        axis: int,
        ratio: float = 1,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """Initializes an object of the class used to generate dot jumps.

        Args:
            scale (Union[float, ParameterSamplingInterface]): Specifies the length of the shifts, gives the mathematical
                expectation for their length. In this case, scale specifies the width of the block in which a shift is
                present. Given in volt per pixel. If the scale is of type ParameterSampling a new scale is sampled
                accordingly per call of noise_function.
            lam (Union[float, ParameterSamplingInterface]): Lambda for the poisson distribution which specifies the
                height of the jumps. Given in volt per pixel. If lam is of type ParameterSampling a new lam is sampled
                accordingly per call of noise_function.
            axis (int): The axis along which dot jumps will be generated. That means, the values are also shifted along
                this axis. For axis=0 this results into shifts along the y/g2-axis and for axis=1 into shifts along
                the x/g1-axis.
            ratio (float): The ratio defining how often this type of distortion should be active. For each simulation a
                random number decides if the distortion type is active, based on the supplied ratio. Default is 1.
            rng (np.random.Generator): The random number generator used for the simulation of random numbers. Default is
                np.random.default_rng().
        """
        self.scale = scale
        self.lam = lam
        self.axis = axis
        self.ratio = ratio
        self.rng = rng
        self.__activated = None

    @property
    def scale(self) -> Union[float, ParameterSamplingInterface]:
        """Specifies the length of the shifts, gives the mathematical expectation for their length.

        In this case, scale specifies the width of the block in which a shift is present. Given in volt per pixel.
        """
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    @property
    def lam(self) -> Union[float, ParameterSamplingInterface]:
        """Lambda for the poisson distribution which specifies the height of the jumps. Given in volt per pixel."""
        return self.__lam

    @lam.setter
    def lam(self, lam):
        self.__lam = lam

    @property
    def axis(self) -> int:
        """The axis along which dot jumps will be generated.

        That means, the values are also shifted along this axis. For axis=0 this results into shifts along the y/g2-axis
        and for axis=1 into shifts along the x/g1-axis.
        """
        return self.__axis

    @axis.setter
    def axis(self, axis):
        self.__axis = axis

    @property
    def ratio(self) -> float:
        """The ratio defining how often this type of distortion should be active."""
        return self.__ratio

    @ratio.setter
    def ratio(self, ratio):
        self.__ratio = ratio

    @property
    def rng(self) -> np.random.Generator:
        """The random number generator used for the simulation of random numbers."""
        return self.__rng

    @rng.setter
    def rng(self, rng):
        self.__rng = rng

    @property
    def activated(self) -> Union[bool, None]:
        """This is true if the noise was activated during the last call of noise function."""
        return self.__activated

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
        """This function is used to add dot jumps to the supplied occupations and lead transitions.

        The distortion parameters are adjusted to the resolution, as they are specified in voltage space but must be
        applied in pixel space.
        Warning: This function changes the supplied occupation and lead_transitions arrays. If you want to keep the
        original ones, please give a copy of them into this function.

        Args:
            occupations (np.ndarray): Contains the original occupations to which the distortions are added.
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
                lead_transitions before. Defaults to None.
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
        if not bool(self.rng.binomial(1, self.ratio)):
            self.__activated = False
            return occupations, lead_transitions
        else:
            self.__activated = True

        resolution = occupations.shape[0:2]
        if freeze:
            scale = None
            lam = None
        else:
            try:
                scale = self.scale.sample_parameter()
            except AttributeError:
                scale = self.scale
            try:
                lam = self.lam.sample_parameter()
            except AttributeError:
                lam = self.lam
            if self.axis == 1:  # dot jumps in x direction
                scale *= resolution[0] / (np.max(volt_limits_g1) - np.min(volt_limits_g1))
                lam *= resolution[0] / (np.max(volt_limits_g1) - np.min(volt_limits_g1))
            elif self.axis == 0:  # dot jumps in y direction
                if len(occupations.shape) == 3:
                    scale *= resolution[1] / (np.max(volt_limits_g2) - np.min(volt_limits_g2))
                    lam *= resolution[1] / (np.max(volt_limits_g2) - np.min(volt_limits_g2))
                else:
                    scale *= resolution[0] / (np.max(volt_limits_g2) - np.min(volt_limits_g2))
                    lam *= resolution[0] / (np.max(volt_limits_g2) - np.min(volt_limits_g2))
            if scale < 1:
                # smallest possible value is 1, so set to 1. As this might not be, what the user wants, warn about it
                warnings.warn(
                    "The given scale in voltages results in a pixelwise scale of less than 1 and is thus set to 1. \
    That means the length of dot jumps is always 1. Consider using a larger scale or disabling dot jumps for small scans."
                )
                scale = 1

        # Dot jumps are always applied from low to high voltages because otherwise many new transition lines
        # would have to be added to be able to simulate the jumps to higher voltages regions
        # sort the voltages and with the same sorting index also the occupations and lead_transitions
        sorted_limits_g1 = np.sort(volt_limits_g1)
        sorted_limits_g2 = np.sort(volt_limits_g2)
        sorted_occupations = occupations
        sorted_transitions = lead_transitions
        if len(occupations.shape) == 2:  # 1d scan
            if volt_limits_g2[0] > volt_limits_g2[1] or volt_limits_g1[0] > volt_limits_g1[1]:
                sorted_occupations = occupations[::-1]
                if lead_transitions is not None:
                    sorted_transitions = lead_transitions[::-1]
            self._noise_function_1d(
                sorted_occupations,
                sorted_transitions,
                sorted_limits_g1,
                sorted_limits_g2,
                generate_csd,
                scale,
                lam,
                freeze,
            )
            if volt_limits_g2[0] > volt_limits_g2[1] or volt_limits_g1[0] > volt_limits_g1[1]:
                occupations = sorted_occupations[::-1]
                if lead_transitions is not None:
                    lead_transitions = sorted_transitions[::-1]
        if len(occupations.shape) == 3:  # 2d scan
            if volt_limits_g2[0] > volt_limits_g2[1]:
                sorted_occupations = occupations[::-1, :]
                if lead_transitions is not None:
                    sorted_transitions = lead_transitions[::-1, :]
            if volt_limits_g1[0] > volt_limits_g1[1]:
                sorted_occupations = occupations[:, ::-1]
                if lead_transitions is not None:
                    sorted_transitions = lead_transitions[:, ::-1]
            self._noise_function_2d(
                sorted_occupations,
                sorted_transitions,
                sorted_limits_g1,
                sorted_limits_g2,
                generate_csd,
                scale,
                lam,
                freeze,
            )
            # bring back into original order
            if volt_limits_g2[0] > volt_limits_g2[1]:
                occupations = sorted_occupations[::-1, :]
                if lead_transitions is not None:
                    lead_transitions = sorted_transitions[::-1, :]
            if volt_limits_g1[0] > volt_limits_g1[1]:
                occupations = sorted_occupations[:, ::-1]
                if lead_transitions is not None:
                    lead_transitions = sorted_transitions[:, ::-1]
        return occupations, lead_transitions

    def _noise_function_1d(
        self,
        occupations: np.ndarray,
        lead_transitions: np.ndarray,
        volt_limits_g1: np.ndarray,
        volt_limits_g2: np.ndarray,
        generate_csd: Callable[
            [np.ndarray, np.ndarray, Union[int, np.ndarray]],
            Tuple[np.ndarray, np.ndarray],
        ],
        scale: float,
        lam: float,
        use_previous_noise: bool,
    ) -> None:
        """Helper function to add dot jumps to the supplied occupations and lead transitions for 1D scans.

        It is called by the main noise_function if a 1D scan is supplied. The distortion parameters are
        adjusted to the resolution, as they are specified in voltage space but must be applied in pixel
        space.
        Warning: This function changes the supplied occupation and lead_transitions arrays. If you want to keep the
        original ones, please give a copy of them into this function.

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
            generate_csd (Callable): Function which generates data points outside the swept gate range. This is
                especially required for distortions, which shift the CSD structure. The generated data points also have
                to contain the distortions, which have already been added to the occupation and lead_transitions before.
            scale (float): Specifies the length of the shifts, gives the mathematical expectation for their length. In
                this case, scale specifies the width of the block in which a shift is present. Given in pixels.
            lam (float): Lambda for the poisson distribution which specifies the height of the jumps. Given in pixels.
            use_previous_noise (bool): indicates if the previously generated noise should be used again. This is
                especially needed, if the noise function is used inside a generate_csd-Function

        """
        # create a two-dimensional array, so that the same dot jumps function can be used
        occupation_2d = np.full([occupations.shape[0], occupations.shape[0], 2], np.nan)
        np.fill_diagonal(occupation_2d[:, :, 0], occupations[:, 0])
        np.fill_diagonal(occupation_2d[:, :, 1], occupations[:, 1])
        if lead_transitions is not None:
            lead_transitions_2d = np.full([occupations.shape[0], occupations.shape[0]], np.nan)
            np.fill_diagonal(lead_transitions_2d, lead_transitions)

        if use_previous_noise:
            noise = self.__previous_noise
        else:
            noise = generate_dot_jumps_blockwise(occupation_2d[:, :, 0].shape, scale, lam, axis=self.axis, rng=self.rng)
            self.__previous_noise = noise
        max_jump = np.max(noise)
        if generate_csd is not None and max_jump > 0:
            added_pixels = max_jump
            if self.axis == 1:
                # generate the necessary part to the "left" of the CSD and fill into 2d csd
                step_size = (volt_limits_g1[1] - volt_limits_g1[0]) / occupations.shape[0]
                for i in range(max_jump):
                    left_occupation, left_charge_transitions = generate_csd(
                        volt_limits_g1=(
                            volt_limits_g1[0] - (i + 1) * step_size,
                            volt_limits_g1[1] - (i + 1) * step_size,
                        ),
                        volt_limits_g2=volt_limits_g2,
                        resolution=occupations.shape[0],
                    )
                    # add one column to the 2d image space
                    occupation_2d = np.hstack((np.full([occupations.shape[0], 1, 2], np.nan), occupation_2d))
                    # insert the i-th diagonal there
                    np.fill_diagonal(occupation_2d[:, :, 0], left_occupation[:, 0])
                    np.fill_diagonal(occupation_2d[:, :, 1], left_occupation[:, 1])
                    if lead_transitions is not None:
                        # add one column to the 2d image space
                        lead_transitions_2d = np.hstack(
                            (
                                np.full([occupations.shape[0], 1], np.nan),
                                lead_transitions_2d,
                            )
                        )
                        # insert the i-th diagonal there
                        np.fill_diagonal(lead_transitions_2d, left_charge_transitions)
            if self.axis == 0:
                step_size = (volt_limits_g1[1] - volt_limits_g1[0]) / occupations.shape[0]
                for i in range(max_jump):
                    top_occupation, top_charge_transitions = generate_csd(
                        volt_limits_g1=volt_limits_g1,
                        volt_limits_g2=(
                            volt_limits_g2[0] - (i + 1) * step_size,
                            volt_limits_g2[1] - (i + 1) * step_size,
                        ),
                        resolution=occupations.shape[0],
                    )
                    # add one row to the 2d image space
                    occupation_2d = np.vstack((np.full([1, occupations.shape[0], 2], np.nan), occupation_2d))
                    # insert the i-th diagonal there
                    np.fill_diagonal(occupation_2d[:, :, 0], top_occupation[:, 0])
                    np.fill_diagonal(occupation_2d[:, :, 1], top_occupation[:, 1])
                    if lead_transitions is not None:
                        # add one row to the 2d image space
                        lead_transitions_2d = np.vstack(
                            (
                                np.full([1, occupations.shape[0]], np.nan),
                                lead_transitions_2d,
                            )
                        )
                        # insert the i-th diagonal there
                        np.fill_diagonal(lead_transitions_2d, top_charge_transitions)
        else:
            added_pixels = 0

        occupation_2d[:, :, 0], _ = dot_jumps_blockwise(
            occupation_2d[:, :, 0],
            scale,
            lam,
            np.concatenate((np.zeros(added_pixels, dtype=int), noise)),
            axis=self.axis,
            rng=self.rng,
        )
        occupation_2d[:, :, 1], _ = dot_jumps_blockwise(
            occupation_2d[:, :, 1],
            scale,
            lam,
            np.concatenate((np.zeros(added_pixels, dtype=int), noise)),
            axis=self.axis,
            rng=self.rng,
        )
        if self.axis == 1:  # dot jumps in x direction
            occupations[:, :] = np.array([occupation_2d[i, i + added_pixels] for i in range(occupations.shape[0])])
        if self.axis == 0:  # dot jumps in y direction
            occupations[:, :] = np.array([occupation_2d[i + added_pixels, i] for i in range(occupations.shape[0])])
        # adapt total charge transitions accordingly
        if lead_transitions is not None:
            if self.axis == 1:
                lead_transitions_2d = dot_jumps_blockwise(
                    lead_transitions_2d,
                    scale,
                    lam,
                    np.concatenate((np.zeros(added_pixels, dtype=int), noise)),
                    axis=self.axis,
                    rng=self.rng,
                )[0][:, added_pixels:]
            if self.axis == 0:
                lead_transitions_2d = dot_jumps_blockwise(
                    lead_transitions_2d,
                    scale,
                    lam,
                    np.concatenate((np.zeros(added_pixels, dtype=int), noise)),
                    axis=self.axis,
                    rng=self.rng,
                )[0][added_pixels:, :]
            lead_transitions[:] = np.diag(lead_transitions_2d)

    def _noise_function_2d(
        self,
        occupation: np.ndarray,
        lead_transitions: np.ndarray,
        volt_limits_g1: np.ndarray,
        volt_limits_g2: np.ndarray,
        generate_csd: Callable[
            [np.ndarray, np.ndarray, Union[int, np.ndarray]],
            Tuple[np.ndarray, np.ndarray],
        ],
        scale: float,
        lam: float,
        use_previous_noise: bool,
    ) -> None:
        """Helper function to add dot jumps to the supplied occupations and lead transitions for 2D scans.

        It is called by the main noise_function if a 2D scan is supplied. The distortion parameters are
        adjusted to the resolution, as they are specified in voltage space but must be applied in pixel space.
        Warning: This function changes the supplied occupation and lead_transitions arrays. If you want to keep the
        original ones, please give a copy of them into this function.

        Args:
            occupations (np.ndarray): Contains the original occupation to which the distortions are added.
                If there are other types of distortions which appear first, they should already be included in the
                occupations.
            lead_transitions (np.ndarray): Contains the original lead transition mask to which the distortions are
                added. If there are other types of distortions which appear first, they should already be included in
                the lead transitions.
            volt_limits_g1 (np.ndarray): Contains the beginning and ending of the swept range for the first plunger
                gate.
            volt_limits_g2 (np.ndarray): Contains the beginning and ending of the swept range for the second plunger
                gate.
            generate_csd (Callable): Function which generates data points outside the swept gate range. This is
                especially required for distortions, which shift the CSD structure. The generated data points also have
                to contain the distortions, which have already been added to the occupation and lead_transitions before.
            scale (float): Specifies the length of the shifts, gives the mathematical expectation for their length. In
                this case, scale specifies the width of the block in which a shift is present. Given in pixels.
            lam (float): Lambda for the poisson distribution which specifies the height of the jumps. Given in pixels.
            use_previous_noise (bool): indicates if the previously generated noise should be used again. This is
                especially needed, if the noise function is used inside a generate_csd-function

        """
        if use_previous_noise:
            noise = self.__previous_noise
        else:
            noise = generate_dot_jumps_blockwise(occupation[:, :, 0].shape, scale, lam, axis=self.axis, rng=self.rng)
            self.__previous_noise = noise
        max_jump = np.max(noise)
        if generate_csd is not None and max_jump > 0:
            added_pixels = max_jump
            if self.axis == 1:  # jumps in x direction
                # generate the necessary part to the "left" of the CSD and stack to CSD
                step_size = (volt_limits_g1[1] - volt_limits_g1[0]) / occupation.shape[1]
                left_occupation, left_charge_transitions = generate_csd(
                    volt_limits_g1=(
                        volt_limits_g1[0] - max_jump * step_size,
                        volt_limits_g1[0] - step_size,
                    ),
                    volt_limits_g2=volt_limits_g2,
                    resolution=(max_jump, occupation.shape[0]) if max_jump > 1 else (occupation.shape[0]),
                )
                left_occupation = np.reshape(left_occupation, (occupation.shape[1], max_jump, 2))
                left_charge_transitions = np.reshape(left_charge_transitions, (occupation.shape[1], max_jump))
                occupation_stacked = np.hstack((left_occupation, occupation))
                if lead_transitions is not None:
                    total_charge_transitions_stacked = np.hstack((left_charge_transitions, lead_transitions))
            elif self.axis == 0:  # jumps in y direction
                # generate the necessary part to the "top" of the CSD array and stack to CSD
                step_size = (volt_limits_g2[1] - volt_limits_g2[0]) / occupation.shape[0]
                top_occupation, top_charge_transitions = generate_csd(
                    volt_limits_g1=volt_limits_g1,
                    volt_limits_g2=(
                        volt_limits_g2[0] - max_jump * step_size,
                        volt_limits_g2[0] - step_size,
                    ),
                    resolution=(occupation.shape[1], max_jump) if max_jump > 1 else (occupation.shape[1]),
                )
                top_occupation = np.reshape(top_occupation, (max_jump, occupation.shape[1], 2))
                top_charge_transitions = np.reshape(top_charge_transitions, (max_jump, occupation.shape[1]))
                occupation_stacked = np.vstack((top_occupation, occupation))
                if lead_transitions is not None:
                    total_charge_transitions_stacked = np.vstack((top_charge_transitions, lead_transitions))
        else:
            added_pixels = 0
            occupation_stacked = occupation
            if lead_transitions is not None:
                total_charge_transitions_stacked = lead_transitions
        # apply distortions and write the original part of the occupations back to the
        # given occupation array again
        if self.axis == 1:
            occupation[:, :, 0] = dot_jumps_blockwise(
                occupation_stacked[:, :, 0],
                scale,
                lam,
                np.concatenate((np.zeros(added_pixels, dtype=int), noise)),
                axis=self.axis,
                rng=self.rng,
            )[0][:, added_pixels:]
            occupation[:, :, 1] = dot_jumps_blockwise(
                occupation_stacked[:, :, 1],
                scale,
                lam,
                np.concatenate((np.zeros(added_pixels, dtype=int), noise)),
                axis=self.axis,
                rng=self.rng,
            )[0][:, added_pixels:]
            # adapt total charge transitions accordingly
            if lead_transitions is not None:
                lead_transitions[:, :] = dot_jumps_blockwise(
                    total_charge_transitions_stacked,
                    scale,
                    lam,
                    np.concatenate((np.zeros(added_pixels, dtype=int), noise)),
                    axis=self.axis,
                    rng=self.rng,
                )[0][:, added_pixels:]
        elif self.axis == 0:
            occupation[:, :, 0] = dot_jumps_blockwise(
                occupation_stacked[:, :, 0],
                scale,
                lam,
                np.concatenate((np.zeros(added_pixels, dtype=int), noise)),
                axis=self.axis,
                rng=self.rng,
            )[0][added_pixels:, :]
            occupation[:, :, 1] = dot_jumps_blockwise(
                occupation_stacked[:, :, 1],
                scale,
                lam,
                np.concatenate((np.zeros(added_pixels, dtype=int), noise)),
                axis=self.axis,
                rng=self.rng,
            )[0][added_pixels:, :]
            # adapt total charge transitions accordingly
            if lead_transitions is not None:
                lead_transitions[:, :] = dot_jumps_blockwise(
                    total_charge_transitions_stacked,
                    scale,
                    lam,
                    np.concatenate((np.zeros(added_pixels, dtype=int), noise)),
                    axis=self.axis,
                    rng=self.rng,
                )[0][added_pixels:, :]

    def __repr__(self):
        return self.__class__.__name__ + (
            f"(scale={self.scale}, lam={self.lam}, axis={self.axis}, "
            f"ratio={self.ratio}, rng={self.rng}){'[activated]' if self.activated else '[deactivated]'}"
        )


def generate_dot_jumps_blockwise(
    shape: tuple,
    scale: float,
    lam: float,
    axis: int = 1,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """Generates blockwise dot jumps, so that they can later be applied for the occupations in dot 1 and dot 2.

    Args:
        shape (tuple): The shape of the CSD for which dot jumps will be generated.
        scale (float): Specifies the length of the shifts, gives the mathematical expectation for their length. In
            this case, scale specifies the width of the block in which a shift is present. Given in pixels.
        lam (float): Lambda for the poisson distribution which specifies the height of the jumps. Given in pixels.
        axis (int): The axis along which dot jumps will be generated. That means the values are shifted along this axis.
        rng (np.random.Generator) : The random number generator used for the simulation of random numbers. Default is
            np.random.default_rng().

    Returns:
        np.ndarray: Generated dot jumps, which can be applied to the image with the help of the function
        dot_jumps_blockwise.

    """
    if scale == 0:
        return np.zeros(shape[axis])
    size = shape[axis]
    # width of a shift in the CSD follows a geometric distribution
    width = rng.geometric(p=1 / scale, size=size)

    noise = np.array([])
    bi = 0  # set binary states, 0 means no shift, 1 means shift
    for i in range(size):
        if width[i] > 0:
            noise = np.concatenate((noise, np.array([bi] * int(width[i])) * rng.poisson(lam)))
            bi = 1 if bi == 0 else 0
        if len(noise) >= size:
            break

    noise = noise[:size]
    noise = noise.astype("uint8")
    return noise


def dot_jumps_blockwise(
    original: np.ndarray,
    scale: float,
    lam: float,
    noise: np.ndarray = None,
    axis: int = 1,
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Adds dot jumps to the original image.

    In this case that means, that a whole block of columns can be shifted in x direction (axis=1) or a block of lines
    can be shifted in y direction (axis=0).

    When this is applied to charge stability diagrams (CSDs) it should be applied before any other distortion type
    directly to the occupation of the dots. It can be applied for the occupation of the first dot and after that the
    generated distortion which is returned can be used for the second dot. This should be done because otherwise
    different blocks are shifted for the occupations of the first and the second dot.

    Args:
        original (np.ndarray): Original image where the distortions should be added. Can be lead transition mask or
            occupations.
        scale (float): Specifies the length of the shifts, gives the mathematical expectation for their length. In
            this case, scale specifies the width of the block in which a shift is present. Given in pixels.
        lam (float): Lambda for the poisson distribution which specifies the height of the jumps. Given in pixels.
        noise (np.ndarray): Noise which was generated with this method for a prior image and should be reapplied for
            this image. Default is None.
        axis (int): The axis along which dot jumps will be generated. That means the values are shifted along this axis.
        rng (np.random.Generator): The random number generator used for the simulation of random numbers. Default is
            np.random.default_rng().

    Returns:
        np.ndarray, np.ndarray: Distorted version of the original image and pure distortion which was used for this
        image and could be used again for the second dot of the dataset.

    """
    if scale == 0:
        return original, np.zeros(original.shape[axis])
    if noise is None:
        noise = generate_dot_jumps_blockwise(original.shape, scale, lam, axis, rng)

    shifted = np.empty(original.shape)
    rows = original.shape[0]
    cols = original.shape[1]
    if axis == 1:
        for i in range(rows):
            for j in range(cols):
                # if there is a jump present the pixels are shifted by distortions[j]
                # in all rows
                for k in range(0, noise[j] + 1):
                    if j + k >= cols:
                        break
                    # Nan values are used as placeholder for 1d scans, so ignore them
                    if np.isnan(original[i, np.maximum(0, j - noise[j] + k)]):
                        shifted[i, j + k] = original[i, j + k]
                        continue
                    shifted[i, j + k] = original[i, np.maximum(0, j - noise[j] + k)]
                j = j + noise[j]
    elif axis == 0:
        for i in range(cols):
            for j in range(rows):
                # if there is a jump present the pixels are shifted by distortions[j]
                # in all columns
                for k in range(0, noise[j] + 1):
                    if j + k >= rows:
                        break
                    # Nan values are used as placeholder for 1d scans, so ignore them
                    if np.isnan(original[np.maximum(0, j - noise[j] + k), i]):
                        shifted[j + k, i] = original[j + k, i]
                        continue
                    shifted[j + k, i] = original[np.maximum(0, j - noise[j] + k), i]
                j = j + noise[j]
    else:
        raise NotImplementedError("Dot jumps are only supported for 2d array. Axis should be 0 or 1.")

    return shifted, noise


def dot_jumps_pixelwise(
    original: np.ndarray,
    scale: float,
    lam: float,
    noise: Union[np.ndarray, None] = None,
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Adds dot jumps to the original image (pixelwise).

    In this case, that means, that shifts are occurring during the measuring of the lines. That means the voltages
    applied at the plunger gates do not have any effect on the occurrence of the shifts. Using this pixelwise
    implementation, the jump is not happening in every row/column at a specific voltages, but the data is rather seen
    as 1d data in which the jumps happened randomly.

    When this is applied to charge stability diagrams (CSDs) it should be applied before any other distortion type
    directly to the occupation of the dots. It can be applied for the occupation of the first dot and after that the
    generated distortion which is returned can be used for the second dot.

    Args:
        original (np.ndarray): Original image where the distortions should be added. Can be lead transition mask or
            occupations.
        scale (float): Specifies the length of the shifts, gives the mathematical expectation for their length. In
            this case, scale specifies the length in pixels, so that a jump can start in the middle of a line and end in
            the middle of the next line. Given in pixels.
        lam (float): Lambda for the poisson distribution which specifies the height of the jumps. Given in pixels.
        noise (Union[np.ndarray, None]): Noise which was generated with this method for a prior image and should be reapplied for
            this image. Default is None.
        rng (np.random.Generator): The random number generator used for the simulation of random numbers. Default is
            np.random.default_rng().

    Returns:
        np.ndarray, np.ndarray: Distorted version of the original image and pure distortion which was used for this
        image and could be used again for the second dot of the dataset.

    """
    if scale == 0:
        return original, np.zeros(original.shape)
    if noise is None:
        size = np.prod(original.shape)
        # length of a shift in the CSD follows a geometric distribution
        length = rng.geometric(p=1 / scale, size=size)

        noise = np.array([])
        bi = 0  # set binary states, 0 means no shift, 1 means shift
        for i in range(size):
            if length[i] > 0:
                noise = np.concatenate((noise, np.array([bi] * int(length[i])) * rng.poisson(lam)))
                bi = 1 if bi == 0 else 0
            if len(noise) >= size:
                break

        noise = noise[:size]
        noise = noise.reshape(original.shape)
        noise = noise.astype("uint8")

    shifted = np.empty(original.shape)
    rows = original.shape[0]
    cols = original.shape[1]
    for i in range(rows):
        for j in range(cols):
            # if there is a jump present the pixels are shifted by distortions[i, j]
            for k in range(0, noise[i, j] + 1):
                if j + k >= cols:
                    break
                shifted[i, j + k] = original[i, np.maximum(0, j - noise[i, j] + k)]
            j = j + noise[i, j]

    return shifted, noise
