"""
SimCATS subpackage containing all functionalities related to the distortion simulation.
"""

from simcats.distortions._distortion_interfaces import DistortionInterface, OccupationDistortionInterface, \
    SensorPotentialDistortionInterface, SensorResponseDistortionInterface
from simcats.distortions._transition_blurring import OccupationTransitionBlurringGaussian, \
    OccupationTransitionBlurringFermiDirac
from simcats.distortions._dot_jumps import OccupationDotJumps, dot_jumps_blockwise
from simcats.distortions._pink_noise import SensorPotentialPinkNoise, pink_gaussian_noise
from simcats.distortions._random_telegraph_noise import RandomTelegraphNoise, SensorPotentialRTN, SensorResponseRTN, \
    random_telegraph_noise
from simcats.distortions._white_noise import SensorResponseWhiteNoise, white_gaussian_noise

__all__ = ["DistortionInterface", "OccupationDistortionInterface", "SensorPotentialDistortionInterface",
           "SensorResponseDistortionInterface", "OccupationTransitionBlurringGaussian",
           "OccupationTransitionBlurringFermiDirac", "OccupationDotJumps", "SensorPotentialPinkNoise",
           "RandomTelegraphNoise", "SensorPotentialRTN", "SensorResponseRTN",
           "SensorResponseWhiteNoise", "dot_jumps_blockwise", "pink_gaussian_noise", "random_telegraph_noise",
           "white_gaussian_noise"]
