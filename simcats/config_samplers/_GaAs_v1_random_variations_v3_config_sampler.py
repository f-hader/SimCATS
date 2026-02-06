"""
This module contains a sampler for SimCATS configurations, corresponding to the random_variations_v3 parameters.
The configurations can be used to initialize an object of the simulation class.

@author: f.hader, b.papajewski
"""

import math
import random
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

from simcats import Simulation, default_configs
from simcats.distortions import (
    OccupationTransitionBlurringFermiDirac,
    SensorPotentialPinkNoise,
    SensorResponseWhiteNoise,
    OccupationDistortionInterface, SensorPotentialDistortionInterface, SensorResponseDistortionInterface,
    SensorPotentialRTN, SensorResponseRTN, OccupationDotJumps,
)
from simcats.ideal_csd import IdealCSDGeometric
from simcats.sensor import SensorGeneric, SensorPeakLorentzian, SensorScanSensorGeneric, SensorPeakInterface, \
    SensorRiseGLF, SensorRiseInterface
from simcats.sensor.barrier_function import BarrierFunctionGLF, BarrierFunctionInterface
from simcats.sensor.deformation import SensorPeakDeformationLinear, SensorPeakDeformationCircle
from simcats.support_functions import (
    NormalSamplingRange,
    ExponentialSamplingRange,
    rotate_points, UniformSamplingRange,
    inverse_glf
)

__all__ = []


def sample_random_variations_v3_config(sensor_type: str = "SensorGeneric",
                                       quality_multiplier: float = 1.0,
                                       compensated_sensor: bool = False,
                                       set_sensor_potential_offset_to_steepest_point: bool = False) -> Dict:
    """
    Samples a full random_variations_v3 SimCATS config.

    Args:
        sensor_type: The type of sensor implementation to use. Can be either "SensorGeneric" or
            "SensorScanSensorGeneric". Defaults to "SensorGeneric".
        quality_multiplier: The quality multiplier to use, which defines how much of the ranges for distortions etc. is
            used. This basically allows to restrict configurations to better data quality, as future improvements of the
            sample quality are expected. The parameter further restricts the preset ranges for distortions (reducing the
            allowed maximum for distortions) and things like the height of Coulomb peaks (increasing the minimum). The
            factor must be in the range (0.0, 1.0]. Defaults to 1.0.
        compensated_sensor: Whether the effect of the double quantum dot on the sensor dot is compensated (lever-arms of
            the double dot gate voltages on the sensor are close to zero). Defaults to False.
        set_sensor_potential_offset_to_steepest_point: Whether the steepest point of the sensor function is determined
            and the corresponding potential is set as the potential offset (offset_mu_sens). This enables simple
            "retuning" of the sensor by always resetting the potential offset before every CSD measurement. Defaults to
            False.

    Returns:
        Dict: Full config for the SimCATS Simulation class.
    """
    if not (quality_multiplier > 0.0 and quality_multiplier <= 1.0):
        raise ValueError(f"quality_multiplier must be in the range (0.0, 1.0], but {quality_multiplier} is given.")
    config = dict()
    config["volt_limits_g1"] = np.array([-0.2, -0.06])
    config["volt_limits_g2"] = np.array([-0.2, -0.06])
    config["volt_limits_sensor_g1"] = np.array([-1, 0])
    config["volt_limits_sensor_g2"] = np.array([-1, 0])
    config["ideal_csd_config"] = sample_IdealCSDGeometric()
    if sensor_type == "SensorGeneric":
        config["sensor"] = sample_SensorGeneric(quality_multiplier=quality_multiplier,
                                                compensated_sensor=compensated_sensor,
                                                set_sensor_potential_offset_to_steepest_point=set_sensor_potential_offset_to_steepest_point)
    elif sensor_type == "SensorScanSensorGeneric":
        config["sensor"] = sample_SensorScanSensorGeneric(quality_multiplier=quality_multiplier,
                                                          compensated_sensor=compensated_sensor
                                                          )
    else:
        raise ValueError(
            f"The parameter sensor_type must be either 'SensorGeneric' or 'SensorScanSensorGeneric'. The supplied value"
            f" '{sensor_type}' is not supported.")
    config["occupation_distortions"] = sample_occupation_distortions()
    config["sensor_potential_distortions"] = sample_sensor_potential_distortions()
    config["sensor_response_distortions"] = sample_sensor_response_distortions()
    return config


def sample_IdealCSDGeometric() -> IdealCSDGeometric:
    """
    Samples a random_variations_v3 IdealCSDGeometric config.

    Returns:
        IdealCSDGeometric: IdealCSDGeometric object to be used as ideal_csd_config for the SimCATS Simulation class.
    """
    # slopes of the lead transitions in 45°-rotated space
    slope1_min = 0.21429
    slope1_max = 0.54688
    slope2_min = -0.44
    slope2_max = -0.07692
    # angle between lead transitions
    angle_lead_min = 0.43760867228
    angle_lead_max = 1.6789624091
    # lengths of the lead transitions (between minima and maxima of a TCT)
    length_min = 0.01
    length_max = 0.025
    # angle between interdot and first lead transition (slope1 is rotated counter-clockwise by this angle to get the direction of the interdot vector)
    angle_interdot_min = 0.58153916891
    angle_interdot_max = 1.396938844
    # length of the interdot transition
    interdot_length_min = 0.00261
    interdot_length_max = 0.00987
    interdot_length_mean = 0.004  # Currently not used, as we sample interdot length with a uniform distribution. Kept for the sake of completeness.
    interdot_length_std = 0.0015  # Currently not used, as we sample interdot length with a uniform distribution. Kept for the sake of completeness.
    # width of the interdot transition, larger values lead to more curved TCTs
    interdot_width_min = 0.00043
    interdot_width_max = 0.00814
    # relation between width and length of the interdot transition
    rel_interdot_length_width_min = 0.8692269873603532
    rel_interdot_length_width_max = 9.055385138137407
    # limits for the plunger gate voltages used for the simulation
    volt_limits_g1 = np.array([-0.2, -0.06])
    volt_limits_g2 = np.array([-0.2, -0.06])
    # place for the first bezier anchor, which determines were the first interdot transition is located
    first_bezier_point_min, first_bezier_point_max = -0.14, -0.14
    # lookup table entries to use for the calculation of the tct_bezier
    lut_entries = 1000

    # create random number generator
    rng = np.random.default_rng()

    # sample slope
    slope1 = np.array([1, rng.uniform(slope1_min, slope1_max)])
    slope2 = np.array([1, rng.uniform(slope2_min, slope2_max)])

    angle_lead = np.arccos(slope1.dot(slope2) / (np.linalg.norm(slope1) * np.linalg.norm(slope2)))

    # create the IdealCSDGeometric object
    # fresh default config
    ideal_csd_geometric = deepcopy(default_configs["GaAs_v1"]["ideal_csd_config"])

    # basic settings regarding volt_limits and lookup table entries
    ideal_csd_geometric.lut_entries = lut_entries

    # resample slope if angle_lead is not between angle_lead_min and angle_lead_max
    while not (angle_lead_min <= angle_lead <= angle_lead_max):
        slope1 = np.array([1, rng.uniform(slope1_min, slope1_max)])
        slope2 = np.array([1, rng.uniform(slope2_min, slope2_max)])
        angle_lead = np.arccos(slope1.dot(slope2) / (np.linalg.norm(slope1) * np.linalg.norm(slope2)))

    # sample lengths
    lengths = [
        rng.uniform(length_min, length_max),
        rng.uniform(length_min, length_max),
    ]

    # sample interdot_vec
    interdot_vec = rotate_points(slope1, rng.uniform(angle_interdot_min, angle_interdot_max))
    # rotate interdot_vec like it would be in the CSD space
    interdot_vec_rot = rotate_points(interdot_vec, -np.pi / 4)
    while interdot_vec_rot[0] < 0 or interdot_vec_rot[1] < 0:
        # interdot vector should not go backwards in the image space, so resample if the rotated vector has negative entries
        interdot_vec = rotate_points(slope1, rng.uniform(angle_interdot_min, angle_interdot_max))
        # rotate interdot_vec
        interdot_vec_rot = rotate_points(interdot_vec, -np.pi / 4)
    interdot_vec = interdot_vec / np.linalg.norm(interdot_vec)

    # sample interdot_length
    # usage of uniform distribution to get more diverse data
    interdot_length = rng.uniform(interdot_length_min, interdot_length_max)
    while interdot_length < interdot_length_min or interdot_length > interdot_length_max:
        interdot_length = rng.uniform(interdot_length_min, interdot_length_max)

    # sample bezier_width
    bezier_width = rng.uniform(interdot_width_min, interdot_width_max)
    # resample interdot_length and bezier_width until interdot_length / bezier_width is between rel_interdot_length_width_min and rel_interdot_length_width_max
    while not (
            rel_interdot_length_width_min <= interdot_length / bezier_width
            and interdot_length / bezier_width <= rel_interdot_length_width_max
    ):
        interdot_length = rng.uniform(interdot_length_min, interdot_length_max)
        bezier_width = rng.uniform(interdot_width_min, interdot_width_max)
    interdot_vec = interdot_vec * interdot_length

    # place first bezier_point in lower left corner
    bezier_point = rng.uniform(first_bezier_point_min, first_bezier_point_max, size=2)
    bezier_point = rotate_points(bezier_point)

    # calculate the second bezier point, so that it is in the correct direction
    # direction for bezier_point is the mean of both slopes
    direction = np.array([1, (slope1[1] + slope2[1]) / 2])
    direction = direction / np.linalg.norm(direction)
    bezier_point2 = bezier_point + bezier_width * direction

    # set up shift vector to calculate next tct params
    shift_vec = np.array([-lengths[1] + interdot_vec[0], interdot_vec[1] - lengths[1] * slope2[1]])

    # set up first tct params
    temp_params = np.array(
        [
            lengths[0],
            lengths[1],
            slope1[1],
            slope2[1],
            bezier_point[0],
            bezier_point[1],
            bezier_point2[0],
            bezier_point2[1],
        ]
    )

    # Add as many TCTs as required, by using a sensor and testing if sufficient TCTs were added to cover the whole
    # voltage space
    ideal_csd_geometric.tct_params = [temp_params]
    running = True
    while running:
        try:
            # set up simulation
            temp_config = dict()
            temp_config["volt_limits_g1"] = volt_limits_g1
            temp_config["volt_limits_g2"] = volt_limits_g2
            temp_config["ideal_csd_config"] = ideal_csd_geometric
            sim = Simulation(**temp_config)
            # measure
            _ = sim.measure(volt_limits_g1, volt_limits_g2, resolution=np.array([10, 10]))
        except IndexError as e:
            ideal_csd_geometric.tct_params = _add_one_tct(ideal_csd_geometric.tct_params, shift_vec)
        else:
            running = False

    return ideal_csd_geometric


def sample_SensorGeneric(quality_multiplier: float = 1.0,
                         compensated_sensor: bool = False,
                         set_sensor_potential_offset_to_steepest_point: bool = False) -> SensorGeneric:
    """
    Sample parameters and initialize a SensorGeneric object.

    Args:
        quality_multiplier: The quality multiplier to use, which defines how much of the ranges for Coulomb peak heights
            etc. is used. This basically allows to restrict configurations to better data quality, as future
            improvements of the sample quality are expected. The parameter further restricts the preset ranges for
            things like the height of Coulomb peaks (increasing the minimum) and the strength of the lever-arms of the
            double quantum dot gates towards the sensor. The factor must be in the range (0.0, 1.0]. Defaults to 1.0.
        compensated_sensor: Whether the effect of the double quantum dot on the sensor dot is compensated (lever-arms of
            the double dot gate voltages on the sensor are close to zero). Defaults to False.
        set_sensor_potential_offset_to_steepest_point: Whether the steepest point of the sensor function is determined
            and the corresponding potential is set as the potential offset (offset_mu_sens). This enables simple
            "retuning" of the sensor by always resetting the potential offset before every CSD measurement. Defaults to
            False.

    Returns:
        SensorGeneric: Fully initialized SensorGeneric object.
    """
    # limits for the plunger gate voltages used for the simulation
    volt_limits_g1 = np.array([-0.2, -0.06])
    volt_limits_g2 = np.array([-0.2, -0.06])
    num_sensor_peak_min, num_sensor_peak_max = 3, 6
    # s_off is the offset of the peaks (y)
    sensor_offset_min, sensor_offset_max = -0.42275, -0.0838
    # mu_0 (offset_mu_sens) from sensor (x)
    sensor_mu_0_min, sensor_mu_0_max = -0.12168, -0.03824
    sensor_height_min, sensor_height_max = 0.02245, 0.19204
    sensor_gamma_min, sensor_gamma_max = 0.0009636, 0.0029509
    if compensated_sensor:
        sensor_alpha_gate_1_min, sensor_alpha_gate_1_max = 0.0001, 0.0005
        sensor_alpha_gate_2_min, sensor_alpha_gate_2_max = 0.0001, 0.0005  # Currently not used, as we sample alpha gate 2 in dependence on alpha gate 1. Kept for the sake of completeness.
    else:
        sensor_alpha_gate_1_min, sensor_alpha_gate_1_max = 0.02805, 0.15093
        sensor_alpha_gate_2_min, sensor_alpha_gate_2_max = 0.03788, 0.19491  # Currently not used, as we sample alpha gate 2 in dependence on alpha gate 1. Kept for the sake of completeness.
    sensor_alpha_dot_1_min, sensor_alpha_dot_1_max = -0.0007994, -0.0000961
    sensor_alpha_dot_1_max = sensor_alpha_dot_1_min + quality_multiplier * (
            sensor_alpha_dot_1_max - sensor_alpha_dot_1_min)
    sensor_alpha_dot_2_min, sensor_alpha_dot_2_max = -0.0005214, -0.0000630
    sensor_alpha_dot_2_max = sensor_alpha_dot_2_min + quality_multiplier * (
            sensor_alpha_dot_2_max - sensor_alpha_dot_2_min)

    quantile_relative_pos = 0.6 * quality_multiplier
    quantile_percentage = 0.99
    exp_scale_sensor_height = quantile_relative_pos / np.log(1 / (1 - quantile_percentage))

    # create random number generator
    rng = np.random.default_rng()

    # sample alpha gates and alpha dots
    sensor_alpha_gate_1 = rng.uniform(low=sensor_alpha_gate_1_min, high=sensor_alpha_gate_1_max)
    sensor_alpha_gate_2 = rng.uniform(low=0.5 * sensor_alpha_gate_1, high=2 * sensor_alpha_gate_1)

    # sample alpha dot 1 depending on alpha gate 1
    # -0.03 / 100 = at least as much change for occ change (alpha dot) as change for sweeping one pixel (alpha gate)
    # Keep in mind negative means, max is actually min (think of absolute values).
    # Set to 2* alpha gate per pixel value, because stepping one pixel adds 1* and we want to jump back at least
    # 1* pixel step (starting with random_variations_v2).
    # Should actually always ensure to have higher min sensitivity (max = min because of negative values). Therefore,
    # np.min with the actual value has been added (starting with random_variations_v3).
    temp_sensor_alpha_dot_1_max = np.min([2 * sensor_alpha_gate_1 * -0.03 / 100,
                                          sensor_alpha_dot_1_max])
    sensor_alpha_dot_1 = rng.uniform(low=sensor_alpha_dot_1_min, high=temp_sensor_alpha_dot_1_max)

    # sample alpha dot 2 depending on alpha gate 2 and alpha dot 1
    # Set to 2* alpha gate per pixel value, because stepping one pixel adds 1* and we want to jump back at least
    # 1* pixel step (starting with random_variations_v2).
    # Should actually always ensure to have higher min sensitivity (max = min because of negative values). Therefore,
    # np.min with the actual value has been added (starting with random_variations_v3).
    temp_sensor_alpha_dot_2_max = np.min([0.5 * sensor_alpha_dot_1, 2 * sensor_alpha_gate_2 * -0.03 / 100,
                                          sensor_alpha_dot_2_max])
    if temp_sensor_alpha_dot_2_max < sensor_alpha_dot_2_min:
        temp_sensor_alpha_dot_2_max = sensor_alpha_dot_2_max
    # Should actually only ensure to be at most twice as strongly coupled compared to dot 1, but never stronger than
    # given by the pre-defined range. Therefore, np.max with the actual value has been added (starting with
    # random_variations_v3).
    temp_sensor_alpha_dot_2_min = np.max([2 * sensor_alpha_dot_1, sensor_alpha_dot_2_min])
    sensor_alpha_dot_2 = rng.uniform(low=temp_sensor_alpha_dot_2_min, high=temp_sensor_alpha_dot_2_max)

    sensor = SensorGeneric(
        sensor_peak_function=None,
        alpha_dot=np.array([sensor_alpha_dot_1, sensor_alpha_dot_2]),
        alpha_gate=np.array([sensor_alpha_gate_1, sensor_alpha_gate_2]),
    )

    # calculate potential to find mu_sens limits
    occupations = np.array([[0.0, 0.0], [5.0, 5.0]])
    potentials = sensor.sensor_potential(occupations=occupations,
                                         volt_limits_g1=volt_limits_g1,
                                         volt_limits_g2=volt_limits_g2)
    sensor_mu_0_min, sensor_mu_0_max = np.min(potentials), np.max(potentials)

    # sample number of peaks
    num_sensor_peak = rng.choice(np.arange(num_sensor_peak_min, num_sensor_peak_max + 1))

    sensor_peak_functions = list()

    sensor_height_sampler = ExponentialSamplingRange(
        total_range=(sensor_height_max, sensor_height_min),
        scale=exp_scale_sensor_height
    )  # using other sampler starting with random_variations_v2

    # sample individual peaks
    for i in range(num_sensor_peak):
        sensor_gamma = rng.uniform(low=sensor_gamma_min, high=sensor_gamma_max)
        sensor_mu_0 = rng.uniform(low=sensor_mu_0_min, high=sensor_mu_0_max)
        if i > 0:  # added a rule for the variation of mu0 between peaks of the same config (starting with random_variations_v2)
            # make sure that sensors have at least (0.5 * potential_range) / number_of_peaks distance (for mu0)
            counter = 0
            # introduced this counter, because we could end up in a case where it is not possible to place the next peak,
            # because two peaks can be slightly below twice min required distance apart from each other and therefore block 2x
            # and not just 1.5x (sensor_mu_0_max - sensor_mu_0_min) / (num_sensor_peak). Try 1000 times and else allow 2x.
            while (np.min([np.abs(sensor_mu_0 - sampled_peak.mu0) for sampled_peak in sensor_peak_functions]) <= (
                    sensor_mu_0_max - sensor_mu_0_min) / (num_sensor_peak * 1.5) and counter < 1000) or np.min(
                [np.abs(sensor_mu_0 - sampled_peak.mu0) for sampled_peak in sensor_peak_functions]) <= (
                    sensor_mu_0_max - sensor_mu_0_min) / (num_sensor_peak * 2):
                sensor_mu_0 = rng.uniform(low=sensor_mu_0_min, high=sensor_mu_0_max)
                counter += 1
        if i == 0:  # added a rule for the variation of peak_offset between peaks of the same config (starting with random_variations_v2)
            sensor_peak_offset = rng.uniform(low=sensor_offset_min, high=sensor_offset_max)
        else:
            # make sure to have at most 20% difference in peak_offset to first peak_offset
            sensor_peak_offset = rng.uniform(low=sensor_peak_functions[0].offset * 1.2,
                                             high=sensor_peak_functions[0].offset * 0.8)
        if i == 0:  # added a rule for the variation of height between peaks of the same config (starting with random_variations_v2)
            sensor_height = sensor_height_sampler.sample_parameter()
        else:
            # make sure to have at most 20% difference in height to first height
            sensor_height = rng.uniform(low=sensor_peak_functions[0].height * 0.8,
                                        high=sensor_peak_functions[0].height * 1.2)
        sensor_peak_functions.append(
            SensorPeakLorentzian(
                gamma=sensor_gamma,
                mu0=sensor_mu_0,
                height=sensor_height,
                offset=sensor_peak_offset,
            )
        )
    sensor.sensor_peak_function = sensor_peak_functions

    if set_sensor_potential_offset_to_steepest_point:
        # calculate steepest point of sensor peak flank for offset_mu_sens (used for the assumption of sensor retuning
        # to shift sensor to this point)
        sample_mu_sens = np.linspace(sensor_mu_0_min - 0.2 * (sensor_mu_0_max - sensor_mu_0_min),
                                     sensor_mu_0_max + 0.2 * (sensor_mu_0_max - sensor_mu_0_min),
                                     1000000)
        sample_sens_response = sensor.sensor_response(sample_mu_sens)
        sample_sens_response_grad = np.gradient(sample_sens_response)
        sensor.offset_mu_sens = sample_mu_sens[np.argmax(sample_sens_response_grad)]

    return sensor


def _sensor_peak_list_add_peak(peak_list: List[SensorPeakInterface],
                              ref_height: float,
                              base_offset: float = 0) -> list[SensorPeakInterface]:
    """
    Method to add a single lorentzian peak to a list of sensor peaks.

    This function is only required when sampling a SensorScanSensorGeneric.

    Args:
        peak_list (List[SensorPeakInterface]): List of sensor peaks to which a peak is added
        ref_height (float): Reference height of all peaks. Based on this, the height of the new peak is sampled at
            random. The reference height is used as the mean value of a distribution from which the height of the new
            peak is sampled.
        base_offset (float): Potential offset of the first sensor peak. The default value is 0. This value has no effect
            for every other peak.

    Returns:
        List[SensorPeakInterface]: The list of sensor peaks with an extra lorentzian peak added.
    """
    mu_dif_absolute_initial_lower_limit = 0
    mu_dif_absolute_initial_upper_limit = 0.00367179654343025 * 4
    mu_dif_absolute_initial_std = 0.00023884164727643042 / 2 # MAD
    # mu_dif_absolute_initial_mean = 0.002902416771776037 * 1.5  # MEDIAN
    mu_dif_absolute_initial_mean = 0.002902416771776037 * 2.5  # MEDIAN
    mu_dif_relative_std = 0.09509839063162284 / 2 # MAD
    mu_dif_relative_mean_fitted_line_slope = -0.0374000174907983 / 4
    mu_dif_relative_max_deviation = 1
    mu_dif_relative_mean_fitted_line_intercept = 1.0189427890611187
    mu_dif_relative_line_max_peak_num = 10

    gamma_absolute_initial_std = 3.6990706645585366e-05 /2 # MAD
    gamma_absolute_initial_mean = 9.878642853157584e-05 * 2.5  # MEDIAN
    gamma_relative_mean_fitted_line_slope = 0
    gamma_relative_mean_fitted_line_intercept = 1.25
    gamma_relative_std = 0.4695086254881995 / 2
    gamma_relative_lower_limit = 0.85
    gamma_relative_higher_limit = 1.5
    gamma_rel_line_max_peak_num = 5
    gamma_max_value = 0.002 #0.0025
    gamma_max_value_deviation_percentage = 0.1

    height_std = 0.04720315300812465 / 2  # MAD

    if len(peak_list) == 0:
        mu0 = base_offset

        gamma = NormalSamplingRange(
            total_range=(0, np.inf),
            std=gamma_absolute_initial_std,
            mean=gamma_absolute_initial_mean
        ).sample_parameter()

    elif len(peak_list) == 1:
        mu_dif = NormalSamplingRange(
            total_range=(mu_dif_absolute_initial_lower_limit, mu_dif_absolute_initial_upper_limit),
            std=mu_dif_absolute_initial_std,
            mean=mu_dif_absolute_initial_mean
        ).sample_parameter()
        mu0 = peak_list[0].mu0 + mu_dif

        gamma_rel_dif = NormalSamplingRange(
            total_range=(gamma_relative_lower_limit, gamma_relative_higher_limit),
            std=gamma_relative_std,
            mean=gamma_relative_mean_fitted_line_slope * (
                    len(peak_list) - 1) + gamma_relative_mean_fitted_line_intercept
        ).sample_parameter()
        gamma = peak_list[-1].gamma * gamma_rel_dif

    else:
        mu_factor = peak_list[1].mu0 - peak_list[0].mu0
        mu_peak_num = len(peak_list) if len(
            peak_list) < mu_dif_relative_line_max_peak_num else mu_dif_relative_line_max_peak_num
        mu_rel_dif = NormalSamplingRange(
            total_range=(
                -mu_dif_relative_max_deviation * mu_dif_relative_std,
                mu_dif_relative_max_deviation * mu_dif_relative_std),
            std=mu_dif_relative_std,
            mean=0,
        ).sample_parameter() + mu_dif_relative_mean_fitted_line_slope * (
                             mu_peak_num - 1) + mu_dif_relative_mean_fitted_line_intercept

        mu0 = peak_list[-1].mu0 + mu_rel_dif * mu_factor

        gamma_peak_num = len(peak_list) if len(peak_list) < gamma_rel_line_max_peak_num else gamma_rel_line_max_peak_num
        gamma_rel_dif = NormalSamplingRange(
            total_range=(gamma_relative_lower_limit, gamma_relative_higher_limit),
            std=gamma_relative_std,
            mean=gamma_relative_mean_fitted_line_slope * (
                    gamma_peak_num - 1) + gamma_relative_mean_fitted_line_intercept
        ).sample_parameter()
        gamma = peak_list[-1].gamma * gamma_rel_dif

        if gamma > gamma_max_value:
            gamma = UniformSamplingRange(
                total_range=(gamma_max_value * (1 - gamma_max_value_deviation_percentage),
                             gamma_max_value * (1 + gamma_max_value_deviation_percentage)),
            ).sample_parameter()

    height = ref_height + NormalSamplingRange(
        total_range=(-ref_height, np.inf),
        std=height_std,
        mean=0
    ).sample_parameter()

    return peak_list + [SensorPeakLorentzian(
        mu0=mu0,
        gamma=gamma,
        height=height,
        offset=0,  # A minimum sensor response of 0 is assumed
    ), ]


def _sample_sensor_peak_list(num_peaks: int,
                            ref_height: float,
                            offset: float) -> List[SensorPeakLorentzian]:
    """
    Function to sample a list of sensor peaks.
    Objects of the SensorPeakLorentzian class are used as peaks here.

    This function is only required when sampling a SensorScanSensorGeneric.

    Args:
        num_peaks (int): Number of lorentzian peaks to sample.
        ref_height (float): Reference height of all peaks. The reference height is used as the mean for the
            distributions all peaks are sampled from.
        offset (float): Potential offset of the first sensor peak.

    Returns:
        List[SensorPeakLorentzian]: A list of Lorentzian peaks.
    """
    peak_list = []

    for i in range(num_peaks):
        peak_list = _sensor_peak_list_add_peak(peak_list=peak_list, ref_height=ref_height, base_offset=offset)

    return peak_list


def sample_sensor_function(offset: float, quality_multiplier: float) -> Tuple[List[SensorPeakInterface], SensorRiseInterface]:
    """
    Function to sample a whole sensor function.
    A whole sensor function consists of a list of lorentzian sensor peaks and ends with a sigmoid sensor rise, which
    represents an increase to the maximum conductivity.

    This function is only required when sampling a SensorScanSensorGeneric.

    Args:
        offset (float): Potential offset of the first sensor peak.
        quality_multiplier (float): The quality multiplier of the sensor function sampler restricts the reference height
            that specifies the mean of the sensor peak height. This also indirectly restricts the height of the final
            rise, as this depends on the maximum height of the preceding sensor peaks. The factor must be in the range
            (0.0, 1.0]. Defaults to 1.0.

    Returns:
        List[SensorPeakInterface]: A whole sensor function as a list returned as tuple. The tuple contains the sensor
        peaks as a list of multiple lorentzian and the sensor rise, a rise to the maximum sensor response.
    """
    peak_num_range = (20, 30)

    height_ref_mean = 0.07543071921134054 * quality_multiplier # MEDIAN
    height_ref_range = (0, 1)
    height_ref_std = 0.04804196857755959

    sensor_rise_max_multiplier = 1.05
    sensor_rise_growth_rate_range = (400, 600)
    sensor_rise_asymmetry_range = (0.01, 1)
    sensor_rise_shape_factor_range = (0.01, 100)
    # The distance of the sensor rise to the previous lorentzian peak is based on a multiple of the distance of the last
    # two peaks. The value of the multiple is sampled from this range.
    sensor_rise_dif_multiplier_range = (2, 3)  # (4, 6)
    sensor_rise_end_percentage = 0.95

    # Sample reference peak height
    ref_height = NormalSamplingRange(
        total_range=height_ref_range,
        std=height_ref_std,
        mean=height_ref_mean
    ).sample_parameter()

    # Sample Lorentzian peaks
    num_peaks = int(UniformSamplingRange(total_range=peak_num_range).sample_parameter())
    sensor_func = _sample_sensor_peak_list(num_peaks=num_peaks, ref_height=ref_height, offset=offset)

    x_values = np.linspace(-5, 5, 500000)
    sensor_resp = np.zeros(500000)
    for func in sensor_func:
        sensor_resp += func.sensor_function(x_values)
    maximum = np.max(sensor_resp)

    left_asymptote = 0  # A minimum sensor response of 0 is assumed
    right_asymptote = maximum * sensor_rise_max_multiplier

    param_dict = {
        "asymptote_left": left_asymptote,
        "asymptote_right": right_asymptote,
        "growth_rate": UniformSamplingRange(total_range=sensor_rise_growth_rate_range).sample_parameter(),
        "asymmetry": UniformSamplingRange(total_range=sensor_rise_asymmetry_range).sample_parameter(),
        "shape_factor": UniformSamplingRange(total_range=sensor_rise_shape_factor_range).sample_parameter(),
        "mu0": 0
    }

    sensor_rise = SensorRiseGLF(
        fully_conductive_percentage=sensor_rise_end_percentage,
        **param_dict
    )

    return sensor_func, sensor_rise


def sample_barrier_function(
                            offset: float = 0,
                            height: float = 0.25,
                            quality_multiplier: float = 1.0,
                            ) -> BarrierFunctionInterface:
    """
    Method to sample a barrier function.

    This function is only required when sampling a SensorScanSensorGeneric.

    Args:
        offset (float): Potential offset of the barrier function. Without this offset the pinch of point of the barrier
            function is at a potential of 0.
        height (float): Maximal height and conductance value of the barrier function.
        quality_multiplier (float): For the barrier function, the quality multiplier restricts the barrier height,
            therefore, the factor is multiplied with the mean of the distribution the sensors are sampled from. The
            factor must be in the range (0.0, 1.0]. Defaults to 1.0.

    Returns:
        BarrierFunctionInterface: A SimCATS barrier function.
    """
    return _sample_simple_barrier_function(offset=offset, height=height, quality_multiplier=quality_multiplier,
                                           pinch_off_percentage=0.001, fully_conductive_percentage=0.999)


def _sample_simple_barrier_function(offset: float = 0,
                                   height: float = 0.25,
                                   quality_multiplier: float = 1.0,
                                   pinch_off_percentage: float = 0.05,
                                   fully_conductive_percentage: float = 0.95) -> BarrierFunctionGLF:
    """
    Method to sample a simple barrier function without any defects. For that a single GLF is used.

    This function is only required when sampling a SensorScanSensorGeneric.

    Args:
        offset (float): Potential offset of the barrier function. Without this offset the pinch of point of the barrier
            function is at a potential of 0.
        height (float): Maximal height and conductance value of the barrier function.
        quality_multiplier (float): For the barrier function, the quality multiplier restricts the barrier height,
            therefore, the factor is multiplied with the mean of the distribution the sensors are sampled from. The
            factor must be in the range (0.0, 1.0]. Defaults to 1.0.
        pinch_off_percentage (float): Percentage of the barrier conductance range which is considered as the pinch-off
            value. E.g.: If the barrier function goes from zero to one and the pinch_off_percentage is set to 0.01, then
            the pinch-off is at the potential that leads to a conductance of 0.01. Defaults to 0.05.
        fully_conductive_percentage (float): Percentage of the barrier conductance range which is considered as the
            point where the barrier vanishes and becomes fully conductive. E.g.: If the barrier function goes from zero
            to one and the fully_conductive_percentage is set to 0.99, then the fully conductive point is at the
            potential that leads to a conductance of 0.99. Defaults to 0.95.

    Returns:
        BarrierFunctionGLF: A barrier function based on a GLF.
    """
    barrier_func_asymmetry_range = (0.000001, 0.0001)

    barrier_func_shape_factor_range = (0.01, 100)
    barrier_func_growth_rate_range = (20,40)

    param_dict = {
        "asymptote_left": 0,
        "asymptote_right": height * quality_multiplier,
        "growth_rate": UniformSamplingRange(total_range=barrier_func_growth_rate_range).sample_parameter(),
        "asymmetry": UniformSamplingRange(total_range=barrier_func_asymmetry_range).sample_parameter(),
        "shape_factor": UniformSamplingRange(total_range=barrier_func_shape_factor_range).sample_parameter(),
        "denominator_offset": 1,
    }

    pinch_off = inverse_glf(value=param_dict["asymptote_right"] * pinch_off_percentage, **param_dict)
    param_dict["offset"] = offset - pinch_off

    return BarrierFunctionGLF(
        pinch_off_percentage=pinch_off_percentage,
        fully_conductive_percentage=fully_conductive_percentage,
        **param_dict
    )


def sample_SensorScanSensorGeneric(quality_multiplier: float = 1.0,
                                   compensated_sensor: bool = False) -> SensorScanSensorGeneric:
    """
    Sample parameters and initialize a SensorScanSensorGeneric object.

    Args:
        quality_multiplier: The quality multiplier to use, which defines how much of the ranges for Coulomb peak heights
            etc. is used. This basically allows to restrict configurations to better data quality, as future
            improvements of the sample quality are expected. The parameter further restricts the preset ranges for
            things like the height of Coulomb peaks (increasing the minimum) and the strength of the lever-arms of the
            double quantum dot gates towards the sensor. The factor must be in the range (0.0, 1.0]. Defaults to 1.0.
        compensated_sensor: Whether the effect of the double quantum dot on the sensor dot is compensated (lever-arms of
            the double dot gate voltages on the sensor are close to zero). Defaults to False.
        set_sensor_potential_offset_to_steepest_point: Whether the steepest point of the sensor function is determined
            and the corresponding potential is set as the potential offset (offset_mu_sens). This enables simple
            "retuning" of the sensor by always resetting the potential offset before every CSD measurement. Defaults to
            False.

    Returns:
        SensorScanSensorGeneric: Fully initialized SensorScanSensorGeneric object.
    """
    alpha_sensor_size = 0.1
    alpha_barrier_size = 0.99
    barrier_height = 0.25

    alpha_sensor_gate_sensor_range = (0, 2)
    alpha_sensor_gate_sensor_std = 0.0846429965722433
    alpha_sensor_gate_sensor_mean = 0.9986776964521298

    alpha_sensor_gate_barrier_range = (0.1, 0.3)

    base_point_range = (-0.688, -0.509)
    base_point_pot_offset = 0.005 * 10 # Offset that specifies how much sensor potential before the base point potential the wavefronts should begin

    include_deformations = False
    deformation_circular_percentage = 4.5
    deformation_linear_percentage = 4.3
    deformation_circular_radius_range = (1, 1.5)
    deformation_linear_angle_range = (math.radians(85), math.radians(95))  # The angles have to be specified in radian

    # Sample alpha gate sensor
    # Sample alpha gate sensor - sensor
    alpha_sensor_gate_sensor_ratio_sampler = NormalSamplingRange(total_range=alpha_sensor_gate_sensor_range,
                                                                 std=alpha_sensor_gate_sensor_std,
                                                                 mean=alpha_sensor_gate_sensor_mean)
    alpha_sensor_gate_sensor_ratio = alpha_sensor_gate_sensor_ratio_sampler.sample_parameter()
    alpha_sensor_gate_sensor = np.array([1, 1 / alpha_sensor_gate_sensor_ratio])
    alpha_sensor_gate_sensor = (alpha_sensor_gate_sensor / np.max(alpha_sensor_gate_sensor)) * alpha_sensor_size

    # Sample alpha gate sensor - barrier
    alpha_sensor_gate_barrier_sampler = UniformSamplingRange(total_range=alpha_sensor_gate_barrier_range)

    alpha_sensor_gate_barrier1 = np.array([alpha_barrier_size, alpha_sensor_gate_barrier_sampler.sample_parameter()])
    alpha_sensor_gate_barrier2 = np.array([alpha_sensor_gate_barrier_sampler.sample_parameter(), alpha_barrier_size])

    alpha_sensor_gate = np.array([
        alpha_sensor_gate_sensor,
        alpha_sensor_gate_barrier1,
        alpha_sensor_gate_barrier2
    ])

    if compensated_sensor:
        alpha_gate_1_min, alpha_gate_1_max = 0.0001, 0.0005
        alpha_gate_2_min, alpha_gate_2_max = 0.0001, 0.0005  # Currently not used, as we sample alpha gate 2 in dependence on alpha gate 1. Kept for the sake of completeness.
    else:
        alpha_gate_1_min, alpha_gate_1_max = 0.02805, 0.15093
        alpha_gate_2_min, alpha_gate_2_max = 0.03788, 0.19491  # Currently not used, as we sample alpha gate 2 in dependence on alpha gate 1. Kept for the sake of completeness.

    alpha_dot_1_min, alpha_dot_1_max = -0.0007994, -0.0000961
    alpha_dot_1_max = alpha_dot_1_min + quality_multiplier * (
            alpha_dot_1_max - alpha_dot_1_min)
    alpha_dot_2_min, alpha_dot_2_max = -0.0005214, -0.0000630
    alpha_dot_2_max = alpha_dot_2_min + quality_multiplier * (
            alpha_dot_2_max - alpha_dot_2_min)

    # create random number generator
    rng = np.random.default_rng()

    # sample alpha gates and alpha dots
    alpha_gate_1 = rng.uniform(low=alpha_gate_1_min, high=alpha_gate_1_max)
    alpha_gate_2 = rng.uniform(low=0.5 * alpha_gate_1, high=2 * alpha_gate_1)

    # sample alpha dot 1 depending on alpha gate 1
    # -0.03 / 100 = at least as much change for occ change (alpha dot) as change for sweeping one pixel (alpha gate)
    # Keep in mind negative means, max is actually min (think of absolute values).
    # Set to 2* alpha gate per pixel value, because stepping one pixel adds 1* and we want to jump back at least
    # 1* pixel step (starting with random_variations_v2).
    # Should actually always ensure to have higher min sensitivity (max = min because of negative values). Therefore,
    # np.min with the actual value has been added (starting with random_variations_v3).
    temp_alpha_dot_1_max = np.min([2 * alpha_gate_1 * -0.03 / 100,
                                   alpha_dot_1_max])
    alpha_dot_1 = rng.uniform(low=alpha_dot_1_min, high=temp_alpha_dot_1_max)

    # sample alpha dot 2 depending on alpha gate 2 and alpha dot 1
    # Set to 2* alpha gate per pixel value, because stepping one pixel adds 1* and we want to jump back at least
    # 1* pixel step (starting with random_variations_v2).
    # Should actually always ensure to have higher min sensitivity (max = min because of negative values). Therefore,
    # np.min with the actual value has been added (starting with random_variations_v3).
    temp_sensor_alpha_dot_2_max = np.min([0.5 * alpha_dot_1, 2 * alpha_gate_2 * -0.03 / 100,
                                          alpha_dot_2_max])
    if temp_sensor_alpha_dot_2_max < alpha_dot_2_min:
        temp_sensor_alpha_dot_2_max = alpha_dot_2_max
    # Should actually only ensure to be at most twice as strongly coupled compared to dot 1, but never stronger than
    # given by the pre-defined range. Therefore, np.max with the actual value has been added (starting with
    # random_variations_v3).
    temp_sensor_alpha_dot_2_min = np.max([2 * alpha_dot_1, alpha_dot_2_min])
    alpha_dot_2 = rng.uniform(low=temp_sensor_alpha_dot_2_min, high=temp_sensor_alpha_dot_2_max)

    alpha_gate = np.array((alpha_gate_1, alpha_gate_2))
    alpha_dot = np.array((alpha_dot_1, alpha_dot_2))

    # Base Point
    base_point_sampler = UniformSamplingRange(total_range=base_point_range)
    base_point = np.array([base_point_sampler.sample_parameter(), base_point_sampler.sample_parameter()])

    sensor_func_offset = np.sum(alpha_sensor_gate_sensor * base_point) - base_point_pot_offset

    barrier_1_start = np.sum(alpha_sensor_gate_barrier1 * base_point)
    barrier_2_start = np.sum(alpha_sensor_gate_barrier2 * base_point)

    sensor_func, final_rise = sample_sensor_function(
        offset=sensor_func_offset,
        quality_multiplier=quality_multiplier
    )

    deformation_dict = {}
    # Sample deformations
    if include_deformations:
        angle_sampler = UniformSamplingRange(
            total_range=deformation_linear_angle_range)
        radius_sampler = UniformSamplingRange(total_range=deformation_circular_radius_range)

        for wavefront_num in range(len(sensor_func) - 1):
            deformation_ran_num = random.random() * 100

            if deformation_ran_num < deformation_linear_percentage:
                deformation_dict[wavefront_num] = SensorPeakDeformationLinear(
                    angle=angle_sampler.sample_parameter()
                )

            elif deformation_ran_num < deformation_linear_percentage + deformation_circular_percentage:
                sign = random.choice([-1, 1])

                deformation_dict[wavefront_num] = SensorPeakDeformationCircle(
                    radius=sign * radius_sampler.sample_parameter()
                )

    return SensorScanSensorGeneric(
        barrier_functions=(
            sample_barrier_function(
                offset=barrier_1_start,
                height=barrier_height,
                quality_multiplier=quality_multiplier
            ),
            sample_barrier_function(
                offset=barrier_2_start,
                height=barrier_height,
                quality_multiplier=quality_multiplier
            )

        ),
        sensor_peak_function=sensor_func,
        final_rise = final_rise,
        sensor_peak_deformations=deformation_dict,
        alpha_sensor_gate=alpha_sensor_gate,
        alpha_gate=alpha_gate,
        alpha_dot=alpha_dot
    )


def sample_occupation_distortions(quality_multiplier: float = 1.0) -> List[OccupationDistortionInterface]:
    """
    Sample the list of occupation distortions for the SimCATS Simulation class.

    Args:
        quality_multiplier: The quality multiplier to use, which defines how much of the ranges for the strength of
            distortions is used. This basically allows to restrict configurations to better data quality, as future
            improvements of the sample quality are expected. The factor must be in the range (0.0, 1.0]. Defaults to
            1.0.

    Returns:
        List[OccupationDistortionInterface]: List of all occupation distortions for the GaAs_v1_random_variations_v3
        configuration.
    """
    return [_sample_occupation_transition_blurring_fermi_dirac(),
            _sample_occupation_dot_jumps_g2(),
            _sample_occupation_dot_jumps_g1()]


def sample_sensor_potential_distortions(quality_multiplier: float = 1.0) -> List[SensorPotentialDistortionInterface]:
    """
    Sample the list of sensor potential distortions for the SimCATS Simulation class.

    Args:
        quality_multiplier: The quality multiplier to use, which defines how much of the ranges for the strength of
            distortions is used. This basically allows to restrict configurations to better data quality, as future
            improvements of the sample quality are expected. The factor must be in the range (0.0, 1.0]. Defaults to
            1.0.

    Returns:
        List[SensorPotentialDistortionInterface]: List of all sensor potential distortions for the
        GaAs_v1_random_variations_v3 configuration.
    """
    return [_sample_sensor_potential_pink_noise(quality_multiplier=quality_multiplier),
            _sample_sensor_potential_rtn(quality_multiplier=quality_multiplier)]


def sample_sensor_response_distortions(quality_multiplier: float = 1.0) -> List[SensorResponseDistortionInterface]:
    """
    Sample the list of sensor response distortions for the SimCATS Simulation class.

    Args:
        quality_multiplier: The quality multiplier to use, which defines how much of the ranges for the strength of
            distortions is used. This basically allows to restrict configurations to better data quality, as future
            improvements of the sample quality are expected. The factor must be in the range (0.0, 1.0]. Defaults to
            1.0.

    Returns:
        List[SensorResponseDistortionInterface]: List of all sensor response distortions for the
        GaAs_v1_random_variations_v3 configuration.
    """
    return [_sample_sensor_response_rtn(quality_multiplier=quality_multiplier),
            _sample_sensor_response_white_noise(quality_multiplier=quality_multiplier)]


def _add_one_tct(tct_params: List[np.ndarray],
                 shift_vec: np.ndarray) -> List[np.ndarray]:
    """
    Helper function that adds one TCT to the list of TCTs, by shifting the last TCT by the shift vector.

    Args:
        tct_params: List of TCTs (or more precisely TCT parameters).
        shift_vec: The vector that describes the shift between two adjacent TCTs.

    Returns:
        List[np.ndarray]: The list of TCTs including one additional TCT.
    """
    temp_params = tct_params[-1].copy()
    temp_params[4] += shift_vec[0]
    temp_params[5] += shift_vec[1]
    temp_params[6] += shift_vec[0]
    temp_params[7] += shift_vec[1]
    tct_params.append(temp_params)
    return tct_params


def _sample_occupation_transition_blurring_fermi_dirac() -> OccupationTransitionBlurringFermiDirac:
    """
    Sample parameters and initialize an OccupationTransitionBlurringFermiDirac object.

    Returns:
        OccupationTransitionBlurringFermiDirac: Fully initialized OccupationTransitionBlurringFermiDirac object.
    """
    occ_trans_blur_fermi_dirac_sigma_min = (
            0.25 * 0.03 / 100
    )
    occ_trans_blur_fermi_dirac_sigma_max = 2 * 0.03 / 100  # changed min and max starting with random_variations_v2
    # use 3*sigma distance from mean to borders
    std_transition_blurring = (occ_trans_blur_fermi_dirac_sigma_max - occ_trans_blur_fermi_dirac_sigma_min) / 6
    return OccupationTransitionBlurringFermiDirac(
        sigma=NormalSamplingRange(
            total_range=(occ_trans_blur_fermi_dirac_sigma_min, occ_trans_blur_fermi_dirac_sigma_max),
            std=std_transition_blurring,
        )  # using other sampler starting with random_variations_v2
    )


def _sample_occupation_dot_jumps_g1() -> OccupationDotJumps:
    """
    Sample parameters and initialize an OccupationDotJumps object affecting g1.

    Returns:
        OccupationDotJumps: Fully initialized OccupationDotJumps object.
    """
    return OccupationDotJumps(ratio=0.01 / 6,
                              scale=100 * 0.03 / 100,
                              lam=6 * 0.03 / 100,
                              axis=1)


def _sample_occupation_dot_jumps_g2() -> OccupationDotJumps:
    """
    Sample parameters and initialize an OccupationDotJumps object affecting g2.

    Returns:
        OccupationDotJumps: Fully initialized OccupationDotJumps object.
    """
    return OccupationDotJumps(ratio=0.01,
                              scale=100 * 0.03 / 100,
                              lam=6 * 0.03 / 100,
                              axis=0)


def _sample_sensor_potential_pink_noise(quality_multiplier: float = 1.0) -> SensorPotentialPinkNoise:
    """
    Sample parameters and initialize a SensorPotentialPinkNoise object.

    Args:
        quality_multiplier: The quality multiplier to use, which defines how much of the ranges for the strength of
            distortions is used. This basically allows to restrict configurations to better data quality, as future
            improvements of the sample quality are expected. The factor must be in the range (0.0, 1.0]. Defaults to
            1.0.

    Returns:
        SensorPotentialPinkNoise: Fully initialized SensorPotentialPinkNoise object.
    """
    sensor_pot_pink_sigma_min, sensor_pot_pink_sigma_max = 1e-10, 0.0005  # reduced max starting with random_variations_v2
    quantile_relative_pos = 0.6 * quality_multiplier
    quantile_percentage = 0.99
    exp_scale_sensor_dot_pink = quantile_relative_pos / np.log(1 / (1 - quantile_percentage))
    return SensorPotentialPinkNoise(
        sigma=ExponentialSamplingRange(
            total_range=(sensor_pot_pink_sigma_min, sensor_pot_pink_sigma_max),
            scale=exp_scale_sensor_dot_pink
        )  # using other sampler starting with random_variations_v2
    )


def _sample_sensor_potential_rtn(quality_multiplier: float = 1.0) -> SensorPotentialRTN:
    """
    Sample parameters and initialize a SensorPotentialRTN object.

    Args:
        quality_multiplier: The quality multiplier to use, which defines how much of the ranges for the strength of
            distortions is used. This basically allows to restrict configurations to better data quality, as future
            improvements of the sample quality are expected. The factor must be in the range (0.0, 1.0]. Defaults to
            1.0.

    Returns:
        SensorPotentialRTN: Fully initialized SensorPotentialRTN object.
    """
    return SensorPotentialRTN(
        scale=74.56704 * 0.03 / 100,
        std=3.491734e-05,
        height=2.53855325e-05,
        ratio=1 / 6 * quality_multiplier,
    )


def _sample_sensor_response_rtn(quality_multiplier: float = 1.0) -> SensorResponseRTN:
    """
    Sample parameters and initialize a SensorResponseRTN object.

    Args:
        quality_multiplier: The quality multiplier to use, which defines how much of the ranges for the strength of
            distortions is used. This basically allows to restrict configurations to better data quality, as future
            improvements of the sample quality are expected. The factor must be in the range (0.0, 1.0]. Defaults to
            1.0.

    Returns:
        SensorResponseRTN: Fully initialized SensorResponseRTN object.
    """
    return SensorResponseRTN(
        scale=10000 * 0.03 / 100,
        std=0.047453767599999995,
        height=0.0152373696,
        ratio=0.01 * quality_multiplier,
        # reduced to 0.01 from 0.03 (not required as often with lots of images)
    )


def _sample_sensor_response_white_noise(quality_multiplier: float = 1.0) -> SensorResponseWhiteNoise:
    """
    Sample parameters and initialize a SensorResponseWhiteNoise object.

    Args:
        quality_multiplier: The quality multiplier to use, which defines how much of the ranges for the strength of
            distortions is used. This basically allows to restrict configurations to better data quality, as future
            improvements of the sample quality are expected. The factor must be in the range (0.0, 1.0]. Defaults to
            1.0.

    Returns:
        SensorResponseWhiteNoise: Fully initialized SensorResponseWhiteNoise object.
    """
    sensor_res_white_sigma_min, sensor_res_white_sigma_max = 1e-10, 0.0005  # reduced max starting with random_variations_v2
    quantile_relative_pos = 0.6 * quality_multiplier
    quantile_percentage = 0.99
    exp_scale_res_white = quantile_relative_pos / np.log(1 / (1 - quantile_percentage))
    return SensorResponseWhiteNoise(
        sigma=ExponentialSamplingRange(
            total_range=(sensor_res_white_sigma_min, sensor_res_white_sigma_max),
            scale=exp_scale_res_white
        )  # using other sampler starting with random_variations_v2
    )
