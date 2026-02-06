"""
This module contains the default configurations, that can be used to initialize an object of the simulation class to
perform charge stability diagram (CSD) simulations.

Contributors should add their parameter sets to the default_configs dictionary so that they are available in a central
location.

@author: f.hader, b.papajewski
"""

import numpy as np

from simcats.distortions import OccupationTransitionBlurringGaussian, OccupationDotJumps, SensorPotentialPinkNoise, \
    SensorPotentialRTN, SensorResponseRTN, SensorResponseWhiteNoise
from simcats.ideal_csd import IdealCSDGeometric
from simcats.sensor import SensorPeakLorentzian, SensorGeneric, SensorScanSensorGeneric, SensorRiseGLF
from simcats.sensor.barrier_function import BarrierFunctionGLF

from simcats.support_functions import NormalSamplingRange, UniformSamplingRange

__all__ = []


default_configs = {"GaAs_v1": {"volt_limits_g1": np.array([-0.2, -0.087]),
                               "volt_limits_g2": np.array([-0.2, -0.047]),
                               "ideal_csd_config": IdealCSDGeometric(tct_params=[np.array(
                                   [0.01075474, 0.01549732, 0.42465033, -0.38038481, -0.02750187, -0.17179705,
                                    -0.02674207, -0.17171497]), np.array(
                                   [0.01075474, 0.01549732, 0.40341781, -0.36136557, -0.04171389, -0.16351316,
                                    -0.04012621, -0.16343108]), np.array(
                                   [0.01075474, 0.01549732, 0.38324692, -0.34329729, -0.05592591, -0.15564346,
                                    -0.05351035, -0.15556138]), np.array(
                                   [0.01075474, 0.01549732, 0.36408457, -0.32613243, -0.07013794, -0.14816725,
                                    -0.06689448, -0.14808517]), np.array(
                                   [0.01075474, 0.01549732, 0.34588034, -0.30982581, -0.08434996, -0.14106485,
                                    -0.08027862, -0.14098277]), np.array(
                                   [0.01075474, 0.01549732, 0.32858633, -0.29433452, -0.09856198, -0.13431757,
                                    -0.09366276, -0.13423549]), np.array(
                                   [0.01075474, 0.01549732, 0.31215701, -0.27961779, -0.112774, -0.12790766, -0.1070469,
                                    -0.12782558]), np.array(
                                   [0.01075474, 0.01549732, 0.20758441, -0.18594583, -0.12698603, -0.12181824,
                                    -0.12125892, -0.12173616]), np.array(
                                   [0.01075474, 0.01549732, 0.13804363, -0.12365398, -0.14119805, -0.11603329,
                                    -0.13547094, -0.11595121]), np.array(
                                   [0.01075474, 0.01549732, 0.09179902, -0.0822299, -0.15541007, -0.11053759,
                                    -0.14968296, -0.1104555]), np.array(
                                   [0.01075474, 0.01549732, 0.06104635, -0.05468288, -0.16962209, -0.10531667,
                                    -0.16389499, -0.10523459]), np.array(
                                   [0.01075474, 0.01549732, 0.04059582, -0.03636412, -0.18383412, -0.1003568,
                                    -0.17810701, -0.10027472]), np.array(
                                   [0.01075474, 0.01549732, 0.02699622, -0.02418214, -0.19804614, -0.09564492,
                                    -0.19231903, -0.09556284]), np.array(
                                   [0.01075474, 0.01549732, 0.01795249, -0.01608112, -0.21225816, -0.09116864,
                                    -0.20653105, -0.09108656])],
                                   rotation=-np.pi / 4,
                                   lut_entries=1000,
                                   cdf_type="sigmoid",
                                   cdf_gamma_factor=None),
                               "sensor": SensorGeneric(sensor_peak_function=[
                                   SensorPeakLorentzian(mu0=-0.12096, gamma=0.00095, height=0.026245, offset=-0.253275),
                                   SensorPeakLorentzian(mu0=-0.11596, gamma=0.001, height=0.027245, offset=-0.253275),
                                   SensorPeakLorentzian(mu0=-0.11096, gamma=0.00105, height=0.028245, offset=-0.253275),
                                   SensorPeakLorentzian(mu0=-0.10596, gamma=0.0011, height=0.029245, offset=-0.253275),
                                   SensorPeakLorentzian(mu0=-0.10096, gamma=0.00115, height=0.030245, offset=-0.253275)
                               ],
                                   alpha_dot=np.array([-0.00044775, -0.0002922]),
                                   alpha_gate=np.array([0.08949, 0.116395]),
                                   offset_mu_sens=-0.0818745),
                               "occupation_distortions": [OccupationTransitionBlurringGaussian(0.75 * 0.03 / 100),
                                                          # in g2 (y in 2D) direction
                                                          OccupationDotJumps(ratio=0.01, scale=100 * 0.03 / 100,
                                                                             lam=6 * 0.03 / 100, axis=0),
                                                          # in g1 (x in 2D) direction
                                                          OccupationDotJumps(ratio=0.01 / 6, scale=100 * 0.03 / 100,
                                                                             lam=6 * 0.03 / 100, axis=1)],
                               "sensor_potential_distortions": [
                                   SensorPotentialPinkNoise(
                                       sigma=UniformSamplingRange((1.8250268077765864e-12, 9.125134038882932e-05)),
                                       fmin=0),
                                   SensorPotentialRTN(scale=74.56704 * 0.03 / 100, std=3.491734e-05,
                                                      height=2.53855325e-05, ratio=1 / 6)],
                               "sensor_response_distortions": [
                                   SensorResponseRTN(scale=10000 * 0.03 / 100, std=0.047453767599999995,
                                                     height=0.0152373696, ratio=0.03),
                                   SensorResponseWhiteNoise(
                                       sigma=NormalSamplingRange((1e-10, 0.003), sampling_range=0.001,
                                                                 std=0.0003))]
                               },
                   "GaAs_v2_extended_sensor": {
                       "volt_limits_g1": np.array([-0.2, -0.087]),
                       "volt_limits_g2": np.array([-0.2, -0.047]),
                       "volt_limits_sensor_g1": np.array([-1, 0.4]),
                       "volt_limits_sensor_g2": np.array([-1, 0.4]),
                       "ideal_csd_config": IdealCSDGeometric(tct_params=[np.array(
                           [0.01075474, 0.01549732, 0.42465033, -0.38038481, -0.02750187, -0.17179705,
                            -0.02674207, -0.17171497]), np.array(
                           [0.01075474, 0.01549732, 0.40341781, -0.36136557, -0.04171389, -0.16351316,
                            -0.04012621, -0.16343108]), np.array(
                           [0.01075474, 0.01549732, 0.38324692, -0.34329729, -0.05592591, -0.15564346,
                            -0.05351035, -0.15556138]), np.array(
                           [0.01075474, 0.01549732, 0.36408457, -0.32613243, -0.07013794, -0.14816725,
                            -0.06689448, -0.14808517]), np.array(
                           [0.01075474, 0.01549732, 0.34588034, -0.30982581, -0.08434996, -0.14106485,
                            -0.08027862, -0.14098277]), np.array(
                           [0.01075474, 0.01549732, 0.32858633, -0.29433452, -0.09856198, -0.13431757,
                            -0.09366276, -0.13423549]), np.array(
                           [0.01075474, 0.01549732, 0.31215701, -0.27961779, -0.112774, -0.12790766, -0.1070469,
                            -0.12782558]), np.array(
                           [0.01075474, 0.01549732, 0.20758441, -0.18594583, -0.12698603, -0.12181824,
                            -0.12125892, -0.12173616]), np.array(
                           [0.01075474, 0.01549732, 0.13804363, -0.12365398, -0.14119805, -0.11603329,
                            -0.13547094, -0.11595121]), np.array(
                           [0.01075474, 0.01549732, 0.09179902, -0.0822299, -0.15541007, -0.11053759,
                            -0.14968296, -0.1104555]), np.array(
                           [0.01075474, 0.01549732, 0.06104635, -0.05468288, -0.16962209, -0.10531667,
                            -0.16389499, -0.10523459]), np.array(
                           [0.01075474, 0.01549732, 0.04059582, -0.03636412, -0.18383412, -0.1003568,
                            -0.17810701, -0.10027472]), np.array(
                           [0.01075474, 0.01549732, 0.02699622, -0.02418214, -0.19804614, -0.09564492,
                            -0.19231903, -0.09556284]), np.array(
                           [0.01075474, 0.01549732, 0.01795249, -0.01608112, -0.21225816, -0.09116864,
                            -0.20653105, -0.09108656])],
                           rotation=-np.pi / 4,
                           lut_entries=1000,
                           cdf_type="sigmoid",
                           cdf_gamma_factor=None),
                       "sensor": SensorScanSensorGeneric(
                           barrier_functions=(
                               BarrierFunctionGLF(pinch_off_percentage=0.001,
                                                  fully_conductive_percentage=0.999,
                                                  asymptote_left=0,
                                                  asymptote_right=0.3,
                                                  growth_rate=80,
                                                  asymmetry=8.948499659663527e-05,
                                                  shape_factor=25.64528635013381,
                                                  denominator_offset=1,
                                                  offset=-0.78),#-0.83),
                               BarrierFunctionGLF(pinch_off_percentage=0.001,
                                                  fully_conductive_percentage=0.999,
                                                  asymptote_left=0,
                                                  asymptote_right=0.3,
                                                  growth_rate=72,
                                                  asymmetry=7.784390241302876e-05,
                                                  shape_factor=97.5062459999433,
                                                  denominator_offset=1,
                                                  offset=-0.88)#-0.93)
                           ),
                           sensor_peak_function=[
                               SensorPeakLorentzian(mu0=-0.151, gamma=0.0002, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.148, gamma=0.000225, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.145, gamma=0.00025, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.142, gamma=0.000275, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.139, gamma=0.0003, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.136, gamma=0.000325, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.133, gamma=0.00035, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.130, gamma=0.000375, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.127, gamma=0.0004, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.124, gamma=0.000425, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.121, gamma=0.00045, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.118, gamma=0.000475, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.115, gamma=0.0005, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.112, gamma=0.00055, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.109, gamma=0.0006, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.106, gamma=0.00065, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.103, gamma=0.0007, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.100, gamma=0.00075, height=0.09, offset=0),
                               SensorPeakLorentzian(mu0=-0.097, gamma=0.0008, height=0.09, offset=0),
                           ],
                           final_rise=SensorRiseGLF(fully_conductive_percentage=0.999,
                                                    mu0=0,
                                                    asymptote_left=0,
                                                    asymptote_right=0.1,
                                                    growth_rate=2500,
                                                    asymmetry=0.035,
                                                    shape_factor=2),
                           alpha_sensor_gate=np.array([[0.096754, 0.1], [0.99, 0.15], [0.15, 0.99]]),
                           alpha_dot=np.array([-0.00044775, -0.0002922]),
                           alpha_gate=np.array([0.08949, 0.116395]),
                           offset_mu_sens=np.array([0, 0, 0]),
                           sensor_peak_deformations={}
                       ),
                       "occupation_distortions": [OccupationTransitionBlurringGaussian(0.75 * 0.03 / 100),
                                                  # in g2 (y in 2D) direction
                                                  OccupationDotJumps(ratio=0.01, scale=100 * 0.03 / 100,
                                                                     lam=6 * 0.03 / 100, axis=0),
                                                  # in g1 (x in 2D) direction
                                                  OccupationDotJumps(ratio=0.01 / 6, scale=100 * 0.03 / 100,
                                                                     lam=6 * 0.03 / 100, axis=1)],
                       "sensor_potential_distortions": [
                           SensorPotentialPinkNoise(
                               sigma=UniformSamplingRange((1.8250268077765864e-12, 9.125134038882932e-05)),
                               fmin=0),
                           SensorPotentialRTN(scale=74.56704 * 0.03 / 100, std=3.491734e-05,
                                              height=2.53855325e-05, ratio=1 / 6)],
                       "sensor_response_distortions": [
                           SensorResponseWhiteNoise(
                               sigma=NormalSamplingRange((1e-10, 0.003), sampling_range=0.001,
                                                         std=0.0003))]
                   }
                   }
"""Dict: Default configurations for the Simulation class.
Includes the default configurations "GaAs_v1" and "GaAs_v2_extended_sensor", which can be used to instantiate an object 
of the Simulation class (have a look at the jupyter notebook `example_SimCATS_Simulation_class.ipynb` for an example).\n
**Information for contributors / developers**: The dictionary itself is stored in the file _default_configs.py to ensure
a clean separation between the Simulation class and the configurations.
"""
