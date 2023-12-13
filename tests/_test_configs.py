"""
Module containing test configurations.

Module that contains the test configurations, that can be used to initialize an object of the simulation class to
perform charge stability diagram (CSD) simulations. The tests have their own configuration due to stability reasons,
in case the default config gets changed in the future.

@author: f.hader
"""

import numpy as np

from simcats.distortions import (
    OccupationDotJumps,
    OccupationTransitionBlurringGaussian,
    SensorPotentialPinkNoise,
    SensorPotentialRTN,
    SensorResponseRTN,
    SensorResponseWhiteNoise,
)
from simcats.ideal_csd import IdealCSDGeometric
from simcats.sensor import SensorGeneric, SensorPeakLorentzian
from simcats.support_functions import (
    NormalSamplingRange,
    UniformSamplingRange,
)

test_configs = {
    "GaAs_v1": {
        "volt_limits_g1": np.array([-0.05, 0.058]),
        "volt_limits_g2": np.array([-0.05, 0.06]),
        "ideal_csd_config": IdealCSDGeometric(
            **{
                "tct_params": [
                    np.array(
                        [
                            0.00848958,
                            0.00848497,
                            0.53666303,
                            -0.53738033,
                            -0.00153027,
                            0.00571683,
                            0.00155205,
                            0.00569933,
                        ],
                    ),
                    np.array(
                        [
                            0.00848958,
                            0.00848497,
                            0.53666303,
                            -0.53738033,
                            -0.01000635,
                            0.01444064,
                            -0.00692403,
                            0.01442314,
                        ],
                    ),
                    np.array(
                        [
                            0.00848958,
                            0.00848497,
                            0.53666303,
                            -0.53738033,
                            -0.01848243,
                            0.02316445,
                            -0.01540011,
                            0.02314695,
                        ],
                    ),
                    np.array(
                        [
                            0.00848958,
                            0.00848497,
                            0.53666303,
                            -0.53738033,
                            -0.02695851,
                            0.03188826,
                            -0.02387619,
                            0.03187076,
                        ],
                    ),
                    np.array(
                        [
                            0.00848958,
                            0.00848497,
                            0.53666303,
                            -0.53738033,
                            -0.03543459,
                            0.04061207,
                            -0.03235227,
                            0.04059457,
                        ],
                    ),
                    np.array(
                        [
                            0.00848958,
                            0.00848497,
                            0.53666303,
                            -0.53738033,
                            -0.04391067,
                            0.04933588,
                            -0.04082835,
                            0.04931838,
                        ],
                    ),
                    np.array(
                        [
                            0.00848958,
                            0.00848497,
                            0.53666303,
                            -0.53738033,
                            -0.05238675,
                            0.05805969,
                            -0.04930443,
                            0.05804219,
                        ],
                    ),
                    np.array(
                        [
                            0.00848958,
                            0.00848497,
                            0.53666303,
                            -0.53738033,
                            -0.06086283,
                            0.0667835,
                            -0.05778051,
                            0.066766,
                        ],
                    ),
                    np.array(
                        [
                            0.00848958,
                            0.00848497,
                            0.53666303,
                            -0.53738033,
                            -0.06933891,
                            0.07550731,
                            -0.06625659,
                            0.07548981,
                        ],
                    ),
                    np.array(
                        [
                            0.00848958,
                            0.00848497,
                            0.53666303,
                            -0.53738033,
                            -0.07781499,
                            0.08423112,
                            -0.07473267,
                            0.08421362,
                        ],
                    ),
                    np.array(
                        [
                            0.00848958,
                            0.00848497,
                            0.53666303,
                            -0.53738033,
                            -0.08629107,
                            0.09295493,
                            -0.08320875,
                            0.09293743,
                        ],
                    ),
                ],
                "rotation": -np.pi / 4,
                "lut_entries": 1000,
                "cdf_type": "sigmoid",
                "cdf_gamma_factor": None,
            },
        ),
        "sensor": SensorGeneric(
            sensor_peak_function=SensorPeakLorentzian(mu0=-0.07996, gamma=0.00195725, height=0.107245,
                                                      offset=-0.253275),
            alpha_dot=np.array([-0.00044775, -0.0002922]),
            alpha_gate=np.array([0.08949, 0.116395]),
            offset_mu_sens=-0.07996 - 2 * 0.00195725,
        ),
        # mu0-2*gamma
        "occupation_distortions": [
            OccupationTransitionBlurringGaussian(0.75 * 0.03 / 100),
            # in P2 (y in 2D) direction
            OccupationDotJumps(
                ratio=0.01,
                scale=100 * 0.03 / 100,
                lam=6 * 0.03 / 100,
                axis=0,
            ),
            # in P1 (x in 2D) direction
            OccupationDotJumps(
                ratio=0.01 / 6,
                scale=100 * 0.03 / 100,
                lam=6 * 0.03 / 100,
                axis=1,
            ),
        ],
        "sensor_potential_distortions": [
            SensorPotentialPinkNoise(
                sigma=UniformSamplingRange(
                    (1.8250268077765864e-12, 9.125134038882932e-05),
                ),
                fmin=0,
            ),
            SensorPotentialRTN(
                scale=74.56704 * 0.03 / 100,
                std=3.491734e-05,
                height=2.53855325e-05,
                ratio=1 / 6,
            ),
        ],
        "sensor_response_distortions": [
            SensorResponseRTN(
                scale=10000 * 0.03 / 100,
                std=0.047453767599999995,
                height=0.0152373696,
                ratio=0.03,
            ),
            SensorResponseWhiteNoise(
                sigma=NormalSamplingRange(
                    (1e-10, 0.003),
                    sampling_range=0.001,
                    std=0.0003,
                ),
            ),
        ],
    },
}
