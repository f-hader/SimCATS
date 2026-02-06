"""Module to test the simulation of sensor scans."""
try:
    from importlib.resources import files as imp_files
except ImportError:
    from importlib_resources import files as imp_files

from copy import deepcopy
from unittest import TestCase
import numpy as np

from simcats import Simulation
from simcats.sensor import SensorScanSensorGeneric
from simcats.sensor.deformation import SensorPeakDeformationCircle
from simcats.sensor.deformation._sensor_peak_deformation_linear import SensorPeakDeformationLinear
from tests._test_configs import test_configs


class SensorSimulationTests(TestCase):
    """Class to test the sensor simulation."""

    def test_undistorted_simulation_no_deformation(self) -> None:
        config = deepcopy(test_configs["GaAs_v2_extended_sensor"])

        simulation = Simulation(**config)

        # turn off noise
        simulation.occupation_distortions = None
        simulation.sensor_potential_distortions = None
        simulation.sensor_response_distortions = None

        scan, cond_mask, wave_mask, metadata = simulation.measure_sensor_scan(
            sweep_range_sensor_g1=np.array([-0.6, -0.3]),
            sweep_range_sensor_g2=np.array([-0.6, -0.3]),
            resolution=np.array((1664 * 2, 20 * 2))
        )

        file_path = str(
            imp_files("tests.comparison_files").joinpath(
                "sensor_scan_simulation_test_undistorted.npz",
            ),
        )

        references = np.load(file=file_path)
        reference_scan = references['scan']
        reference_wave_mask = references['wave_mask']
        reference_cond_mask = references['cond_mask']

        # Check if the simulated scan itself is the same as the reference scan
        np.testing.assert_allclose(reference_scan, scan)
        # Check if the simulated scan itself is the same as the reference mask of the wavefronts
        assert (reference_wave_mask == wave_mask).all()
        # Check if the simulated scan itself is the same as the reference mask of the conductive area
        assert (reference_cond_mask == cond_mask).all()

    def test_undistorted_simulation_with_deformation_linear(self) -> None:
        config = deepcopy(test_configs["GaAs_v2_extended_sensor"])

        new_sensor = SensorScanSensorGeneric(
            sensor_peak_function=config["sensor"].sensor_peak_function,
            final_rise=config["sensor"].final_rise,
            alpha_sensor_gate=config["sensor"].alpha_sensor_gate,
            sensor_peak_deformations={1: SensorPeakDeformationLinear(angle=(np.pi / 2) - 0.2)},
            barrier_functions=config["sensor"].barrier_functions,
        )

        config["sensor"] = new_sensor

        simulation = Simulation(**config)

        # turn off noise
        simulation.occupation_distortions = None
        simulation.sensor_potential_distortions = None
        simulation.sensor_response_distortions = None

        scan, cond_mask, wave_mask, metadata = simulation.measure_sensor_scan(
            sweep_range_sensor_g1=np.array([-0.6, -0.3]),
            sweep_range_sensor_g2=np.array([-0.6, -0.3]),
            resolution=np.array((1664 * 2, 20 * 2))
        )

        file_path = str(
            imp_files("tests.comparison_files").joinpath(
                "sensor_scan_simulation_test_undistorted_deform_lin.npz",
            ),
        )

        references = np.load(file=file_path)
        reference_scan = references['scan']
        reference_wave_mask = references['wave_mask']
        reference_cond_mask = references['cond_mask']

        # Check if the simulated scan itself is the same as the reference scan
        np.testing.assert_allclose(reference_scan, scan)
        # Check if the simulated scan itself is the same as the reference mask of the wavefronts
        assert (reference_wave_mask == wave_mask).all()
        # Check if the simulated scan itself is the same as the reference mask of the conductive area
        assert (reference_cond_mask == cond_mask).all()

    def test_undistorted_simulation_with_deformation_circular(self) -> None:
        config = deepcopy(test_configs["GaAs_v2_extended_sensor"])

        new_sensor = SensorScanSensorGeneric(
            sensor_peak_function=config["sensor"].sensor_peak_function,
            final_rise= config["sensor"].final_rise,
            alpha_sensor_gate=config["sensor"].alpha_sensor_gate,
            sensor_peak_deformations={1: SensorPeakDeformationCircle(radius=0.25)},
            barrier_functions=config["sensor"].barrier_functions,
        )

        config["sensor"] = new_sensor

        simulation = Simulation(**config)

        # turn off noise
        simulation.occupation_distortions = None
        simulation.sensor_potential_distortions = None
        simulation.sensor_response_distortions = None

        scan, cond_mask, wave_mask, metadata = simulation.measure_sensor_scan(
            sweep_range_sensor_g1=np.array([-0.6, -0.3]),
            sweep_range_sensor_g2=np.array([-0.6, -0.3]),
            resolution=np.array((1664 * 2, 20 * 2))
        )

        file_path = str(
            imp_files("tests.comparison_files").joinpath(
                "sensor_scan_simulation_test_undistorted_deform_circ.npz",
            ),
        )


        references = np.load(file=file_path)
        reference_scan = references['scan']
        reference_wave_mask = references['wave_mask']
        reference_cond_mask = references['cond_mask']

        # Check if the simulated scan itself is the same as the reference scan
        np.testing.assert_allclose(reference_scan, scan)
        # Check if the simulated scan itself is the same as the reference mask of the wavefronts
        assert (reference_wave_mask == wave_mask).all()
        # Check if the simulated scan itself is the same as the reference mask of the conductive area
        assert (reference_cond_mask == cond_mask).all()
