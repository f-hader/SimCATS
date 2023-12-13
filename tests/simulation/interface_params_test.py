"""Module to test behavior of the simulation class when some parameters are wrong or missing."""
from unittest import TestCase

import pytest

from simcats import Simulation


class InterfaceParamsTests(TestCase):
    """Class to test behavior of the simulation class when some parameters are wrong or missing."""

    def test_ideal_cds_interface(self) -> None:
        """Test that for the parameter ideal_csd_config values that do not implement IdealCSDInterface are intercepted and the correct error is raised."""
        ideal_csd_config = 3
        with pytest.raises(
            ValueError,
            match=r"The provided ideal CSD configuration is not supported, as it doesn't implement the interface 'IdealCSDInterface'.",
        ):
            Simulation(ideal_csd_config=ideal_csd_config)

    def test_sensor_interface(self) -> None:
        """Test that for the parameter sensor values that do not implement SensorInterface are intercepted and the correct error is raised."""
        sensor = 3
        with pytest.raises(
            ValueError,
            match=r"The provided sensor configuration is not supported, as it doesn't implement the interface 'SensorInterface'.",
        ):
            Simulation(sensor=sensor)

    def test_occupation_distortion_interface(self) -> None:
        """Test that for the parameter occupation_distortions values that do not implement OccupationDistortionInterface are intercepted and the correct error is raised."""
        occupation_distortions = [3]
        with pytest.raises(
            ValueError,
            match=r"The provided occupation distortion configuration is not supported, as not all list objects implement the interface 'OccupationDistortionInterface'.",
        ):
            Simulation(occupation_distortions=occupation_distortions)

    def test_sensor_potential_distortions_interface(self) -> None:
        """Test that for the parameter sensor_potential_distortions values that do not implement SensorPotentialDistortionInterface are intercepted and the correct error is raised."""
        sensor_potential_distortions = [3]
        with pytest.raises(
            ValueError,
            match=r"The provided sensor potential distortion configuration is not supported, as not all list objects implement the interface 'SensorPotentialDistortionInterface'.",
        ):
            Simulation(
                sensor_potential_distortions=sensor_potential_distortions,
            )

    def test_sensor_response_distortions_interface(self) -> None:
        """Test that for the parameter sensor_response_distortions values that do not implement SensorResponseDistortionInterface are intercepted and the correct error is raised."""
        sensor_response_distortions = [3]
        with pytest.raises(
            ValueError,
            match=r"The provided sensor response distortion configuration is not supported, as not all list objects implement the interface 'SensorResponseDistortionInterface'.",
        ):
            Simulation(
                sensor_response_distortions=sensor_response_distortions,
            )
