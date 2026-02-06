"""
SimCATS subpackage of the sensor package containing all functionalities related to sensor peak deformations.
"""

from simcats.sensor.deformation._sensor_peak_deformation_interface import SensorPeakDeformationInterface
from simcats.sensor.deformation._sensor_peak_deformation_circle import SensorPeakDeformationCircle
from simcats.sensor.deformation._sensor_peak_deformation_linear import SensorPeakDeformationLinear

__all__ = ['SensorPeakDeformationInterface', 'SensorPeakDeformationCircle', 'SensorPeakDeformationLinear']
