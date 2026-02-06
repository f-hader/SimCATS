"""
SimCATS subpackage containing all functionalities related to the sensor simulation.
"""

from simcats.sensor._sensor_interface import SensorPeakInterface, SensorRiseInterface, SensorInterface, \
    SensorScanSensorInterface
from simcats.sensor._sensor_peak_gaussian import SensorPeakGaussian, sensor_response_gauss
from simcats.sensor._sensor_peak_lorentzian import SensorPeakLorentzian, sensor_response_lorentz
from simcats.sensor._sensor_rise_glf import SensorRiseGLF
from simcats.sensor._sensor_generic import SensorGeneric
from simcats.sensor._sensor_scan_sensor_generic import SensorScanSensorGeneric

__all__ = ["SensorPeakInterface", "SensorRiseInterface", "SensorInterface", "SensorScanSensorInterface",
           "SensorPeakGaussian", "SensorPeakLorentzian", "SensorRiseGLF", "SensorGeneric",
           "SensorScanSensorGeneric", "sensor_response_gauss", "sensor_response_lorentz"]
