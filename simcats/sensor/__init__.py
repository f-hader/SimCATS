"""
SimCATS subpackage containing all functionalities related to the sensor simulation.
"""

from simcats.sensor._sensor_interface import SensorPeakInterface, SensorInterface
from simcats.sensor._gaussian_sensor_peak import SensorPeakGaussian, sensor_response_gauss
from simcats.sensor._lorentzian_sensor_peak import SensorPeakLorentzian, sensor_response_lorentz
from simcats.sensor._generic_sensor import SensorGeneric

__all__ = ["SensorPeakInterface", "SensorInterface", "SensorPeakGaussian", "SensorPeakLorentzian", "SensorGeneric",
           "sensor_response_gauss", "sensor_response_lorentz"]
