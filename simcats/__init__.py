"""SimCATS is a python framework for simulating charge stability diagrams (CSDs) typically measured during the tuning
process of qubits.
"""

from simcats._simulation import Simulation
from simcats._default_configs import default_configs

__all__ = ["Simulation", "default_configs"]

__version__ = "2.0.0"
