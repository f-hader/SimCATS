"""SimCATS is a python framework for simulating charge stability diagrams (CSDs) typically measured during the tuning
process of qubits.
"""

from ._simulation import Simulation
from ._default_configs import default_configs

__all__ = ["Simulation", "default_configs"]
__version__ = "1.1.0"
