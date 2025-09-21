"""
ViTfly - Vision Transformer based Obstacle Avoidance for Quadrotor UAVs
A minimal implementation for Windows AirSim environment
"""

__version__ = "1.0.0"
__author__ = "ViTfly Team"
__email__ = ""

from .vitfly import ViTflySystem
from .models import create_minimal_vit_model
from .navigation import NavigationController
from .airsim_interface import AirSimDroneInterface

__all__ = [
    "ViTflySystem",
    "create_minimal_vit_model", 
    "NavigationController",
    "AirSimDroneInterface"
]