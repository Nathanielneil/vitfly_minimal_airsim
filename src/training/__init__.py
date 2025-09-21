"""
训练和数据收集模块
"""

from .simple_trainer import SimpleTrainer, SimpleDataCollector
from .model_adapter import ModelAdapter

__all__ = [
    "SimpleTrainer",
    "SimpleDataCollector", 
    "ModelAdapter"
]