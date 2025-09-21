#!/usr/bin/env python3
"""
ViTfly训练系统入口
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.simple_trainer import main

if __name__ == "__main__":
    main()