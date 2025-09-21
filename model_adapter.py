#!/usr/bin/env python3
"""
模型权重适配器入口
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.model_adapter import main

if __name__ == "__main__":
    main()