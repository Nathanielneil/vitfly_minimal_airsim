#!/usr/bin/env python3
"""
ViTfly导航系统入口 - 目标点导航 + 避障
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vitfly.vitfly_navigation import main

if __name__ == "__main__":
    main()