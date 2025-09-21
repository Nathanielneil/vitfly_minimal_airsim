@echo off
REM ViTfly Conda安装脚本
REM 推荐的安装方式，解决所有依赖问题

echo ========================================
echo ViTfly Conda环境安装
echo ========================================

REM 检查conda是否安装
conda --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到conda，请先安装Anaconda或Miniconda
    echo 下载地址: https://www.anaconda.com/products/distribution
    pause
    exit /b 1
)

echo 当前conda版本:
conda --version

REM 进入项目目录
cd /d "%~dp0.."

echo.
echo 1. 创建vitfly conda环境...
conda env create -f environment.yml

if errorlevel 1 (
    echo 环境创建失败，尝试更新现有环境...
    conda env update -f environment.yml
)

echo.
echo 2. 激活环境...
call conda activate vitfly

echo.
echo 3. 验证安装...
python -c "
import torch
import torchvision
import numpy as np
import cv2
import matplotlib
import pandas as pd
import yaml
print('=== 安装验证成功 ===')
print(f'Python版本: {torch.__version__}')
print(f'PyTorch版本: {torch.__version__}')
print(f'NumPy版本: {np.__version__}')
print(f'OpenCV版本: {cv2.__version__}')
"

if errorlevel 1 (
    echo 安装验证失败
    pause
    exit /b 1
)

echo.
echo 4. 安装AirSim...
pip install airsim>=1.8.1

echo.
echo 5. 创建模型权重...
python model_adapter.py --mode simple

echo.
echo ========================================
echo ViTfly Conda安装完成!
echo ========================================
echo.
echo 使用方法:
echo 1. 每次使用前激活环境: conda activate vitfly
echo 2. 启动AirSim环境
echo 3. 运行: python vitfly_main.py --test
echo.
echo 环境管理:
echo - 激活环境: conda activate vitfly
echo - 停用环境: conda deactivate
echo - 删除环境: conda env remove -n vitfly
echo.
pause