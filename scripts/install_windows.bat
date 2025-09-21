@echo off
REM ViTfly Windows安装脚本
REM 解决SSL和编码问题

echo ======================================
echo ViTfly Windows环境安装
echo ======================================

REM 设置编码为UTF-8
chcp 65001

REM 进入项目目录
cd /d "%~dp0.."

echo 1. 检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8-3.10
    pause
    exit /b 1
)

echo.
echo 2. 升级pip...
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

echo.
echo 3. 配置pip镜像源...
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

echo.
echo 4. 安装核心依赖...

REM 逐个安装避免编码问题
echo 安装PyTorch...
pip install torch>=1.12.0 torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cpu

echo 安装数值计算库...
pip install numpy>=1.21.0

echo 安装图像处理库...
pip install opencv-python>=4.5.0 Pillow>=8.3.0

echo 安装AirSim...
pip install airsim>=1.8.1

echo 安装工具库...
pip install PyYAML>=6.0 tqdm>=4.62.0 matplotlib>=3.5.0 pandas>=1.3.0

echo.
echo 5. 验证安装...
python -c "
import torch
import airsim
import cv2
import numpy as np
import yaml
print('所有依赖安装成功!')
print(f'PyTorch版本: {torch.__version__}')
print(f'NumPy版本: {np.__version__}')
print(f'OpenCV版本: {cv2.__version__}')
"

if errorlevel 1 (
    echo 安装验证失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo 6. 创建初始模型权重...
python model_adapter.py --mode simple

echo.
echo ======================================
echo ViTfly安装完成!
echo ======================================
echo.
echo 使用方法:
echo 1. 启动AirSim环境
echo 2. 运行: scripts\run_vitfly.bat
echo 3. 或直接运行: python vitfly_main.py --test
echo.
pause