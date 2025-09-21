@echo off
REM ViTfly Windows启动脚本
REM 适用于Windows + AirSim 1.8.1 + UE4.7.2

echo ========================================
echo ViTfly Vision Transformer Obstacle Avoidance
echo Windows AirSim版本
echo ========================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请确保Python已正确安装并添加到PATH
    pause
    exit /b 1
)

REM 检查依赖包
echo 检查依赖包...
python -c "import torch, airsim, cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误: 依赖包安装失败
        pause
        exit /b 1
    )
)

REM 检查AirSim连接
echo 检查AirSim连接...
python -c "
import airsim
try:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print('AirSim连接成功')
except:
    print('警告: AirSim连接失败，请确保AirSim环境正在运行')
    exit(1)
" 
if errorlevel 1 (
    echo.
    echo 请确保：
    echo 1. AirSim已正确安装
    echo 2. UE4环境正在运行
    echo 3. 防火墙允许连接
    echo.
    set /p choice="是否继续？(y/n): "
    if /i not "%choice%"=="y" exit /b 1
)

echo.
echo 选择运行模式：
echo 1. 测试模式（验证系统功能）
echo 2. 标准避障飞行（3m/s, 3m高度, 60s）
echo 3. 高速避障飞行（5m/s, 4m高度, 120s）
echo 4. 自定义参数飞行
echo 5. 退出

set /p mode="请选择 (1-5): "

if "%mode%"=="1" (
    echo 启动测试模式...
    python vitfly_main.py --test
) else if "%mode%"=="2" (
    echo 启动标准避障飞行...
    echo 按Ctrl+C可随时安全停止
    python vitfly_main.py
) else if "%mode%"=="3" (
    echo 启动高速避障飞行...
    echo 按Ctrl+C可随时安全停止
    python vitfly_main.py --velocity 5.0 --altitude 4.0 --duration 120
) else if "%mode%"=="4" (
    echo 自定义参数设置：
    set /p velocity="期望速度 (m/s, 默认3.0): "
    set /p altitude="飞行高度 (m, 默认3.0): "
    set /p duration="飞行时间 (s, 默认60): "
    
    if "%velocity%"=="" set velocity=3.0
    if "%altitude%"=="" set altitude=3.0
    if "%duration%"=="" set duration=60
    
    echo 启动自定义避障飞行...
    echo 参数: 速度=%velocity%m/s, 高度=%altitude%m, 时间=%duration%s
    echo 按Ctrl+C可随时安全停止
    python vitfly_main.py --velocity %velocity% --altitude %altitude% --duration %duration%
) else if "%mode%"=="5" (
    echo 退出
    exit /b 0
) else (
    echo 无效选择
    pause
    goto :eof
)

echo.
echo ViTfly运行完成
pause