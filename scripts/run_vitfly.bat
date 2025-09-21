@echo off
REM ViTfly Windows启动脚本 (重构版本)
REM 适用于Windows + AirSim 1.8.1 + UE4.7.2

echo ========================================
echo ViTfly Vision Transformer Obstacle Avoidance
echo Windows AirSim版本 (重构)
echo ========================================

REM 进入项目根目录
cd /d "%~dp0.."

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
echo 1. 系统测试（验证所有功能）
echo 2. 基础避障飞行（纯避障，60s）
echo 3. 导航任务 - 正方形巡航
echo 4. 导航任务 - 探索模式
echo 5. 模型权重管理
echo 6. 训练新模型
echo 7. 退出

set /p mode="请选择 (1-7): "

if "%mode%"=="1" (
    echo 启动系统测试...
    python vitfly_main.py --test
) else if "%mode%"=="2" (
    echo 启动基础避障飞行...
    echo 按Ctrl+C可随时安全停止
    python model_adapter.py --mode simple
    python vitfly_main.py --model vitfly_simple_policy.pth
) else if "%mode%"=="3" (
    echo 启动正方形巡航任务...
    echo 按Ctrl+C可随时安全停止
    python model_adapter.py --mode simple
    python vitfly_navigation.py --model vitfly_simple_policy.pth --mission-type square
) else if "%mode%"=="4" (
    echo 启动探索任务...
    echo 按Ctrl+C可随时安全停止
    python model_adapter.py --mode simple
    python vitfly_navigation.py --model vitfly_simple_policy.pth --mission-file mission_exploration.yaml
) else if "%mode%"=="5" (
    echo 模型权重管理...
    echo 1. 创建简单策略权重
    echo 2. 创建虚拟训练权重
    echo 3. 适配原始ViTfly权重
    set /p weight_mode="选择权重类型 (1-3): "
    
    if "%weight_mode%"=="1" (
        python model_adapter.py --mode simple
    ) else if "%weight_mode%"=="2" (
        python model_adapter.py --mode dummy
    ) else if "%weight_mode%"=="3" (
        set /p original_path="请输入原始权重路径: "
        python model_adapter.py --mode adapt --original "%original_path%"
    )
) else if "%mode%"=="6" (
    echo 训练新模型...
    echo 1. 仅收集数据
    echo 2. 仅训练模型
    echo 3. 收集数据+训练
    set /p train_mode="选择训练模式 (1-3): "
    
    set /p duration="数据收集时间(秒，默认300): "
    if "%duration%"=="" set duration=300
    
    if "%train_mode%"=="1" (
        python train.py --mode collect --duration %duration%
    ) else if "%train_mode%"=="2" (
        python train.py --mode train
    ) else if "%train_mode%"=="3" (
        python train.py --mode both --duration %duration%
    )
) else if "%mode%"=="7" (
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