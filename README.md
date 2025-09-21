# ViTfly最小实现 - Windows AirSim版本

基于Vision Transformer的端到端无人机避障系统，实现从深度图像直接预测三维速度指令。此版本专为Windows + AirSim 1.8.1 + UE4.7.2环境设计。

## 系统架构

```
深度图像(60×90) → ViT特征提取 → LSTM时序建模 → 3D速度指令(vx,vy,vz)
     ↓              ↓               ↓              ↓
   预处理     自注意力机制      多模态融合       安全控制
```

## 核心特性

- **端到端学习**: 直接从深度图像学习避障策略
- **Vision Transformer**: 全局感受野，优异的空间理解能力  
- **实时推理**: 推理时间 <50ms，支持10Hz控制频率
- **安全机制**: 多层安全保障，包括渐进加速和紧急制动
- **AirSim集成**: 完全兼容Windows AirSim环境

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保AirSim 1.8.1已正确安装并运行UE4环境
```

### 2. 测试系统

```bash
# 系统连接测试
python vitfly_main.py --test

# 参数说明:
# --test: 启动测试模式，验证AirSim连接和模型功能
```

### 3. 开始避障飞行

```bash
# 基础飞行（默认参数）
python vitfly_main.py

# 自定义参数飞行
python vitfly_main.py --velocity 5.0 --altitude 4.0 --duration 120

# 参数说明:
# --velocity: 期望飞行速度 (m/s)，默认3.0
# --altitude: 飞行高度 (m)，默认3.0  
# --duration: 飞行持续时间 (s)，默认60.0
# --model: 预训练模型路径（可选）
# --device: 计算设备 (cuda/cpu/auto)，默认auto
```

## 文件结构

```
vitfly_minimal_airsim/
├── vit_model.py           # Vision Transformer模型实现
├── airsim_interface.py    # AirSim接口和安全控制
├── vitfly_main.py         # 主程序
├── config.yaml           # 配置文件
├── requirements.txt      # Python依赖
└── README.md            # 说明文档
```

## 技术细节

### Vision Transformer模型

- **输入**: 深度图像(60×90) + 期望速度 + 四元数姿态
- **架构**: 重叠补丁嵌入 → 高效自注意力 → 混合FFN → LSTM → 速度预测
- **参数量**: ~150K（轻量化设计）
- **特色**: 降维注意力机制，计算复杂度从O(N²)降到O(N×N/r²)

### 安全机制

1. **渐进加速**: 起飞后3秒内逐渐加速到目标速度
2. **速度限制**: 最大速度10m/s，最大加速度5m/s²
3. **高度保护**: 0.5-50m高度范围限制
4. **碰撞检测**: 实时碰撞监测和紧急制动
5. **优雅退出**: Ctrl+C安全降落

### AirSim配置

在AirSim的`settings.json`中建议使用以下配置：

```json
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "DefaultVehicleState": "Armed",
      "EnableCollisionPassthrogh": false,
      "EnableCollisions": true,
      "AllowAPIAlways": true,
      "RC": {
        "RemoteControlID": 0,
        "AllowAPIWhenDisconnected": true
      }
    }
  },
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 640,
        "Height": 480,
        "FOV_Degrees": 90
      },
      {
        "ImageType": 2,
        "Width": 640, 
        "Height": 480,
        "FOV_Degrees": 90
      }
    ]
  }
}
```

## 性能优化

- **CPU推理**: 优化的卷积和注意力计算，支持实时推理
- **内存优化**: 轻量化模型设计，降低内存占用
- **批处理**: 支持批量推理（如果需要）

## 扩展功能

- **模型训练**: 使用模仿学习训练自定义模型
- **多相机**: 支持多相机输入融合
- **路径规划**: 集成更高级的路径规划算法
- **多机协同**: 扩展到多无人机协同避障

## 故障排除

### 常见问题

1. **AirSim连接失败**
   - 确保AirSim和UE4环境正在运行
   - 检查防火墙设置
   - 验证IP地址和端口配置

2. **模型推理速度慢**
   - 确保PyTorch已正确安装CUDA支持
   - 检查GPU内存是否充足
   - 考虑使用CPU推理作为备选

3. **无人机行为异常**
   - 检查AirSim物理参数设置
   - 验证相机标定和位置
   - 调整控制参数

### 调试模式

```bash
# 启用详细日志
python vitfly_main.py --test

# 检查系统状态
python -c "
from airsim_interface import AirSimDroneInterface
interface = AirSimDroneInterface()
print('连接状态:', interface.connect())
print('无人机状态:', interface.get_state())
"
```

## 许可证

基于原始ViTfly项目进行简化和适配，用于学习和研究目的。

## 参考文献

- Bhattacharya, A., et al. "Vision transformers for end-to-end vision-based quadrotor obstacle avoidance." ICRA 2025.
- 原始项目: https://github.com/anish-bhattacharya/vitfly