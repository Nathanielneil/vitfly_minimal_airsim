# Windows部署指南

## 🚨 常见问题解决

### 问题1: SSL证书错误
```
Could not fetch URL https://pypi.org/simple/pip/: HTTPSConnectionPool SSL error
```

### 问题2: 编码错误
```
UnicodeDecodeError: 'gbk' codec can't decode byte 0x80
```

## 🛠️ 解决方案

### 方案1: 使用安装脚本（推荐）

```bash
# 直接运行Windows安装脚本
scripts\install_windows.bat
```

这个脚本会自动：
- 设置UTF-8编码
- 配置可信镜像源
- 逐个安装依赖包
- 验证安装结果

### 方案2: 手动解决

#### 步骤1: 设置编码
```bash
# 在PowerShell中设置编码
$env:PYTHONIOENCODING="utf-8"
chcp 65001
```

#### 步骤2: 配置pip镜像源
```bash
# 配置清华镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 或使用阿里云镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
```

#### 步骤3: 逐个安装依赖
```bash
# 升级pip
python -m pip install --upgrade pip --trusted-host pypi.org

# 安装PyTorch (CPU版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install numpy opencv-python airsim PyYAML tqdm matplotlib pandas
```

### 方案3: 使用conda环境

```bash
# 创建conda环境
conda create -n vitfly python=3.8
conda activate vitfly

# 安装conda包
conda install numpy opencv matplotlib pandas pyyaml
conda install pytorch torchvision cpuonly -c pytorch

# 安装pip包
pip install airsim tqdm
```

### 方案4: 离线安装

如果网络问题严重，可以下载wheel文件离线安装：

```bash
# 下载所需的.whl文件到本地，然后：
pip install --find-links ./wheels --no-index package_name
```

## 🔧 环境配置

### Python环境检查
```bash
# 检查Python版本（应为3.8-3.10）
python --version

# 检查pip版本
pip --version

# 检查编码设置
python -c "import sys; print(sys.getdefaultencoding())"
```

### AirSim配置
确保AirSim settings.json配置正确：
```json
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "DefaultVehicleState": "Armed",
      "EnableCollisions": true,
      "AllowAPIAlways": true
    }
  }
}
```

## 📋 验证安装

### 基础验证
```bash
# 运行系统测试
python vitfly_main.py --test
```

### 详细验证
```bash
# 检查所有导入
python -c "
import torch
import airsim
import cv2
import numpy as np
import yaml
print('所有依赖导入成功!')
"
```

## 🚀 快速启动

### 完整流程
```bash
# 1. 安装环境
scripts\install_windows.bat

# 2. 启动AirSim
# (手动启动UE4 AirSim环境)

# 3. 运行ViTfly
scripts\run_vitfly.bat
```

## ⚠️ 常见问题

### 问题: ModuleNotFoundError
```bash
# 解决方案: 检查环境激活
conda activate vitfly
# 或
Scripts\activate.bat
```

### 问题: AirSim连接失败
```bash
# 解决方案: 
# 1. 确保AirSim正在运行
# 2. 检查防火墙设置
# 3. 验证IP和端口配置
```

### 问题: CUDA相关错误
```bash
# 解决方案: 使用CPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 📞 技术支持

如果遇到其他问题：
1. 检查Python版本是否为3.8-3.10
2. 确保使用管理员权限运行
3. 尝试使用不同的镜像源
4. 查看详细错误日志