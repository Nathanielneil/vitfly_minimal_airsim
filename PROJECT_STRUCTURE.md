# ViTfly项目结构

## 📁 目录结构

```
vitfly_minimal_airsim/
├── src/                          # 源代码目录
│   ├── __init__.py              # 包初始化
│   ├── models/                   # 模型模块
│   │   ├── __init__.py
│   │   └── vit_model.py         # Vision Transformer模型
│   ├── navigation/              # 导航模块
│   │   ├── __init__.py
│   │   ├── navigation_controller.py  # 导航控制器
│   │   └── mission_planner.py   # 任务规划器
│   ├── airsim_interface/        # AirSim接口模块
│   │   ├── __init__.py
│   │   └── airsim_interface.py  # AirSim连接和控制
│   ├── vitfly/                  # 核心系统模块
│   │   ├── __init__.py
│   │   ├── vitfly_system.py     # 基础避障系统
│   │   └── vitfly_navigation.py # 导航系统
│   └── training/                # 训练模块
│       ├── __init__.py
│       ├── simple_trainer.py    # 简单训练器
│       └── model_adapter.py     # 模型权重适配器
├── configs/                     # 配置文件目录
│   ├── config.yaml             # 主配置文件
│   ├── mission_square.yaml     # 正方形任务
│   └── mission_exploration.yaml # 探索任务
├── examples/                    # 示例代码目录
│   └── advanced_avoidance_demo.py # 高级避障演示
├── scripts/                     # 脚本目录
│   └── run_vitfly.bat          # Windows启动脚本
├── docs/                        # 文档目录
├── tests/                       # 测试目录
├── data/                        # 数据目录
├── models/                      # 预训练模型目录
├── logs/                        # 日志目录
├── vitfly_main.py              # 基础避障入口
├── vitfly_navigation.py        # 导航系统入口
├── train.py                    # 训练入口
├── model_adapter.py            # 模型适配入口
├── requirements.txt            # Python依赖
├── setup.py                    # 安装脚本
├── README.md                   # 项目说明
├── PROJECT_STRUCTURE.md        # 项目结构说明
└── .gitignore                  # Git忽略文件
```

## 🧩 模块说明

### 核心模块

#### 1. `src/models/` - 模型模块
- **vit_model.py**: Vision Transformer核心实现
  - `MinimalViTObstacleAvoidance`: 主模型类
  - `OverlapPatchEmbedding`: 重叠补丁嵌入
  - `EfficientMultiHeadAttention`: 高效自注意力
  - `MixFFN`: 混合前馈网络

#### 2. `src/navigation/` - 导航模块
- **navigation_controller.py**: 导航控制器
  - `NavigationController`: 主控制器
  - `Waypoint`: 航点数据结构
- **mission_planner.py**: 任务规划器
  - `MissionPlanner`: 任务生成和优化

#### 3. `src/airsim_interface/` - AirSim接口模块
- **airsim_interface.py**: AirSim连接和控制
  - `AirSimDroneInterface`: 主接口类
  - `SafetyController`: 安全控制器

#### 4. `src/vitfly/` - 核心系统模块
- **vitfly_system.py**: 基础避障系统
  - `ViTflySystem`: 纯避障系统
- **vitfly_navigation.py**: 导航系统
  - `ViTflyNavigationSystem`: 导航+避障系统

#### 5. `src/training/` - 训练模块
- **simple_trainer.py**: 训练和数据收集
  - `SimpleTrainer`: 训练器
  - `SimpleDataCollector`: 数据收集器
- **model_adapter.py**: 模型权重适配
  - `ModelAdapter`: 权重适配器

### 配置和数据

#### 1. `configs/` - 配置文件
- **config.yaml**: 主配置文件（模型、飞行、安全参数）
- **mission_*.yaml**: 任务配置文件

#### 2. `examples/` - 示例代码
- **advanced_avoidance_demo.py**: 避障策略演示

#### 3. `scripts/` - 工具脚本
- **run_vitfly.bat**: Windows便捷启动脚本

## 🚀 入口文件

### 主要入口
- **vitfly_main.py**: 基础避障系统
- **vitfly_navigation.py**: 导航+避障系统
- **train.py**: 训练系统
- **model_adapter.py**: 模型权重适配

### 使用方式
```bash
# 基础避障
python vitfly_main.py --model weights.pth

# 导航系统
python vitfly_navigation.py --mission-type square

# 模型训练
python train.py --mode both --duration 300

# 权重适配
python model_adapter.py --mode simple
```

## 📦 包管理

### 安装方式
```bash
# 开发安装
pip install -e .

# 生产安装
pip install .

# 依赖安装
pip install -r requirements.txt
```

### 导入方式
```python
# 从包导入
from vitfly import ViTflySystem
from vitfly.models import create_minimal_vit_model
from vitfly.navigation import NavigationController

# 直接导入
from src.vitfly.vitfly_system import ViTflySystem
```

## 🔧 开发指南

### 添加新功能
1. 在相应模块下创建新文件
2. 更新对应的`__init__.py`
3. 创建相应的测试文件
4. 更新文档

### 测试
```bash
# 运行测试
pytest tests/

# 系统测试
python vitfly_main.py --test
```

### 代码风格
- 使用类型注解
- 遵循PEP 8规范
- 添加docstring文档
- 适当的错误处理

## 📝 配置管理

### 环境配置
- 开发环境: `configs/dev_config.yaml`
- 生产环境: `configs/prod_config.yaml`
- 测试环境: `configs/test_config.yaml`

### 模型配置
- 模型参数在`configs/config.yaml`中配置
- 支持命令行参数覆盖
- 支持环境变量配置

这种结构的优势：
- ✅ 模块化设计，职责清晰
- ✅ 易于维护和扩展
- ✅ 符合Python包开发标准
- ✅ 支持测试和CI/CD
- ✅ 配置和代码分离