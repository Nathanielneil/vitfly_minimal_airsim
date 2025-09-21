# 远程Windows计算机部署指南

## 🎯 远程部署场景

当您需要在远程Windows计算机上部署ViTfly时，推荐使用conda方式，这样可以避免网络和权限问题。

## 🐍 Conda部署方案（推荐）

### 方案1: 使用environment.yml文件

```bash
# 在远程Windows机器上执行

# 1. 克隆或下载项目
git clone https://github.com/Nathanielneil/vitfly_minimal_airsim.git
cd vitfly_minimal_airsim

# 2. 使用conda创建环境
conda env create -f environment.yml

# 3. 激活环境
conda activate vitfly

# 4. 验证安装
python vitfly_main.py --test
```

### 方案2: 手动conda安装

```bash
# 创建新环境
conda create -n vitfly python=3.8

# 激活环境
conda activate vitfly

# 安装主要依赖
conda install pytorch torchvision cpuonly -c pytorch
conda install numpy opencv matplotlib pandas pyyaml tqdm -c conda-forge

# 安装AirSim
pip install airsim

# 验证
python -c "import torch, airsim, cv2; print('安装成功!')"
```

### 方案3: 运行自动脚本

```bash
# Windows
scripts\install_conda.bat

# 或PowerShell
.\scripts\install_conda.bat
```

## 🔧 conda的优势

### 对比pip安装的优势：
- ✅ **依赖管理更可靠** - conda自动解决版本冲突
- ✅ **网络问题更少** - conda有更好的镜像支持
- ✅ **环境隔离** - 不会影响系统Python
- ✅ **跨平台兼容** - Windows/Linux/Mac统一方案
- ✅ **预编译包** - 安装速度更快
- ✅ **科学计算优化** - 针对NumPy/PyTorch优化

### 解决常见问题：
- ❌ SSL证书错误 → ✅ conda镜像源更稳定
- ❌ 编码问题 → ✅ conda处理更好
- ❌ 依赖冲突 → ✅ conda自动解决
- ❌ 权限问题 → ✅ 用户环境安装

## 📋 远程部署完整流程

### 步骤1: 准备远程环境
```bash
# 确认conda已安装
conda --version

# 如果没有conda，下载Miniconda (推荐)
# https://docs.conda.io/en/latest/miniconda.html
```

### 步骤2: 下载项目
```bash
# 方式1: Git克隆
git clone https://github.com/Nathanielneil/vitfly_minimal_airsim.git

# 方式2: 下载ZIP (如果没有git)
# 从GitHub下载ZIP并解压
```

### 步骤3: 创建conda环境
```bash
cd vitfly_minimal_airsim
conda env create -f environment.yml
```

### 步骤4: 激活并测试
```bash
conda activate vitfly
python vitfly_main.py --test
```

### 步骤5: 配置AirSim
```bash
# 确保AirSim正在运行
# 运行ViTfly
python vitfly_navigation.py --mission-type square
```

## 🌐 网络受限环境

### 离线安装包
如果远程机器网络受限，可以：

```bash
# 在有网络的机器上导出环境
conda env export -n vitfly > vitfly_env.yml

# 或打包整个环境
conda pack -n vitfly -o vitfly_env.tar.gz

# 在目标机器上恢复
conda env create -f vitfly_env.yml
# 或
conda unpack vitfly_env.tar.gz
```

### 使用国内镜像源
```bash
# 配置conda国内镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes
```

## 🔍 故障排除

### 常见问题

1. **conda命令未找到**
   ```bash
   # 重新初始化conda
   conda init
   # 重启命令行
   ```

2. **环境创建失败**
   ```bash
   # 更新conda
   conda update conda
   # 清理缓存
   conda clean --all
   ```

3. **AirSim连接失败**
   ```bash
   # 检查防火墙
   # 确认AirSim正在运行
   python -c "import airsim; client = airsim.MultirotorClient(); client.confirmConnection()"
   ```

## 📞 远程支持

### 获取环境信息
```bash
# 发送这些信息用于远程诊断
conda info
conda list
python --version
python vitfly_main.py --test
```

### 日志收集
```bash
# 运行并收集日志
python vitfly_main.py --test > test_log.txt 2>&1
```

## 🎯 推荐配置

### 最佳实践配置
```yaml
# environment.yml 最小配置
name: vitfly
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.8
  - pytorch-cpu
  - numpy
  - opencv
  - pip
  - pip:
    - airsim
```

**conda是远程部署的最佳选择，特别适合企业网络环境和受限网络场景。**