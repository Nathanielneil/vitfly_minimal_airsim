"""
简单训练脚本 - 在AirSim中收集数据并训练ViT模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import cv2
from collections import deque
from pathlib import Path

from models.vit_model import create_minimal_vit_model
from airsim_interface.airsim_interface import AirSimDroneInterface


class SimpleDataCollector:
    """简单的数据收集器"""
    
    def __init__(self, max_samples=1000):
        self.max_samples = max_samples
        self.depth_images = []
        self.velocity_commands = []
        self.desired_velocities = []
        self.quaternions = []
        
    def add_sample(self, depth_image, velocity_cmd, desired_vel, quaternion):
        """添加训练样本"""
        if len(self.depth_images) >= self.max_samples:
            # 移除最早的样本
            self.depth_images.pop(0)
            self.velocity_commands.pop(0)
            self.desired_velocities.pop(0)
            self.quaternions.pop(0)
            
        self.depth_images.append(depth_image.copy())
        self.velocity_commands.append(velocity_cmd.copy())
        self.desired_velocities.append(desired_vel)
        self.quaternions.append(quaternion.copy())
        
    def get_dataset(self):
        """获取训练数据集"""
        if len(self.depth_images) == 0:
            return None
            
        # 转换为numpy数组
        depth_array = np.array(self.depth_images)
        velocity_array = np.array(self.velocity_commands)
        desired_vel_array = np.array(self.desired_velocities)
        quaternion_array = np.array(self.quaternions)
        
        return depth_array, velocity_array, desired_vel_array, quaternion_array
        
    def save_dataset(self, path):
        """保存数据集"""
        dataset = self.get_dataset()
        if dataset is not None:
            np.savez(path, 
                    depth_images=dataset[0],
                    velocity_commands=dataset[1], 
                    desired_velocities=dataset[2],
                    quaternions=dataset[3])
            print(f"数据集已保存: {path}")
            
    def load_dataset(self, path):
        """加载数据集"""
        try:
            data = np.load(path)
            self.depth_images = list(data['depth_images'])
            self.velocity_commands = list(data['velocity_commands'])
            self.desired_velocities = list(data['desired_velocities'])
            self.quaternions = list(data['quaternions'])
            print(f"数据集已加载: {path}, 样本数: {len(self.depth_images)}")
        except Exception as e:
            print(f"加载数据集失败: {e}")


def simple_obstacle_avoidance_expert(depth_image, desired_velocity=3.0):
    """简单的专家避障策略"""
    height, width = depth_image.shape
    
    # 分析前方区域的深度
    front_region = depth_image[height//3:2*height//3, width//3:2*width//3]
    left_region = depth_image[height//3:2*height//3, :width//3]
    right_region = depth_image[height//3:2*height//3, 2*width//3:]
    
    # 计算平均深度（深度值越大表示越远）
    front_depth = np.mean(front_region)
    left_depth = np.mean(left_region)
    right_depth = np.mean(right_region)
    
    # 避障逻辑
    obstacle_threshold = 0.3  # 深度阈值
    
    vx = desired_velocity * 0.8  # 基础前进速度
    vy = 0.0
    vz = 0.0
    
    # 前方有障碍物
    if front_depth < obstacle_threshold:
        vx = desired_velocity * 0.3  # 减速
        
        # 选择左右避让方向
        if left_depth > right_depth:
            vy = -2.0  # 向左
        else:
            vy = 2.0   # 向右
            
        # 如果左右都有障碍，尝试上升
        if left_depth < obstacle_threshold and right_depth < obstacle_threshold:
            vz = -1.0  # 上升（AirSim中Z轴向下为正）
    
    # 侧方避障
    elif left_depth < obstacle_threshold:
        vy = 1.0   # 向右避让
    elif right_depth < obstacle_threshold:
        vy = -1.0  # 向左避让
    
    return np.array([vx, vy, vz])


def collect_training_data(interface, collector, duration=300, desired_velocity=3.0):
    """收集训练数据"""
    print(f"开始收集训练数据，持续时间: {duration}秒")
    
    start_time = time.time()
    sample_count = 0
    
    while time.time() - start_time < duration:
        try:
            # 获取传感器数据
            depth_image = interface.get_depth_image()
            state = interface.get_state()
            
            if depth_image is None or not state:
                time.sleep(0.1)
                continue
                
            # 生成专家命令
            expert_velocity = simple_obstacle_avoidance_expert(depth_image, desired_velocity)
            
            # 执行专家命令
            interface.move_by_velocity(*expert_velocity, 0.2)
            
            # 收集数据
            collector.add_sample(
                depth_image,
                expert_velocity,
                desired_velocity,
                state['orientation_quaternion']
            )
            
            sample_count += 1
            
            if sample_count % 50 == 0:
                print(f"已收集 {sample_count} 个样本")
                
            time.sleep(0.1)  # 10Hz数据收集
            
        except KeyboardInterrupt:
            print("数据收集被中断")
            break
        except Exception as e:
            print(f"数据收集错误: {e}")
            time.sleep(0.5)
    
    print(f"数据收集完成，总样本数: {sample_count}")
    return sample_count


def train_model(model, dataset, epochs=50, batch_size=32, lr=0.001):
    """训练模型"""
    depth_images, velocity_commands, desired_velocities, quaternions = dataset
    
    # 转换为PyTorch张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 预处理数据
    depth_tensor = torch.FloatTensor(depth_images).unsqueeze(1).to(device)  # (N, 1, H, W)
    velocity_tensor = torch.FloatTensor(velocity_commands).to(device)
    desired_vel_tensor = torch.FloatTensor(desired_velocities).unsqueeze(1).to(device)
    quaternion_tensor = torch.FloatTensor(quaternions).to(device)
    
    # 归一化速度命令
    velocity_normalized = velocity_tensor / desired_vel_tensor
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 训练循环
    model.train()
    dataset_size = len(depth_images)
    
    print(f"开始训练，数据集大小: {dataset_size}, 批次大小: {batch_size}")
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        # 随机打乱数据
        indices = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # 批次数据
            batch_depth = depth_tensor[batch_indices]
            batch_velocity = velocity_normalized[batch_indices]
            batch_desired = desired_vel_tensor[batch_indices]
            batch_quaternion = quaternion_tensor[batch_indices]
            
            # 前向传播
            optimizer.zero_grad()
            predicted_velocity, _ = model(batch_depth, batch_desired, batch_quaternion)
            
            # 计算损失
            loss = criterion(predicted_velocity, batch_velocity)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, 平均损失: {avg_loss:.6f}")
        
        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'vitfly_training_epoch_{epoch+1}.pth')
    
    # 保存最终模型
    torch.save(model.state_dict(), 'vitfly_trained_model.pth')
    print("训练完成，模型已保存: vitfly_trained_model.pth")


def main():
    """主训练流程"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ViTfly简单训练")
    parser.add_argument('--mode', choices=['collect', 'train', 'both'], default='both',
                       help='模式: collect(收集数据), train(训练模型), both(收集+训练)')
    parser.add_argument('--duration', type=int, default=300, help='数据收集时间(秒)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--data', type=str, default='training_data.npz', help='数据文件路径')
    
    args = parser.parse_args()
    
    # 创建数据收集器
    collector = SimpleDataCollector(max_samples=2000)
    
    if args.mode in ['collect', 'both']:
        # 连接AirSim收集数据
        interface = AirSimDroneInterface()
        
        try:
            if interface.connect():
                if interface.takeoff(altitude=3.0):
                    print("开始专家数据收集...")
                    collect_training_data(interface, collector, args.duration)
                    collector.save_dataset(args.data)
                    
                interface.land()
            else:
                print("AirSim连接失败")
                return
                
        finally:
            interface.disconnect()
    
    if args.mode in ['train', 'both']:
        # 训练模型
        if args.mode == 'train':
            collector.load_dataset(args.data)
            
        dataset = collector.get_dataset()
        if dataset is not None:
            print("开始训练模型...")
            model = create_minimal_vit_model()
            train_model(model, dataset, epochs=args.epochs)
        else:
            print("没有可用的训练数据")


if __name__ == "__main__":
    main()