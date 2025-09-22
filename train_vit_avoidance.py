#!/usr/bin/env python3
"""
ViT避障模型训练脚本
使用收集的深度图像和避障动作训练端到端模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
import glob
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from vit_model import create_minimal_vit_model


class ViTAvoidanceDataset(Dataset):
    """ViT避障训练数据集"""
    
    def __init__(self, data_dir, sequence_length=10, train=True):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.train = train
        
        # 找到所有轨迹文件夹
        self.trajectories = sorted(glob.glob(str(self.data_dir / "trajectory_*")))
        
        # 构建序列样本
        self.samples = self._build_sequences()
        
        print(f"加载 {'训练' if train else '验证'} 数据集: {len(self.samples)} 个序列")
        
    def _build_sequences(self):
        """构建序列样本"""
        samples = []
        
        for traj_path in self.trajectories:
            # 读取元数据
            csv_path = os.path.join(traj_path, "data.csv")
            if not os.path.exists(csv_path):
                continue
                
            metadata = pd.read_csv(csv_path)
            
            # 过滤碰撞样本
            if self.train:
                metadata = metadata[metadata['collision'] == False]
                
            if len(metadata) < self.sequence_length:
                continue
                
            # 创建序列样本
            for i in range(len(metadata) - self.sequence_length + 1):
                sequence_data = metadata.iloc[i:i+self.sequence_length]
                
                sample = {
                    'trajectory_path': traj_path,
                    'start_frame': i,
                    'sequence_data': sequence_data
                }
                samples.append(sample)
                
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        traj_path = sample['trajectory_path']
        start_frame = sample['start_frame']
        sequence_data = sample['sequence_data']
        
        # 加载序列深度图像
        depth_images = []
        desired_velocities = []
        quaternions = []
        velocity_commands = []
        
        for _, row in sequence_data.iterrows():
            # 加载深度图像
            img_path = os.path.join(traj_path, f"depth_{row['frame_id']:06d}.png")
            depth_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            depth_img = depth_img.astype(np.float32) / 255.0
            depth_images.append(depth_img)
            
            # 其他数据
            desired_velocities.append(row['desired_vel'])
            quaternions.append([row['quat_w'], row['quat_x'], row['quat_y'], row['quat_z']])
            velocity_commands.append([row['cmd_vx'], row['cmd_vy'], row['cmd_vz']])
            
        # 转换为张量
        depth_images = torch.FloatTensor(np.array(depth_images)).unsqueeze(1)  # (seq, 1, H, W)
        desired_velocities = torch.FloatTensor(desired_velocities).unsqueeze(-1)  # (seq, 1)
        quaternions = torch.FloatTensor(quaternions)  # (seq, 4)
        velocity_commands = torch.FloatTensor(velocity_commands)  # (seq, 3)
        
        return {
            'depth_images': depth_images,
            'desired_velocities': desired_velocities,
            'quaternions': quaternions,
            'velocity_commands': velocity_commands
        }


class ViTTrainer:
    """ViT避障训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = create_minimal_vit_model().to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        
        # 数据加载器
        self._setup_data_loaders()
        
        print(f"训练器初始化完成，使用设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def _setup_data_loaders(self):
        """设置数据加载器"""
        # 划分训练集和验证集
        all_trajectories = sorted(glob.glob(str(Path(self.args.data_dir) / "trajectory_*")))
        split_idx = int(len(all_trajectories) * 0.8)
        
        train_trajectories = all_trajectories[:split_idx]
        val_trajectories = all_trajectories[split_idx:]
        
        # 创建数据集
        train_dataset = ViTAvoidanceDataset(self.args.data_dir, train=True)
        val_dataset = ViTAvoidanceDataset(self.args.data_dir, train=False)
        
        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size, 
            shuffle=False,
            num_workers=4
        )
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc="训练"):
            # 移动数据到设备
            depth_images = batch['depth_images'].to(self.device)  # (batch, seq, 1, H, W)
            desired_velocities = batch['desired_velocities'].to(self.device)
            quaternions = batch['quaternions'].to(self.device)
            velocity_commands = batch['velocity_commands'].to(self.device)
            
            batch_size, seq_length = depth_images.shape[:2]
            
            # 重置LSTM状态
            hidden_state = self.model.reset_lstm_state(batch_size, self.device)
            
            self.optimizer.zero_grad()
            total_loss_batch = 0.0
            
            # 序列训练
            for t in range(seq_length):
                depth_img = depth_images[:, t]  # (batch, 1, H, W)
                desired_vel = desired_velocities[:, t]  # (batch, 1)
                quaternion = quaternions[:, t]  # (batch, 4)
                target_vel = velocity_commands[:, t]  # (batch, 3)
                
                # 前向传播
                pred_vel, hidden_state = self.model(depth_img, desired_vel, quaternion, hidden_state)
                
                # 计算损失
                loss = self.criterion(pred_vel, target_vel)
                total_loss_batch += loss
                
            # 反向传播
            avg_loss = total_loss_batch / seq_length
            avg_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += avg_loss.item()
            num_batches += 1
            
        return total_loss / num_batches
        
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                depth_images = batch['depth_images'].to(self.device)
                desired_velocities = batch['desired_velocities'].to(self.device)
                quaternions = batch['quaternions'].to(self.device)
                velocity_commands = batch['velocity_commands'].to(self.device)
                
                batch_size, seq_length = depth_images.shape[:2]
                hidden_state = self.model.reset_lstm_state(batch_size, self.device)
                
                total_loss_batch = 0.0
                
                for t in range(seq_length):
                    depth_img = depth_images[:, t]
                    desired_vel = desired_velocities[:, t] 
                    quaternion = quaternions[:, t]
                    target_vel = velocity_commands[:, t]
                    
                    pred_vel, hidden_state = self.model(depth_img, desired_vel, quaternion, hidden_state)
                    loss = self.criterion(pred_vel, target_vel)
                    total_loss_batch += loss
                    
                avg_loss = total_loss_batch / seq_length
                total_loss += avg_loss.item()
                num_batches += 1
                
        return total_loss / num_batches
        
    def train(self):
        """训练主循环"""
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(self.args.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.args.num_epochs}")
            
            # 训练
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            print(f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.args.output_path)
                print(f"保存最佳模型: {self.args.output_path}")
                
        # 绘制训练曲线
        self._plot_training_curves(train_losses, val_losses)
        
    def _plot_training_curves(self, train_losses, val_losses):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('ViT避障训练曲线')
        plt.savefig('training_curves.png')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='ViT避障模型训练')
    parser.add_argument('--data_dir', type=str, default='./training_data', help='训练数据目录')
    parser.add_argument('--output_path', type=str, default='vitfly_trained_model.pth', help='输出模型路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"数据目录不存在: {args.data_dir}")
        print("请先使用 data_collector.py 收集训练数据")
        return
        
    # 开始训练
    trainer = ViTTrainer(args)
    trainer.train()
    
    print("训练完成！")


if __name__ == "__main__":
    main()