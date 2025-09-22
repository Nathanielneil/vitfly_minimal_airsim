#!/usr/bin/env python3
"""
ViT避障训练数据收集器
收集深度图像和对应的避障动作用于训练
"""

import airsim
import numpy as np
import cv2
import time
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import threading


class ViTDataCollector:
    """ViT避障训练数据收集器"""
    
    def __init__(self, data_dir="./training_data"):
        self.client = airsim.MultirotorClient()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 数据收集参数
        self.collection_frequency = 10  # Hz
        self.image_size = (60, 90)
        
        # 当前收集状态
        self.is_collecting = False
        self.current_trajectory = []
        self.trajectory_count = 0
        
        # 控制输入
        self.current_velocity_cmd = np.array([0.0, 0.0, 0.0])
        
        print("数据收集器初始化完成")
        
    def connect(self):
        """连接AirSim"""
        try:
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            print("AirSim连接成功")
            return True
        except Exception as e:
            print(f"AirSim连接失败: {e}")
            return False
            
    def get_sensor_data(self):
        """获取传感器数据"""
        try:
            # 获取深度图像
            request = airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)
            response = self.client.simGetImages([request])[0]
            
            if response.image_data_float:
                depth_array = np.array(response.image_data_float, dtype=np.float32)
                depth_array = depth_array.reshape(response.height, response.width)
                
                # 调整到模型输入尺寸
                depth_image = cv2.resize(depth_array, (self.image_size[1], self.image_size[0]))
                
                # 归一化到[0,1]
                depth_image = np.clip(depth_image, 0, 100) / 100.0
            else:
                return None
                
            # 获取无人机状态
            state = self.client.getMultirotorState()
            pose = state.kinematics_estimated.pose
            velocity = state.kinematics_estimated.linear_velocity
            
            # 组织传感器数据
            sensor_data = {
                'depth_image': depth_image,
                'position': np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val]),
                'quaternion': np.array([pose.orientation.w_val, pose.orientation.x_val, 
                                      pose.orientation.y_val, pose.orientation.z_val]),
                'velocity': np.array([velocity.x_val, velocity.y_val, velocity.z_val]),
                'timestamp': time.time()
            }
            
            return sensor_data
            
        except Exception as e:
            print(f"传感器数据获取失败: {e}")
            return None
            
    def handle_terminal_input(self):
        """处理终端输入（非阻塞）"""
        import select
        import sys
        
        # 检查是否有输入
        if select.select([sys.stdin], [], [], 0)[0]:
            command = sys.stdin.readline().strip().lower()
            return self.process_command(command)
        return True
    
    def process_command(self, command):
        """处理命令"""
        base_speed = 2.0
        
        if command == 'w':
            self.current_velocity_cmd = np.array([base_speed, 0, 0])
            print("前进")
        elif command == 's':
            self.current_velocity_cmd = np.array([-base_speed, 0, 0])
            print("后退")
        elif command == 'a':
            self.current_velocity_cmd = np.array([0, -base_speed, 0])
            print("左移")
        elif command == 'd':
            self.current_velocity_cmd = np.array([0, base_speed, 0])
            print("右移")
        elif command == 'q':
            self.current_velocity_cmd = np.array([0, 0, -base_speed])
            print("上升")
        elif command == 'e':
            self.current_velocity_cmd = np.array([0, 0, base_speed])
            print("下降")
        elif command == 'stop':
            self.current_velocity_cmd = np.array([0, 0, 0])
            print("停止")
        elif command == 'start' or command == 'space':
            if not self.is_collecting:
                self.start_collection()
            else:
                print("已在收集中")
        elif command == 'end':
            if self.is_collecting:
                self.stop_collection()
            else:
                print("未在收集中")
        elif command == 'quit' or command == 'exit':
            return False
        elif command == 'help':
            self.print_help()
        else:
            print(f"未知命令: {command}")
            
        return True
    
    def print_help(self):
        """打印帮助信息"""
        print("\n可用命令:")
        print("  w/s - 前进/后退")
        print("  a/d - 左移/右移") 
        print("  q/e - 上升/下降")
        print("  stop - 停止移动")
        print("  start/space - 开始数据收集")
        print("  end - 停止数据收集")
        print("  help - 显示帮助")
        print("  quit/exit - 退出程序")
        print("输入命令后按回车执行\n")
        
    def toggle_collection(self):
        """切换数据收集状态"""
        if self.is_collecting:
            self.stop_collection()
        else:
            self.start_collection()
            
    def start_collection(self):
        """开始数据收集"""
        self.is_collecting = True
        self.current_trajectory = []
        print(f"开始收集轨迹 {self.trajectory_count}")
        
    def stop_collection(self):
        """停止数据收集并保存"""
        if self.is_collecting and len(self.current_trajectory) > 0:
            self.save_trajectory()
            self.trajectory_count += 1
            
        self.is_collecting = False
        self.current_trajectory = []
        print("数据收集已停止")
        
    def collect_sample(self):
        """收集单个数据样本"""
        sensor_data = self.get_sensor_data()
        if sensor_data is None:
            return
            
        # 执行当前速度指令
        self.client.moveByVelocityAsync(
            float(self.current_velocity_cmd[0]),
            float(self.current_velocity_cmd[1]), 
            float(self.current_velocity_cmd[2]),
            0.1  # 持续时间
        )
        
        # 检查碰撞
        collision_info = self.client.simGetCollisionInfo()
        
        # 构建训练样本
        sample = {
            'depth_image': sensor_data['depth_image'],
            'position': sensor_data['position'],
            'quaternion': sensor_data['quaternion'],
            'current_velocity': sensor_data['velocity'],
            'desired_velocity': 2.0,  # 期望前进速度
            'velocity_command': self.current_velocity_cmd.copy(),
            'collision': collision_info.has_collided,
            'timestamp': sensor_data['timestamp']
        }
        
        if self.is_collecting:
            self.current_trajectory.append(sample)
            print(f"收集样本 {len(self.current_trajectory)}: 速度{self.current_velocity_cmd}")
            
    def save_trajectory(self):
        """保存当前轨迹数据"""
        if len(self.current_trajectory) == 0:
            return
            
        # 创建轨迹文件夹
        traj_dir = self.data_dir / f"trajectory_{self.trajectory_count:06d}"
        traj_dir.mkdir(exist_ok=True)
        
        # 保存深度图像和元数据
        metadata = []
        
        for i, sample in enumerate(self.current_trajectory):
            # 保存深度图像
            depth_img = (sample['depth_image'] * 255).astype(np.uint8)
            img_path = traj_dir / f"depth_{i:06d}.png"
            cv2.imwrite(str(img_path), depth_img)
            
            # 收集元数据
            metadata.append({
                'frame_id': i,
                'timestamp': sample['timestamp'],
                'pos_x': sample['position'][0],
                'pos_y': sample['position'][1], 
                'pos_z': sample['position'][2],
                'quat_w': sample['quaternion'][0],
                'quat_x': sample['quaternion'][1],
                'quat_y': sample['quaternion'][2],
                'quat_z': sample['quaternion'][3],
                'vel_x': sample['current_velocity'][0],
                'vel_y': sample['current_velocity'][1],
                'vel_z': sample['current_velocity'][2],
                'desired_vel': sample['desired_velocity'],
                'cmd_vx': sample['velocity_command'][0],
                'cmd_vy': sample['velocity_command'][1],
                'cmd_vz': sample['velocity_command'][2],
                'collision': sample['collision']
            })
            
        # 保存元数据CSV
        df = pd.DataFrame(metadata)
        df.to_csv(traj_dir / "data.csv", index=False)
        
        print(f"轨迹 {self.trajectory_count} 已保存: {len(self.current_trajectory)} 样本")
        
    def run_collection(self):
        """运行数据收集主循环"""
        if not self.connect():
            return
            
        print("数据收集器启动!")
        print("在终端中输入命令控制无人机:")
        self.print_help()
        
        try:
            dt = 1.0 / self.collection_frequency
            
            while True:
                loop_start = time.time()
                
                # 处理终端输入
                if not self.handle_terminal_input():
                    break
                    
                # 收集数据
                self.collect_sample()
                
                # 控制频率
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n收到中断信号")
        finally:
            self.stop_collection()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("数据收集结束")


def main():
    """主函数"""
    collector = ViTDataCollector()
    collector.run_collection()


if __name__ == "__main__":
    main()