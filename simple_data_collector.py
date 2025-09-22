#!/usr/bin/env python3
"""
简化版数据收集器 - 使用AirSim内置控制
"""

import airsim
import numpy as np
import cv2
import time
import os
import pandas as pd
from datetime import datetime
from pathlib import Path


class SimpleDataCollector:
    """简化数据收集器 - 使用AirSim手柄/键盘控制"""
    
    def __init__(self, data_dir="./training_data"):
        self.client = airsim.MultirotorClient()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 收集参数
        self.collection_frequency = 10  # Hz
        self.image_size = (60, 90)
        
        # 状态
        self.is_collecting = False
        self.current_trajectory = []
        self.trajectory_count = 0
        self.last_velocity = np.array([0.0, 0.0, 0.0])
        
        print("简化数据收集器初始化完成")
        print("使用方法:")
        print("1. 在AirSim中手动控制无人机 (WASD键)")
        print("2. 运行此程序自动记录飞行数据")
        print("3. 按Ctrl+C停止收集")
        
    def connect(self):
        """连接AirSim"""
        try:
            self.client.confirmConnection()
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
                depth_image = np.clip(depth_image, 0, 100) / 100.0
            else:
                return None
                
            # 获取无人机状态
            state = self.client.getMultirotorState()
            pose = state.kinematics_estimated.pose
            velocity = state.kinematics_estimated.linear_velocity
            
            # 组织数据
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
            return None
            
    def detect_movement(self, current_velocity):
        """检测无人机是否在移动（表示用户在控制）"""
        velocity_magnitude = np.linalg.norm(current_velocity)
        velocity_change = np.linalg.norm(current_velocity - self.last_velocity)
        
        # 如果速度大于阈值或速度变化明显，认为在控制中
        is_moving = velocity_magnitude > 0.1 or velocity_change > 0.05
        
        self.last_velocity = current_velocity.copy()
        return is_moving
        
    def collect_sample(self, sensor_data):
        """收集数据样本"""
        # 检查碰撞
        collision_info = self.client.simGetCollisionInfo()
        
        # 估算期望前进速度（基于当前速度）
        velocity_magnitude = np.linalg.norm(sensor_data['velocity'])
        desired_velocity = max(1.0, velocity_magnitude) if velocity_magnitude > 0.1 else 1.0
        
        # 构建样本
        sample = {
            'depth_image': sensor_data['depth_image'],
            'position': sensor_data['position'],
            'quaternion': sensor_data['quaternion'],
            'current_velocity': sensor_data['velocity'],
            'desired_velocity': desired_velocity,
            'velocity_command': sensor_data['velocity'].copy(),  # 用当前速度作为"专家动作"
            'collision': collision_info.has_collided,
            'timestamp': sensor_data['timestamp']
        }
        
        self.current_trajectory.append(sample)
        
        if len(self.current_trajectory) % 50 == 0:
            print(f"已收集 {len(self.current_trajectory)} 个样本")
            
    def start_new_trajectory(self):
        """开始新轨迹"""
        if len(self.current_trajectory) > 0:
            self.save_trajectory()
            
        self.current_trajectory = []
        self.is_collecting = True
        print(f"开始收集轨迹 {self.trajectory_count}")
        
    def save_trajectory(self):
        """保存当前轨迹"""
        if len(self.current_trajectory) < 10:  # 太短的轨迹不保存
            print("轨迹太短，跳过保存")
            return
            
        # 创建轨迹文件夹
        traj_dir = self.data_dir / f"trajectory_{self.trajectory_count:06d}"
        traj_dir.mkdir(exist_ok=True)
        
        # 保存数据
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
            
        # 保存CSV
        df = pd.DataFrame(metadata)
        df.to_csv(traj_dir / "data.csv", index=False)
        
        print(f"轨迹 {self.trajectory_count} 已保存: {len(self.current_trajectory)} 样本")
        self.trajectory_count += 1
        
    def run_auto_collection(self):
        """运行自动数据收集"""
        if not self.connect():
            return
            
        print("\n自动数据收集开始!")
        print("请在AirSim中手动控制无人机飞行")
        print("程序会自动检测飞行并记录数据")
        print("按Ctrl+C停止收集\n")
        
        dt = 1.0 / self.collection_frequency
        no_movement_count = 0
        movement_threshold = 30  # 3秒无移动就结束当前轨迹
        
        try:
            while True:
                loop_start = time.time()
                
                # 获取传感器数据
                sensor_data = self.get_sensor_data()
                if sensor_data is None:
                    time.sleep(dt)
                    continue
                    
                # 检测是否在移动
                is_moving = self.detect_movement(sensor_data['velocity'])
                
                if is_moving:
                    # 有移动，开始或继续收集
                    if not self.is_collecting:
                        self.start_new_trajectory()
                        
                    self.collect_sample(sensor_data)
                    no_movement_count = 0
                    
                else:
                    # 无移动
                    no_movement_count += 1
                    
                    # 如果长时间无移动且正在收集，则结束当前轨迹
                    if self.is_collecting and no_movement_count > movement_threshold:
                        print("检测到停止移动，结束当前轨迹")
                        self.save_trajectory()
                        self.is_collecting = False
                        no_movement_count = 0
                        
                # 控制频率
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n收到停止信号")
        finally:
            if self.is_collecting:
                self.save_trajectory()
            print(f"数据收集结束，共收集 {self.trajectory_count} 条轨迹")


def main():
    """主函数"""
    collector = SimpleDataCollector()
    collector.run_auto_collection()


if __name__ == "__main__":
    main()