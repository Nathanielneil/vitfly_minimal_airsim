#!/usr/bin/env python3
"""
手动API控制数据收集器
使用程序内置控制代替AirSim键盘控制
"""

import airsim
import numpy as np
import cv2
import time
import os
import pandas as pd
import msvcrt  # Windows键盘输入
from pathlib import Path


class ManualDataCollector:
    """手动API控制数据收集器"""
    
    def __init__(self, data_dir="./training_data"):
        self.client = airsim.MultirotorClient()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 收集参数
        self.collection_frequency = 10
        self.image_size = (60, 90)
        self.base_speed = 2.0
        
        # 状态
        self.is_collecting = False
        self.current_trajectory = []
        self.trajectory_count = 0
        self.current_velocity_cmd = np.array([0.0, 0.0, 0.0])
        
        print("手动API控制数据收集器初始化完成")
        print("\n控制说明:")
        print("  W/S: 前进/后退")
        print("  A/D: 左移/右移")
        print("  Q/E: 上升/下降")
        print("  SPACE: 开始/停止数据收集")
        print("  X: 停止移动")
        print("  ESC: 退出程序")
        print("\n注意: 请保持此终端窗口为活动窗口以接收按键输入")
        
    def connect(self):
        """连接AirSim"""
        try:
            self.client.confirmConnection()
            self.client.enableApiControl(True, vehicle_name="Drone1")
            self.client.armDisarm(True, vehicle_name="Drone1")
            print("AirSim连接成功，API控制已启用")
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
                depth_image = cv2.resize(depth_array, (self.image_size[1], self.image_size[0]))
                depth_image = np.clip(depth_image, 0, 100) / 100.0
            else:
                return None
                
            # 获取无人机状态
            state = self.client.getMultirotorState()
            pose = state.kinematics_estimated.pose
            velocity = state.kinematics_estimated.linear_velocity
            
            return {
                'depth_image': depth_image,
                'position': np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val]),
                'quaternion': np.array([pose.orientation.w_val, pose.orientation.x_val, 
                                      pose.orientation.y_val, pose.orientation.z_val]),
                'velocity': np.array([velocity.x_val, velocity.y_val, velocity.z_val]),
                'timestamp': time.time()
            }
            
        except Exception as e:
            return None
            
    def handle_keyboard_input(self):
        """处理键盘输入（Windows版本）"""
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8').lower()
            
            if key == 'w':
                self.current_velocity_cmd = np.array([self.base_speed, 0, 0])
                print("前进")
            elif key == 's':
                self.current_velocity_cmd = np.array([-self.base_speed, 0, 0])
                print("后退")
            elif key == 'a':
                self.current_velocity_cmd = np.array([0, -self.base_speed, 0])
                print("左移")
            elif key == 'd':
                self.current_velocity_cmd = np.array([0, self.base_speed, 0])
                print("右移")
            elif key == 'q':
                self.current_velocity_cmd = np.array([0, 0, -self.base_speed])
                print("上升")
            elif key == 'e':
                self.current_velocity_cmd = np.array([0, 0, self.base_speed])
                print("下降")
            elif key == 'x':
                self.current_velocity_cmd = np.array([0, 0, 0])
                print("停止")
            elif key == ' ':  # 空格键
                self.toggle_collection()
            elif key == '\x1b':  # ESC键
                return False
                
        return True
        
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
        print(f"\n🟢 开始收集轨迹 {self.trajectory_count}")
        
    def stop_collection(self):
        """停止数据收集"""
        if self.is_collecting and len(self.current_trajectory) > 0:
            self.save_trajectory()
            self.trajectory_count += 1
            
        self.is_collecting = False
        print("🔴 数据收集已停止")
        
    def collect_sample(self):
        """收集数据样本"""
        sensor_data = self.get_sensor_data()
        if sensor_data is None:
            return
            
        # 执行速度指令
        if np.any(self.current_velocity_cmd != 0):
            self.client.moveByVelocityAsync(
                float(self.current_velocity_cmd[0]),
                float(self.current_velocity_cmd[1]),
                float(self.current_velocity_cmd[2]),
                0.1
            )
        
        # 收集数据
        if self.is_collecting:
            collision_info = self.client.simGetCollisionInfo()
            
            sample = {
                'depth_image': sensor_data['depth_image'],
                'position': sensor_data['position'],
                'quaternion': sensor_data['quaternion'],
                'current_velocity': sensor_data['velocity'],
                'desired_velocity': 2.0,
                'velocity_command': self.current_velocity_cmd.copy(),
                'collision': collision_info.has_collided,
                'timestamp': sensor_data['timestamp']
            }
            
            self.current_trajectory.append(sample)
            
            if len(self.current_trajectory) % 20 == 0:
                print(f"已收集 {len(self.current_trajectory)} 个样本")
                
    def save_trajectory(self):
        """保存轨迹数据"""
        if len(self.current_trajectory) < 10:
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
        
        print(f"✅ 轨迹 {self.trajectory_count} 已保存: {len(self.current_trajectory)} 样本")
        
    def run_collection(self):
        """运行数据收集主循环"""
        if not self.connect():
            return
            
        print("\n🚁 手动控制数据收集开始!")
        print("请在此终端中按键控制无人机")
        
        dt = 1.0 / self.collection_frequency
        
        try:
            while True:
                loop_start = time.time()
                
                # 处理键盘输入
                if not self.handle_keyboard_input():
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
            print(f"数据收集结束，共收集 {self.trajectory_count} 条轨迹")


def main():
    """主函数"""
    collector = ManualDataCollector()
    collector.run_collection()


if __name__ == "__main__":
    main()