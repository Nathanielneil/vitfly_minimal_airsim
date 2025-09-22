#!/usr/bin/env python3
"""
GUI数据收集器 - 图形界面控制
"""

import tkinter as tk
from tkinter import ttk
import airsim
import numpy as np
import cv2
import time
import os
import pandas as pd
import threading
from pathlib import Path
from PIL import Image, ImageTk


class GUIDataCollector:
    """GUI数据收集器"""
    
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.data_dir = Path("./training_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # 收集参数
        self.collection_frequency = 10
        self.image_size = (60, 90)
        self.base_speed = 2.0
        
        # 状态
        self.is_collecting = False
        self.is_connected = False
        self.current_trajectory = []
        self.trajectory_count = 0
        self.current_velocity_cmd = np.array([0.0, 0.0, 0.0])
        
        # 创建GUI
        self.setup_gui()
        
        # 启动后台线程
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
        
    def setup_gui(self):
        """设置GUI界面"""
        self.root = tk.Tk()
        self.root.title("ViT避障数据收集器")
        self.root.geometry("800x600")
        
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 连接状态
        self.status_var = tk.StringVar(value="未连接")
        ttk.Label(main_frame, text="状态:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=0, column=1, sticky=tk.W)
        
        # 连接按钮
        ttk.Button(main_frame, text="连接AirSim", command=self.connect_airsim).grid(row=0, column=2)
        
        # 分隔线
        ttk.Separator(main_frame, orient='horizontal').grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="飞行控制", padding="10")
        control_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        
        # 控制按钮布局
        #     Q
        #   A W D  
        #     S
        #     E
        
        ttk.Button(control_frame, text="Q\n上升", command=lambda: self.set_velocity(0, 0, -1)).grid(row=0, column=1)
        ttk.Button(control_frame, text="A\n左移", command=lambda: self.set_velocity(0, -1, 0)).grid(row=1, column=0)
        ttk.Button(control_frame, text="W\n前进", command=lambda: self.set_velocity(1, 0, 0)).grid(row=1, column=1)
        ttk.Button(control_frame, text="D\n右移", command=lambda: self.set_velocity(0, 1, 0)).grid(row=1, column=2)
        ttk.Button(control_frame, text="S\n后退", command=lambda: self.set_velocity(-1, 0, 0)).grid(row=2, column=1)
        ttk.Button(control_frame, text="E\n下降", command=lambda: self.set_velocity(0, 0, 1)).grid(row=3, column=1)
        
        # 停止按钮
        ttk.Button(control_frame, text="停止", command=self.stop_movement).grid(row=1, column=3, padx=10)
        
        # 速度滑块
        ttk.Label(control_frame, text="速度:").grid(row=4, column=0, sticky=tk.W, pady=(10,0))
        self.speed_var = tk.DoubleVar(value=2.0)
        speed_scale = ttk.Scale(control_frame, from_=0.5, to=5.0, variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(10,0))
        
        # 数据收集面板
        data_frame = ttk.LabelFrame(main_frame, text="数据收集", padding="10")
        data_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        
        # 收集控制
        self.collect_button = ttk.Button(data_frame, text="开始收集", command=self.toggle_collection)
        self.collect_button.grid(row=0, column=0, padx=5)
        
        ttk.Button(data_frame, text="保存轨迹", command=self.save_current_trajectory).grid(row=0, column=1, padx=5)
        
        # 收集状态
        self.collect_status_var = tk.StringVar(value="未收集")
        ttk.Label(data_frame, textvariable=self.collect_status_var).grid(row=0, column=2, padx=10)
        
        # 统计信息
        self.stats_var = tk.StringVar(value="轨迹: 0, 样本: 0")
        ttk.Label(data_frame, textvariable=self.stats_var).grid(row=1, column=0, columnspan=3, pady=5)
        
        # 深度图像显示
        image_frame = ttk.LabelFrame(main_frame, text="深度图像", padding="5")
        image_frame.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        
        self.image_label = ttk.Label(image_frame, text="无图像")
        self.image_label.grid(row=0, column=0)
        
        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="日志", padding="5")
        log_frame.grid(row=5, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, width=80)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 配置权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def log(self, message):
        """添加日志"""
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        print(message)
        
    def connect_airsim(self):
        """连接AirSim"""
        try:
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.is_connected = True
            self.status_var.set("已连接")
            self.log("AirSim连接成功")
        except Exception as e:
            self.status_var.set("连接失败")
            self.log(f"AirSim连接失败: {e}")
            
    def set_velocity(self, vx, vy, vz):
        """设置速度指令"""
        speed = self.speed_var.get()
        self.current_velocity_cmd = np.array([vx * speed, vy * speed, vz * speed])
        direction = ["停止", "前进", "后退", "左移", "右移", "上升", "下降"]
        if vx > 0: self.log("前进")
        elif vx < 0: self.log("后退")
        elif vy > 0: self.log("右移")
        elif vy < 0: self.log("左移")
        elif vz > 0: self.log("下降")
        elif vz < 0: self.log("上升")
        
    def stop_movement(self):
        """停止移动"""
        self.current_velocity_cmd = np.array([0.0, 0.0, 0.0])
        self.log("停止移动")
        
    def toggle_collection(self):
        """切换数据收集状态"""
        if self.is_collecting:
            self.is_collecting = False
            self.collect_button.config(text="开始收集")
            self.collect_status_var.set("已停止")
            self.log("数据收集已停止")
        else:
            self.current_trajectory = []
            self.is_collecting = True
            self.collect_button.config(text="停止收集")
            self.collect_status_var.set("收集中...")
            self.log(f"开始收集轨迹 {self.trajectory_count}")
            
    def save_current_trajectory(self):
        """保存当前轨迹"""
        if len(self.current_trajectory) > 0:
            self.save_trajectory()
            self.log(f"轨迹 {self.trajectory_count-1} 已手动保存")
        else:
            self.log("无数据可保存")
            
    def get_sensor_data(self):
        """获取传感器数据"""
        if not self.is_connected:
            return None
            
        try:
            # 深度图像
            request = airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)
            response = self.client.simGetImages([request])[0]
            
            if response.image_data_float:
                depth_array = np.array(response.image_data_float, dtype=np.float32)
                depth_array = depth_array.reshape(response.height, response.width)
                depth_image = cv2.resize(depth_array, (self.image_size[1], self.image_size[0]))
                depth_image = np.clip(depth_image, 0, 100) / 100.0
            else:
                return None
                
            # 无人机状态
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
            
    def update_image_display(self, depth_image):
        """更新深度图像显示"""
        try:
            # 转换为显示格式
            display_img = (depth_image * 255).astype(np.uint8)
            display_img = cv2.resize(display_img, (180, 120))  # 放大显示
            
            # 转换为PIL图像
            pil_img = Image.fromarray(display_img)
            photo = ImageTk.PhotoImage(pil_img)
            
            # 更新显示
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 保持引用
            
        except Exception as e:
            pass
            
    def save_trajectory(self):
        """保存轨迹数据"""
        if len(self.current_trajectory) < 10:
            return
            
        traj_dir = self.data_dir / f"trajectory_{self.trajectory_count:06d}"
        traj_dir.mkdir(exist_ok=True)
        
        metadata = []
        for i, sample in enumerate(self.current_trajectory):
            # 保存深度图像
            depth_img = (sample['depth_image'] * 255).astype(np.uint8)
            img_path = traj_dir / f"depth_{i:06d}.png"
            cv2.imwrite(str(img_path), depth_img)
            
            # 元数据
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
            
        df = pd.DataFrame(metadata)
        df.to_csv(traj_dir / "data.csv", index=False)
        
        self.trajectory_count += 1
        
    def control_loop(self):
        """控制循环"""
        dt = 1.0 / self.collection_frequency
        
        while self.running:
            try:
                if self.is_connected:
                    # 执行速度指令
                    if np.any(self.current_velocity_cmd != 0):
                        self.client.moveByVelocityAsync(
                            float(self.current_velocity_cmd[0]),
                            float(self.current_velocity_cmd[1]),
                            float(self.current_velocity_cmd[2]),
                            dt
                        )
                    
                    # 获取传感器数据
                    sensor_data = self.get_sensor_data()
                    if sensor_data is not None:
                        # 更新图像显示
                        self.update_image_display(sensor_data['depth_image'])
                        
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
                            
                            # 更新统计
                            self.stats_var.set(f"轨迹: {self.trajectory_count}, 当前样本: {len(self.current_trajectory)}")
                            
                time.sleep(dt)
                
            except Exception as e:
                time.sleep(dt)
                
    def on_closing(self):
        """关闭程序"""
        self.running = False
        if self.is_collecting:
            self.save_trajectory()
        if self.is_connected:
            try:
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
            except:
                pass
        self.root.destroy()
        
    def run(self):
        """运行GUI"""
        self.log("GUI数据收集器启动")
        self.log("1. 点击'连接AirSim'")
        self.log("2. 使用按钮控制无人机")
        self.log("3. 点击'开始收集'记录数据")
        self.root.mainloop()


def main():
    """主函数"""
    collector = GUIDataCollector()
    collector.run()


if __name__ == "__main__":
    main()