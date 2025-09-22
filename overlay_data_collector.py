#!/usr/bin/env python3
"""
覆盖显示数据收集器
在AirSim窗口内显示控制信息，无需切换窗口
"""

import airsim
import numpy as np
import cv2
import time
import os
import pandas as pd
import threading
from pathlib import Path
import queue


class OverlayDataCollector:
    """覆盖显示数据收集器"""
    
    def __init__(self, data_dir="./training_data"):
        self.client = airsim.MultirotorClient()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 收集参数
        self.collection_frequency = 10
        self.image_size = (60, 90)
        self.display_size = (400, 300)
        
        # 状态
        self.is_collecting = False
        self.current_trajectory = []
        self.trajectory_count = 0
        self.is_connected = False
        self.running = True
        
        # 显示窗口
        self.window_name = "ViT数据收集器"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 500, 400)
        
        # 控制队列
        self.command_queue = queue.Queue()
        
        print("覆盖显示数据收集器初始化完成")
        print("特点: 无需切换窗口，所有控制在一个界面完成")
        
    def connect(self):
        """连接AirSim"""
        try:
            print("正在连接AirSim...")
            self.client.confirmConnection()
            print("AirSim连接确认成功")
            
            print("启用API控制...")
            self.client.enableApiControl(True, vehicle_name="Drone1")
            print("API控制启用成功")
            
            print("解锁无人机...")
            self.client.armDisarm(True, vehicle_name="Drone1")
            print("无人机解锁成功")
            
            # 测试图像获取
            print("测试图像获取...")
            test_request = airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
            test_response = self.client.simGetImages([test_request], vehicle_name="Drone1")[0]
            if test_response.image_data_uint8:
                print(f"图像获取测试成功: {test_response.width}x{test_response.height}")
            else:
                print("警告: 图像获取测试失败")
            
            self.is_connected = True
            print("✅ AirSim连接完全成功")
            return True
        except Exception as e:
            print(f"❌ AirSim连接失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def get_sensor_data(self):
        """获取传感器数据"""
        if not self.is_connected:
            return None
            
        try:
            # 获取RGB图像用于显示
            rgb_request = airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
            rgb_response = self.client.simGetImages([rgb_request], vehicle_name="Drone1")[0]
            
            # 获取深度图像用于训练
            depth_request = airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)
            depth_response = self.client.simGetImages([depth_request], vehicle_name="Drone1")[0]
            
            # 处理RGB图像
            rgb_array = None
            if rgb_response.image_data_uint8 and len(rgb_response.image_data_uint8) > 0:
                rgb_1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
                if len(rgb_1d) >= rgb_response.height * rgb_response.width * 3:
                    rgb_array = rgb_1d.reshape(rgb_response.height, rgb_response.width, 3)
                    rgb_array = cv2.resize(rgb_array, self.display_size)
                    # BGR转RGB (OpenCV默认BGR)
                    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
            
            # 如果RGB失败，创建默认图像
            if rgb_array is None:
                print("警告: RGB图像获取失败，使用默认图像")
                rgb_array = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
                cv2.putText(rgb_array, "No Camera Feed", (50, self.display_size[1]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 处理深度图像
            depth_array = None
            if depth_response.image_data_float and len(depth_response.image_data_float) > 0:
                depth_1d = np.array(depth_response.image_data_float, dtype=np.float32)
                if len(depth_1d) >= depth_response.height * depth_response.width:
                    depth_array = depth_1d.reshape(depth_response.height, depth_response.width)
                    depth_array = cv2.resize(depth_array, (self.image_size[1], self.image_size[0]))
                    depth_array = np.clip(depth_array, 0, 100) / 100.0
            
            # 如果深度失败，创建默认深度
            if depth_array is None:
                print("警告: 深度图像获取失败，使用默认深度")
                depth_array = np.ones(self.image_size, dtype=np.float32) * 0.5
                
            # 获取无人机状态
            state = self.client.getMultirotorState(vehicle_name="Drone1")
            pose = state.kinematics_estimated.pose
            velocity = state.kinematics_estimated.linear_velocity
            
            return {
                'rgb_image': rgb_array,
                'depth_image': depth_array,
                'position': np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val]),
                'quaternion': np.array([pose.orientation.w_val, pose.orientation.x_val, 
                                      pose.orientation.y_val, pose.orientation.z_val]),
                'velocity': np.array([velocity.x_val, velocity.y_val, velocity.z_val]),
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"传感器数据获取失败: {e}")
            return None
            
    def create_control_overlay(self, image, sensor_data):
        """创建控制界面覆盖"""
        overlay = image.copy()
        h, w = overlay.shape[:2]
        
        # 半透明背景
        cv2.rectangle(overlay, (10, 10), (w-10, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, overlay)
        
        # 标题
        cv2.putText(overlay, "ViT Data Collector", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 连接状态
        status_color = (0, 255, 0) if self.is_connected else (0, 0, 255)
        status_text = "Connected" if self.is_connected else "Disconnected"
        cv2.putText(overlay, f"Status: {status_text}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # 收集状态
        collect_color = (0, 255, 0) if self.is_collecting else (255, 255, 255)
        collect_text = "COLLECTING" if self.is_collecting else "Stopped"
        cv2.putText(overlay, f"Recording: {collect_text}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, collect_color, 1)
        
        # 统计信息
        if sensor_data:
            vel = sensor_data['velocity']
            speed = np.linalg.norm(vel)
            cv2.putText(overlay, f"Speed: {speed:.2f} m/s", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        cv2.putText(overlay, f"Trajectories: {self.trajectory_count}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(overlay, f"Samples: {len(self.current_trajectory)}", (20, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 控制说明
        controls = [
            "Controls (Press keys in this window):",
            "W/S: Forward/Backward  A/D: Left/Right",
            "Q/E: Up/Down  X: Stop  SPACE: Record",
            "ESC: Exit"
        ]
        
        for i, text in enumerate(controls):
            y_pos = h - 80 + (i * 20)
            cv2.putText(overlay, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return overlay
        
    def handle_key_input(self, key):
        """处理按键输入"""
        base_speed = 2.0
        
        if key == ord('w') or key == ord('W'):
            self.command_queue.put(('move', [base_speed, 0, 0]))
            print("前进")
        elif key == ord('s') or key == ord('S'):
            self.command_queue.put(('move', [-base_speed, 0, 0]))
            print("后退")
        elif key == ord('a') or key == ord('A'):
            self.command_queue.put(('move', [0, -base_speed, 0]))
            print("左移")
        elif key == ord('d') or key == ord('D'):
            self.command_queue.put(('move', [0, base_speed, 0]))
            print("右移")
        elif key == ord('q') or key == ord('Q'):
            self.command_queue.put(('move', [0, 0, -base_speed]))
            print("上升")
        elif key == ord('e') or key == ord('E'):
            self.command_queue.put(('move', [0, 0, base_speed]))
            print("下降")
        elif key == ord('x') or key == ord('X'):
            self.command_queue.put(('move', [0, 0, 0]))
            print("停止")
        elif key == ord(' '):  # 空格键
            self.toggle_collection()
        elif key == 27:  # ESC键
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
        print(f"🟢 开始收集轨迹 {self.trajectory_count}")
        
    def stop_collection(self):
        """停止数据收集"""
        if self.is_collecting and len(self.current_trajectory) > 0:
            self.save_trajectory()
            self.trajectory_count += 1
            
        self.is_collecting = False
        print("🔴 数据收集已停止")
        
    def execute_commands(self):
        """执行运动指令"""
        try:
            while not self.command_queue.empty():
                command, params = self.command_queue.get_nowait()
                if command == 'move' and self.is_connected:
                    vx, vy, vz = params
                    self.client.moveByVelocityAsync(
                        float(vx), float(vy), float(vz), 0.1,
                        vehicle_name="Drone1"
                    )
        except queue.Empty:
            pass
            
    def collect_sample(self, sensor_data):
        """收集数据样本"""
        if not self.is_collecting or not sensor_data:
            return
            
        try:
            collision_info = self.client.simGetCollisionInfo(vehicle_name="Drone1")
            
            sample = {
                'depth_image': sensor_data['depth_image'],
                'position': sensor_data['position'],
                'quaternion': sensor_data['quaternion'],
                'current_velocity': sensor_data['velocity'],
                'desired_velocity': 2.0,
                'velocity_command': sensor_data['velocity'].copy(),
                'collision': collision_info.has_collided,
                'timestamp': sensor_data['timestamp']
            }
            
            self.current_trajectory.append(sample)
            
        except Exception as e:
            pass
            
    def save_trajectory(self):
        """保存轨迹数据"""
        if len(self.current_trajectory) < 10:
            print("轨迹太短，跳过保存")
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
        print(f"✅ 轨迹 {self.trajectory_count} 已保存: {len(self.current_trajectory)} 样本")
        
    def run_collection(self):
        """运行数据收集主循环"""
        if not self.connect():
            return
            
        print("\n🚁 覆盖显示数据收集器启动!")
        print("所有控制在弹出的图像窗口中完成，无需切换")
        print("确保图像窗口获得焦点后按键控制")
        
        dt = 1.0 / self.collection_frequency
        
        try:
            while self.running:
                loop_start = time.time()
                
                # 获取传感器数据
                sensor_data = self.get_sensor_data()
                
                if sensor_data:
                    # 创建控制界面
                    display_image = self.create_control_overlay(
                        sensor_data['rgb_image'], sensor_data
                    )
                    
                    # 显示图像
                    cv2.imshow(self.window_name, display_image)
                    
                    # 处理按键（等待1ms）
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:  # 有按键
                        if not self.handle_key_input(key):
                            break
                    
                    # 执行运动指令
                    self.execute_commands()
                    
                    # 收集数据
                    self.collect_sample(sensor_data)
                    
                else:
                    # 无传感器数据时显示连接界面
                    blank = np.zeros((300, 400, 3), dtype=np.uint8)
                    cv2.putText(blank, "Connecting to AirSim...", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow(self.window_name, blank)
                    cv2.waitKey(1)
                
                # 控制频率
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n收到中断信号")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """清理资源"""
        self.running = False
        if self.is_collecting:
            self.stop_collection()
        if self.is_connected:
            try:
                self.client.armDisarm(False, vehicle_name="Drone1")
                self.client.enableApiControl(False, vehicle_name="Drone1")
            except:
                pass
        cv2.destroyAllWindows()
        print(f"数据收集结束，共收集 {self.trajectory_count} 条轨迹")


def main():
    """主函数"""
    collector = OverlayDataCollector()
    collector.run_collection()


if __name__ == "__main__":
    main()