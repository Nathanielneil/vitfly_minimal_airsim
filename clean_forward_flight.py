#!/usr/bin/env python3
"""
纯净版前进避障脚本
- 起飞到指定高度
- 使用ViT模型避障前进指定距离
- 无调试信息干扰，实时碰撞检测
"""

import torch
import numpy as np
import cv2
import time
import logging
import signal
import sys
import airsim
from typing import Optional, Dict

from vit_model import create_minimal_vit_model


class CleanForwardFlight:
    """纯净版前进避障飞行"""
    
    def __init__(self, model_path: str = "vitfly_simple_policy.pth"):
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = create_minimal_vit_model(model_path).to(self.device)
        self.model.eval()
        print(f"ViT模型已加载 (参数量: {sum(p.numel() for p in self.model.parameters()):,})")
        
        # AirSim客户端（直接使用，无额外封装）
        self.client = airsim.MultirotorClient()
        self.vehicle_name = ""
        self.is_connected = False
        
        # 飞行参数
        self.takeoff_height = 0.5
        self.forward_distance = 4.0
        self.base_velocity = 1.5
        self.control_frequency = 10
        self.image_resize = (60, 90)
        
        # 状态管理
        self.lstm_hidden_state = None
        self.is_running = False
        self.start_position = None
        self.target_reached = False
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """优雅退出"""
        self.logger.info("接收到退出信号，正在安全停止...")
        self.stop()
        sys.exit(0)
        
    def connect(self) -> bool:
        """连接AirSim"""
        try:
            self.client.confirmConnection()
            self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)
            self.is_connected = True
            
            pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            position = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
            self.logger.info(f"AirSim连接成功! 初始位置: {position}")
            return True
            
        except Exception as e:
            self.logger.error(f"AirSim连接失败: {e}")
            return False
            
    def takeoff(self, altitude: float) -> bool:
        """起飞"""
        try:
            self.logger.info(f"起飞到 {altitude}m 高度...")
            future = self.client.takeoffAsync(vehicle_name=self.vehicle_name)
            future.join()
            
            # 移动到目标高度
            future = self.client.moveToZAsync(-altitude, 3.0, vehicle_name=self.vehicle_name)
            future.join()
            
            self.logger.info("起飞完成")
            return True
            
        except Exception as e:
            self.logger.error(f"起飞失败: {e}")
            return False
            
    def land(self):
        """降落"""
        try:
            self.logger.info("正在降落...")
            future = self.client.landAsync(vehicle_name=self.vehicle_name)
            future.join()
            self.logger.info("降落完成")
            
        except Exception as e:
            self.logger.error(f"降落失败: {e}")
            
    def disconnect(self):
        """断开连接"""
        try:
            if self.is_connected:
                self.client.armDisarm(False, vehicle_name=self.vehicle_name)
                self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
                self.is_connected = False
                self.logger.info("AirSim连接已断开")
                
        except Exception as e:
            self.logger.error(f"断开连接失败: {e}")
            
    def get_depth_image(self) -> Optional[np.ndarray]:
        """获取深度图像"""
        try:
            request = airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)
            response = self.client.simGetImages([request], vehicle_name=self.vehicle_name)[0]
            
            if response.image_data_float:
                depth_array = np.array(response.image_data_float, dtype=np.float32)
                depth_array = depth_array.reshape(response.height, response.width)
                return depth_array
            return None
            
        except Exception as e:
            self.logger.error(f"深度图像获取失败: {e}")
            return None
            
    def get_state(self) -> Dict:
        """获取无人机状态"""
        try:
            pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            velocity = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.linear_velocity
            
            position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
            orientation_quaternion = np.array([
                pose.orientation.w_val, pose.orientation.x_val,
                pose.orientation.y_val, pose.orientation.z_val
            ])
            velocity_vec = np.array([velocity.x_val, velocity.y_val, velocity.z_val])
            
            return {
                'position': position,
                'orientation_quaternion': orientation_quaternion,
                'velocity': velocity_vec,
                'height': abs(pose.position.z_val)  # 使用Z值的绝对值作为高度
            }
            
        except Exception as e:
            self.logger.error(f"状态获取失败: {e}")
            return {}
            
    def check_collision(self) -> bool:
        """检查真实碰撞（过滤误报）"""
        try:
            collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
            
            # 检查是否真的有碰撞且碰撞强度足够
            if collision_info.has_collided:
                # 检查碰撞强度和持续时间
                impact_point = collision_info.impact_point
                penetration_depth = collision_info.penetration_depth
                
                # 只有当有明显的碰撞深度时才认为是真碰撞
                if penetration_depth > 0.01:  # 1cm以上的深度
                    self.logger.warning(f"检测到真实碰撞: 深度 {penetration_depth:.3f}m")
                    return True
                    
            return False
            
        except Exception as e:
            # API调用失败不认为是碰撞
            return False
            
    def check_front_obstacle(self, depth_image: np.ndarray) -> bool:
        """检查前方障碍物"""
        try:
            height, width = depth_image.shape
            
            # 检查多个区域
            center_region = depth_image[height//3:2*height//3, width//3:2*width//3]
            left_region = depth_image[height//3:2*height//3, 0:width//3]
            right_region = depth_image[height//3:2*height//3, 2*width//3:width]
            
            center_depth = np.mean(center_region)
            left_depth = np.mean(left_region)
            right_depth = np.mean(right_region)
            
            # 更早的警告距离
            if center_depth < 1.0:  # 1m以内有障碍物
                self.logger.warning(f"⚠️ 前方障碍物: 中心{center_depth:.2f}m, 左{left_depth:.2f}m, 右{right_depth:.2f}m")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"障碍物检测失败: {e}")
            return False
            
    def move_by_velocity(self, vx: float, vy: float, vz: float, duration: float) -> bool:
        """按速度移动"""
        try:
            future = self.client.moveByVelocityAsync(
                float(vx), float(vy), float(vz), float(duration),
                vehicle_name=self.vehicle_name
            )
            return True
        except Exception as e:
            self.logger.error(f"速度控制失败: {e}")
            return False
            
    def preprocess_depth_image(self, depth_image: np.ndarray) -> torch.Tensor:
        """预处理深度图像"""
        try:
            if depth_image.shape != self.image_resize:
                depth_image = cv2.resize(depth_image, 
                                       (self.image_resize[1], self.image_resize[0]))
            
            depth_image = depth_image.astype(np.float32)
            if depth_image.max() > 1.0:
                depth_image = depth_image / 255.0
                
            depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).to(self.device)
            return depth_tensor
            
        except Exception as e:
            self.logger.error(f"深度图像预处理失败: {e}")
            return None
            
    def compute_velocity_command(self, depth_image: np.ndarray, state: dict) -> Optional[np.ndarray]:
        """计算避障速度指令"""
        try:
            depth_tensor = self.preprocess_depth_image(depth_image)
            if depth_tensor is None:
                return None
                
            desired_vel = torch.tensor([[self.base_velocity]], device=self.device, dtype=torch.float32)
            quaternion = torch.tensor([state['orientation_quaternion']], device=self.device, dtype=torch.float32)
            
            with torch.no_grad():
                velocity_cmd, self.lstm_hidden_state = self.model(
                    depth_tensor, desired_vel, quaternion, self.lstm_hidden_state
                )
                
            velocity_np = velocity_cmd.squeeze().cpu().numpy()
            
            # 打印原始ViT输出用于调试
            if hasattr(self, '_debug_frame_count'):
                self._debug_frame_count += 1
            else:
                self._debug_frame_count = 1
                
            if self._debug_frame_count % 10 == 0:  # 每10帧打印一次
                self.logger.info(f"🤖 ViT原始输出: {velocity_np}")
            
            if np.linalg.norm(velocity_np) > 0:
                # 保持ViT输出的方向，但调整幅度
                velocity_direction = velocity_np / np.linalg.norm(velocity_np)
                
                # 根据ViT输出调整速度
                final_velocity = velocity_np.copy()
                
                # 限制速度范围，但保持避障行为
                final_velocity[0] = np.clip(final_velocity[0], 0.2, 2.0)  # 前进速度
                final_velocity[1] = np.clip(final_velocity[1], -1.5, 1.5)  # 左右避障
                final_velocity[2] = np.clip(final_velocity[2], -1.0, 1.0)  # 上下避障
                
                # 轻微的高度稳定（不覆盖ViT的Z轴避障）
                current_height = state['height']
                target_height = self.takeoff_height
                height_error = current_height - target_height
                
                # 只在高度偏差很大时才干预
                if height_error > 1.5:  # 高于目标1.5m以上
                    final_velocity[2] = 0.5  # 强制下降
                elif height_error < -0.8:  # 低于目标0.8m以上
                    final_velocity[2] = -0.3  # 强制上升
                    
            else:
                # ViT没有输出时的保守策略
                final_velocity = np.array([0.3, 0.0, 0.0])
                
            return final_velocity
            
        except Exception as e:
            self.logger.error(f"速度指令计算失败: {e}")
            return None
            
    def check_distance_reached(self, current_position: np.ndarray) -> bool:
        """检查是否到达目标距离"""
        if self.start_position is None:
            return False
            
        forward_distance = current_position[0] - self.start_position[0]
        
        if forward_distance >= self.forward_distance:
            self.logger.info(f"✅ 目标距离已达成: {forward_distance:.2f}m >= {self.forward_distance}m")
            return True
            
        if hasattr(self, '_last_progress_time'):
            if time.time() - self._last_progress_time > 2.0:
                self.logger.info(f"📍 前进进度: {forward_distance:.2f}m / {self.forward_distance}m")
                self._last_progress_time = time.time()
        else:
            self._last_progress_time = time.time()
            
        return False
        
    def control_loop(self):
        """主控制循环"""
        self.logger.info("🚁 开始避障前进控制循环")
        
        control_dt = 1.0 / self.control_frequency
        frame_count = 0
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # 获取传感器数据
                depth_image = self.get_depth_image()
                state = self.get_state()
                
                if depth_image is None or not state:
                    time.sleep(control_dt)
                    continue
                    
                # 检查距离
                if self.check_distance_reached(state['position']):
                    self.target_reached = True
                    break
                    
                # 碰撞检测（仅检测严重碰撞）
                if self.check_collision():
                    self.logger.error("💥 检测到严重碰撞！")
                    break
                    
                # 简单障碍物检测（作为备选）
                if self.check_front_obstacle(depth_image):
                    # 不停止，只是警告，让ViT处理
                    pass
                    
                # 高度安全检查（检查是否过低）
                current_height = state['height']
                if current_height > 5.0:  # 高度超过5m认为异常
                    self.logger.error(f"⚠️ 高度异常: {current_height:.2f}m")
                    break
                elif current_height < 0.1:  # 高度低于10cm认为着陆
                    self.logger.error(f"⚠️ 高度过低: {current_height:.2f}m")
                    break
                    
                # 计算避障指令
                velocity_cmd = self.compute_velocity_command(depth_image, state)
                
                if velocity_cmd is not None:
                    # 速度限制
                    velocity_magnitude = np.linalg.norm(velocity_cmd)
                    if velocity_magnitude > 3.0:
                        velocity_cmd = velocity_cmd / velocity_magnitude * 3.0
                    
                    # 执行速度指令
                    success = self.move_by_velocity(*velocity_cmd, control_dt)
                    
                    if success:
                        frame_count += 1
                        
                    # 每20帧报告一次
                    if frame_count % 20 == 0:
                        current_distance = state['position'][0] - self.start_position[0] if self.start_position is not None else 0
                        current_height = state['height']
                        self.logger.info(f"🎯 帧 {frame_count}: 速度 {velocity_cmd}, 已前进 {current_distance:.2f}m, 高度 {current_height:.2f}m")
                        
                # 控制频率
                loop_time = time.time() - loop_start
                if loop_time < control_dt:
                    time.sleep(control_dt - loop_time)
                    
        except Exception as e:
            self.logger.error(f"控制循环异常: {e}")
        finally:
            self.logger.info("🏁 控制循环结束")
            
    def start_forward_flight(self):
        """开始前进避障飞行"""
        try:
            # 连接AirSim
            if not self.connect():
                return False
                
            # 起飞
            if not self.takeoff(self.takeoff_height):
                return False
                
            # 记录起始位置
            initial_state = self.get_state()
            if initial_state:
                self.start_position = initial_state['position'].copy()
                self.logger.info(f"🏠 起始位置: {self.start_position}")
                
            # 重置LSTM状态
            self.lstm_hidden_state = self.model.reset_lstm_state(1, self.device)
            
            # 开始避障前进
            self.is_running = True
            self.logger.info(f"🎯 目标: 前进{self.forward_distance}m (高度: {self.takeoff_height}m)")
            
            # 主控制循环
            self.control_loop()
            
            # 检查结果
            if self.target_reached:
                final_state = self.get_state()
                if final_state:
                    actual_distance = final_state['position'][0] - self.start_position[0]
                    self.logger.info(f"🎉 任务完成！实际前进: {actual_distance:.2f}m")
                return True
            else:
                self.logger.warning("⚠️ 任务未完成")
                return False
                
        except Exception as e:
            self.logger.error(f"前进避障飞行失败: {e}")
            return False
        finally:
            self.stop()
            
    def stop(self):
        """停止系统"""
        self.is_running = False
        if self.is_connected:
            self.land()
            self.disconnect()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="纯净版前进避障飞行")
    parser.add_argument('--model', type=str, default='vitfly_simple_policy.pth')
    parser.add_argument('--height', type=float, default=0.5)
    parser.add_argument('--distance', type=float, default=4.0)
    parser.add_argument('--speed', type=float, default=1.5)
    
    args = parser.parse_args()
    
    try:
        flight_system = CleanForwardFlight(model_path=args.model)
        flight_system.takeoff_height = args.height
        flight_system.forward_distance = args.distance
        flight_system.base_velocity = args.speed
        
        print("=" * 60)
        print("🚁 ViTfly 纯净版前进避障飞行")
        print("=" * 60)
        print(f"📋 任务参数:")
        print(f"   起飞高度: {args.height}m")
        print(f"   前进距离: {args.distance}m") 
        print(f"   飞行速度: {args.speed}m/s")
        print(f"   模型文件: {args.model}")
        print("=" * 60)
        
        success = flight_system.start_forward_flight()
        
        print("=" * 60)
        if success:
            print("🎉 飞行任务完成！")
        else:
            print("❌ 飞行任务失败")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断，正在安全退出...")
    except Exception as e:
        print(f"💥 系统异常: {e}")


if __name__ == "__main__":
    main()