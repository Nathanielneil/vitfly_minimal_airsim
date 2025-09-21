"""
AirSim接口模块 - 适配Windows AirSim 1.8.1 + UE4.7.2
提供无人机控制和传感器数据获取功能
"""

import airsim
import numpy as np
import cv2
import time
import logging
from typing import Tuple, Optional, Dict


class AirSimDroneInterface:
    """AirSim无人机控制接口"""
    
    def __init__(self, vehicle_name: str = "Drone1", connection_timeout: float = 30.0):
        self.vehicle_name = vehicle_name
        self.client = None
        self.is_connected = False
        self.connection_timeout = connection_timeout
        
        # 飞行状态
        self.is_flying = False
        self.start_position = None
        self.current_position = None
        
        # 安全参数
        self.max_velocity = 10.0  # m/s
        self.min_height = 0.5     # m
        self.max_height = 50.0    # m
        self.emergency_stop = False
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def connect(self) -> bool:
        """连接到AirSim"""
        try:
            self.logger.info(f"正在连接AirSim (超时: {self.connection_timeout}s)...")
            
            # 创建客户端连接
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            
            # 启用API控制
            self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            
            # 解锁无人机
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)
            
            # 获取初始位置
            pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            self.start_position = np.array([
                pose.position.x_val,
                pose.position.y_val,
                pose.position.z_val
            ])
            self.current_position = self.start_position.copy()
            
            self.is_connected = True
            self.logger.info(f"AirSim连接成功! 初始位置: {self.start_position}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"AirSim连接失败: {e}")
            self.is_connected = False
            return False
            
    def disconnect(self):
        """断开AirSim连接"""
        if self.client and self.is_connected:
            try:
                # 紧急停止
                self.emergency_land()
                
                # 禁用API控制
                self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
                
                self.is_connected = False
                self.logger.info("AirSim连接已断开")
                
            except Exception as e:
                self.logger.error(f"断开连接时出错: {e}")
                
    def takeoff(self, altitude: float = 2.0, timeout: float = 10.0) -> bool:
        """起飞到指定高度"""
        if not self.is_connected:
            self.logger.error("未连接到AirSim")
            return False
            
        try:
            self.logger.info(f"起飞到 {altitude}m 高度...")
            
            # 起飞
            future = self.client.takeoffAsync(timeout_sec=timeout, vehicle_name=self.vehicle_name)
            future.join()
            
            # 飞行到指定高度
            future = self.client.moveToZAsync(
                -altitude,  # AirSim使用NED坐标系，Z轴向下为正
                velocity=2.0,
                timeout_sec=timeout,
                vehicle_name=self.vehicle_name
            )
            future.join()
            
            self.is_flying = True
            self.logger.info("起飞完成")
            
            return True
            
        except Exception as e:
            self.logger.error(f"起飞失败: {e}")
            return False
            
    def land(self, timeout: float = 10.0) -> bool:
        """安全降落"""
        if not self.is_connected:
            return False
            
        try:
            self.logger.info("开始降落...")
            
            future = self.client.landAsync(timeout_sec=timeout, vehicle_name=self.vehicle_name)
            future.join()
            
            self.is_flying = False
            self.logger.info("降落完成")
            
            return True
            
        except Exception as e:
            self.logger.error(f"降落失败: {e}")
            return False
            
    def emergency_land(self):
        """紧急降落"""
        if self.client and self.is_connected:
            try:
                self.emergency_stop = True
                self.client.moveByVelocityAsync(0, 0, 0, 0.1, vehicle_name=self.vehicle_name)
                self.land(timeout=5.0)
                self.logger.warning("紧急降落执行")
            except Exception as e:
                self.logger.error(f"紧急降落失败: {e}")
                
    def move_by_velocity(self, vx: float, vy: float, vz: float, duration: float = 0.1) -> bool:
        """按速度移动无人机"""
        if not self.is_connected or self.emergency_stop:
            return False
            
        try:
            # 速度限制
            velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
            if velocity_magnitude > self.max_velocity:
                scale = self.max_velocity / velocity_magnitude
                vx *= scale
                vy *= scale
                vz *= scale
                
            # 高度安全检查
            current_pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            current_height = -current_pose.position.z_val  # 转换为正值高度
            
            # 防止过低或过高
            if current_height < self.min_height and vz > 0:
                vz = max(vz, -1.0)  # 限制下降速度
            elif current_height > self.max_height and vz < 0:
                vz = min(vz, 1.0)   # 限制上升速度
                
            # 执行速度控制
            future = self.client.moveByVelocityAsync(
                vx, vy, vz, duration, 
                vehicle_name=self.vehicle_name
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"速度控制失败: {e}")
            return False
            
    def get_depth_image(self, camera_name: str = "front_center") -> Optional[np.ndarray]:
        """获取深度图像"""
        if not self.is_connected:
            return None
            
        try:
            # 获取深度图像
            responses = self.client.simGetImages([
                airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, False, False)
            ], vehicle_name=self.vehicle_name)
            
            if len(responses) == 0:
                return None
                
            response = responses[0]
            
            # 转换为numpy数组
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            
            # 转换为灰度深度图
            depth_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
            # 归一化到[0, 1]
            depth_image = depth_image.astype(np.float32) / 255.0
            
            return depth_image
            
        except Exception as e:
            self.logger.error(f"获取深度图像失败: {e}")
            return None
            
    def get_rgb_image(self, camera_name: str = "front_center") -> Optional[np.ndarray]:
        """获取RGB图像"""
        if not self.is_connected:
            return None
            
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
            ], vehicle_name=self.vehicle_name)
            
            if len(responses) == 0:
                return None
                
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            
            return img_rgb
            
        except Exception as e:
            self.logger.error(f"获取RGB图像失败: {e}")
            return None
            
    def get_state(self) -> Dict:
        """获取无人机状态信息"""
        if not self.is_connected:
            return {}
            
        try:
            # 位置和姿态
            pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            position = np.array([
                pose.position.x_val,
                pose.position.y_val,
                pose.position.z_val
            ])
            
            # 四元数姿态
            orientation = np.array([
                pose.orientation.w_val,
                pose.orientation.x_val,
                pose.orientation.y_val,
                pose.orientation.z_val
            ])
            
            # 速度信息
            velocity = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.linear_velocity
            velocity_vec = np.array([
                velocity.x_val,
                velocity.y_val,
                velocity.z_val
            ])
            
            # IMU数据
            imu_data = self.client.getImuData(vehicle_name=self.vehicle_name)
            angular_velocity = np.array([
                imu_data.angular_velocity.x_val,
                imu_data.angular_velocity.y_val,
                imu_data.angular_velocity.z_val
            ])
            
            self.current_position = position
            
            return {
                'position': position,
                'orientation_quaternion': orientation,
                'velocity': velocity_vec,
                'angular_velocity': angular_velocity,
                'height': -position[2],  # 转换为正值高度
                'is_flying': self.is_flying,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"获取状态失败: {e}")
            return {}
            
    def reset_to_start(self, timeout: float = 10.0) -> bool:
        """重置到起始位置"""
        if not self.is_connected or self.start_position is None:
            return False
            
        try:
            self.logger.info("重置到起始位置...")
            
            # 移动到起始位置
            future = self.client.moveToPositionAsync(
                self.start_position[0],
                self.start_position[1], 
                self.start_position[2],
                velocity=3.0,
                timeout_sec=timeout,
                vehicle_name=self.vehicle_name
            )
            future.join()
            
            # 重置姿态
            self.client.rotateToYawAsync(0, timeout_sec=5.0, vehicle_name=self.vehicle_name).join()
            
            self.logger.info("重置完成")
            return True
            
        except Exception as e:
            self.logger.error(f"重置失败: {e}")
            return False
            
    def check_collision(self) -> bool:
        """检查碰撞"""
        if not self.is_connected:
            return False
            
        try:
            collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
            return collision_info.has_collided
            
        except Exception as e:
            self.logger.error(f"碰撞检查失败: {e}")
            return False


class SafetyController:
    """安全控制器 - 提供额外的安全保障"""
    
    def __init__(self, interface: AirSimDroneInterface):
        self.interface = interface
        self.max_acceleration = 5.0  # m/s²
        self.previous_velocity = np.zeros(3)
        self.previous_time = time.time()
        
    def safe_velocity_command(self, vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
        """安全速度指令处理"""
        current_time = time.time()
        dt = current_time - self.previous_time
        
        if dt > 0:
            # 当前速度指令
            current_velocity = np.array([vx, vy, vz])
            
            # 计算加速度
            acceleration = (current_velocity - self.previous_velocity) / dt
            acceleration_magnitude = np.linalg.norm(acceleration)
            
            # 限制加速度
            if acceleration_magnitude > self.max_acceleration:
                scale = self.max_acceleration / acceleration_magnitude
                current_velocity = self.previous_velocity + acceleration * scale * dt
                
            self.previous_velocity = current_velocity
            self.previous_time = current_time
            
            return tuple(current_velocity)
        
        return vx, vy, vz
        
    def emergency_check(self) -> bool:
        """紧急情况检查"""
        # 碰撞检查
        if self.interface.check_collision():
            self.interface.emergency_land()
            return True
            
        # 高度检查
        state = self.interface.get_state()
        if state and 'height' in state:
            if state['height'] < 0.3:  # 过低
                self.interface.emergency_land()
                return True
                
        return False


if __name__ == "__main__":
    # 测试AirSim接口
    interface = AirSimDroneInterface()
    
    try:
        # 连接
        if interface.connect():
            print("AirSim连接成功!")
            
            # 起飞
            if interface.takeoff(altitude=3.0):
                print("起飞成功!")
                
                # 获取状态
                state = interface.get_state()
                print(f"当前状态: {state}")
                
                # 获取图像
                depth_img = interface.get_depth_image()
                if depth_img is not None:
                    print(f"深度图像尺寸: {depth_img.shape}")
                    
                # 测试移动
                print("测试移动...")
                for i in range(10):
                    interface.move_by_velocity(1.0, 0.0, 0.0, 0.5)
                    time.sleep(0.5)
                    
                # 降落
                interface.land()
                
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        interface.disconnect()
        print("测试完成")