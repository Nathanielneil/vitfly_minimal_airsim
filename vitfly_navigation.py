#!/usr/bin/env python3
"""
ViTfly导航系统 - 目标点导航 + 避障
"""

import torch
import numpy as np
import cv2
import time
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from vit_model import create_minimal_vit_model
from airsim_interface import AirSimDroneInterface, SafetyController
from navigation_controller import NavigationController


class ViTflyNavigationSystem:
    """ViTfly导航系统 - 结合避障和目标点导航"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        # 基础设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"使用设备: {self.device}")
        
        # 初始化组件
        self.model = create_minimal_vit_model(model_path).to(self.device)
        self.model.eval()
        self.airsim_interface = AirSimDroneInterface()
        self.safety_controller = SafetyController(self.airsim_interface)
        self.navigation_controller = NavigationController()
        
        # 控制参数
        self.base_velocity = 3.0
        self.control_frequency = 10
        self.image_resize = (60, 90)
        
        # 状态管理
        self.lstm_hidden_state = None
        self.is_running = False
        self.navigation_enabled = True
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        self.logger.info("接收到退出信号，正在安全停止...")
        self.stop()
        sys.exit(0)
        
    def load_mission(self, mission_file: str = None, mission_type: str = "square"):
        """加载任务"""
        if mission_file:
            self.navigation_controller.load_mission_from_file(mission_file)
        else:
            self.navigation_controller.create_simple_mission(mission_type)
            
    def detect_obstacles(self, depth_image: np.ndarray) -> bool:
        """检测前方是否有障碍物"""
        height, width = depth_image.shape
        front_region = depth_image[height//3:2*height//3, width//3:2*width//3]
        return np.mean(front_region) < 0.3
        
    def compute_avoidance_command(self, depth_image: np.ndarray, state: Dict) -> Optional[np.ndarray]:
        """计算避障指令"""
        try:
            # 预处理深度图像
            if depth_image.shape != self.image_resize:
                depth_image = cv2.resize(depth_image, 
                                       (self.image_resize[1], self.image_resize[0]))
            
            depth_image = depth_image.astype(np.float32)
            if depth_image.max() > 1.0:
                depth_image = depth_image / 255.0
                
            depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # 模型推理
            desired_vel = torch.tensor([[self.base_velocity]], device=self.device, dtype=torch.float32)
            quaternion = torch.tensor([state['orientation_quaternion']], device=self.device, dtype=torch.float32)
            
            with torch.no_grad():
                velocity_cmd, self.lstm_hidden_state = self.model(
                    depth_tensor, desired_vel, quaternion, self.lstm_hidden_state
                )
                
            velocity_np = velocity_cmd.squeeze().cpu().numpy()
            
            if np.linalg.norm(velocity_np) > 0:
                velocity_direction = velocity_np / np.linalg.norm(velocity_np)
                return velocity_direction * self.base_velocity
            else:
                return np.array([self.base_velocity * 0.5, 0.0, 0.0])
                
        except Exception as e:
            self.logger.error(f"避障指令计算失败: {e}")
            return None
            
    def compute_navigation_command(self, state: Dict) -> tuple:
        """计算导航指令"""
        current_position = state['position']
        current_velocity = state['velocity']
        return self.navigation_controller.compute_navigation_command(current_position, current_velocity)
        
    def fuse_commands(self, navigation_cmd: np.ndarray, avoidance_cmd: np.ndarray, 
                     obstacle_detected: bool, nav_info: Dict) -> np.ndarray:
        """融合导航和避障指令"""
        if nav_info['status'] == 'mission_completed':
            return avoidance_cmd * 0.3
            
        if not obstacle_detected:
            return navigation_cmd
            
        return self.navigation_controller.combine_navigation_and_avoidance(
            navigation_cmd, avoidance_cmd, obstacle_detected
        )
        
    def control_loop(self):
        """主控制循环"""
        self.logger.info("开始ViTfly导航控制循环")
        
        control_dt = 1.0 / self.control_frequency
        
        while self.is_running:
            try:
                # 获取传感器数据
                depth_image = self.airsim_interface.get_depth_image()
                state = self.airsim_interface.get_state()
                
                if depth_image is None or not state:
                    time.sleep(control_dt)
                    continue
                    
                # 障碍物检测
                obstacle_detected = self.detect_obstacles(depth_image)
                
                # 计算指令
                avoidance_cmd = self.compute_avoidance_command(depth_image, state)
                navigation_cmd, nav_info = self.compute_navigation_command(state)
                
                if avoidance_cmd is not None:
                    # 融合指令
                    final_cmd = self.fuse_commands(navigation_cmd, avoidance_cmd, obstacle_detected, nav_info)
                    
                    # 安全检查并执行
                    safe_velocity = self.safety_controller.safe_velocity_command(*final_cmd)
                    self.airsim_interface.move_by_velocity(*safe_velocity, control_dt)
                    
                time.sleep(control_dt)
                
            except Exception as e:
                self.logger.error(f"控制循环异常: {e}")
                break
                
    def start_navigation_mission(self, altitude: float = 3.0, mission_file: str = None, 
                               mission_type: str = "square"):
        """开始导航任务"""
        try:
            # 连接和起飞
            if not self.airsim_interface.connect():
                return False
            if not self.airsim_interface.takeoff(altitude):
                return False
                
            # 设置任务
            state = self.airsim_interface.get_state()
            if state:
                self.navigation_controller.set_home_position(state['position'])
            self.load_mission(mission_file, mission_type)
            
            # 重置LSTM状态
            self.lstm_hidden_state = self.model.reset_lstm_state(1, self.device)
            
            # 开始导航
            self.is_running = True
            self.logger.info(f"开始ViTfly导航任务 (高度: {altitude}m, 类型: {mission_type})")
            self.control_loop()
            
            return True
            
        except Exception as e:
            self.logger.error(f"导航任务失败: {e}")
            return False
        finally:
            self.stop()
            
    def stop(self):
        """停止系统"""
        self.is_running = False
        if self.airsim_interface.is_connected:
            self.airsim_interface.land()
            self.airsim_interface.disconnect()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ViTfly导航系统")
    parser.add_argument('--model', type=str, help='预训练模型路径')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    parser.add_argument('--altitude', type=float, default=2.5, help='飞行高度 (m)')
    parser.add_argument('--mission-file', type=str, help='任务文件路径')
    parser.add_argument('--mission-type', choices=['square', 'line', 'exploration'], 
                       default='square', help='内置任务类型')
    parser.add_argument('--avoidance-only', action='store_true', help='纯避障模式')
    
    args = parser.parse_args()
    
    try:
        vitfly_nav = ViTflyNavigationSystem(model_path=args.model, device=args.device)
        
        if args.avoidance_only:
            vitfly_nav.navigation_enabled = False
            
        print(f"开始ViTfly导航任务 (类型: {args.mission_type})")
        print("按 Ctrl+C 可随时安全停止")
        
        success = vitfly_nav.start_navigation_mission(
            altitude=args.altitude,
            mission_file=args.mission_file,
            mission_type=args.mission_type
        )
        
        print("导航任务结果:", "成功" if success else "失败")
        
    except KeyboardInterrupt:
        print("\n用户中断，正在安全退出...")
    except Exception as e:
        print(f"系统异常: {e}")


if __name__ == "__main__":
    main()