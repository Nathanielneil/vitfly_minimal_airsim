#!/usr/bin/env python3
"""
简单前进避障脚本
- 起飞到0.5m高度
- 使用ViT模型避障前进4米
- 自动降落
"""

import torch
import numpy as np
import time
import logging
import signal
import sys
from typing import Optional

from vit_model import create_minimal_vit_model
from airsim_interface import AirSimDroneInterface, SafetyController


class SimpleForwardFlight:
    """简单前进避障飞行"""
    
    def __init__(self, model_path: str = "vitfly_simple_policy.pth"):
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = create_minimal_vit_model(model_path).to(self.device)
        self.model.eval()
        print(f"ViT模型已加载 (参数量: {sum(p.numel() for p in self.model.parameters()):,})")
        
        # AirSim接口
        self.airsim_interface = AirSimDroneInterface()
        self.safety_controller = SafetyController(self.airsim_interface)
        
        # 飞行参数
        self.takeoff_height = 0.5  # 起飞高度0.5m
        self.forward_distance = 4.0  # 前进距离4m
        self.base_velocity = 1.5  # 基础前进速度 (较慢，更安全)
        self.control_frequency = 10  # 控制频率10Hz
        self.image_resize = (60, 90)  # 模型输入尺寸
        
        # 状态管理
        self.lstm_hidden_state = None
        self.is_running = False
        self.start_position = None
        self.target_reached = False
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """优雅退出"""
        self.logger.info("接收到退出信号，正在安全停止...")
        self.stop()
        sys.exit(0)
        
    def preprocess_depth_image(self, depth_image: np.ndarray) -> torch.Tensor:
        """预处理深度图像"""
        try:
            # 调整尺寸
            if depth_image.shape != self.image_resize:
                import cv2
                depth_image = cv2.resize(depth_image, 
                                       (self.image_resize[1], self.image_resize[0]),
                                       interpolation=cv2.INTER_LINEAR)
            
            # 数据类型和归一化
            depth_image = depth_image.astype(np.float32)
            if depth_image.max() > 1.0:
                depth_image = depth_image / 255.0
                
            # 转换为PyTorch张量
            depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).to(self.device)
            return depth_tensor
            
        except Exception as e:
            self.logger.error(f"深度图像预处理失败: {e}")
            return None
            
    def compute_velocity_command(self, depth_image: np.ndarray, state: dict) -> Optional[np.ndarray]:
        """计算避障速度指令"""
        try:
            # 预处理深度图像
            depth_tensor = self.preprocess_depth_image(depth_image)
            if depth_tensor is None:
                return None
                
            # 准备输入
            desired_vel = torch.tensor([[self.base_velocity]], device=self.device, dtype=torch.float32)
            quaternion = torch.tensor([state['orientation_quaternion']], device=self.device, dtype=torch.float32)
            
            # 模型推理
            with torch.no_grad():
                velocity_cmd, self.lstm_hidden_state = self.model(
                    depth_tensor, desired_vel, quaternion, self.lstm_hidden_state
                )
                
            # 转换输出
            velocity_np = velocity_cmd.squeeze().cpu().numpy()
            
            # 后处理：确保主要是前进
            if np.linalg.norm(velocity_np) > 0:
                velocity_direction = velocity_np / np.linalg.norm(velocity_np)
                final_velocity = velocity_direction * self.base_velocity
                
                # 确保主要方向是前进（X轴正方向）
                if final_velocity[0] < 0.3:  # 如果前进速度太小
                    final_velocity[0] = 0.5  # 设置最小前进速度
                    
                # 高度稳定：保持在0.5m高度
                current_height = -state['position'][2]
                target_height = self.takeoff_height
                height_error = current_height - target_height
                
                if height_error > 0.2:  # 高于目标高度0.2m
                    final_velocity[2] = max(final_velocity[2], 0.3)  # 下降
                elif height_error < -0.2:  # 低于目标高度0.2m  
                    final_velocity[2] = min(final_velocity[2], -0.3)  # 上升
                else:
                    final_velocity[2] = np.clip(final_velocity[2], -0.5, 0.5)  # 轻微调整
                    
            else:
                # 如果模型没有输出，保守前进
                final_velocity = np.array([0.5, 0.0, 0.0])
                
            return final_velocity
            
        except Exception as e:
            self.logger.error(f"速度指令计算失败: {e}")
            return None
            
    def check_distance_reached(self, current_position: np.ndarray) -> bool:
        """检查是否已前进指定距离"""
        if self.start_position is None:
            return False
            
        # 计算前进距离（X轴方向）
        forward_distance = current_position[0] - self.start_position[0]
        
        if forward_distance >= self.forward_distance:
            self.logger.info(f"目标距离已达成: {forward_distance:.2f}m >= {self.forward_distance}m")
            return True
            
        # 定期报告进度
        if hasattr(self, '_last_progress_time'):
            if time.time() - self._last_progress_time > 2.0:  # 每2秒报告一次
                self.logger.info(f"前进进度: {forward_distance:.2f}m / {self.forward_distance}m")
                self._last_progress_time = time.time()
        else:
            self._last_progress_time = time.time()
            
        return False
        
    def control_loop(self):
        """主控制循环"""
        self.logger.info("开始避障前进控制循环")
        
        control_dt = 1.0 / self.control_frequency
        frame_count = 0
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # 获取传感器数据
                depth_image = self.airsim_interface.get_depth_image()
                state = self.airsim_interface.get_state()
                
                if depth_image is None or not state:
                    self.logger.warning("传感器数据获取失败，跳过此帧")
                    time.sleep(control_dt)
                    continue
                    
                # 检查是否到达目标距离
                if self.check_distance_reached(state['position']):
                    self.target_reached = True
                    break
                    
                # 安全检查
                if self.safety_controller.emergency_check():
                    self.logger.warning("检测到紧急情况，停止飞行")
                    break
                    
                # 计算避障速度指令
                velocity_cmd = self.compute_velocity_command(depth_image, state)
                
                if velocity_cmd is not None:
                    # 安全检查
                    safe_velocity = self.safety_controller.safe_velocity_command(*velocity_cmd)
                    
                    # 执行速度指令
                    success = self.airsim_interface.move_by_velocity(*safe_velocity, control_dt)
                    
                    if success:
                        frame_count += 1
                        
                    # 每20帧报告一次状态
                    if frame_count % 20 == 0:
                        current_distance = state['position'][0] - self.start_position[0] if self.start_position is not None else 0
                        self.logger.info(f"帧 {frame_count}: 速度指令 {safe_velocity}, 已前进 {current_distance:.2f}m")
                        
                # 控制频率
                loop_time = time.time() - loop_start
                if loop_time < control_dt:
                    time.sleep(control_dt - loop_time)
                    
        except Exception as e:
            self.logger.error(f"控制循环异常: {e}")
        finally:
            self.logger.info("控制循环结束")
            
    def start_forward_flight(self):
        """开始前进避障飞行"""
        try:
            # 连接AirSim
            if not self.airsim_interface.connect():
                self.logger.error("AirSim连接失败")
                return False
                
            # 起飞
            self.logger.info(f"起飞到 {self.takeoff_height}m 高度...")
            if not self.airsim_interface.takeoff(self.takeoff_height):
                self.logger.error("起飞失败")
                return False
                
            # 记录起始位置
            initial_state = self.airsim_interface.get_state()
            if initial_state:
                self.start_position = initial_state['position'].copy()
                self.logger.info(f"起始位置: {self.start_position}")
                
            # 重置LSTM状态
            self.lstm_hidden_state = self.model.reset_lstm_state(1, self.device)
            
            # 开始避障前进
            self.is_running = True
            self.logger.info(f"开始避障前进 (目标: 前进{self.forward_distance}m, 高度: {self.takeoff_height}m)")
            
            # 主控制循环
            self.control_loop()
            
            # 检查结果
            if self.target_reached:
                final_state = self.airsim_interface.get_state()
                if final_state:
                    actual_distance = final_state['position'][0] - self.start_position[0]
                    self.logger.info(f"✅ 任务完成！实际前进距离: {actual_distance:.2f}m")
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
        
        if self.airsim_interface.is_connected:
            self.logger.info("正在降落...")
            self.airsim_interface.land()
            self.airsim_interface.disconnect()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="简单前进避障飞行")
    parser.add_argument('--model', type=str, default='vitfly_simple_policy.pth', 
                       help='预训练模型路径')
    parser.add_argument('--height', type=float, default=0.5, 
                       help='起飞高度 (m)')
    parser.add_argument('--distance', type=float, default=4.0, 
                       help='前进距离 (m)')
    parser.add_argument('--speed', type=float, default=1.5,
                       help='基础前进速度 (m/s)')
    
    args = parser.parse_args()
    
    try:
        # 创建飞行系统
        flight_system = SimpleForwardFlight(model_path=args.model)
        
        # 应用参数
        flight_system.takeoff_height = args.height
        flight_system.forward_distance = args.distance
        flight_system.base_velocity = args.speed
        
        print("=" * 60)
        print("🚁 ViTfly 简单前进避障飞行")
        print("=" * 60)
        print(f"📋 任务参数:")
        print(f"   起飞高度: {args.height}m")
        print(f"   前进距离: {args.distance}m") 
        print(f"   飞行速度: {args.speed}m/s")
        print(f"   模型文件: {args.model}")
        print("=" * 60)
        print("🎮 按 Ctrl+C 可随时安全停止")
        print("=" * 60)
        
        # 开始飞行
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
    finally:
        print("🔒 ViTfly系统已关闭")


if __name__ == "__main__":
    main()