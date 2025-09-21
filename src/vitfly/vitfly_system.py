"""
ViTfly主程序 - 基于Vision Transformer的端到端无人机避障系统
整合ViT模型和AirSim接口，实现实时避障飞行
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

from models.vit_model import create_minimal_vit_model
from airsim_interface.airsim_interface import AirSimDroneInterface, SafetyController


class ViTflySystem:
    """ViTfly避障系统主类"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = create_minimal_vit_model(model_path).to(self.device)
        self.model.eval()
        print(f"ViT模型已加载 (参数量: {sum(p.numel() for p in self.model.parameters()):,})")
        
        # AirSim接口
        self.airsim_interface = AirSimDroneInterface()
        self.safety_controller = SafetyController(self.airsim_interface)
        
        # 控制参数
        self.desired_velocity = 3.0  # m/s 期望前进速度
        self.control_frequency = 10  # Hz 控制频率
        self.image_resize = (60, 90)  # 模型输入尺寸
        
        # 状态管理
        self.lstm_hidden_state = None
        self.is_running = False
        self.performance_stats = {
            'inference_times': [],
            'total_frames': 0,
            'collision_count': 0,
            'start_time': None
        }
        
        # 安全机制
        self.gradual_acceleration = True
        self.startup_duration = 3.0  # 渐进加速时间
        self.emergency_stop = False
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 信号处理（优雅退出）
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """信号处理器 - 优雅退出"""
        self.logger.info("接收到退出信号，正在安全停止...")\n        self.emergency_stop = True
        self.stop()
        sys.exit(0)
        
    def preprocess_depth_image(self, depth_image: np.ndarray) -> torch.Tensor:
        """预处理深度图像"""
        try:
            # 调整尺寸
            if depth_image.shape != self.image_resize:
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
            
    def compute_velocity_command(self, depth_image: np.ndarray, state: Dict) -> Optional[np.ndarray]:
        """计算速度指令"""
        try:
            start_time = time.time()
            
            # 预处理深度图像
            depth_tensor = self.preprocess_depth_image(depth_image)
            if depth_tensor is None:
                return None
                
            # 准备输入
            desired_vel = torch.tensor([[self.desired_velocity]], device=self.device)
            quaternion = torch.tensor([state['orientation_quaternion']], device=self.device)
            
            # 模型推理
            with torch.no_grad():
                velocity_cmd, self.lstm_hidden_state = self.model(
                    depth_tensor, desired_vel, quaternion, self.lstm_hidden_state
                )
                
            # 转换输出
            velocity_np = velocity_cmd.squeeze().cpu().numpy()
            
            # 后处理：归一化和缩放
            if np.linalg.norm(velocity_np) > 0:
                velocity_direction = velocity_np / np.linalg.norm(velocity_np)
                final_velocity = velocity_direction * self.desired_velocity
            else:
                final_velocity = np.array([self.desired_velocity * 0.5, 0.0, 0.0])  # 保守前进
                
            # 记录性能
            inference_time = time.time() - start_time
            self.performance_stats['inference_times'].append(inference_time)
            
            return final_velocity
            
        except Exception as e:
            self.logger.error(f"速度指令计算失败: {e}")
            return None
            
    def apply_gradual_acceleration(self, velocity_cmd: np.ndarray, 
                                 elapsed_time: float) -> np.ndarray:
        """应用渐进加速策略"""
        if not self.gradual_acceleration or elapsed_time > self.startup_duration:
            return velocity_cmd
            
        # 计算加速比例
        acceleration_factor = min(1.0, elapsed_time / self.startup_duration)
        
        # 最小前进速度
        min_forward_speed = 0.5
        
        # 应用渐进加速
        scaled_velocity = velocity_cmd * acceleration_factor
        scaled_velocity[0] = max(scaled_velocity[0], min_forward_speed * acceleration_factor)
        
        return scaled_velocity
        
    def control_loop(self):
        """主控制循环"""
        self.logger.info("开始ViTfly控制循环")
        
        try:
            control_dt = 1.0 / self.control_frequency
            
            while self.is_running and not self.emergency_stop:
                loop_start = time.time()
                
                # 紧急情况检查
                if self.safety_controller.emergency_check():
                    self.logger.warning("检测到紧急情况，停止飞行")
                    break
                    
                # 获取传感器数据
                depth_image = self.airsim_interface.get_depth_image()
                state = self.airsim_interface.get_state()
                
                if depth_image is None or not state:
                    self.logger.warning("传感器数据获取失败，跳过此帧")
                    time.sleep(control_dt)
                    continue
                    
                # 计算速度指令
                velocity_cmd = self.compute_velocity_command(depth_image, state)
                
                if velocity_cmd is not None:
                    # 渐进加速
                    elapsed_time = time.time() - self.performance_stats['start_time']
                    velocity_cmd = self.apply_gradual_acceleration(velocity_cmd, elapsed_time)
                    
                    # 安全检查
                    safe_velocity = self.safety_controller.safe_velocity_command(*velocity_cmd)
                    
                    # 执行速度指令
                    success = self.airsim_interface.move_by_velocity(*safe_velocity, control_dt)
                    
                    if success:
                        self.performance_stats['total_frames'] += 1
                    
                    # 日志记录
                    if self.performance_stats['total_frames'] % 50 == 0:
                        avg_inference_time = np.mean(self.performance_stats['inference_times'][-50:]) * 1000
                        self.logger.info(
                            f"帧 {self.performance_stats['total_frames']}: "
                            f"速度指令 {safe_velocity}, "
                            f"平均推理时间 {avg_inference_time:.1f}ms"
                        )
                        
                # 控制频率
                loop_time = time.time() - loop_start
                if loop_time < control_dt:
                    time.sleep(control_dt - loop_time)
                    
        except Exception as e:
            self.logger.error(f"控制循环异常: {e}")
        finally:
            self.logger.info("控制循环结束")
            
    def start_autonomous_flight(self, altitude: float = 3.0, duration: float = 60.0):
        """开始自主避障飞行"""
        try:
            # 连接AirSim
            if not self.airsim_interface.connect():
                self.logger.error("AirSim连接失败")
                return False
                
            # 起飞
            if not self.airsim_interface.takeoff(altitude):
                self.logger.error("起飞失败")
                return False
                
            # 重置LSTM状态
            self.lstm_hidden_state = self.model.reset_lstm_state(1, self.device)
            
            # 开始飞行
            self.is_running = True
            self.performance_stats['start_time'] = time.time()
            
            self.logger.info(f"开始自主避障飞行 (高度: {altitude}m, 持续时间: {duration}s)")
            
            # 设置超时
            end_time = time.time() + duration
            
            # 主控制循环
            self.control_loop()
            
            # 检查超时
            if time.time() >= end_time:
                self.logger.info("飞行时间到达，准备降落")
                
            return True
            
        except Exception as e:
            self.logger.error(f"自主飞行失败: {e}")
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
            
        self._print_performance_stats()
        
    def _print_performance_stats(self):
        """打印性能统计"""
        if self.performance_stats['total_frames'] > 0:
            total_time = time.time() - self.performance_stats['start_time']
            avg_fps = self.performance_stats['total_frames'] / total_time
            avg_inference = np.mean(self.performance_stats['inference_times']) * 1000
            
            print("\n" + "="*50)
            print("ViTfly性能统计")
            print("="*50)
            print(f"总帧数: {self.performance_stats['total_frames']}")
            print(f"总时间: {total_time:.1f}s")
            print(f"平均FPS: {avg_fps:.1f}")
            print(f"平均推理时间: {avg_inference:.1f}ms")
            print(f"碰撞次数: {self.performance_stats['collision_count']}")
            print("="*50)
            
    def test_mode(self):
        """测试模式 - 验证系统功能"""
        self.logger.info("启动测试模式")
        
        try:
            # 连接测试
            if not self.airsim_interface.connect():
                return False
                
            # 传感器测试
            depth_image = self.airsim_interface.get_depth_image()
            state = self.airsim_interface.get_state()
            
            if depth_image is not None:
                print(f"深度图像尺寸: {depth_image.shape}")
                
            if state:
                print(f"无人机状态: {state}")
                
            # 模型测试
            if depth_image is not None and state:
                velocity_cmd = self.compute_velocity_command(depth_image, state)
                if velocity_cmd is not None:
                    print(f"速度指令: {velocity_cmd}")
                    
            print("系统测试完成")
            return True
            
        except Exception as e:
            self.logger.error(f"测试失败: {e}")
            return False
        finally:
            self.airsim_interface.disconnect()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ViTfly避障系统")
    parser.add_argument('--model', type=str, help='预训练模型路径')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (cuda/cpu/auto)')
    parser.add_argument('--velocity', type=float, default=3.0, help='期望飞行速度 (m/s)')
    parser.add_argument('--altitude', type=float, default=3.0, help='飞行高度 (m)')
    parser.add_argument('--duration', type=float, default=60.0, help='飞行持续时间 (s)')
    parser.add_argument('--test', action='store_true', help='测试模式')
    
    args = parser.parse_args()
    
    try:
        # 创建系统
        vitfly = ViTflySystem(model_path=args.model, device=args.device)
        vitfly.desired_velocity = args.velocity
        
        if args.test:
            # 测试模式
            success = vitfly.test_mode()
            print("测试结果:", "成功" if success else "失败")
        else:
            # 自主飞行模式
            print(f"开始ViTfly自主避障飞行")
            print(f"参数: 速度={args.velocity}m/s, 高度={args.altitude}m, 持续时间={args.duration}s")
            print("按 Ctrl+C 可随时安全停止")
            
            success = vitfly.start_autonomous_flight(
                altitude=args.altitude,
                duration=args.duration
            )
            
            print("飞行结果:", "成功" if success else "失败")
            
    except KeyboardInterrupt:
        print("\n用户中断，正在安全退出...")
    except Exception as e:
        print(f"系统异常: {e}")
    finally:
        print("ViTfly系统已关闭")


if __name__ == "__main__":
    main()