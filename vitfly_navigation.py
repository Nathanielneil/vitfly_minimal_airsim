"""
ViTfly导航版本 - 集成目标点导航和避障功能
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
        
        # 导航控制器
        self.navigation_controller = NavigationController()
        
        # 控制参数
        self.base_velocity = 3.0  # 基础飞行速度
        self.control_frequency = 10  # Hz 控制频率
        self.image_resize = (60, 90)  # 模型输入尺寸
        
        # 融合参数
        self.obstacle_detection_threshold = 0.3  # 障碍物检测阈值
        self.navigation_weight = 0.6  # 导航权重
        self.avoidance_weight = 0.4   # 避障权重
        
        # 状态管理
        self.lstm_hidden_state = None
        self.is_running = False
        self.navigation_enabled = True
        self.pure_avoidance_mode = False
        
        # 性能统计
        self.performance_stats = {
            'inference_times': [],
            'navigation_times': [],
            'total_frames': 0,
            'waypoints_reached': 0,
            'start_time': None
        }
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info("接收到退出信号，正在安全停止...")
        self.stop()
        sys.exit(0)
        
    def load_mission(self, mission_file: str = None, mission_type: str = "square"):
        """加载任务"""
        if mission_file:
            self.navigation_controller.load_mission_from_file(mission_file)
        else:
            self.navigation_controller.create_simple_mission(mission_type)
            
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
            
    def detect_obstacles(self, depth_image: np.ndarray) -> bool:
        """检测前方是否有障碍物"""
        height, width = depth_image.shape
        
        # 检查前方中央区域
        front_region = depth_image[height//3:2*height//3, width//3:2*width//3]
        front_depth = np.mean(front_region)
        
        return front_depth < self.obstacle_detection_threshold
        
    def compute_avoidance_command(self, depth_image: np.ndarray, state: Dict) -> Optional[np.ndarray]:
        """计算避障指令"""
        try:
            start_time = time.time()
            
            # 预处理深度图像
            depth_tensor = self.preprocess_depth_image(depth_image)
            if depth_tensor is None:
                return None
                
            # 准备输入
            desired_vel = torch.tensor([[self.base_velocity]], device=self.device)
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
                final_velocity = velocity_direction * self.base_velocity
            else:
                final_velocity = np.array([self.base_velocity * 0.5, 0.0, 0.0])
                
            # 记录性能
            inference_time = time.time() - start_time
            self.performance_stats['inference_times'].append(inference_time)
            
            return final_velocity
            
        except Exception as e:
            self.logger.error(f"避障指令计算失败: {e}")
            return None
            
    def compute_navigation_command(self, state: Dict) -> Tuple[np.ndarray, Dict]:
        """计算导航指令"""
        start_time = time.time()
        
        current_position = state['position']
        current_velocity = state['velocity']
        
        # 计算导航指令
        nav_cmd, nav_info = self.navigation_controller.compute_navigation_command(
            current_position, current_velocity
        )
        
        # 记录性能
        nav_time = time.time() - start_time
        self.performance_stats['navigation_times'].append(nav_time)
        
        return nav_cmd, nav_info
        
    def fuse_commands(self, navigation_cmd: np.ndarray, avoidance_cmd: np.ndarray, 
                     obstacle_detected: bool, nav_info: Dict) -> np.ndarray:
        """融合导航和避障指令"""
        
        if self.pure_avoidance_mode or not self.navigation_enabled:
            # 纯避障模式
            return avoidance_cmd
            
        if nav_info['status'] == 'mission_completed':
            # 任务完成，使用避障保持位置
            return avoidance_cmd * 0.3
            
        if not obstacle_detected:
            # 无障碍物，主要使用导航指令
            return navigation_cmd
            
        # 有障碍物时，使用导航控制器的融合逻辑
        return self.navigation_controller.combine_navigation_and_avoidance(
            navigation_cmd, avoidance_cmd, obstacle_detected
        )
        
    def control_loop(self):
        """主控制循环"""
        self.logger.info("开始ViTfly导航控制循环")
        
        try:
            control_dt = 1.0 / self.control_frequency
            
            while self.is_running:
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
                    
                # 障碍物检测
                obstacle_detected = self.detect_obstacles(depth_image)
                
                # 计算避障指令
                avoidance_cmd = self.compute_avoidance_command(depth_image, state)
                if avoidance_cmd is None:
                    time.sleep(control_dt)
                    continue
                    
                # 计算导航指令
                navigation_cmd, nav_info = self.compute_navigation_command(state)
                
                # 融合指令
                final_cmd = self.fuse_commands(
                    navigation_cmd, avoidance_cmd, obstacle_detected, nav_info
                )
                
                # 安全检查
                safe_velocity = self.safety_controller.safe_velocity_command(*final_cmd)
                
                # 执行速度指令
                success = self.airsim_interface.move_by_velocity(*safe_velocity, control_dt)
                
                if success:
                    self.performance_stats['total_frames'] += 1
                    
                # 检查航点到达
                if nav_info.get('waypoint_reached', False):
                    self.performance_stats['waypoints_reached'] += 1
                
                # 日志记录
                if self.performance_stats['total_frames'] % 30 == 0:
                    self._log_status(safe_velocity, nav_info, obstacle_detected)
                    
                # 控制频率
                loop_time = time.time() - loop_start
                if loop_time < control_dt:
                    time.sleep(control_dt - loop_time)
                    
        except Exception as e:
            self.logger.error(f"控制循环异常: {e}")
        finally:
            self.logger.info("控制循环结束")
            
    def _log_status(self, velocity_cmd: np.ndarray, nav_info: Dict, obstacle_detected: bool):
        """记录状态信息"""
        avg_inference_time = np.mean(self.performance_stats['inference_times'][-30:]) * 1000
        avg_nav_time = np.mean(self.performance_stats['navigation_times'][-30:]) * 1000
        
        status_msg = (
            f"帧 {self.performance_stats['total_frames']}: "
            f"速度 [{velocity_cmd[0]:.1f}, {velocity_cmd[1]:.1f}, {velocity_cmd[2]:.1f}], "
            f"航点 {nav_info.get('current_waypoint', 0)}/{nav_info.get('total_waypoints', 0)}, "
            f"障碍物: {'是' if obstacle_detected else '否'}, "
            f"推理: {avg_inference_time:.1f}ms, 导航: {avg_nav_time:.1f}ms"
        )
        
        self.logger.info(status_msg)
        
    def start_navigation_mission(self, altitude: float = 3.0, mission_file: str = None, 
                               mission_type: str = "square"):
        """开始导航任务"""
        try:
            # 连接AirSim
            if not self.airsim_interface.connect():
                self.logger.error("AirSim连接失败")
                return False
                
            # 起飞
            if not self.airsim_interface.takeoff(altitude):
                self.logger.error("起飞失败")
                return False
                
            # 设置起始位置
            state = self.airsim_interface.get_state()
            if state:
                self.navigation_controller.set_home_position(state['position'])
                
            # 加载任务
            self.load_mission(mission_file, mission_type)
            
            # 重置LSTM状态
            self.lstm_hidden_state = self.model.reset_lstm_state(1, self.device)
            
            # 开始导航
            self.is_running = True
            self.performance_stats['start_time'] = time.time()
            
            self.logger.info(f"开始ViTfly导航任务 (高度: {altitude}m, 类型: {mission_type})")
            
            # 主控制循环
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
            avg_navigation = np.mean(self.performance_stats['navigation_times']) * 1000
            
            print("\n" + "="*60)
            print("ViTfly导航系统性能统计")
            print("="*60)
            print(f"总帧数: {self.performance_stats['total_frames']}")
            print(f"总时间: {total_time:.1f}s")
            print(f"平均FPS: {avg_fps:.1f}")
            print(f"平均推理时间: {avg_inference:.1f}ms")
            print(f"平均导航时间: {avg_navigation:.1f}ms")
            print(f"到达航点数: {self.performance_stats['waypoints_reached']}")
            
            # 任务状态
            mission_status = self.navigation_controller.get_mission_status()
            print(f"任务状态: {mission_status}")
            print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ViTfly导航系统")
    parser.add_argument('--model', type=str, help='预训练模型路径')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    parser.add_argument('--altitude', type=float, default=3.0, help='飞行高度 (m)')
    parser.add_argument('--mission-file', type=str, help='任务文件路径')
    parser.add_argument('--mission-type', choices=['square', 'line', 'exploration'], 
                       default='square', help='内置任务类型')
    parser.add_argument('--avoidance-only', action='store_true', help='纯避障模式')
    
    args = parser.parse_args()
    
    try:
        # 创建系统
        vitfly_nav = ViTflyNavigationSystem(model_path=args.model, device=args.device)
        
        if args.avoidance_only:
            vitfly_nav.pure_avoidance_mode = True
            print("纯避障模式启动")
        
        print(f"开始ViTfly导航任务")
        print(f"参数: 高度={args.altitude}m, 任务类型={args.mission_type}")
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
    finally:
        print("ViTfly导航系统已关闭")


if __name__ == "__main__":
    main()