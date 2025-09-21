"""
导航控制器 - 添加目标点导航功能到ViTfly系统
"""

import numpy as np
import math
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class Waypoint:
    """航点定义"""
    x: float
    y: float
    z: float
    tolerance: float = 2.0  # 到达容差 (米)
    max_velocity: float = 3.0  # 该点最大速度
    
    def distance_to(self, position: np.ndarray) -> float:
        """计算到该航点的距离"""
        return np.linalg.norm(np.array([self.x, self.y, self.z]) - position)
    
    def is_reached(self, position: np.ndarray) -> bool:
        """判断是否到达航点"""
        return self.distance_to(position) <= self.tolerance


class NavigationController:
    """导航控制器 - 结合避障和目标点导航"""
    
    def __init__(self):
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_index = 0
        self.navigation_mode = "goto"  # "goto", "patrol", "return_home"
        
        # 导航参数
        self.max_distance_for_direct_flight = 50.0  # 最大直飞距离
        self.approach_slowdown_distance = 10.0      # 接近减速距离
        self.min_approach_velocity = 1.0            # 最小接近速度
        
        # 路径规划参数
        self.path_planning_enabled = True
        self.avoidance_priority = 0.7  # 避障优先级 (0-1)
        self.navigation_priority = 0.3 # 导航优先级 (0-1)
        
        # 状态
        self.home_position = np.array([0.0, 0.0, -3.0])
        self.mission_completed = False
        
    def set_home_position(self, position: np.ndarray):
        """设置起始位置"""
        self.home_position = position.copy()
        
    def add_waypoint(self, x: float, y: float, z: float, 
                    tolerance: float = 2.0, max_velocity: float = 3.0):
        """添加航点"""
        waypoint = Waypoint(x, y, z, tolerance, max_velocity)
        self.waypoints.append(waypoint)
        print(f"添加航点: ({x:.1f}, {y:.1f}, {z:.1f}), 容差: {tolerance}m")
        
    def load_mission_from_file(self, filepath: str):
        """从文件加载任务航点"""
        try:
            import yaml
            with open(filepath, 'r', encoding='utf-8') as f:
                mission_data = yaml.safe_load(f)
                
            self.waypoints.clear()
            for wp_data in mission_data.get('waypoints', []):
                self.add_waypoint(
                    x=wp_data['x'],
                    y=wp_data['y'], 
                    z=wp_data['z'],
                    tolerance=wp_data.get('tolerance', 2.0),
                    max_velocity=wp_data.get('max_velocity', 3.0)
                )
                
            self.navigation_mode = mission_data.get('mode', 'goto')
            print(f"任务加载完成: {len(self.waypoints)}个航点, 模式: {self.navigation_mode}")
            
        except Exception as e:
            print(f"任务文件加载失败: {e}")
            
    def create_simple_mission(self, mission_type: str = "square"):
        """创建简单任务"""
        self.waypoints.clear()
        
        if mission_type == "square":
            # 正方形巡航
            self.add_waypoint(10, 0, -3)   # 前进
            self.add_waypoint(10, 10, -3)  # 右转
            self.add_waypoint(0, 10, -3)   # 后退
            self.add_waypoint(0, 0, -3)    # 回起点
            self.navigation_mode = "patrol"
            
        elif mission_type == "line":
            # 直线往返
            self.add_waypoint(20, 0, -3)
            self.add_waypoint(0, 0, -3)
            self.navigation_mode = "patrol"
            
        elif mission_type == "exploration":
            # 探索模式
            self.add_waypoint(15, 5, -3)
            self.add_waypoint(20, -5, -4)
            self.add_waypoint(10, -10, -5)
            self.add_waypoint(0, 0, -3)
            self.navigation_mode = "goto"
            
        print(f"创建{mission_type}任务: {len(self.waypoints)}个航点")
        
    def get_current_target(self) -> Optional[Waypoint]:
        """获取当前目标航点"""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return None
        return self.waypoints[self.current_waypoint_index]
        
    def compute_navigation_command(self, current_position: np.ndarray, 
                                 current_velocity: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        计算导航指令
        返回: (目标速度向量, 导航信息)
        """
        # 获取当前目标
        target = self.get_current_target()
        if target is None:
            return np.array([0.0, 0.0, 0.0]), {"status": "mission_completed"}
            
        # 计算到目标的向量
        target_position = np.array([target.x, target.y, target.z])
        to_target = target_position - current_position
        distance_to_target = np.linalg.norm(to_target)
        
        # 检查是否到达当前航点
        if target.is_reached(current_position):
            self._advance_to_next_waypoint()
            return self.compute_navigation_command(current_position, current_velocity)
            
        # 计算导航速度
        if distance_to_target > 0:
            direction = to_target / distance_to_target
        else:
            direction = np.array([1.0, 0.0, 0.0])  # 默认前进方向
            
        # 根据距离调整速度
        desired_speed = self._compute_desired_speed(distance_to_target, target.max_velocity)
        navigation_velocity = direction * desired_speed
        
        # 导航信息
        nav_info = {
            "status": "navigating",
            "current_waypoint": self.current_waypoint_index,
            "total_waypoints": len(self.waypoints),
            "distance_to_target": distance_to_target,
            "target_position": target_position,
            "desired_speed": desired_speed,
            "navigation_mode": self.navigation_mode
        }
        
        return navigation_velocity, nav_info
        
    def _compute_desired_speed(self, distance: float, max_speed: float) -> float:
        """根据距离计算期望速度"""
        if distance <= self.approach_slowdown_distance:
            # 接近目标时减速
            speed_ratio = max(distance / self.approach_slowdown_distance, 0.2)
            return max(max_speed * speed_ratio, self.min_approach_velocity)
        else:
            return max_speed
            
    def _advance_to_next_waypoint(self):
        """前进到下一个航点"""
        if self.navigation_mode == "goto":
            # 顺序执行模式
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                self.mission_completed = True
                print("任务完成!")
                
        elif self.navigation_mode == "patrol":
            # 巡航模式 - 循环执行
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
            print(f"巡航: 前往航点 {self.current_waypoint_index}")
            
        elif self.navigation_mode == "return_home":
            # 返回起点模式
            self.waypoints = [Waypoint(
                self.home_position[0], 
                self.home_position[1], 
                self.home_position[2]
            )]
            self.current_waypoint_index = 0
            self.navigation_mode = "goto"
            print("返回起点模式激活")
            
    def combine_navigation_and_avoidance(self, 
                                       navigation_cmd: np.ndarray,
                                       avoidance_cmd: np.ndarray,
                                       obstacle_detected: bool = False) -> np.ndarray:
        """
        融合导航和避障指令
        
        Args:
            navigation_cmd: 导航指令
            avoidance_cmd: 避障指令  
            obstacle_detected: 是否检测到障碍物
        """
        if not obstacle_detected:
            # 无障碍物时，主要使用导航指令
            return navigation_cmd
            
        # 有障碍物时，融合两个指令
        # 避障在横向(y)和垂直(z)方向有更高优先级
        # 导航在前进(x)方向保持一定权重
        
        combined_cmd = np.zeros(3)
        
        # X方向 (前进): 避障和导航的加权平均
        combined_cmd[0] = (self.avoidance_priority * avoidance_cmd[0] + 
                          self.navigation_priority * navigation_cmd[0])
                          
        # Y方向 (左右): 主要使用避障指令
        combined_cmd[1] = avoidance_cmd[1] + 0.2 * navigation_cmd[1]
        
        # Z方向 (上下): 主要使用避障指令
        combined_cmd[2] = avoidance_cmd[2] + 0.1 * navigation_cmd[2]
        
        return combined_cmd
        
    def get_mission_status(self) -> Dict:
        """获取任务状态"""
        target = self.get_current_target()
        
        return {
            "mission_completed": self.mission_completed,
            "current_waypoint": self.current_waypoint_index,
            "total_waypoints": len(self.waypoints),
            "navigation_mode": self.navigation_mode,
            "current_target": {
                "x": target.x if target else None,
                "y": target.y if target else None,
                "z": target.z if target else None
            } if target else None
        }
        
    def emergency_return_home(self):
        """紧急返回起点"""
        self.navigation_mode = "return_home"
        self.current_waypoint_index = 0
        print("紧急返回起点模式激活")
        
    def reset_mission(self):
        """重置任务"""
        self.current_waypoint_index = 0
        self.mission_completed = False
        print("任务已重置")


def create_sample_missions():
    """创建示例任务文件"""
    import yaml
    
    # 示例任务1: 正方形巡航
    square_mission = {
        "mode": "patrol",
        "waypoints": [
            {"x": 10, "y": 0, "z": -3, "tolerance": 2.0, "max_velocity": 3.0},
            {"x": 10, "y": 10, "z": -3, "tolerance": 2.0, "max_velocity": 3.0},
            {"x": 0, "y": 10, "z": -3, "tolerance": 2.0, "max_velocity": 3.0},
            {"x": 0, "y": 0, "z": -3, "tolerance": 2.0, "max_velocity": 3.0}
        ]
    }
    
    # 示例任务2: 探索任务
    exploration_mission = {
        "mode": "goto",
        "waypoints": [
            {"x": 15, "y": 5, "z": -3, "tolerance": 3.0, "max_velocity": 4.0},
            {"x": 20, "y": -5, "z": -4, "tolerance": 2.0, "max_velocity": 3.0},
            {"x": 10, "y": -10, "z": -5, "tolerance": 2.0, "max_velocity": 2.0},
            {"x": 0, "y": 0, "z": -3, "tolerance": 2.0, "max_velocity": 3.0}
        ]
    }
    
    # 保存任务文件
    with open('mission_square.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(square_mission, f, default_flow_style=False, allow_unicode=True)
        
    with open('mission_exploration.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(exploration_mission, f, default_flow_style=False, allow_unicode=True)
        
    print("示例任务文件已创建:")
    print("- mission_square.yaml: 正方形巡航")
    print("- mission_exploration.yaml: 探索任务")


if __name__ == "__main__":
    # 测试导航控制器
    nav_controller = NavigationController()
    
    # 创建示例任务
    nav_controller.create_simple_mission("square")
    
    # 模拟导航
    current_pos = np.array([0.0, 0.0, -3.0])
    current_vel = np.array([0.0, 0.0, 0.0])
    
    for i in range(10):
        nav_cmd, nav_info = nav_controller.compute_navigation_command(current_pos, current_vel)
        print(f"步骤 {i}: 导航指令 {nav_cmd}, 状态: {nav_info['status']}")
        
        # 模拟移动
        current_pos += nav_cmd * 0.1
        
    # 创建示例任务文件
    create_sample_missions()