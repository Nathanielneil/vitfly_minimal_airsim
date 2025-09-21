"""
任务规划器 - 高级任务生成和管理
"""

import numpy as np
import yaml
from typing import List, Dict, Tuple
from .navigation_controller import Waypoint


class MissionPlanner:
    """任务规划器"""
    
    def __init__(self):
        self.predefined_missions = {
            "square": self._create_square_mission,
            "line": self._create_line_mission,
            "exploration": self._create_exploration_mission,
            "spiral": self._create_spiral_mission,
            "zigzag": self._create_zigzag_mission
        }
    
    def _create_square_mission(self, size: float = 10.0, altitude: float = 3.0) -> List[Waypoint]:
        """创建正方形巡航任务"""
        return [
            Waypoint(size, 0, -altitude, tolerance=2.0, max_velocity=3.0),
            Waypoint(size, size, -altitude, tolerance=2.0, max_velocity=3.0),
            Waypoint(0, size, -altitude, tolerance=2.0, max_velocity=3.0),
            Waypoint(0, 0, -altitude, tolerance=2.0, max_velocity=3.0)
        ]
    
    def _create_line_mission(self, distance: float = 20.0, altitude: float = 3.0) -> List[Waypoint]:
        """创建直线往返任务"""
        return [
            Waypoint(distance, 0, -altitude, tolerance=2.0, max_velocity=4.0),
            Waypoint(0, 0, -altitude, tolerance=2.0, max_velocity=3.0)
        ]
    
    def _create_exploration_mission(self, altitude: float = 3.0) -> List[Waypoint]:
        """创建探索任务"""
        return [
            Waypoint(15, 5, -altitude, tolerance=3.0, max_velocity=4.0),
            Waypoint(20, -5, -altitude-1, tolerance=2.0, max_velocity=3.0),
            Waypoint(10, -10, -altitude-2, tolerance=2.0, max_velocity=2.0),
            Waypoint(0, 0, -altitude, tolerance=2.0, max_velocity=3.0)
        ]
    
    def _create_spiral_mission(self, radius: float = 15.0, altitude: float = 3.0) -> List[Waypoint]:
        """创建螺旋任务"""
        waypoints = []
        for i in range(8):
            angle = i * np.pi / 4  # 45度间隔
            r = radius * (1 - i * 0.1)  # 半径递减
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = -altitude - i * 0.5  # 高度递增
            waypoints.append(Waypoint(x, y, z, tolerance=2.0, max_velocity=3.0))
        return waypoints
    
    def _create_zigzag_mission(self, width: float = 20.0, depth: float = 15.0, 
                              altitude: float = 3.0) -> List[Waypoint]:
        """创建之字形搜索任务"""
        waypoints = []
        for i in range(4):
            y = i * depth / 3
            if i % 2 == 0:
                x = width
            else:
                x = 0
            waypoints.append(Waypoint(x, y, -altitude, tolerance=2.0, max_velocity=3.0))
        return waypoints
    
    def create_mission(self, mission_type: str, **kwargs) -> List[Waypoint]:
        """创建预定义任务"""
        if mission_type in self.predefined_missions:
            return self.predefined_missions[mission_type](**kwargs)
        else:
            raise ValueError(f"未知任务类型: {mission_type}")
    
    def save_mission(self, waypoints: List[Waypoint], filepath: str, mode: str = "goto"):
        """保存任务到YAML文件"""
        mission_data = {
            "mode": mode,
            "waypoints": [
                {
                    "x": wp.x,
                    "y": wp.y,
                    "z": wp.z,
                    "tolerance": wp.tolerance,
                    "max_velocity": wp.max_velocity
                }
                for wp in waypoints
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(mission_data, f, default_flow_style=False, allow_unicode=True)
    
    def optimize_mission(self, waypoints: List[Waypoint], 
                        start_position: np.ndarray) -> List[Waypoint]:
        """优化任务路径（简单的最近邻优化）"""
        if len(waypoints) <= 1:
            return waypoints
        
        optimized = []
        remaining = waypoints.copy()
        current_pos = start_position
        
        while remaining:
            # 找到距离当前位置最近的航点
            distances = [
                np.linalg.norm(np.array([wp.x, wp.y, wp.z]) - current_pos)
                for wp in remaining
            ]
            nearest_idx = np.argmin(distances)
            nearest_wp = remaining.pop(nearest_idx)
            
            optimized.append(nearest_wp)
            current_pos = np.array([nearest_wp.x, nearest_wp.y, nearest_wp.z])
        
        return optimized
    
    def estimate_mission_time(self, waypoints: List[Waypoint]) -> float:
        """估算任务执行时间"""
        if len(waypoints) <= 1:
            return 0.0
        
        total_time = 0.0
        prev_pos = np.array([0, 0, 0])  # 假设从原点开始
        
        for wp in waypoints:
            curr_pos = np.array([wp.x, wp.y, wp.z])
            distance = np.linalg.norm(curr_pos - prev_pos)
            travel_time = distance / wp.max_velocity
            
            # 加上悬停时间
            hover_time = 2.0  # 假设每个航点悬停2秒
            
            total_time += travel_time + hover_time
            prev_pos = curr_pos
        
        return total_time
    
    def validate_mission(self, waypoints: List[Waypoint]) -> Dict[str, bool]:
        """验证任务的安全性"""
        validation = {
            "altitude_safe": True,
            "distance_reasonable": True,
            "velocity_safe": True,
            "waypoint_count_ok": True
        }
        
        # 检查高度
        for wp in waypoints:
            if wp.z > -0.5 or wp.z < -100:  # AirSim坐标系
                validation["altitude_safe"] = False
        
        # 检查距离
        prev_pos = np.array([0, 0, 0])
        for wp in waypoints:
            curr_pos = np.array([wp.x, wp.y, wp.z])
            distance = np.linalg.norm(curr_pos - prev_pos)
            if distance > 100:  # 单段距离不超过100米
                validation["distance_reasonable"] = False
            prev_pos = curr_pos
        
        # 检查速度
        for wp in waypoints:
            if wp.max_velocity > 10.0 or wp.max_velocity < 0.5:
                validation["velocity_safe"] = False
        
        # 检查航点数量
        if len(waypoints) > 20:
            validation["waypoint_count_ok"] = False
        
        return validation


if __name__ == "__main__":
    # 测试任务规划器
    planner = MissionPlanner()
    
    # 创建各种任务
    missions = {
        "square": planner.create_mission("square", size=15.0),
        "spiral": planner.create_mission("spiral", radius=20.0),
        "zigzag": planner.create_mission("zigzag", width=25.0, depth=20.0)
    }
    
    for name, waypoints in missions.items():
        print(f"\n{name}任务:")
        print(f"  航点数: {len(waypoints)}")
        print(f"  估算时间: {planner.estimate_mission_time(waypoints):.1f}秒")
        
        validation = planner.validate_mission(waypoints)
        print(f"  安全验证: {all(validation.values())}")
        
        # 保存任务文件
        planner.save_mission(waypoints, f"mission_{name}.yaml", mode="patrol")
        print(f"  已保存: mission_{name}.yaml")