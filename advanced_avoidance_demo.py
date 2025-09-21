"""
高级避障策略演示 - 展示ViTfly的全方位3D避障能力
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


class AdvancedObstacleAvoidance:
    """高级避障策略 - 全方位3D避障"""
    
    def __init__(self):
        # 避障参数
        self.obstacle_threshold = 0.3     # 障碍物检测阈值
        self.safety_margin = 0.2          # 安全边距
        self.preferred_clearance = 0.5    # 期望间隙
        
        # 避障策略权重
        self.forward_weight = 0.6         # 前进权重
        self.lateral_weight = 0.3         # 侧向权重  
        self.vertical_weight = 0.1        # 垂直权重
        
        # 动作空间
        self.avoidance_actions = {
            "forward": np.array([1.0, 0.0, 0.0]),      # 继续前进
            "slow_forward": np.array([0.3, 0.0, 0.0]), # 减速前进
            "stop": np.array([0.0, 0.0, 0.0]),         # 停止
            "backward": np.array([-0.5, 0.0, 0.0]),    # 后退
            "left": np.array([0.3, -2.0, 0.0]),        # 左转避障
            "right": np.array([0.3, 2.0, 0.0]),        # 右转避障
            "up": np.array([0.3, 0.0, -1.5]),          # 上升避障
            "down": np.array([0.3, 0.0, 1.0]),         # 下降避障
            "left_up": np.array([0.2, -1.5, -1.0]),    # 左上避障
            "right_up": np.array([0.2, 1.5, -1.0]),    # 右上避障
            "left_down": np.array([0.2, -1.5, 0.8]),   # 左下避障
            "right_down": np.array([0.2, 1.5, 0.8]),   # 右下避障
        }
    
    def analyze_depth_environment(self, depth_image: np.ndarray) -> Dict:
        """
        分析深度图像环境
        返回各个方向的障碍物信息
        """
        height, width = depth_image.shape
        
        # 定义检测区域
        regions = {
            # 前方区域 (分为上中下)
            "front_top": depth_image[0:height//3, width//3:2*width//3],
            "front_center": depth_image[height//3:2*height//3, width//3:2*width//3],
            "front_bottom": depth_image[2*height//3:height, width//3:2*width//3],
            
            # 左侧区域 (分为上中下)
            "left_top": depth_image[0:height//3, 0:width//3],
            "left_center": depth_image[height//3:2*height//3, 0:width//3],
            "left_bottom": depth_image[2*height//3:height, 0:width//3],
            
            # 右侧区域 (分为上中下)
            "right_top": depth_image[0:height//3, 2*width//3:width],
            "right_center": depth_image[height//3:2*height//3, 2*width//3:width],
            "right_bottom": depth_image[2*height//3:height, 2*width//3:width],
            
            # 上方和下方区域
            "top": depth_image[0:height//4, :],
            "bottom": depth_image[3*height//4:height, :]
        }
        
        # 计算各区域的深度统计
        environment = {}
        for region_name, region in regions.items():
            mean_depth = np.mean(region)
            min_depth = np.min(region)
            std_depth = np.std(region)
            
            # 障碍物检测
            has_obstacle = mean_depth < self.obstacle_threshold
            is_safe = min_depth > self.preferred_clearance
            
            environment[region_name] = {
                "mean_depth": mean_depth,
                "min_depth": min_depth,
                "std_depth": std_depth,
                "has_obstacle": has_obstacle,
                "is_safe": is_safe,
                "clearance_score": min(mean_depth, 1.0)  # 0-1之间的间隙分数
            }
        
        return environment
    
    def compute_3d_avoidance_strategy(self, environment: Dict, 
                                    desired_velocity: float = 3.0) -> Tuple[np.ndarray, str]:
        """
        计算3D避障策略
        返回: (速度指令, 策略描述)
        """
        
        # 提取关键区域信息
        front_blocked = environment["front_center"]["has_obstacle"]
        left_blocked = environment["left_center"]["has_obstacle"]
        right_blocked = environment["right_center"]["has_obstacle"]
        top_clear = environment["top"]["is_safe"]
        bottom_clear = environment["bottom"]["is_safe"]
        
        # 计算各方向的间隙分数
        front_clearance = environment["front_center"]["clearance_score"]
        left_clearance = environment["left_center"]["clearance_score"]
        right_clearance = environment["right_center"]["clearance_score"]
        top_clearance = environment["top"]["clearance_score"]
        bottom_clearance = environment["bottom"]["clearance_score"]
        
        # 避障决策逻辑
        if not front_blocked and front_clearance > 0.6:
            # 前方畅通 - 继续前进
            action = "forward"
            strategy = "前方畅通，继续前进"
            
        elif not front_blocked and front_clearance > 0.4:
            # 前方有小障碍 - 减速前进
            action = "slow_forward"
            strategy = "前方有小障碍，减速前进"
            
        elif front_blocked:
            # 前方阻塞 - 需要避障
            
            if not left_blocked and not right_blocked:
                # 左右都可以，选择间隙更大的方向
                if left_clearance > right_clearance:
                    action = "left"
                    strategy = "前方阻塞，向左避障"
                else:
                    action = "right" 
                    strategy = "前方阻塞，向右避障"
                    
            elif not left_blocked:
                # 只能向左
                action = "left"
                strategy = "前方和右侧阻塞，向左避障"
                
            elif not right_blocked:
                # 只能向右
                action = "right"
                strategy = "前方和左侧阻塞，向右避障"
                
            else:
                # 左右前方都阻塞 - 考虑垂直避障
                if top_clear and top_clearance > bottom_clearance:
                    if left_clearance > right_clearance:
                        action = "left_up"
                        strategy = "三面阻塞，左上方避障"
                    else:
                        action = "right_up"
                        strategy = "三面阻塞，右上方避障"
                        
                elif bottom_clear:
                    if left_clearance > right_clearance:
                        action = "left_down"
                        strategy = "三面阻塞，左下方避障"
                    else:
                        action = "right_down"
                        strategy = "三面阻塞，右下方避障"
                        
                elif top_clear:
                    action = "up"
                    strategy = "多面阻塞，上升避障"
                    
                else:
                    # 极端情况 - 后退
                    action = "backward"
                    strategy = "四面阻塞，后退重新规划"
        else:
            # 默认保守策略
            action = "slow_forward"
            strategy = "默认保守前进"
        
        # 获取速度指令
        base_velocity = self.avoidance_actions[action]
        scaled_velocity = base_velocity * desired_velocity
        
        return scaled_velocity, strategy
    
    def visualize_avoidance_analysis(self, depth_image: np.ndarray, 
                                   environment: Dict, velocity_cmd: np.ndarray, 
                                   strategy: str):
        """可视化避障分析"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 原始深度图
        axes[0].imshow(depth_image, cmap='viridis')
        axes[0].set_title('深度图像')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 区域分析
        height, width = depth_image.shape
        overlay = np.zeros((height, width, 3))
        
        # 前方区域 (红色)
        overlay[height//3:2*height//3, width//3:2*width//3, 0] = 0.3
        
        # 左侧区域 (绿色)
        overlay[height//3:2*height//3, 0:width//3, 1] = 0.3
        
        # 右侧区域 (蓝色)
        overlay[height//3:2*height//3, 2*width//3:width, 2] = 0.3
        
        # 上下区域 (黄色)
        overlay[0:height//4, :, [0,1]] = 0.2
        overlay[3*height//4:height, :, [0,1]] = 0.2
        
        axes[1].imshow(depth_image, cmap='gray', alpha=0.7)
        axes[1].imshow(overlay, alpha=0.5)
        axes[1].set_title('区域分析')
        
        # 3. 避障决策可视化
        axes[2].bar(['vx', 'vy', 'vz'], velocity_cmd, 
                   color=['red', 'green', 'blue'], alpha=0.7)
        axes[2].set_ylabel('速度 (m/s)')
        axes[2].set_title(f'避障策略: {strategy}')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细分析
        print("\n" + "="*50)
        print("避障环境分析")
        print("="*50)
        
        for region, info in environment.items():
            status = "阻塞" if info["has_obstacle"] else "畅通"
            safety = "安全" if info["is_safe"] else "危险"
            print(f"{region:12}: 深度={info['mean_depth']:.2f}, {status}, {safety}")
            
        print(f"\n避障策略: {strategy}")
        print(f"速度指令: vx={velocity_cmd[0]:.1f}, vy={velocity_cmd[1]:.1f}, vz={velocity_cmd[2]:.1f}")
        print("="*50)


def demonstrate_avoidance_scenarios():
    """演示各种避障场景"""
    
    avoidance = AdvancedObstacleAvoidance()
    
    # 场景1: 前方有障碍，左右畅通
    print("场景1: 前方障碍，左右畅通")
    depth1 = np.ones((60, 90)) * 0.8  # 远距离背景
    depth1[20:40, 30:60] = 0.2        # 前方障碍物
    
    env1 = avoidance.analyze_depth_environment(depth1)
    vel1, strategy1 = avoidance.compute_3d_avoidance_strategy(env1)
    print(f"策略: {strategy1}")
    print(f"速度: {vel1}\n")
    
    # 场景2: 三面阻塞，需要上升
    print("场景2: 三面阻塞，需要上升避障")
    depth2 = np.ones((60, 90)) * 0.8
    depth2[20:40, 25:65] = 0.15       # 前方大障碍
    depth2[20:40, 0:30] = 0.2         # 左侧障碍
    depth2[20:40, 60:90] = 0.2        # 右侧障碍
    
    env2 = avoidance.analyze_depth_environment(depth2)
    vel2, strategy2 = avoidance.compute_3d_avoidance_strategy(env2)
    print(f"策略: {strategy2}")
    print(f"速度: {vel2}\n")
    
    # 场景3: 狭窄通道
    print("场景3: 狭窄通道导航")
    depth3 = np.ones((60, 90)) * 0.8
    depth3[:, 0:35] = 0.2             # 左侧墙壁
    depth3[:, 55:90] = 0.2            # 右侧墙壁
    depth3[0:15, :] = 0.25            # 顶部障碍
    
    env3 = avoidance.analyze_depth_environment(depth3)
    vel3, strategy3 = avoidance.compute_3d_avoidance_strategy(env3)
    print(f"策略: {strategy3}")
    print(f"速度: {vel3}\n")
    
    return [(depth1, env1, vel1, strategy1),
            (depth2, env2, vel2, strategy2), 
            (depth3, env3, vel3, strategy3)]


if __name__ == "__main__":
    print("ViTfly高级避障策略演示")
    print("="*50)
    
    # 演示各种避障场景
    scenarios = demonstrate_avoidance_scenarios()
    
    # 如果有matplotlib，可视化第一个场景
    try:
        avoidance = AdvancedObstacleAvoidance()
        depth, env, vel, strategy = scenarios[0]
        avoidance.visualize_avoidance_analysis(depth, env, vel, strategy)
    except ImportError:
        print("提示: 安装matplotlib可以看到可视化结果")
        print("pip install matplotlib")
    
    print("\nViTfly避障能力总结:")
    print("- ✅ 前后移动 (加速/减速/后退)")
    print("- ✅ 左右避让 (左转/右转)")  
    print("- ✅ 上下避障 (上升/下降)")
    print("- ✅ 组合避障 (左上/右上/左下/右下)")
    print("- ✅ 智能决策 (根据环境选择最佳策略)")
    print("- ✅ 安全后退 (极端情况下的保守策略)")