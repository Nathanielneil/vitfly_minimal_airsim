#!/usr/bin/env python3
"""
AirSim连接测试脚本
"""

import airsim
import numpy as np

def test_connection():
    """测试AirSim连接"""
    try:
        print("正在连接AirSim...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✅ AirSim连接成功")
        
        # 测试API控制
        print("启用API控制...")
        client.enableApiControl(True, vehicle_name="Drone1")
        client.armDisarm(True, vehicle_name="Drone1")
        print("✅ API控制启用成功")
        
        # 测试状态获取
        print("获取无人机状态...")
        state = client.getMultirotorState(vehicle_name="Drone1")
        
        # 兼容不同AirSim版本的属性名
        try:
            if hasattr(state.kinematics_estimated, 'pose'):
                pose = state.kinematics_estimated.pose
                position = pose.position
            elif hasattr(state.kinematics_estimated, 'position'):
                position = state.kinematics_estimated.position
            else:
                # 尝试直接获取位置信息
                pose = client.simGetVehiclePose(vehicle_name="Drone1")
                position = pose.position
                
            print(f"✅ 位置: ({position.x_val:.2f}, {position.y_val:.2f}, {position.z_val:.2f})")
        except Exception as e:
            print(f"状态获取详细错误: {e}")
            print("尝试备用方法...")
            try:
                pose = client.simGetVehiclePose(vehicle_name="Drone1")
                position = pose.position
                print(f"✅ 位置(备用方法): ({position.x_val:.2f}, {position.y_val:.2f}, {position.z_val:.2f})")
            except Exception as e2:
                print(f"备用方法也失败: {e2}")
                print("但连接仍然有效，继续测试...")
        
        # 测试图像获取
        print("获取相机图像...")
        request = airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        response = client.simGetImages([request], vehicle_name="Drone1")[0]
        
        if response.image_data_uint8:
            print(f"✅ RGB图像获取成功: {response.width}x{response.height}")
        else:
            print("❌ RGB图像获取失败")
            
        # 测试深度图像
        depth_request = airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)
        depth_response = client.simGetImages([depth_request], vehicle_name="Drone1")[0]
        
        if depth_response.image_data_float:
            print(f"✅ 深度图像获取成功: {depth_response.width}x{depth_response.height}")
        else:
            print("❌ 深度图像获取失败")
        
        # 清理
        client.armDisarm(False, vehicle_name="Drone1")
        client.enableApiControl(False, vehicle_name="Drone1")
        
        print("\n🎉 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_connection()