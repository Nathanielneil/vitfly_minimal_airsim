#!/usr/bin/env python3
"""
模型权重适配器 - 生成简单的策略权重
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from pathlib import Path

from vit_model import create_minimal_vit_model


def create_simple_policy_weights(model, output_path="vitfly_simple_policy.pth"):
    """创建简单的避障策略权重"""
    print("创建简单避障策略权重...")
    
    # 初始化权重
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'linear' in name.lower() or 'fc' in name.lower():
                    # 线性层使用Xavier初始化
                    nn.init.xavier_uniform_(param)
                elif 'conv' in name.lower():
                    # 卷积层使用He初始化
                    nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                else:
                    # 其他层使用正态分布
                    nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    # 调整最后一层的权重，使其倾向于前进
    if hasattr(model, 'velocity_predictor'):
        with torch.no_grad():
            # 设置偏向前进的偏置
            if hasattr(model.velocity_predictor, 'bias') and model.velocity_predictor.bias is not None:
                model.velocity_predictor.bias.data[0] = 0.5  # 前进偏置
                model.velocity_predictor.bias.data[1] = 0.0  # 左右平衡
                model.velocity_predictor.bias.data[2] = 0.0  # 上下平衡
    
    # 保存模型权重
    torch.save(model.state_dict(), output_path)
    print(f"简单策略权重已保存: {output_path}")
    return True


def create_dummy_trained_weights(model, output_path="vitfly_dummy_trained.pth"):
    """创建虚拟训练权重（用于测试）"""
    print("创建虚拟训练权重...")
    
    # 加载简单策略权重作为基础
    create_simple_policy_weights(model, "temp_simple.pth")
    model.load_state_dict(torch.load("temp_simple.pth", map_location='cpu'))
    
    # 添加一些随机扰动模拟训练效果
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * 0.01  # 小幅随机扰动
            param.add_(noise)
    
    # 保存权重
    torch.save(model.state_dict(), output_path)
    print(f"虚拟训练权重已保存: {output_path}")
    
    # 清理临时文件
    try:
        Path("temp_simple.pth").unlink()
    except:
        pass
    
    return True


def adapt_original_weights(original_path, model, output_path="vitfly_adapted.pth"):
    """适配原始ViTfly权重"""
    print(f"适配原始权重: {original_path}")
    
    try:
        # 加载原始权重
        original_weights = torch.load(original_path, map_location='cpu')
        
        # 获取模型的状态字典
        model_state = model.state_dict()
        
        # 尝试匹配和适配权重
        adapted_state = {}
        
        for name, param in model_state.items():
            if name in original_weights:
                original_param = original_weights[name]
                if param.shape == original_param.shape:
                    adapted_state[name] = original_param
                    print(f"✓ 匹配权重: {name}")
                else:
                    print(f"✗ 形状不匹配: {name} - 模型: {param.shape}, 原始: {original_param.shape}")
                    adapted_state[name] = param  # 使用模型默认权重
            else:
                print(f"✗ 未找到权重: {name}")
                adapted_state[name] = param  # 使用模型默认权重
        
        # 加载适配后的权重
        model.load_state_dict(adapted_state)
        
        # 保存适配后的权重
        torch.save(model.state_dict(), output_path)
        print(f"适配权重已保存: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"权重适配失败: {e}")
        print("生成简单策略权重作为备选...")
        return create_simple_policy_weights(model, output_path)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ViTfly模型权重适配器")
    parser.add_argument('--mode', choices=['simple', 'dummy', 'adapt'], default='simple',
                       help='权重生成模式: simple(简单策略), dummy(虚拟训练), adapt(适配原始)')
    parser.add_argument('--original', type=str, help='原始权重文件路径 (adapt模式)')
    parser.add_argument('--output', type=str, help='输出权重文件路径')
    
    args = parser.parse_args()
    
    # 创建模型
    try:
        model = create_minimal_vit_model()
        print(f"模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"模型创建失败: {e}")
        return False
    
    # 根据模式生成权重
    success = False
    
    if args.mode == 'simple':
        output_path = args.output or "vitfly_simple_policy.pth"
        success = create_simple_policy_weights(model, output_path)
        
    elif args.mode == 'dummy':
        output_path = args.output or "vitfly_dummy_trained.pth"
        success = create_dummy_trained_weights(model, output_path)
        
    elif args.mode == 'adapt':
        if not args.original:
            print("adapt模式需要指定 --original 参数")
            return False
        if not Path(args.original).exists():
            print(f"原始权重文件不存在: {args.original}")
            return False
        output_path = args.output or "vitfly_adapted.pth"
        success = adapt_original_weights(args.original, model, output_path)
    
    if success:
        print("✓ 权重生成/适配完成")
        
        # 验证权重文件
        try:
            test_model = create_minimal_vit_model()
            test_model.load_state_dict(torch.load(output_path, map_location='cpu'))
            print("✓ 权重文件验证通过")
        except Exception as e:
            print(f"✗ 权重文件验证失败: {e}")
            return False
            
    else:
        print("✗ 权重生成/适配失败")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)