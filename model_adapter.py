"""
模型权重适配器 - 将原始ViTfly权重适配到最小实现
"""

import torch
import torch.nn as nn
import numpy as np
from vit_model import create_minimal_vit_model


def create_dummy_weights():
    """创建虚拟的预训练权重用于测试"""
    model = create_minimal_vit_model()
    
    # 初始化权重为更合理的值
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    # 保存虚拟权重
    torch.save(model.state_dict(), 'vitfly_dummy_weights.pth')
    print("已创建虚拟预训练权重: vitfly_dummy_weights.pth")
    
    return model


def adapt_original_vitfly_weights(original_model_path: str, output_path: str = 'vitfly_adapted_weights.pth'):
    """
    适配原始ViTfly模型权重到我们的最小实现
    
    Args:
        original_model_path: 原始ViTfly模型权重路径
        output_path: 输出适配后权重的路径
    """
    try:
        # 加载原始权重
        original_weights = torch.load(original_model_path, map_location='cpu')
        print(f"原始模型权重键: {list(original_weights.keys())}")
        
        # 创建我们的模型
        our_model = create_minimal_vit_model()
        our_state_dict = our_model.state_dict()
        print(f"我们的模型权重键: {list(our_state_dict.keys())}")
        
        # 权重映射字典
        weight_mapping = {
            # 示例映射 - 需要根据实际的原始模型结构调整
            'patch_embed.proj.weight': 'patch_embed.proj.weight',
            'patch_embed.proj.bias': 'patch_embed.proj.bias',
            'patch_embed.norm.weight': 'patch_embed.norm.weight',
            'patch_embed.norm.bias': 'patch_embed.norm.bias',
            
            # Transformer层映射
            'transformer_layers.0.attn.q_proj.weight': 'encoder_blocks.0.attn.q_proj.weight',
            'transformer_layers.0.attn.kv_proj.weight': 'encoder_blocks.0.attn.kv_proj.weight',
            
            # LSTM映射
            'lstm.weight_ih_l0': 'lstm.weight_ih_l0',
            'lstm.weight_hh_l0': 'lstm.weight_hh_l0',
            'lstm.bias_ih_l0': 'lstm.bias_ih_l0',
            'lstm.bias_hh_l0': 'lstm.bias_hh_l0',
            
            # 输出层映射
            'velocity_head.0.weight': 'nn_fc2.weight',
            'velocity_head.0.bias': 'nn_fc2.bias',
        }
        
        # 适配权重
        adapted_weights = {}
        
        for our_key, our_param in our_state_dict.items():
            if our_key in weight_mapping:
                original_key = weight_mapping[our_key]
                if original_key in original_weights:
                    original_param = original_weights[original_key]
                    
                    # 检查形状兼容性
                    if our_param.shape == original_param.shape:
                        adapted_weights[our_key] = original_param
                        print(f"✅ 成功映射: {our_key} <- {original_key}")
                    else:
                        # 尝试形状适配
                        adapted_param = adapt_parameter_shape(our_param, original_param)
                        if adapted_param is not None:
                            adapted_weights[our_key] = adapted_param
                            print(f"🔄 形状适配: {our_key} {our_param.shape} <- {original_key} {original_param.shape}")
                        else:
                            adapted_weights[our_key] = our_param
                            print(f"❌ 形状不匹配，使用随机初始化: {our_key}")
                else:
                    adapted_weights[our_key] = our_param
                    print(f"❓ 原始模型中未找到对应权重，使用随机初始化: {our_key}")
            else:
                adapted_weights[our_key] = our_param
                print(f"➡️ 使用随机初始化: {our_key}")
        
        # 保存适配后的权重
        torch.save(adapted_weights, output_path)
        print(f"\n✅ 权重适配完成，已保存到: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ 权重适配失败: {e}")
        return None


def adapt_parameter_shape(target_param: torch.Tensor, source_param: torch.Tensor) -> torch.Tensor:
    """尝试适配参数形状"""
    target_shape = target_param.shape
    source_shape = source_param.shape
    
    # 如果维度相同，尝试截取或填充
    if len(target_shape) == len(source_shape):
        adapted = source_param
        
        for dim in range(len(target_shape)):
            if target_shape[dim] != source_shape[dim]:
                if target_shape[dim] < source_shape[dim]:
                    # 截取
                    slices = [slice(None)] * len(target_shape)
                    slices[dim] = slice(0, target_shape[dim])
                    adapted = adapted[tuple(slices)]
                else:
                    # 填充
                    pad_size = target_shape[dim] - source_shape[dim]
                    pad_dims = [0] * (2 * len(target_shape))
                    pad_dims[-(2*dim+1)] = pad_size
                    adapted = torch.nn.functional.pad(adapted, pad_dims)
        
        return adapted
    
    return None


def create_simple_obstacle_avoidance_policy():
    """创建简单的避障策略权重（基于规则的初始化）"""
    model = create_minimal_vit_model()
    
    # 为最终的速度预测层设置更保守的初始权重
    with torch.no_grad():
        # 让模型倾向于输出前进+微调的速度
        if hasattr(model, 'velocity_head'):
            final_layer = model.velocity_head[-1]  # 最后一层
            
            # 前进方向权重更大
            final_layer.weight[0, :] = torch.randn_like(final_layer.weight[0, :]) * 0.5 + 0.5  # vx: 偏向前进
            final_layer.weight[1, :] = torch.randn_like(final_layer.weight[1, :]) * 0.2      # vy: 小幅左右
            final_layer.weight[2, :] = torch.randn_like(final_layer.weight[2, :]) * 0.1      # vz: 轻微升降
            
            # 偏置设置
            final_layer.bias[0] = 0.7   # 默认前进
            final_layer.bias[1] = 0.0   # 不偏向左右
            final_layer.bias[2] = 0.0   # 保持高度
    
    # 保存简单策略权重
    torch.save(model.state_dict(), 'vitfly_simple_policy.pth')
    print("已创建简单避障策略权重: vitfly_simple_policy.pth")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ViTfly模型权重适配器")
    parser.add_argument('--mode', choices=['dummy', 'adapt', 'simple'], default='simple',
                       help='模式: dummy(虚拟权重), adapt(适配原始权重), simple(简单策略)')
    parser.add_argument('--original', type=str, help='原始ViTfly权重路径')
    parser.add_argument('--output', type=str, help='输出权重路径')
    
    args = parser.parse_args()
    
    if args.mode == 'dummy':
        create_dummy_weights()
    elif args.mode == 'adapt':
        if not args.original:
            print("❌ 需要指定原始权重路径: --original <path>")
        else:
            output_path = args.output or 'vitfly_adapted_weights.pth'
            adapt_original_vitfly_weights(args.original, output_path)
    elif args.mode == 'simple':
        create_simple_obstacle_avoidance_policy()
    
    print("\n使用方法:")
    print("python vitfly_main.py --model vitfly_simple_policy.pth")