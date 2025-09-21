"""
ViTfly最小实现 - Vision Transformer避障模型
基于原始ViTfly项目的简化版本，实现深度图像到3D速度指令的端到端转换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OverlapPatchEmbedding(nn.Module):
    """重叠补丁嵌入模块"""
    
    def __init__(self, in_channels=1, embed_dim=64, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=stride, 
            padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H', W')
        x = self.proj(x)
        B, C, H, W = x.shape
        
        # 展平空间维度: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x, H, W


class EfficientMultiHeadAttention(nn.Module):
    """高效多头自注意力机制"""
    
    def __init__(self, embed_dim=64, num_heads=2, reduction_ratio=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.reduction_ratio = reduction_ratio
        
        # Query从原始序列生成
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        
        # Key和Value从降维序列生成（减少计算量）
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        
        # 序列降维卷积
        self.sr_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.sr_norm = nn.LayerNorm(embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # Query从原始序列
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Key和Value从降维序列（减少计算复杂度）
        x_sr = x.transpose(1, 2).reshape(B, C, H, W)
        x_sr = self.sr_conv(x_sr).reshape(B, C, -1).transpose(1, 2)
        x_sr = self.sr_norm(x_sr)
        
        kv = self.kv_proj(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # 缩放点积注意力
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        
        return x


class MixFFN(nn.Module):
    """混合前馈网络 - 结合MLP和深度可分离卷积"""
    
    def __init__(self, embed_dim=64, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # MLP扩展
        x = self.fc1(x)
        
        # 深度可分离卷积
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = x.flatten(2).transpose(1, 2)
        
        # MLP压缩
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, embed_dim=64, num_heads=2, reduction_ratio=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EfficientMultiHeadAttention(embed_dim, num_heads, reduction_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MixFFN(embed_dim, embed_dim * mlp_ratio, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, H, W):
        # 自注意力 + 残差连接
        x = x + self.dropout(self.attn(self.norm1(x), H, W))
        
        # 前馈网络 + 残差连接
        x = x + self.dropout(self.mlp(self.norm2(x), H, W))
        
        return x


class MinimalViTObstacleAvoidance(nn.Module):
    """最小ViT避障模型 - 深度图像到3D速度指令"""
    
    def __init__(self, input_size=(60, 90), patch_size=7, embed_dim=64, num_layers=2, 
                 num_heads=2, reduction_ratio=4, lstm_hidden=128):
        super().__init__()
        
        self.input_size = input_size
        self.embed_dim = embed_dim
        
        # 补丁嵌入
        self.patch_embed = OverlapPatchEmbedding(
            in_channels=1, embed_dim=embed_dim, 
            patch_size=patch_size, stride=4, padding=3
        )
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, reduction_ratio)
            for _ in range(num_layers)
        ])
        
        # 特征聚合和解码
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # LSTM时序建模
        self.lstm = nn.LSTM(
            input_size=128 + 1 + 4,  # 视觉特征 + 速度 + 姿态
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # 最终速度预测
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # (vx, vy, vz)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
    def preprocess_inputs(self, depth_image, desired_velocity, quaternion):
        """输入预处理和标准化"""
        # 深度图像调整到标准尺寸
        if depth_image.shape[-2:] != self.input_size:
            depth_image = F.interpolate(
                depth_image, size=self.input_size, 
                mode='bilinear', align_corners=False
            )
        
        # 确保四元数完整性
        if quaternion is None:
            batch_size = depth_image.shape[0]
            quaternion = torch.zeros(batch_size, 4, device=depth_image.device)
            quaternion[:, 0] = 1  # 单位四元数 [1, 0, 0, 0]
            
        # 期望速度归一化
        desired_velocity = desired_velocity / 10.0
        
        return depth_image, desired_velocity, quaternion
        
    def forward(self, depth_image, desired_velocity, quaternion, hidden_state=None):
        """
        前向推理
        Args:
            depth_image: (B, 1, H, W) 深度图像
            desired_velocity: (B, 1) 期望速度
            quaternion: (B, 4) 四元数姿态
            hidden_state: LSTM隐藏状态 (可选)
        Returns:
            velocity_command: (B, 3) 3D速度指令
            hidden_state: 更新的LSTM隐藏状态
        """
        # 输入预处理
        depth_image, desired_velocity, quaternion = self.preprocess_inputs(
            depth_image, desired_velocity, quaternion
        )
        
        # Vision Transformer特征提取
        x, H, W = self.patch_embed(depth_image)
        
        # 逐层Transformer编码
        for layer in self.transformer_layers:
            x = layer(x, H, W)
            
        # 全局特征聚合
        visual_features = self.global_pool(x.transpose(1, 2)).squeeze(-1)  # (B, embed_dim)
        visual_features = self.feature_decoder(visual_features)  # (B, 128)
        
        # 多模态特征融合
        multimodal_input = torch.cat([
            visual_features,    # 视觉特征 (B, 128)
            desired_velocity,   # 期望速度 (B, 1)
            quaternion         # 姿态信息 (B, 4)
        ], dim=-1).unsqueeze(1)  # (B, 1, 133)
        
        # LSTM时序建模
        if hidden_state is not None:
            lstm_out, hidden_state = self.lstm(multimodal_input, hidden_state)
        else:
            lstm_out, hidden_state = self.lstm(multimodal_input)
            
        # 速度指令预测
        velocity_command = self.velocity_head(lstm_out.squeeze(1))  # (B, 3)
        
        return velocity_command, hidden_state
        
    def reset_lstm_state(self, batch_size, device):
        """重置LSTM隐藏状态"""
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        return (h0, c0)


def create_minimal_vit_model(pretrained_weights=None):
    """创建最小ViT避障模型"""
    model = MinimalViTObstacleAvoidance(
        input_size=(60, 90),
        patch_size=7,
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        reduction_ratio=4,
        lstm_hidden=128
    )
    
    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights, map_location='cpu'))
        print(f"已加载预训练权重: {pretrained_weights}")
    
    return model


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_minimal_vit_model().to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向推理
    batch_size = 1
    depth_image = torch.randn(batch_size, 1, 60, 90, device=device)
    desired_velocity = torch.tensor([[5.0]], device=device, dtype=torch.float32)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    
    # 初始推理
    with torch.no_grad():
        velocity_cmd, hidden = model(depth_image, desired_velocity, quaternion)
        print(f"输出速度指令: {velocity_cmd}")
        
        # 后续推理（带隐藏状态）
        velocity_cmd, hidden = model(depth_image, desired_velocity, quaternion, hidden)
        print(f"带状态输出: {velocity_cmd}")