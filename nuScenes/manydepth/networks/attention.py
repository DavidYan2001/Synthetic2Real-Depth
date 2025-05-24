import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 Channel Attention 机制
class ChannelAttention(nn.Module):
    def __init__(self, img_channels, cost_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 结合图像特征和 cost volume 进行 channel attention
        self.fc1 = nn.Conv2d(img_channels + cost_channels, (img_channels + cost_channels) // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d((img_channels + cost_channels) // reduction, cost_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features, cost_volume):
        # 拼接图像特征和 cost volume
        combined_features = torch.cat((img_features, cost_volume), dim=1)
        
        # 通道维度上的全局平均池化
        avg_pool = torch.mean(combined_features, dim=[2, 3], keepdim=True)  # B * (C+cost_channels) * 1 * 1
        attn = self.fc1(avg_pool)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)
        return cost_volume * attn

# 定义 Spatial Attention 机制
class SpatialAttention(nn.Module):
    def __init__(self, img_channels, cost_channels):
        super(SpatialAttention, self).__init__()
        # 结合图像特征和 cost volume 进行 spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features, cost_volume):
        # 将图像特征与 cost volume 拼接并进行最大池化和平均池化
        combined_features = torch.cat((img_features, cost_volume), dim=1)
        avg_pool = torch.mean(combined_features, dim=1, keepdim=True)
        max_pool, _ = torch.max(combined_features, dim=1, keepdim=True)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        attn = self.conv(combined)
        attn = self.sigmoid(attn)
        return cost_volume * attn

# 调制 Cost Volume 的模块
class CostVolumeModulation(nn.Module):
    def __init__(self, img_channels, cost_channels):
        super(CostVolumeModulation, self).__init__()
        # 初始化 channel 和 spatial attention 模块
        self.channel_attn = ChannelAttention(img_channels, cost_channels)
        self.spatial_attn = SpatialAttention(img_channels, cost_channels)

        # 调制的卷积层
        self.fusion_conv = nn.Conv2d(img_channels + cost_channels, cost_channels, kernel_size=1)

    def forward(self, img_features, cost_volume):
        # Channel Attention 调制
        modulated_cost_volume = self.channel_attn(img_features, cost_volume)
        
        # 拼接图像特征和调制后的 cost volume
        combined_features = torch.cat((img_features, modulated_cost_volume), dim=1)
        
        # 调制后的融合
        fused = self.fusion_conv(combined_features)

        # Spatial Attention 调制
        modulated_cost_volume = self.spatial_attn(img_features, fused)
        
        return modulated_cost_volume


# 使用示例
# 假设 img_features: B*C*H*W, cost_volume: B*D*H*W
img_channels = 64  # 图像特征的通道数
cost_channels = 128  # cost volume 的通道数
B, C, H, W = 4, img_channels, 40, 72  # Batch, Channel, Height, Width
D = cost_channels  # cost volume 的通道数

# 随机初始化 img_features 和 cost_volume
img_features = torch.rand(B, C, H, W)
cost_volume = torch.rand(B, D, H, W)

# 初始化调制模块并进行 forward
modulation_module = CostVolumeModulation(img_channels, cost_channels)
modulated_cost_volume = modulation_module(img_features, cost_volume)

print(modulated_cost_volume.shape)  # 输出的 cost volume shape 应该为 B*D*H*W
