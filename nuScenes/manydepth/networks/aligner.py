import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AlignModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AlignModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))
    

class AlignModule_Large(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AlignModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # 输入特征
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 进一步特征提取
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x