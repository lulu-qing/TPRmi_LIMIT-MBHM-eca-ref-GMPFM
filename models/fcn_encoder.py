import torch
from torch import nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.se(self.avg_pool(x))
        max_out = self.se(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class ConvWide(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=8):
        super(ConvWide, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class ConvMultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvMultiScale, self).__init__()
        if out_channels % 4 != 0:
            raise ValueError('out_channels should be divisible by 4')
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv1d(in_channels, mid_channels, 1, 1, padding=0)
        self.conv3 = nn.Conv1d(in_channels, mid_channels, 3, 1, padding=1)
        self.conv5 = nn.Conv1d(in_channels, mid_channels, 5, 1, padding=2)
        self.conv7 = nn.Conv1d(in_channels, mid_channels, 7, 1, padding=3)
        self.norm = nn.BatchNorm1d(mid_channels * 3)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(mid_channels * 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        
        x_multi = torch.cat([x3, x5, x7], dim=1)
        x_multi = self.norm(x_multi)
        x_multi = self.relu(x_multi)
        x_multi = self.ca(x_multi) * x_multi
        
        out = torch.cat([x1, x_multi], dim=1)
        return out

class GMPFM(nn.Module):
    def __init__(self, in_channels=128):
        super(GMPFM, self).__init__()
        # 多尺度分支：每个分支输出 32 通道，总共 96 通道
        self.conv_k3 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=1)
        )
        # 这里的 BN 必须对应拼接后的总通道数。
        # 如果你拼接的是 96 (融合) + 32 (原特征的一部分)，则总数是 128
        self.bn = nn.BatchNorm1d(128) 

    def forward(self, x):
        # x: (Batch, 128, L)
        f3 = F.relu(self.conv_k3(x))
        f5 = F.relu(self.conv_k5(x))
        f7 = F.relu(self.conv_k7(x))
        
        weights = self.gate(x).unsqueeze(-1) # (Batch, 3, 1)
        
        # 融合后的特征维度是 (Batch, 32, L)
        fused = weights[:, 0:1, :] * f3 + weights[:, 1:2, :] * f5 + weights[:, 2:3, :] * f7
        
        # 为了凑够 128 维：
        # 融合特征 32 维 + 原始特征 96 维 = 128 维
        out = torch.cat([fused, x[:, 32:, :]], dim=1) 
        
        return self.bn(out)

class FCN_Encoder(nn.Module):
    def __init__(self):
        super(FCN_Encoder, self).__init__()
        # 初始特征提取 [cite: 254]
        self.conv_in = ConvWide(1, 128, kernel_size=8, stride=8) 
        
        # 递进式结构：前两层 MSCAB，第三层替换为 GMPFM [cite: 254]
        self.mscab1 = ConvMultiScale(128, 128)
        self.mscab2 = ConvMultiScale(128, 128)
        self.gmpfm = GMPFM(in_channels=128)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mscab1(x)
        x = self.mscab2(x)
        x = self.gmpfm(x)
        return x
