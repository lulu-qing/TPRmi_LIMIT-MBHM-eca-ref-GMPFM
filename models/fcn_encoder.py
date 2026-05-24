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
    # 增加 stride 参数，默认设为 4 (如果你希望降维更快，可以设为 2)
    def __init__(self, in_channels, out_channels, stride=4):
        super(ConvMultiScale, self).__init__()
        if out_channels % 4 != 0:
            raise ValueError('out_channels should be divisible by 4')
        mid_channels = out_channels // 4
        
        # 将 stride 应用到所有卷积分支
        self.conv1 = nn.Conv1d(in_channels, mid_channels, 1, stride=stride, padding=0)
        self.conv3 = nn.Conv1d(in_channels, mid_channels, 3, stride=stride, padding=1)
        self.conv5 = nn.Conv1d(in_channels, mid_channels, 5, stride=stride, padding=2)
        self.conv7 = nn.Conv1d(in_channels, mid_channels, 7, stride=stride, padding=3)
        
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
    def __init__(self, in_channels=128, stride=1): # 默认 stride=1
        super(GMPFM, self).__init__()
        self.stride = stride
        
        # 卷积分支
        self.conv_k3 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=stride, padding=1)
        self.conv_k5 = nn.Conv1d(in_channels, 32, kernel_size=5, stride=stride, padding=2)
        self.conv_k7 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=stride, padding=3)
        
        
        
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
        # f3, f5, f7 经过了卷积层，长度由 stride 决定
        f3 = F.relu(self.conv_k3(x))
        f5 = F.relu(self.conv_k5(x))
        f7 = F.relu(self.conv_k7(x))
        
        # gate_weights 计算的是全局上下文，不需要考虑 stride
        gate_weights = self.gate(x) 
        self.last_lambda = gate_weights
        weights = gate_weights.unsqueeze(-1) 
        
        # 融合逻辑
        fused = weights[:, 0:1, :] * f3 + weights[:, 1:2, :] * f5 + weights[:, 2:3, :] * f7
        
        # ========================================================
        # ⚠️ 维度匹配的核心点：
        # 如果 fused 因为 stride > 1 变短了，这里的 x[:, 32:, :] 也必须同样变短
        # ========================================================
        if self.stride > 1:
            # 同样使用和卷积层一样的下采样方式 (这里用 MaxPool1d 模拟)
            residual = F.max_pool1d(x[:, 32:, :], kernel_size=self.stride, stride=self.stride)
        else:
            residual = x[:, 32:, :]
            
        out = torch.cat([fused, residual], dim=1) 
        
        return self.bn(out)

class FCN_Encoder(nn.Module):
    def __init__(self):
        super(FCN_Encoder, self).__init__()
        self.conv_in = ConvWide(1, 128, kernel_size=8, stride=8) 
        
        # 每层都进行 4 倍下采样
        self.mscab1 = ConvMultiScale(128, 128, stride=4)
        self.mscab2 = ConvMultiScale(128, 128, stride=4)
        # 如果你希望在 GMPFM 也下采样，可以仿照上面的逻辑修改 GMPFM 类
        self.gmpfm = GMPFM(in_channels=128, stride=4) # 显式传入 stride=4 

    def forward(self, x):
        x = self.conv_in(x)
       # print(f"Shape after conv_in: {x.shape}") # 应该输出 [B, 128, 3000]
        x = self.mscab1(x)
       # print(f"Shape after mscab1: {x.shape}")  # 应该输出 [B, 128, 750]
        x = self.mscab2(x)
       # print(f"Shape after mscab2: {x.shape}")  # 应该输出 [B, 128, 188]
        x = self.gmpfm(x)
       # print(f"Shape after gmpfm: {x.shape}")   # 应该输出 [B, 128, 47]
        return x
