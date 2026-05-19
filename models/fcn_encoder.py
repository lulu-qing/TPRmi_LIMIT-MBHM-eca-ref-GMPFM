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
        out_channels = out_channels // 4
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, 4, padding=0)
        self.conv3 = nn.Conv1d(in_channels, out_channels, 3, 4, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, 5, 4, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, 7, 4, padding=3)
        self.norm = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(out_channels * 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.norm(x)
        x = self.relu(x)
        x = self.ca(x) * x
        x = torch.cat([x1, x], dim=1)
        return x

class GMPFM(nn.Module):
    """
    Gated Multi-Scale Prototype Fusion Mechanism (GMPFM)
    用于替换第三个 MSCAB 模块，实现多尺度特征的自适应门控融合与协同双注意力增强。
    """
    def __init__(self, in_channels, out_channels, temperature=1.0):
        super(GMPFM, self).__init__()
        if out_channels % 4 != 0:
            raise ValueError('out_channels should be divisible by 4')
        branch_channels = out_channels // 4
        
        # 原有残差分支 (k=1)
        self.conv1 = nn.Conv1d(in_channels, branch_channels, 1, 4, padding=0)
        
        # 多尺度特征提取 F_high, F_mid, F_low (k=3, 5, 7)
        self.conv3 = nn.Conv1d(in_channels, branch_channels, 3, 4, padding=1)
        self.norm3 = nn.BatchNorm1d(branch_channels)
        
        self.conv5 = nn.Conv1d(in_channels, branch_channels, 5, 4, padding=2)
        self.norm5 = nn.BatchNorm1d(branch_channels)
        
        self.conv7 = nn.Conv1d(in_channels, branch_channels, 7, 4, padding=3)
        self.norm7 = nn.BatchNorm1d(branch_channels)
        
        self.relu = nn.ReLU()
        
        # 门控网络 Gate Network
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(branch_channels * 3, branch_channels * 3 // 2)
        self.fc2 = nn.Linear(branch_channels * 3 // 2, 3)
        self.temperature = temperature
        
        # 通道注意力 Channel Attention
        self.ca = ChannelAttention(branch_channels * 3)

    def forward(self, x, phase=1):
        # 1. 提取残差分支特征
        x1 = self.conv1(x)
        
        # 2. 提取多尺度特征
        x3 = self.relu(self.norm3(self.conv3(x)))
        x5 = self.relu(self.norm5(self.conv5(x)))
        x7 = self.relu(self.norm7(self.conv7(x)))
        
        # 拼接多尺度特征 F_in
        f_in = torch.cat([x3, x5, x7], dim=1)
        
        # 3. 门控网络计算尺度权重 \lambda
        s = self.gap(f_in).squeeze(-1)
        z = self.fc2(self.relu(self.fc1(s)))
        
        # 阶段依赖的归一化策略: phase=0 (如预训练阶段) 时权重全为1，其他阶段使用 softmax 动态加权
        if phase == 0:
            lam = torch.ones_like(z).to(z.device)
        else:
            lam = F.softmax(z / self.temperature, dim=-1)
            
        lam3 = lam[:, 0].unsqueeze(-1).unsqueeze(-1)
        lam5 = lam[:, 1].unsqueeze(-1).unsqueeze(-1)
        lam7 = lam[:, 2].unsqueeze(-1).unsqueeze(-1)
        
        # 4. 协同双注意力融合
        f_scaled = torch.cat([x3 * lam3, x5 * lam5, x7 * lam7], dim=1)
        w_ch = self.ca(f_in)
        f_fused = f_scaled * w_ch
        
        # 5. 输出：拼接残差分支得到最终增强特征 F_out
        out = torch.cat([x1, f_fused], dim=1)
        return out

class FCN_Encoder(nn.Module):
    def __init__(self):
        super(FCN_Encoder, self).__init__()
        # 将 BearLLM 中针对双信号拼接的 3 分支输入简化为单分支输入
        # 1 个输入通道，128 个输出通道 (替代原本的 60+8+60)
        self.conv_in = ConvWide(1, 128, kernel_size=8, stride=8) 
        self.conv = nn.Sequential(
            ConvMultiScale(128, 128),  # 第 1 个 MSCAB
            ConvMultiScale(128, 128),  # 第 2 个 MSCAB
            GMPFM(128, 128)            # 第 3 个替换为门控多尺度原型融合机制 (GMPFM)
        )

    def forward(self, x):
        # 期待的输入 x 维度为: (Batch, 1, Length)
        x = self.conv_in(x)
        x = self.conv(x)
        return x
