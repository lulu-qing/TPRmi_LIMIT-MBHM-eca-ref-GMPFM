import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.Network import MYNET as Net
import numpy as np
from copy import deepcopy
import math


class WeightECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(WeightECA, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, w):
        y = self.avg_pool(w)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y.transpose(-1, -2))
        return w * y


class ECA_RefConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, map_k=3):
        super(ECA_RefConv1d, self).__init__()
        assert map_k <= kernel_size

        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))

        self.num_kernels = out_channels * in_channels // groups
        G = in_channels * out_channels // (groups ** 2)

        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding if padding is not None else kernel_size // 2
        self.bias = None

        self.convmap1 = nn.Conv1d(
            in_channels=self.num_kernels, out_channels=self.num_kernels,
            kernel_size=map_k, stride=1, padding=map_k // 2, groups=G, bias=False
        )
        self.eca = WeightECA(channels=self.num_kernels)
        self.convmap2 = nn.Conv1d(
            in_channels=self.num_kernels, out_channels=self.num_kernels,
            kernel_size=map_k, stride=1, padding=map_k // 2, groups=G, bias=False
        )

        nn.init.dirac_(self.convmap1.weight)
        # 赋予微小扰动，打破零梯度僵局
        nn.init.normal_(self.convmap2.weight, mean=0.0, std=0.01)

    def get_equivalent_kernel(self):
        origin_weight = self.weight.view(1, self.num_kernels, self.kernel_size)
        w1 = self.convmap1(origin_weight)
        w2 = self.eca(w1)
        w_vi = self.weight + self.convmap2(w2).view(*self.origin_kernel_shape)
        return w_vi

    def forward(self, inputs):
        kernel = self.get_equivalent_kernel()
        return F.conv1d(inputs, kernel, stride=self.stride, padding=self.padding, groups=self.groups, bias=self.bias)


def sample_task_ids(support_label, num_task, num_shot, num_way, num_class):
    basis_matrix = torch.arange(num_shot).long().view(-1, 1).repeat(1, num_way).view(-1) * num_class
    permuted_ids = torch.zeros(num_task, num_shot * num_way).long()
    permuted_labels = []
    for i in range(num_task):
        clsmap = torch.randperm(num_class)[:num_way]
        permuted_labels.append(support_label[clsmap])
        permuted_ids[i, :].copy_(basis_matrix + clsmap.repeat(num_shot))
    return permuted_ids, permuted_labels


def one_hot(indices, depth):
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    return encoded_indicies


class MYNET(Net):
    def __init__(self, args, mode=None):
        super().__init__(args, mode)
        # 🚀 核心修复 1：强制关闭 buggy 的 seminorm，启用纯正余弦相似度
        self.seminorm = False
        self._replace_wide_convs(self.encoder)

    # def _replace_wide_convs(self, module):
    #     for name, child in module.named_children():
    #         k_size = child.kernel_size[0] if isinstance(getattr(child, 'kernel_size', None), tuple) else getattr(child,
    #                                                                                                              'kernel_size',
    #                                                                                                              0)
    #         if isinstance(child, nn.Conv1d) and k_size == 16:
    #             refconv = ECA_RefConv1d(
    #                 in_channels=child.in_channels, out_channels=child.out_channels,
    #                 kernel_size=k_size, stride=child.stride[0] if isinstance(child.stride, tuple) else child.stride,
    #                 padding=child.padding[0] if isinstance(child.padding, tuple) else child.padding, groups=child.groups
    #             )
    #             refconv.weight.copy_(child.weight.data)
    #             if child.bias is not None:
    #                 refconv.bias = nn.Parameter(child.bias.data.clone())
    #             setattr(module, name, refconv)
    #         else:
    #             self._replace_wide_convs(child)

    def _replace_wide_convs(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv1d):
                k_size = child.kernel_size[0] if isinstance(child.kernel_size, tuple) else child.kernel_size
                
                # 🚀 极其精准的打击：只替换那三个感受野为 8 的宽卷积！
                if k_size == 8:
                    refconv = ECA_RefConv1d(
                        in_channels=child.in_channels, out_channels=child.out_channels,
                        kernel_size=k_size, stride=child.stride[0] if isinstance(child.stride, tuple) else child.stride,
                        padding=child.padding[0] if isinstance(child.padding, tuple) else child.padding, groups=child.groups
                    )
                    refconv.weight.copy_(child.weight.data)
                    if child.bias is not None:
                        refconv.bias = nn.Parameter(child.bias.data.clone())
                    setattr(module, name, refconv)
                    # 打印确认
                    print(f"✅ 精准注入 ECA-RefConv 到宽卷积层: {name} (感受野={k_size})")
            else:
                self._replace_wide_convs(child)


    def split_instances(self, support_label, epoch):
        # 🚀 终极修复：不再强制全盘覆盖，而是模拟 2 个增量新类，恢复新旧知识碰撞！
        self.current_way = self.args.meta_new_class
        permuted_ids, permuted_labels = sample_task_ids(support_label, self.args.num_tasks, num_shot=self.args.sample_shot,
                                                        num_way=self.current_way, num_class=self.args.sample_class)
        return (permuted_ids.view(self.args.num_tasks, self.args.sample_shot, self.current_way), torch.stack(permuted_labels))

    def forward(self, x_shot, x_query=None, shot_label=None, epoch=None):
        if self.mode == 'encoder':
            return self.encode(x_shot)
        else:
            return self._forward(self.encode(x_shot), self.encode(x_query), self.split_instances(shot_label, epoch))

    def _forward(self, support, query, index_label):
        support_idx, support_labels = index_label
        num_task = support_idx.shape[0]
        support = support[support_idx.view(-1)].view(*(support_idx.shape + (-1,)))
        proto = support.mean(dim=1)

        num_proto = self.args.num_classes
        query = query.unsqueeze(1)

        logit = []
        for tt in range(num_task):
            global_mask = torch.eye(num_proto).cuda()
            whole_support_index = support_labels[tt, :]
            global_mask[:, whole_support_index] = 0
            local_mask = one_hot(whole_support_index, num_proto)

            current_classifier = torch.mm(self.fc.weight.t(), global_mask) + torch.mm(proto[tt, :].t(), local_mask)
            current_classifier = current_classifier.t().unsqueeze(0).expand(query.shape[0], num_proto, support.size(-1))

            # 🚀 核心修复 2：正确使用温度系数乘法，放大梯度信号！
            logits = F.cosine_similarity(query, current_classifier, dim=-1)
            logits = logits * self.args.temperature
            logit.append(logits)

        logit = torch.cat(logit, 1)
        return logit.view(-1, self.args.num_classes)

    def updateclf(self, data, label):
        support_embs = self.encode(data)
        proto = support_embs.reshape(5, -1, support_embs.shape[-1]).mean(dim=0)
        self.fc.weight.data[torch.min(label):torch.max(label) + 1] = proto

    def forward_many(self, query):
        num_proto = self.args.num_classes
        query = query.view(-1, 1, query.size(-1))
        current_classifier = self.fc.weight.unsqueeze(0).expand(query.shape[0], num_proto, query.size(-1))

        # 🚀 核心修复 3：正确测试环境匹配
        logits = F.cosine_similarity(query, current_classifier, dim=-1)
        logits = logits * self.args.temperature
        return logits