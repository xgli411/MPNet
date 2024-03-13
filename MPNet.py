from collections import OrderedDict
from functools import partial
from typing import List

import torch
import torch.nn as nn
from thop import profile

from torch.nn import functional as F
# from thop import profile
from torch import Tensor
from timm.models.layers import DropPath


# Multi-branch feature fusion module
class MBFM(nn.Module):

    def __init__(self, dim, in_dim):

        super(MBFM, self).__init__()

        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(nn.Conv2d(dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(dim, down_dim, kernel_size=3, dilation=2, padding=2),
                                   nn.BatchNorm2d(down_dim), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(dim, down_dim, kernel_size=3, dilation=3, padding=3),
                                   nn.BatchNorm2d(down_dim), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(dim, down_dim, kernel_size=3, dilation=5, padding=5),
                                   nn.BatchNorm2d(down_dim), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim),  nn.ReLU())
        self.fuse = nn.Sequential(nn.Conv2d(5 * down_dim, dim, kernel_size=1), nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.interpolate(self.conv5(F.avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)
        x = self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
        return x


class SPA(nn.Module):

    def __init__(self, channel, reduction=8):
        super(SPA, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.weight = nn.Parameter(torch.ones((1, 3, 1, 1, 1)))
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.BatchNorm2d(channel//reduction),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool2(x)
        y4 = self.avg_pool4(x)
        y = torch.cat(
            (y4.unsqueeze(1),
             F.interpolate(y2, scale_factor=2).unsqueeze(1),
             F.interpolate(y1, scale_factor=4).unsqueeze(1)),
            dim=1
        )
        y = (y * self.weight).sum(axis=1, keepdim=False)
        y = self.transform(y)
        y = F.interpolate(y, size=x.shape[2:])
        return x * y


class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                                 stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                                 stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                                 stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3] // 2,
                                 stride=stride, groups=pyconv_groups[3])
        assert planes % min(pyconv_groups) == 0, "planes must be divisible by the minimum value of pyconv_groups"

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 act: str = 'ReLU'):
        super(ConvBNLayer, self).__init__()
        assert act in ('ReLU', 'GELU')
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = getattr(nn, act)()

    # 将卷积层和批量归一化（Batch Normalization）层融合（fuse）在一起，并返回融合后的卷积核权重和偏置项
    # 减少模型中的计算量和内存占用，同时保持模型在训练和推理过程中的一致性。通过融合卷积层和批量归一化层，可以提高模型的执行效率。
    def _fuse_bn_tensor(self) -> None:
        kernel = self.conv.weight
        bias = self.conv.bias if hasattr(self.conv, 'bias') and self.conv.bias is not None else 0
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        self.conv.weight.data = kernel * t
        self.conv.bias = nn.Parameter(beta - (running_mean - bias) * gamma / std, requires_grad=False)
        self.bn = nn.Identity()
        return self.conv.weight.data, self.conv.bias.data

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class PRBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 inner_channels: int = None,
                 kernel_size: int = 1,
                 bias=False,
                 act: str = 'ReLU',
                 drop_path: float = 0.,
                 ):
        super(PRBlock, self).__init__()
        inner_channels = inner_channels or in_channels * 2

        self.conv1 = PyConv4(inplans=in_channels, planes=in_channels,
                             pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 2, 5, 10])
        self.conv2 = ConvBNLayer(in_channels, inner_channels, bias=bias, act=act)
        self.conv3 = nn.Conv2d(inner_channels, in_channels, kernel_size=kernel_size)

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.conv1(x)
        y1 = self.bn(y1)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        out = x + self.drop_path(y3)
        out = self.act(out)
        # return y3
        return out


class MPNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=18,
                 last_channels=1280,
                 inner_channels: list = [40, 80, 160, 320],
                 blocks: list = [1, 2, 8, 2],
                 bias=False,
                 act='GeLU',
                 drop_path=0.,
                 ):
        super(MPNet, self).__init__()

        self.mbfm = MBFM(dim=in_channels, in_dim=16)
        self.stem = ConvBNLayer(in_channels,
                                inner_channels[0],
                                kernel_size=4,
                                stride=4,
                                bias=bias)
        self.spa0 = SPA(channel=inner_channels[0], reduction=8)

        self.stage1 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             PRBlock(inner_channels[0],
                            bias=bias,
                            act=act,
                            drop_path=drop_path)) for idx in range(blocks[0])]))

        self.merging1 = ConvBNLayer(inner_channels[0],
                                    inner_channels[1],
                                    kernel_size=2,
                                    stride=2,
                                    bias=bias)
        self.spa1 = SPA(channel=inner_channels[1], reduction=8)

        self.stage2 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             PRBlock(inner_channels[1],
                            bias=bias,
                            act=act,
                            drop_path=drop_path)) for idx in range(blocks[1])]))

        self.merging2 = ConvBNLayer(inner_channels[1],
                                    inner_channels[2],
                                    kernel_size=2,
                                    stride=2,
                                    bias=bias)
        self.spa2 = SPA(channel=inner_channels[2], reduction=8)

        self.stage3 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             PRBlock(inner_channels[2],
                            bias=bias,
                            act=act,
                            drop_path=drop_path)) for idx in range(blocks[2])]))

        self.merging3 = ConvBNLayer(inner_channels[2],
                                    inner_channels[3],
                                    kernel_size=2,
                                    stride=2,
                                    bias=bias)
        self.spa3 = SPA(channel=inner_channels[3], reduction=8)

        self.stage4 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             PRBlock(inner_channels[3],
                     bias=bias,
                     act=act,
                     drop_path=drop_path)) for idx in range(blocks[3])]))

        self.classifier = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(inner_channels[-1], last_channels, kernel_size=1, bias=False)),
            ('act', getattr(nn, act)()),
            ('flat', nn.Flatten()),
            ('fc', nn.Linear(last_channels, out_channels, bias=True))
        ]))
        self.feature_channels = inner_channels

    def fuse_bn_tensor(self):
        for m in self.modules():
            if isinstance(m, ConvBNLayer):
                m._fuse_bn_tensor()

    def forward_feature(self, x: Tensor) -> List[Tensor]:
        x = self.mbfm(x)
        x1 = self.stage1(self.spa0(self.stem(x)))
        x2 = self.stage2(self.spa1(self.merging1(x1)))
        x3 = self.stage3(self.spa2(self.merging2(x2)))
        x4 = self.stage4(self.spa3(self.merging3(x3)))
        return [x1, x2, x3, x4]

    def forward(self, x: Tensor) -> Tensor:
        _, _, _, x = self.forward_feature(x)
        avg = nn.AdaptiveAvgPool2d(1)
        max = nn.AdaptiveMaxPool2d(1)
        avg_out = avg(x)
        max_out = max(x)
        output = (avg_out + max_out) / 2.0
        x = self.classifier(output)
        return x

MPNetT0 = partial(MPNet, inner_channels=[40, 80, 120, 160], blocks=[2, 2, 2, 2], act='GELU', drop_path=0.02)


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MPNetT0(drop_path=0.).to(device)

    model.eval()
    x = torch.randn((1, 3, 224, 224), device=device)
    model.fuse_bn_tensor()
    y = model(x)

    flops, params = profile(model, inputs=(x,))

    print(y.size())
    print('FLOPs = %.2f G ' % ((flops / 1000 ** 3)))
    print('Params = %.2f M' % (params / 1000 ** 2))
