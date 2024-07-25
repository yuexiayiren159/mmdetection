import torch
import torch.nn as nn
from torchsummary import summary
from mmcv.cnn import ConvModule

class DetectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(DetectionBlock, self).__init__()
        double_out_channels = out_channels * 2
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels, out_channels, 1, **cfg)
        self.conv2 = ConvModule(out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv3 = ConvModule(double_out_channels, out_channels, 1, **cfg)
        self.conv4 = ConvModule(out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv5 = ConvModule(double_out_channels, out_channels, 1, **cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class YOLOV3Neck(nn.Module):
    def __init__(self, num_scales, in_channels, out_channels, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(YOLOV3Neck, self).__init__()
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.detect1 = DetectionBlock(in_channels[0], out_channels[0], **cfg)
        for i in range(1, self.num_scales):
            in_c, out_c = self.in_channels[i], self.out_channels[i]
            inter_c = out_channels[i - 1]
            self.add_module(f'conv{i}', ConvModule(inter_c, out_c, 1, **cfg))
            self.add_module(f'detect{i+1}', DetectionBlock(in_c + out_c, out_c, **cfg))

    def forward(self, feats):
        assert len(feats) == self.num_scales
        outs = []
        out = self.detect1(feats[-1])
        outs.append(out)
        for i, x in enumerate(reversed(feats[:-1])):
            conv = getattr(self, f'conv{i+1}')
            tmp = conv(out)
            tmp = nn.functional.interpolate(tmp, scale_factor=2)
            tmp = torch.cat((tmp, x), 1)
            detect = getattr(self, f'detect{i+2}')
            out = detect(tmp)
            outs.append(out)
        return tuple(outs)

# 创建网络实例
num_scales = 3
in_channels = [320, 96, 32]
out_channels = [96, 96, 96]

model = YOLOV3Neck(num_scales, in_channels, out_channels)

# 使用 torchsummary 打印模型结构
input_data = [(in_channels[0], 32, 32), (in_channels[1], 32, 32), (in_channels[2], 32, 32)]
print(summary(model, input_size=input_data))
