import torch
import torch.nn as nn
import warnings
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.registry import MODELS

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=1, stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,stride=1):
        super(Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        if out_filters != in_filters or stride != 1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=stride,bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        rep = []
        rep.append(SeparableConv2d(in_filters,out_filters,3,stride=stride,padding=1,bias=False))
        rep.append(nn.BatchNorm2d(out_filters))
        rep.append(nn.ReLU(inplace=True))
        rep.append(SeparableConv2d(out_filters,out_filters,3,stride=1,padding=1,bias=False))
        rep.append(nn.BatchNorm2d(out_filters))

        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return self.relu(x)

@MODELS.register_module()
class Xceptionb0(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 widen_factor=1.0,
                 out_indices=(4, 7 ,12),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super(Xceptionb0, self).__init__()

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.widen_factor = widen_factor
        self.out_indices = out_indices
        if not set(out_indices).issubset(set(range(0, 14))):
            raise ValueError('out_indices must be a subset of range'
                             f'(0, 14). But received {out_indices}')

        if frozen_stages not in range(-1, 14):
            raise ValueError('frozen_stages must be in range(-1, 14). '
                             f'But received {frozen_stages}')
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.num_classes = num_classes

        self.layers = []

        # Define the layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.block1 = Block(32,96,stride=2)
        self.block2 = Block(96,96,stride=1)
        self.block3 = Block(96,96,stride=1)
        self.block4 = Block(96,96,stride=1)
        self.block5 = Block(96,192,stride=2)
        self.block6 = Block(192,192,stride=1)
        self.block7 = Block(192,192,stride=1)
        self.block8 = Block(192,192,stride=1)
        self.block9 = Block(192,384,stride=2)
        self.block10 = Block(384,384,stride=1)
        self.block11 = Block(384,384,stride=1)
        self.block12 = Block(384,384,stride=1)


    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.layer_1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        if 4 in self.out_indices:
            outs.append(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        if 7 in self.out_indices:
            outs.append(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        if 12 in self.out_indices:
            outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super(Xceptionb0, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

# # 创建模型实例
# model = Xceptionb0().to('cuda')

# 打印模型结构
# from torchsummary import summary
# summary(model, (3, 224, 224), device='cuda')

# input = torch.randn(1, 3, 416, 736).to('cuda')
# output = model(input)
# print(output[0].shape)
# print(output[1].shape)
# print(output[2].shape)
# torch.Size([1, 96, 52, 92])
# torch.Size([1, 192, 26, 46])
# torch.Size([1, 384, 13, 23])


# 创建模型实例
# model = Xceptionb0()

# 设置模型为训练模式
# model.eval()

# 创建示例输入
# dummy_input = torch.randn(1, 3, 416, 736)
# output = model(dummy_input)
# print(output[0].shape)
# print(output[1].shape)
# print(output[2].shape)


# 导出模型
# torch.onnx.export(
#     model,
#     dummy_input,
#     "xceptionb0.onnx",
#     keep_initializers_as_inputs=False,  # 保留初始值作为输入
#     opset_version=11,                  # ONNX opset 版本
#     input_names=["input"],             # 输入名称
#     output_names=["output"],           # 输出名称
#     dynamic_axes={
#         "input": {0: "batch_size"},    # 动态轴设置
#         "output": {0: "batch_size"}
#     },
#     export_params=True,                # 导出所有参数
#     training=2
# )

# from onnxsim import simplify
# import onnx
# onnx_model = onnx.load("xceptionb0.onnx")
# model_sim, check = simplify(onnx_model,skip_fuse_bn=True)
# onnx.save(model_sim, "xceptionb0_sim.onnx")
# print("finished exporting model to onnx!")