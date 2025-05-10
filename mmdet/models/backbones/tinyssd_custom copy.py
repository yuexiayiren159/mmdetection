# my_tiny_ssd.py

from mmengine.model import BaseModule
import torch.nn as nn


from mmdet.registry import MODELS

# 1. TinySSD Backbone
@MODELS.register_module()
class TinySSD_Custom(BaseModule):
    def __init__(self, out_channels=[64, 128, 128, 128, 128], init_cfg=None):
        super().__init__(init_cfg)
        self.out_channels = out_channels
        self.blocks = nn.ModuleList()
        self._make_layers()

    def _make_layers(self):
        num_filters = [3, 16, 32, 64]
        self.blocks.append(self._make_downsample(num_filters[0], num_filters[1]))
        self.blocks.append(self._make_downsample(num_filters[1], num_filters[2]))
        self.blocks.append(self._make_downsample(num_filters[2], num_filters[3]))
        self.blocks.append(self._make_downsample(64, 128))
        self.blocks.append(self._make_downsample(128, 128))
        self.blocks.append(self._make_downsample(128, 128))
        self.blocks.append(nn.AdaptiveMaxPool2d((1,1)))

    def _make_downsample(self, in_c, out_c):
        layers = []
        for _ in range(2):
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in [2,3,4,5,6]:  # 取5个尺度的特征
                outputs.append(x)
        return tuple(outputs)


