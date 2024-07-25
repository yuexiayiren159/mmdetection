# 在`mmdet/models/necks`目录下新建一个文件，命名为`custom_neck.py`

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS

@MODELS.register_module()
class CustomNeck(nn.Module):
    def __init__(self):
        super(CustomNeck, self).__init__()
        self.fpn_convs = nn.ModuleList()
        for in_channels in [128, 256, 512]:
            l_conv = nn.Conv2d(in_channels, 256, kernel_size=1)
            self.fpn_convs.append(l_conv)
        self.lateral_convs = nn.ModuleList()
        for in_channels in [256, 256, 256]:
            l_conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)

    def forward(self, inputs):
        assert len(inputs) == 3
        laterals = [l_conv(inputs[i]) for i, l_conv in enumerate(self.fpn_convs)]
        
        used_backbone_levels = len(laterals)
        
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')
        
        outs = [self.lateral_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        return tuple(outs)

# Example configuration in mmdet config file
# neck=dict(
#     type='CustomNeck',
#     ...
# )
