# 在`mmdet/models/dense_heads`目录下新建一个文件，命名为`custom_head.py`

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
# from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead

from mmdet.registry import MODELS

@MODELS.register_module()
class CustomHead(BaseDenseHead):
    def __init__(self, num_classes, in_channels=256):
        super(CustomHead, self).__init__()
        self.conv_cls = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.conv_reg = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.conv_centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        centerness = []
        for feat in feats:
            cls_scores.append(self.conv_cls(feat))
            bbox_preds.append(self.conv_reg(feat))
            centerness.append(self.conv_centerness(feat))
        return cls_scores, bbox_preds, centerness

    def loss(self, cls_scores, bbox_preds, centerness, gt_bboxes, gt_labels, img_metas, cfg, gt_bboxes_ignore=None):
        # Define your loss computation here
        pass

# Example configuration in mmdet config file
# head=dict(
#     type='CustomHead',
#     num_classes=80,
#     ...
# )
