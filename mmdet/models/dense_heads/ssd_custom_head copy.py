import torch
import torch.nn as nn
from mmdet.registry import MODELS

@MODELS.register_module()
class TinySSDHead(nn.Module):
    def __init__(self,
                 num_classes,
                 num_anchors=4,
                 in_channels=(16, 32, 64),
                 **kwargs):
        super(TinySSDHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_channels = in_channels

        # 分类预测器
        self.cls_preds = nn.ModuleList()
        # 边界框预测器
        self.bbox_preds = nn.ModuleList()

        for channels in in_channels:
            self.cls_preds.append(
                nn.Conv2d(channels, num_anchors * (num_classes + 1), kernel_size=3, padding=1)
            )
            self.bbox_preds.append(
                nn.Conv2d(channels, num_anchors * 4, kernel_size=3, padding=1)
            )
        
        # 使用AdaptiveMaxPool2d来确保所有尺度的特征图大小一致
        # self.pool = nn.AdaptiveMaxPool2d((32, 32))  # 假设我们想要输出大小为 (32, 32)

    def forward(self, feats):
        cls_outputs = []
        bbox_outputs = []

        for feat, cls_pred, bbox_pred in zip(feats, self.cls_preds, self.bbox_preds):
            cls_outputs.append(cls_pred(feat))
            bbox_outputs.append(bbox_pred(feat))

        # 将所有尺度的预测结果连接在一起
        cls_outputs = torch.cat([torch.flatten(p.permute(0,2,3,1), start_dim=1) for p in cls_outputs], dim=1)
        bbox_outputs = torch.cat([torch.flatten(p.permute(0,2,3,1), start_dim=1) for p in bbox_outputs], dim=1)

        return cls_outputs, bbox_outputs
