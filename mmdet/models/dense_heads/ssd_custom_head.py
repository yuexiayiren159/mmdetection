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
        self.pool = nn.AdaptiveMaxPool2d((32, 32))  # 假设我们想要输出大小为 (32, 32)

    def forward(self, feats):
        cls_outputs = []
        bbox_outputs = []

        for feat, cls_pred, bbox_pred in zip(feats, self.cls_preds, self.bbox_preds):
            # 分类输出
            cls_out = cls_pred(feat).permute(0, 2, 3, 1)  # 将 channels 移到最后
            # 边界框输出
            bbox_out = bbox_pred(feat).permute(0, 2, 3, 1)  # 将 channels 移到最后
            
            # 确保输出具有正确的形状 [batch_size, num_anchors * num_classes, H, W]
            cls_out = cls_out.reshape(feat.size(0), -1, feat.size(2), feat.size(3))  # 保留空间维度
            bbox_out = bbox_out.reshape(feat.size(0), -1, feat.size(2), feat.size(3))  # 保留空间维度
            
            # 使用AdaptiveMaxPool2d确保特征图大小一致
            cls_out = self.pool(cls_out)
            bbox_out = self.pool(bbox_out)

            # 打印输出形状，用于调试
            print(f"cls_out形状: {cls_out.shape}")
            print(f"bbox_out形状: {bbox_out.shape}")
            
            cls_outputs.append(cls_out)
            bbox_outputs.append(bbox_out)

        # 将所有尺度的预测结果连接在一起
        cls_outputs = torch.cat(cls_outputs, dim=1)
        bbox_outputs = torch.cat(bbox_outputs, dim=1)

        # 确保批次大小一致
        batch_size = cls_outputs.size(0)  # 获取批次大小
        print(f"批次大小: {batch_size}")  # 确认批次大小

        # 保持批次大小一致，不要改变批次维度
        cls_outputs = cls_outputs.view(batch_size, -1)  # 展平
        bbox_outputs = bbox_outputs.view(batch_size, -1)  # 展平

        # 打印输出形状，用于调试
        print(f"连接后的 cls_outputs 形状: {cls_outputs.shape}")
        print(f"连接后的 bbox_outputs 形状: {bbox_outputs.shape}")

        return cls_outputs, bbox_outputs
