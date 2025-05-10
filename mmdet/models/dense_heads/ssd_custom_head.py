# ssd_custom_head.py
from mmdet.registry import MODELS, TASK_UTILS # 添加 TASK_UTILS
from mmdet.models.dense_heads import SSDHead
from mmdet.utils import ConfigType # 用于类型提示
import torch.nn as nn # 如果你需要覆盖 _init_layers
from typing import List, Tuple # 用于类型提示
import torch # 用于类型提示
from mmdeploy.core import mark # 导入 mark 函数

# print("DEBUG: LOADING ssd_custom_head.py - VERSION <YOUR_NEW_UNIQUE_MARKER_HERE>")

@MODELS.register_module()
class TinySSDHead(SSDHead):
    def __init__(self,
                 # 从 kwargs 中显式取出 AnchorHead 需要的损失参数
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss', use_sigmoid=False, # SSD 通常 use_sigmoid=False
                     reduction='sum', loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='SmoothL1Loss', beta=1.0,
                     reduction='sum', loss_weight=1.0),
                 **kwargs): # 其他参数（如 num_classes, in_channels, anchor_generator, bbox_coder, train_cfg, test_cfg 等）仍在 kwargs 中

        # print(f"DEBUG: TinySSDHead __init__ called. Received explicit loss_cls: {loss_cls}")
        # print(f"DEBUG: TinySSDHead __init__ called. Received explicit loss_bbox: {loss_bbox}")
        # print(f"DEBUG: TinySSDHead __init__ called. Received other kwargs keys: {list(kwargs.keys())}")

        # 调用父类 SSDHead 的 __init__，kwargs 中现在不应再包含 loss_cls, loss_bbox
        # 因为 SSDHead 的 __init__ 签名中没有它们
        # train_cfg 和 test_cfg 是 SSDHead __init__ 的参数，所以它们应该在 kwargs 中并被传递
        super().__init__(**kwargs)
        # print("DEBUG: TinySSDHead super().__init__(**kwargs) for SSDHead COMPLETED.")

        # 手动构建并覆盖 self.loss_cls 和 self.loss_bbox
        # 这样做是因为我们假设 SSDHead 没有正确地将这些传递给 AnchorHead
        # 或者 SSDHead 的 __init__ 签名中不包含它们。
        # 我们现在直接使用 AnchorHead 初始化损失函数的方式。
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        # print(f"DEBUG: TinySSDHead self.loss_cls explicitly built: {self.loss_cls}")
        # print(f"DEBUG: TinySSDHead self.loss_bbox explicitly built: {self.loss_bbox}")

        # 可选：如果你需要自定义卷积层，可以在这里覆盖 _init_layers
        # self._init_my_custom_layers() # 如果你需要

    # def forward(self, x: Tuple[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    #     """
    #     重写 forward 方法以标记输出，用于 MMDeploy 导出原始多层级输出。
    #     """
    #     # 调用父类 (AnchorHead) 的 forward 方法
    #     cls_scores, bbox_preds = super().forward(x)

    #     # 标记每个层级的 cls_score 和 bbox_pred
    #     # 这些标记名需要与 MMDeploy 部署配置文件中的 output_names 对应
    #     marked_cls_scores = []
    #     for i, score in enumerate(cls_scores):
    #         marked_cls_scores.append(mark(f'cls_score_{i}', score))
        
    #     marked_bbox_preds = []
    #     for i, pred in enumerate(bbox_preds):
    #         marked_bbox_preds.append(mark(f'bbox_pred_{i}', pred))
            
    #     return tuple(marked_cls_scores), tuple(marked_bbox_preds)
