import torch
from mmdet.registry import MODELS
from mmdet.models.detectors.single_stage import SingleStageDetector

@MODELS.register_module()
class TinySSDDetector(SingleStageDetector):
    def __init__(self, 
                 backbone, 
                 bbox_head, 
                 neck=None,
                 train_cfg=None, 
                 test_cfg=None, 
                 data_preprocessor=None,
                 init_cfg=None):
        super(TinySSDDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )

    # def extract_feat(self, inputs):
    #     """Extract features."""
    #     return self.backbone(inputs)
    def extract_feat(self, inputs):
        """Extract features."""
        if isinstance(inputs, (list, tuple)):  # 新加的判断
            inputs = inputs[0]
        if inputs.dim() == 3:  # (C, H, W)
            inputs = inputs.unsqueeze(0)  # 变成 (1, C, H, W)
        
        # ⭐ 关键！强制转float
        if inputs.dtype != torch.float32:
            inputs = inputs.float()  # 变成 float32
        
        return self.backbone(inputs)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        x = self.extract_feat(inputs)
        cls_preds, bbox_preds = self.bbox_head(x)

        if mode == 'tensor':
            return cls_preds, bbox_preds
        elif mode == 'predict':
            # TODO: 这里可以加推理后处理，比如nms等
            pass
        elif mode == 'loss':
            # 输入 data_samples 中含有 gt标签
            losses = self.loss(cls_preds, bbox_preds, data_samples)
            return losses
        else:
            raise RuntimeError(f'Invalid mode {mode}')

    def loss(self, cls_preds, bbox_preds, data_samples):
        # 这里 data_samples 需要自己组织成 gt_labels 和 gt_bboxes
        # gt_labels = torch.cat([ds.gt_labels for ds in data_samples], dim=0)
        # gt_bboxes = torch.cat([ds.gt_bboxes for ds in data_samples], dim=0)
        '''
        在 OpenMMLab 2.x/3.x 框架里面，DetDataSample 正确的写法应该是：
        gt_instances.labels（标签）
        gt_instances.bboxes（边界框）
        不是直接 .gt_labels！而是 .gt_instances.labels！
        '''
        gt_labels = torch.cat([ds.gt_instances.labels for ds in data_samples], dim=0)
        # gt_bboxes = torch.cat([ds.gt_instances.bboxes for ds in data_samples], dim=0)
        '''
        你在拿 gt_instances.bboxes 的时候，拿到的是 HorizontalBoxes 这种对象，不是 Tensor。
        HorizontalBoxes 是 MMDetection 的一个封装，它里面包着 Tensor。
        要想拼接（cat）成 Tensor，你必须先取出里面的 tensor！
        '''
        gt_bboxes = torch.cat([ds.gt_instances.bboxes.tensor for ds in data_samples], dim=0)

        cls_loss = torch.nn.functional.cross_entropy(
            cls_preds.reshape(-1, cls_preds.shape[-1]),
            gt_labels
        )

        bbox_loss = torch.nn.functional.l1_loss(
            bbox_preds.reshape(-1, 4),
            gt_bboxes,
            reduction='mean'
        )

        return dict(loss_cls=cls_loss, loss_bbox=bbox_loss)
