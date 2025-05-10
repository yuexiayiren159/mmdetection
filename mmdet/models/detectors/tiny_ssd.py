# tiny_ssd.py (确保使用这个简化版本！)
from mmdet.registry import MODELS
from mmdet.models.detectors.single_stage import SingleStageDetector

@MODELS.register_module()
class TinySSDDetector(SingleStageDetector):
    """
    自定义的 SSD 检测器 - 简化版。
    完全依赖父类 SingleStageDetector 的标准行为。
    """
    def __init__(self,
                 backbone,
                 bbox_head,
                 neck=None,
                 train_cfg=None, # detector 级别的 train_cfg (通常为 None 或用于特定 hook)
                 test_cfg=None,  # detector 级别的 test_cfg
                 data_preprocessor=None,
                 init_cfg=None):
        # 直接调用父类的 __init__ 方法
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg, # 传递 detector 级别的 train_cfg
            test_cfg=test_cfg,   # 传递 detector 级别的 test_cfg
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )
    # --- 不重写 extract_feat, forward, loss, predict ---