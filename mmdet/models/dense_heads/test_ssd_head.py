# test_head.py
# test_ssd_head.py
import torch
from mmengine.config import Config
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample # DetDataSample 依然从这里导入
from mmengine.structures import InstanceData # <--- InstanceData 从 mmengine.structures 导入
import mmengine # 用于执行 custom_imports

def get_dummy_input_and_features(cfg, device='cpu'):
    """生成模拟的输入和来自Backbone的特征图"""
    img_size = cfg.model.bbox_head.anchor_generator.get('input_size', None)
    if img_size is None: # 尝试从其他地方获取，或使用默认值
        # 尝试从数据预处理中获取
        if hasattr(cfg, 'train_pipeline'):
            for transform in cfg.train_pipeline:
                if transform['type'] == 'Resize' and 'scale' in transform:
                    img_size = transform['scale']
                    if isinstance(img_size, tuple) and len(img_size) == 2 and isinstance(img_size[0], int):
                        img_size = img_size[0] # 取较小边或假设方形
                    break
        if img_size is None:
            img_size = 256 # 默认值
            print(f"Warning: Could not determine input_size for anchor_generator, using default {img_size}")

    # 1. 模拟 Backbone
    # 我们需要知道 backbone 输出特征图的通道数和下采样率 (strides)
    # 这些应该与 head 的 in_channels 和 anchor_generator.strides 匹配
    backbone_out_channels = cfg.model.bbox_head.in_channels
    strides = cfg.model.bbox_head.anchor_generator.strides

    dummy_feats = []
    current_size = img_size
    for i in range(len(backbone_out_channels)):
        # 根据 stride 计算特征图的空间尺寸
        # 注意：实际的 stride 应用可能更复杂，这里是简化模拟
        spatial_size = current_size // strides[i] if strides[i] > 0 else current_size # 简单处理 stride
        feat_map = torch.randn(1, backbone_out_channels[i], spatial_size, spatial_size, device=device)
        dummy_feats.append(feat_map)
        # current_size = spatial_size # 下一个特征图基于当前特征图尺寸计算 (可选的简化)

    return tuple(dummy_feats), img_size

def get_dummy_data_samples(img_shape_hw, num_classes, device='cpu', batch_size=1): # 参数名改为 img_shape_hw
    """生成模拟的 data_samples，包含 pad_shape"""
    data_samples = []
    img_h, img_w = img_shape_hw # (H, W)

    for _ in range(batch_size):
        data_sample = DetDataSample()
        img_meta = dict(
            img_shape=(img_h, img_w),           # (H, W)
            ori_shape=(img_h, img_w),           # (H, W), 假设与处理后相同
            pad_shape=(img_h, img_w),           # (H, W) <--- 添加这个关键字段！假设没有padding
            scale_factor=(1.0, 1.0, 1.0, 1.0),  # (scale_w, scale_h, scale_w, scale_h)
            flip=False,
            batch_input_shape=(img_h, img_w)    # (H, W)
        )
        data_sample.set_metainfo(img_meta)

        # 模拟一些真实框和标签 (对于损失计算是必要的)
        gt_instances = InstanceData()
        num_gt = torch.randint(1, 4, (1,)).item() # 至少有一个GT框，避免avg_factor为0
        # 确保 GT boxes 在图像范围内
        gt_bboxes = torch.rand(num_gt, 4, device=device)
        gt_bboxes[:, 0] *= img_w * 0.8 # x1, 限制在 0 到 0.8*W 之间
        gt_bboxes[:, 1] *= img_h * 0.8 # y1, 限制在 0 到 0.8*H 之间
        gt_bboxes[:, 2] = gt_bboxes[:, 0] + torch.rand(num_gt, device=device) * (img_w - gt_bboxes[:, 0]) # x2 > x1
        gt_bboxes[:, 3] = gt_bboxes[:, 1] + torch.rand(num_gt, device=device) * (img_h - gt_bboxes[:, 1]) # y2 > y1
        # Clamp to image boundaries again just in case
        gt_bboxes[:, 0::2].clamp_(min=0, max=img_w)
        gt_bboxes[:, 1::2].clamp_(min=0, max=img_h)
        # Ensure x1 < x2 and y1 < y2 strictly to avoid zero-area boxes
        gt_bboxes[:, 2] = torch.max(gt_bboxes[:, 2], gt_bboxes[:, 0] + 1e-2)
        gt_bboxes[:, 3] = torch.max(gt_bboxes[:, 3], gt_bboxes[:, 1] + 1e-2)


        gt_labels = torch.randint(0, num_classes, (num_gt,), device=device)

        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels
        data_sample.gt_instances = gt_instances
        data_samples.append(data_sample)
    return data_samples


def main():
    config_file = 'E:/workspace/lanyun_work/openmmlab/mmdetection/configs/ssd/ssd_custom.py'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        cfg = Config.fromfile(config_file)
        # ... (custom_imports) ...
        print("\n--- Testing Head Instantiation ---")
        bbox_head_cfg = cfg.model.bbox_head

        # +++ 新增打印 +++
        print("DEBUG: Configuration for bbox_head that will be built:")
        import json
        print(json.dumps(bbox_head_cfg, indent=4)) # 以易于阅读的JSON格式打印
        print(f"DEBUG: Keys in bbox_head_cfg: {list(bbox_head_cfg.keys())}")
        # +++ 结束新增打印 +++

        # 确保 anchor_generator 和 bbox_coder 等嵌套配置的 'type' 存在 (这部分检查可以保留)
        # ...

        # 执行 custom_imports
        if cfg.get('custom_imports', None):
            mmengine.utils.import_modules_from_strings(**cfg['custom_imports'])
        else:
            import mmdet.models # noqa

        # --- 1. 实例化 Head ---
        print("\n--- Testing Head Instantiation ---")
        # 我们需要完整的 bbox_head 配置，包括 anchor_generator, bbox_coder, losses, train_cfg 等
        # 这些应该已经在你的 ssd_custom.py 中配置好了
        bbox_head_cfg = cfg.model.bbox_head
        # 确保 anchor_generator 和 bbox_coder 等嵌套配置的 'type' 存在
        if 'anchor_generator' not in bbox_head_cfg or 'type' not in bbox_head_cfg.anchor_generator:
            raise ValueError("bbox_head.anchor_generator config with 'type' is required.")
        if 'bbox_coder' not in bbox_head_cfg or 'type' not in bbox_head_cfg.bbox_coder:
            raise ValueError("bbox_head.bbox_coder config with 'type' is required.")
        # 对于损失测试，还需要 train_cfg, loss_cls, loss_bbox
        if 'train_cfg' not in bbox_head_cfg:
             print("Warning: bbox_head.train_cfg not found. Loss calculation might fail if it's needed by SSDHead.")
        if 'loss_cls' not in bbox_head_cfg or 'loss_bbox' not in bbox_head_cfg:
             print("Warning: bbox_head.loss_cls or loss_bbox not found. Loss calculation will fail.")


        bbox_head = MODELS.build(bbox_head_cfg)
        bbox_head.to(device)
        bbox_head.train() # 设置为训练模式
        print("BBox Head instantiated successfully.")
        print(f"BBox Head type: {type(bbox_head)}")
        # 打印内部卷积层结构 (如果 bbox_head 是 SSDHead 或其子类)
        if hasattr(bbox_head, 'cls_convs') and hasattr(bbox_head, 'reg_convs'):
            print("\nHead Class Prediction Convolutional Layers:")
            for i, conv_module in enumerate(bbox_head.cls_convs):
                print(f"  Feature Level {i} (Input Channels: {bbox_head.in_channels[i]}): {conv_module}")
            print("\nHead Bounding Box Prediction Convolutional Layers:")
            for i, conv_module in enumerate(bbox_head.reg_convs):
                print(f"  Feature Level {i} (Input Channels: {bbox_head.in_channels[i]}): {conv_module}")
        else:
            print("Could not find cls_convs or reg_convs to print detailed structure.")


        # --- 2. 前向传播测试 ---
        print("\n--- Testing Head Forward Pass ---")
        # 模拟来自 backbone 的特征图输入
        # 特征图的通道数和数量应与 bbox_head.in_channels 匹配
        # 空间尺寸应与 anchor_generator.strides 和 input_size 对应
        dummy_feats, img_size_for_test = get_dummy_input_and_features(cfg, device=device)
        print(f"Generated {len(dummy_feats)} dummy feature maps for input size {img_size_for_test}:")
        for i, feat in enumerate(dummy_feats):
            print(f"  Feat {i} shape: {feat.shape}")

        # 调用 head 的 forward 方法
        # SSDHead.forward 返回 Tuple[List[Tensor], List[Tensor]]
        cls_scores, bbox_preds = bbox_head(dummy_feats)

        print("\nOutput from head.forward():")
        print(f"Number of cls_scores levels: {len(cls_scores)}")
        for i, score in enumerate(cls_scores):
            print(f"  Cls Score {i} shape: {score.shape}") # 应为 (N, num_anchors*num_classes, H, W)
        print(f"Number of bbox_preds levels: {len(bbox_preds)}")
        for i, pred in enumerate(bbox_preds):
            print(f"  BBox Pred {i} shape: {pred.shape}")   # 应为 (N, num_anchors*4, H, W)

        # --- 3. 损失计算测试 (如果 TinySSDHead 继承了 SSDHead/AnchorHead 并配置了损失) ---
        # 这一步的前提是 TinySSDHead 继承了 SSDHead 或 AnchorHead，并正确实现了 loss 方法
        # 或者你的配置文件为 TinySSDHead 提供了完整的 loss_cls, loss_bbox, train_cfg (含 assigner)
        if hasattr(bbox_head, 'loss') and callable(getattr(bbox_head, 'loss')) and \
           'loss_cls' in bbox_head_cfg and 'loss_bbox' in bbox_head_cfg and 'train_cfg' in bbox_head_cfg:
            print("\n--- Testing Head Loss Calculation (requires proper inheritance and config) ---")
            # 模拟批处理的 data_samples
            # img_shape 应该与生成 dummy_feats 时使用的 input_size 对应
            img_h = img_w = img_size_for_test
            dummy_data_samples = get_dummy_data_samples((img_h, img_w), bbox_head.num_classes, device=device, batch_size=dummy_feats[0].size(0))
            print(f"Generated {len(dummy_data_samples)} dummy data_samples.")
            for i, ds in enumerate(dummy_data_samples):
                 print(f"  Data_sample {i} gt_instances: {ds.gt_instances}")


            # 调用 loss 方法
            # SSDHead.loss 通常接收的是 forward 的输出 (cls_scores, bbox_preds)
            # 和 data_samples (其中包含 gt_instances 和 img_metas)
            # 但更底层的 loss_by_feat 接收的是 cls_scores, bbox_preds, batch_gt_instances, batch_img_metas
            # 我们直接调用 head.loss，它内部会处理
            # 注意: SSDHead 的 loss 方法定义在 AnchorHead 中，它期望的输入是：
            # loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict
            # 所以我们应该传递 dummy_feats 给 loss，而不是 cls_scores, bbox_preds
            losses = bbox_head.loss(dummy_feats, dummy_data_samples)

            print("\nOutput from head.loss():")
            if isinstance(losses, dict):
                for loss_name, loss_value in losses.items():
                    if isinstance(loss_value, list): # 有些损失可能是列表 (每个FPN层一个)
                        for i, lv in enumerate(loss_value):
                            print(f"  {loss_name}_{i}: {lv.item()} (shape: {lv.shape})")
                    elif hasattr(loss_value, 'item'):
                         print(f"  {loss_name}: {loss_value.item()} (shape: {loss_value.shape})")
                    else:
                        print(f"  {loss_name}: {loss_value}")
                if 'loss_cls' in losses and 'loss_bbox' in losses:
                    print("Loss calculation seems to produce expected keys.")
                else:
                    print("Warning: loss_cls or loss_bbox not found in losses dict.")
            else:
                print(f"Error: head.loss() did not return a dict. Got: {type(losses)}")
        else:
            print("\nSkipping Head Loss Calculation: head.loss method not found or prerequisites (loss_cls, loss_bbox, train_cfg in config) missing.")


        # --- 4. 预测测试 (如果 TinySSDHead 继承了 SSDHead/AnchorHead) ---
        if hasattr(bbox_head, 'predict') and callable(getattr(bbox_head, 'predict')):
            print("\n--- Testing Head Prediction (requires proper inheritance) ---")
            bbox_head.eval() # 设置为评估模式
            # 模拟 data_samples (主要用 img_metas)
            # batch_size 应该与 dummy_feats 的 batch_size 匹配
            img_h = img_w = img_size_for_test
            dummy_data_samples_for_predict = get_dummy_data_samples((img_h, img_w), bbox_head.num_classes, device=device, batch_size=dummy_feats[0].size(0))
            # 在预测时，通常不需要 gt_instances，但 DataSample 结构需要存在
            for ds in dummy_data_samples_for_predict:
                if hasattr(ds, 'gt_instances'):
                    del ds.gt_instances # 预测时通常没有或不需要GT

            # 临时修改 test_cfg 以获取更多输出
            original_test_cfg = bbox_head.test_cfg
            # 创建一个新的test_cfg字典进行修改，避免直接修改原始配置对象（如果是共享的）
            modified_test_cfg_dict = original_test_cfg.copy() if original_test_cfg is not None else {}

            # 确保 nms_cfg 存在且是字典类型
            if 'nms' not in modified_test_cfg_dict or not isinstance(modified_test_cfg_dict['nms'], dict):
                modified_test_cfg_dict['nms'] = dict(type='nms', iou_threshold=0.99) # 使用非常宽松的NMS
            else:
                modified_test_cfg_dict['nms']['iou_threshold'] = 0.99 # 使NMS非常宽松

            modified_test_cfg_dict['score_thr'] = 0.001 # 非常低的得分阈值
            modified_test_cfg_dict['max_per_img'] = 300 # 允许更多框

            print(f"DEBUG: Using modified test_cfg for predict: {modified_test_cfg_dict}")

            # 调用 predict 方法
            # AnchorHead.predict 期望的输入是 x: Tuple[Tensor], batch_data_samples: SampleList, rescale: bool = True
            with torch.no_grad(): # 预测时不需要梯度
                # results_list = bbox_head.predict(dummy_feats, dummy_data_samples_for_predict, rescale=False) # rescale=False 通常用于测试时不缩放到原图
                # MMDetection 3.x SampleList 包含 rescale 信息
                results_list = bbox_head.predict(dummy_feats, dummy_data_samples_for_predict)


            print(f"\nOutput from head.predict() (length: {len(results_list)}):")
            if results_list and isinstance(results_list[0], DetDataSample):
                for i, det_data_sample in enumerate(results_list):
                    print(f"  Sample {i}:")
                    if hasattr(det_data_sample, 'pred_instances') and det_data_sample.pred_instances:
                        print(f"    pred_instances.bboxes shape: {det_data_sample.pred_instances.bboxes.shape}")
                        print(f"    pred_instances.scores shape: {det_data_sample.pred_instances.scores.shape}")
                        print(f"    pred_instances.labels shape: {det_data_sample.pred_instances.labels.shape}")
                    else:
                        print(f"    No pred_instances found or empty in DetDataSample {i}")
                print("Prediction seems to produce DetDataSample list.")
            else:
                print("Error: head.predict() did not return a list of DetDataSample or the list is empty.")
        else:
            print("\nSkipping Head Prediction: head.predict method not found (likely due to not inheriting from AnchorHead/SSDHead).")


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()