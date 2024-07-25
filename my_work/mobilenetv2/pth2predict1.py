from mmdet.apis import init_detector, inference_detector
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


from torch import Tensor
from typing import List, Optional, Sequence, Tuple
from mmengine.structures import InstanceData
import copy

def batched_nms(boxes: Tensor,
                scores: Tensor,
                idxs: Tensor,
                nms_cfg: Optional[Dict],
                class_agnostic: bool = False) -> Tuple[Tensor, Tensor]:
    r"""Performs non-maximum suppression in a batched fashion.

    Modified from `torchvision/ops/boxes.py#L39
    <https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Note:
        In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
        returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
            is None, otherwise it should specify nms type and other
            parameters like `iou_thr`. Possible keys includes the following.

            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
              number of boxes is large (e.g., 200k). To avoid OOM during
              training, the users could set `split_thr` to a small value.
              If the number of boxes is greater than the threshold, it will
              perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.

    Returns:
        tuple: kept dets and indice.

        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]],
                                      dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep

def get_box_tensor(boxes: Union[Tensor, BaseBoxes]) -> Tensor:
    """Get tensor data from box type boxes.

    Args:
        boxes (Tensor or BaseBoxes): boxes with type of tensor or box type.
            If its type is a tensor, the boxes will be directly returned.
            If its type is a box type, the `boxes.tensor` will be returned.

    Returns:
        Tensor: boxes tensor.
    """
    if isinstance(boxes, BaseBoxes):
        boxes = boxes.tensor
    return boxes


def get_box_wh(boxes: Union[Tensor, BaseBoxes]) -> Tuple[Tensor, Tensor]:
    """Get the width and height of boxes with type of tensor or box type.

    Args:
        boxes (Tensor or :obj:`BaseBoxes`): boxes with type of tensor
            or box type.

    Returns:
        Tuple[Tensor, Tensor]: the width and height of boxes.
    """
    if isinstance(boxes, BaseBoxes):
        w = boxes.widths
        h = boxes.heights
    else:
        # Tensor boxes will be treated as horizontal boxes by defaults
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
    return w, h

def scale_boxes(boxes: Union[Tensor, BaseBoxes],
                scale_factor: Tuple[float, float]) -> Union[Tensor, BaseBoxes]:
    """Scale boxes with type of tensor or box type.

    Args:
        boxes (Tensor or :obj:`BaseBoxes`): boxes need to be scaled. Its type
            can be a tensor or a box type.
        scale_factor (Tuple[float, float]): factors for scaling boxes.
            The length should be 2.

    Returns:
        Union[Tensor, :obj:`BaseBoxes`]: Scaled boxes.
    """
    if isinstance(boxes, BaseBoxes):
        boxes.rescale_(scale_factor)
        return boxes
    else:
        # Tensor boxes will be treated as horizontal boxes
        repeat_num = int(boxes.size(-1) / 2)
        scale_factor = boxes.new_tensor(scale_factor).repeat((1, repeat_num))
        return boxes * scale_factor


def _bbox_post_process(self,
                       results: InstanceData,
                       cfg: ConfigDict,
                       rescale: bool = False,
                       with_nms: bool = True,
                       img_meta: Optional[dict] = None) -> InstanceData:
    """bbox post-processing method.

    The boxes would be rescaled to the original image scale and do
    the nms operation. Usually `with_nms` is False is used for aug test.

    Args:
        results (:obj:`InstaceData`): Detection instance results,
            each item has shape (num_bboxes, ).
        cfg (ConfigDict): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.
            Default to False.
        with_nms (bool): If True, do nms before return boxes.
            Default to True.
        img_meta (dict, optional): Image meta info. Defaults to None.

    Returns:
        :obj:`InstanceData`: Detection results of each image
        after the post process.
        Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
    """
    if rescale:
        assert img_meta.get('scale_factor') is not None
        scale_factor = [1 / s for s in img_meta['scale_factor']]
        results.bboxes = scale_boxes(results.bboxes, scale_factor)

    if hasattr(results, 'score_factors'):
        # TODO: Add sqrt operation in order to be consistent with
        #  the paper.
        score_factors = results.pop('score_factors')
        results.scores = results.scores * score_factors

    # filter small size bboxes
    if cfg.get('min_bbox_size', -1) >= 0:
        w, h = get_box_wh(results.bboxes)
        valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
        if not valid_mask.all():
            results = results[valid_mask]

    # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
    if with_nms and results.bboxes.numel() > 0:
        bboxes = get_box_tensor(results.bboxes)
        det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                            results.labels, cfg.nms)
        results = results[keep_idxs]
        # some nms would reweight the score, such as softnms
        results.scores = det_bboxes[:, -1]
        results = results[:cfg.max_per_img]

    return results


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results


def predict_by_feat(pred_maps: Sequence[Tensor],
                    batch_img_metas: Optional[List[dict]],
                    bbox_coder, prior_generator,
                    num_classes: int,
                    featmap_strides: List[int],
                    test_cfg: Optional[dict] = None,
                    rescale: bool = False,
                    with_nms: bool = True) -> List[InstanceData]:
    """Transform a batch of output features extracted from the head into
    bbox results.

    Args:
        pred_maps (Sequence[Tensor]): Raw predictions for a batch of images.
        batch_img_metas (list[dict], Optional): Batch image meta info.
            Defaults to None.
        bbox_coder: Bounding box coder to decode predictions.
        prior_generator: Prior (anchor) generator.
        num_classes (int): Number of classes.
        featmap_strides (List[int]): Strides of the feature maps.
        test_cfg (dict, optional): Test / postprocessing configuration.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes. Defaults to True.

    Returns:
        list[:obj:`InstanceData`]: Object detection results of each image
        after the post process. Each item usually contains following keys.

        - scores (Tensor): Classification scores, has a shape
          (num_instance, )
        - labels (Tensor): Labels of bboxes, has a shape
          (num_instances, ).
        - bboxes (Tensor): Has a shape (num_instances, 4),
          the last dimension 4 arrange as (x1, y1, x2, y2).
    """
    cfg = test_cfg if test_cfg else {}
    cfg = copy.deepcopy(cfg)

    num_imgs = len(batch_img_metas)
    featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]

    mlvl_anchors = prior_generator.grid_priors(
        featmap_sizes, device=pred_maps[0].device)

    flatten_preds = []
    flatten_strides = []

    for pred, stride in zip(pred_maps, featmap_strides):
        pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 5 + num_classes)
        pred[..., :2].sigmoid_()
        flatten_preds.append(pred)
        flatten_strides.append(
            pred.new_tensor(stride).expand(pred.size(1)))

    flatten_preds = torch.cat(flatten_preds, dim=1)
    flatten_bbox_preds = flatten_preds[..., :4]
    flatten_objectness = flatten_preds[..., 4].sigmoid()
    flatten_cls_scores = flatten_preds[..., 5:].sigmoid()
    flatten_anchors = torch.cat(mlvl_anchors)
    flatten_strides = torch.cat(flatten_strides)
    flatten_bboxes = bbox_coder.decode(flatten_anchors,
                                       flatten_bbox_preds,
                                       flatten_strides.unsqueeze(-1))
    results_list = []
    for (bboxes, scores, objectness,
         img_meta) in zip(flatten_bboxes, flatten_cls_scores,
                          flatten_objectness, batch_img_metas):
        # Filtering out all predictions with conf < conf_thr
        conf_thr = cfg.get('conf_thr', -1)
        if conf_thr > 0:
            conf_inds = objectness >= conf_thr
            bboxes = bboxes[conf_inds, :]
            scores = scores[conf_inds, :]
            objectness = objectness[conf_inds]

        score_thr = cfg.get('score_thr', 0)
        nms_pre = cfg.get('nms_pre', -1)
        scores, labels, keep_idxs, _ = filter_scores_and_topk(
            scores, score_thr, nms_pre)

        results = InstanceData(
            scores=scores,
            labels=labels,
            bboxes=bboxes[keep_idxs],
            score_factors=objectness[keep_idxs],
        )
        results = _bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
        results_list.append(results)
    return results_list


# COCO 数据集的 80 类
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# 定义每个尺度的锚框
anchors = [
    [(116, 90), (156, 198), (373, 326)],  # 13x13 特征图
    [(30, 61), (62, 45), (59, 119)],     # 26x26 特征图
    [(10, 13), (16, 30), (33, 23)]       # 52x52 特征图
]

# 配置文件和模型权重路径
config_file = 'E:\workspace\lanyun_work\openmmlab\mmdetection\configs\yolo\yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py'
checkpoint_file = 'F:/model/mmdet/yolov3/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 设置模型的 CLASSES 属性
model.CLASSES = coco_classes

# 加载模型权重
checkpoint = torch.load(f=checkpoint_file, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)

# 读取测试图像
img_path = r"E:\workspace\lanyun_work\openmmlab\mmdetection\demo\demo.jpg"

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    return img.unsqueeze(0)  # 添加批次维度

img_size = 416
img = preprocess_image(img_path).cuda()

def process_yolo_outputs(outputs, img_size, conf_threshold=0.5, nms_threshold=0.4):
    boxes = []
    confidences = []
    class_ids = []

    # 遍历每个尺度的输出
    for scale_index, scale_output in enumerate(outputs[0]):
        output = scale_output
        batch_size, num_channels, grid_height, grid_width = output.shape
        num_classes = (num_channels // 3) - 5

        # 调整形状以适应 YOLO 格式
        output = output.view(batch_size, 3, 5 + num_classes, grid_height, grid_width)
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        # 获取当前尺度的锚框
        scale_anchors = anchors[scale_index]

        # 处理每个锚点
        for batch in range(batch_size):
            for anchor in range(3):
                for y in range(grid_height):
                    for x in range(grid_width):
                        pred = output[batch, anchor, y, x]

                        # 获取对象置信度
                        obj_conf = torch.sigmoid(pred[4])

                        # 获取类别概率
                        class_scores = torch.sigmoid(pred[5:])
                        class_score, class_id = torch.max(class_scores, dim=0)
                        confidence = obj_conf * class_score

                        if confidence.item() > conf_threshold:
                            # 获取边界框坐标和尺寸
                            bx = (torch.sigmoid(pred[0]) + x) / grid_width
                            by = (torch.sigmoid(pred[1]) + y) / grid_height

                            # 获取锚框的宽度和高度
                            anchor_width, anchor_height = scale_anchors[anchor]
                            bw = torch.exp(pred[2]) * anchor_width / img_size
                            bh = torch.exp(pred[3]) * anchor_height / img_size

                            # 转换为左上角坐标
                            x1 = int((bx - bw / 2) * img_size)
                            y1 = int((by - bh / 2) * img_size)
                            x2 = int((bx + bw / 2) * img_size)
                            y2 = int((by + bh / 2) * img_size)

                            boxes.append([x1, y1, x2 - x1, y2 - y1])
                            confidences.append(float(confidence))
                            class_ids.append(class_id.item())

    # 使用 NMS 来移除重复的边界框
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            result = [(boxes[i], confidences[i], class_ids[i]) for i in indices]
        else:
            result = []
    else:
        result = []

    return result

# 不需要梯度计算
with torch.no_grad():
    outputs = model(img)

# 调用解码函数
detections = process_yolo_outputs(outputs, img_size)

# 可视化结果
def show_result(img_path, detections):
    if not detections:
        print("No detections found.")

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for (box, score, class_id) in detections:
        x, y, w, h = box
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            print(f"Invalid bounding box detected: {box}")
            continue

        label = f'{coco_classes[class_id]}: {score:.2f}'
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

show_result(img_path, detections)
