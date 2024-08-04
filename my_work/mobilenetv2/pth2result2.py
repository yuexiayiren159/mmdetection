import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mmdet.apis import init_detector

from typing import List, Union

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(e)

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

# 定义每个尺度的锚框（假设这里使用了YOLOv3常见的锚框大小）
anchors = [
    [(116, 90), (156, 198), (373, 326)],  # 13x13 特征图
    [(30, 61), (62, 45), (59, 119)],     # 26x26 特征图
    [(10, 13), (16, 30), (33, 23)]       # 52x52 特征图
]

# 配置文件和模型权重路径
config_file = 'E:\workspace\lanyun_work\openmmlab\mmdetection\configs\yolo\yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py'
checkpoint_file = 'F:/model/mmdet/yolov3/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'



model = init_detector(config_file, checkpoint_file, device='cpu')
# checkpoint = torch.load(checkpoint_file)
# model.load_state_dict(checkpoint['state_dict'])



img_path = r"E:\workspace\lanyun_work\openmmlab\mmdetection\demo\demo.jpg"
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def stack_batch(tensor_list: List[torch.Tensor],
                pad_size_divisor: int = 1,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """将多个张量堆叠成一个批次，并填充到最大形状，使用右下填充模式。如果
    `pad_size_divisor > 0`，填充确保每个维度的大小可被 `pad_size_divisor` 整除。

    Args:
        tensor_list (List[Tensor]): 张量列表，维度相同。
        pad_size_divisor (int): 如果 `pad_size_divisor > 0`，填充确保每个维度的大小可被 `pad_size_divisor` 整除。
        pad_value (int, float): 填充值。默认为 0。

    Returns:
       Tensor: 堆叠的 n 维张量。
    """
    assert isinstance(tensor_list, list), (f'Expected input type to be list, but got {type(tensor_list)}')
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({tensor.ndim for tensor in tensor_list}) == 1, (
        f'Expected the dimensions of all tensors must be the same, '
        f'but got {[tensor.ndim for tensor in tensor_list]}')

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes = torch.Tensor([tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes

    # 不填充第一个维度（通常是通道数）
    padded_sizes[:, 0] = 0

    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)

    # `pad` 是 `F.pad` 的第二个参数。如果 pad 是 (1, 2, 3, 4),
    # 表示最后一个维度左填充1，右填充2，倒数第二个维度上填充3，下填充4。
    # `padded_sizes` 的顺序与 `pad` 相反，因此 `padded_sizes` 需要反转，
    # 并且只有奇数索引的 pad 需要赋值，以保持填充 "右" 和 "下"。
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)

def preprocess_image(img_path, target_size=(288, 416), intermediate_size=(278, 416)):
    # 读取图像
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图像：{img_path}")

    # BGR 转 RGB
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # img_pil = Image.fromarray(img_rgb)

    cv2_interp_codes = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }
    interpolation = 'bilinear'
    # 调整图像大小
    # print(img_bgr)
    # 这里传入原始图片的bgr图片
    resized_img = cv2.resize(
        img_bgr, (416, 278),dst=None, interpolation=cv2_interp_codes[interpolation])
    # print(resized_img)
    # print(resized_img.shape)
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
    # print("img_tensor: ",img_tensor)
    # print(img_tensor.shape)

    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]
    mean_tensor = torch.tensor(mean).float().view(3, 1, 1)
    std_tensor = torch.tensor(std).float().view(3, 1, 1)
    _batch_input = (img_tensor - mean_tensor) / std_tensor
    # print("img_tensor: ",_batch_input)
    # print(_batch_input.shape)

    # 示例调用
    # tensor_list = [_batch_input, _batch_input]  # 示例张量列表
    tensor_list = [_batch_input]  # 示例张量列表
    stacked_tensor = stack_batch(tensor_list, pad_size_divisor=32, pad_value=0)
    # print("stacked_tensor: ",stacked_tensor.shape)
    # print(stacked_tensor)
    return stacked_tensor


def process_yolo_outputs(outputs, img_size=(288, 416), conf_threshold=0.5, nms_threshold=0.45):
    boxes = []
    confidences = []
    class_ids = []

    for scale_index, scale_output in enumerate(outputs[0]):
        batch_size, num_channels, grid_height, grid_width = scale_output.shape
        num_classes = (num_channels // 3) - 5

        # 调整形状以适应 YOLO 格式
        # scale_output = torch.Size([1, 3, 85, 9, 13])
        scale_output = scale_output.view(batch_size, 3, 5 + num_classes, grid_height, grid_width)
        # torch.Size([1, 3, 85, 9, 13]) -> torch.Size([1, 3, 9, 13, 85])
        scale_output = scale_output.permute(0, 1, 3, 4, 2).contiguous()

        # 获取当前尺度的锚框
        scale_anchors = anchors[scale_index]

        for batch in range(batch_size):
            for anchor in range(3):
                for y in range(grid_height):
                    for x in range(grid_width):
                        pred = scale_output[batch, anchor, y, x]

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
                            img_h, img_w = img_size
                            bw = torch.exp(pred[2]) * anchor_width / img_w
                            bh = torch.exp(pred[3]) * anchor_height / img_h

                            # 转换为左上角坐标
                            x1 = int((bx - bw / 2) * img_w)
                            y1 = int((by - bh / 2) * img_h)
                            x2 = int((bx + bw / 2) * img_w)
                            y2 = int((by + bh / 2) * img_h)

                            boxes.append([x1, y1, x2 - x1, y2 - y1])
                            confidences.append(float(confidence))
                            class_ids.append(class_id.item())

    # 使用 NMS 来移除重复的边界框
    if len(boxes) > 0:
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_threshold, nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            result = [(boxes[i], confidences[i], coco_classes[class_ids[i]]) for i in indices]
        else:
            result = []
    else:
        result = []

    return result



img_size = (288,416)

stacked_tensor = preprocess_image(img_path)
print(stacked_tensor.shape)
# print(img)
# print(img.shape)

# print(img.shape)
with torch.no_grad():
    outs = model(stacked_tensor)
    # print(outs)

# 调用解码函数
detections = process_yolo_outputs(outs, img_size)
print(detections)

def show_result(img_path, detections, original_size=(427, 640), input_size=(288, 416)):
    if not detections:
        print("No detections found.")
        return

    # 读取原始图片
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 计算宽度和高度的缩放比率
    orig_h, orig_w = original_size
    inp_h, inp_w = input_size
    scale_w = orig_w / inp_w
    scale_h = orig_h / inp_h

    for (box, score, class_name) in detections:
        # 将检测框的坐标映射回原始图片尺寸
        x, y, w, h = box
        x = int(x * scale_w)
        y = int(y * scale_h)
        w = int(w * scale_w)
        h = int(h * scale_h)

        # 绘制边框和标签
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f'{class_name}: {score:.2f}'
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(img)
    plt.axis('off')
    # plt.savefig("demo_output.png")
    plt.show()

show_result(img_path, detections)