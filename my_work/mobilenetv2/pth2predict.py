from mmdet.apis import init_detector, inference_detector
import mmcv
import torch

import cv2
import numpy as np

# 预处理输入数据
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# print(model)

# 测试图像路径
# img = 'path/to/your/test_image.jpg'

# 进行推理
# result = inference_detector(model, img)

# 显示结果
# show_result_pyplot(model, img, result, score_thr=0.3)

# 设置模型的 CLASSES 属性
model.CLASSES = coco_classes
print(f"Model loaded with classes: {model.CLASSES}")

checkpoint = torch.load(f=checkpoint_file, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)
print(f"Model loaded with classes: {model.CLASSES}")

# 假设类别信息存储在检查点的 'meta' 部分
if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    print("Classes not found in checkpoint.")

# print(checkpoint)
model.eval()
model.cuda()

#
img_path = r"E:\workspace\lanyun_work\openmmlab\mmdetection\demo\demo.jpg"
#
# img_bgr = cv2.imread(img_path)
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# print(img.shape)

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((416, 416)),  # 确保图像尺寸与模型的输入尺寸一致
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    return img.unsqueeze(0)  # 添加批次维度

img_size = 416
img = preprocess_image(img_path).cuda()
print(img.shape)

# 解析output以获得检测结果
# 这里output的结构取决于模型的实现
# def process_yolo_outputs(outputs, img_size, conf_threshold=0.5, nms_threshold=0.4):
#     boxes = []
#     confidences = []
#     class_ids = []
#
#     for scale_output in outputs[0]:
#         # output = scale_output[1]  # 获取torch.Tensor
#         output = scale_output
#         batch_size, num_channels, grid_height, grid_width = output.shape
#         num_classes = (num_channels // 3) - 5
#
#         # 调整形状以适应 YOLO 格式
#         output = output.view(batch_size, 3, 5 + num_classes, grid_height, grid_width)
#         output = output.permute(0, 1, 3, 4, 2).contiguous()  # 调整维度顺序
#
#         # 处理每个锚点
#         for batch in range(batch_size):
#             for anchor in range(3):
#                 for y in range(grid_height):
#                     for x in range(grid_width):
#                         pred = output[batch, anchor, y, x]
#
#                         # 获取对象置信度
#                         obj_conf = torch.sigmoid(pred[4])
#
#                         # 获取类别概率
#                         class_scores = torch.sigmoid(pred[5:])
#
#                         # 获取最大类别概率及对应类别ID
#                         class_score, class_id = torch.max(class_scores, dim=0)
#                         confidence = obj_conf * class_score
#
#                         if confidence.item() > conf_threshold:
#                             # 获取边界框坐标和尺寸
#                             bx = torch.sigmoid(pred[0]) + x
#                             by = torch.sigmoid(pred[1]) + y
#                             bw = torch.exp(pred[2])
#                             bh = torch.exp(pred[3])
#
#                             # 归一化为图像尺寸
#                             bx = bx * (img_size / grid_width)
#                             by = by * (img_size / grid_height)
#                             bw = bw * (img_size / grid_width)
#                             bh = bh * (img_size / grid_height)
#
#                             # 转换为左上角坐标
#                             x1 = int(bx - bw / 2)
#                             y1 = int(by - bh / 2)
#                             x2 = int(bx + bw / 2)
#                             y2 = int(by + bh / 2)
#
#                             boxes.append([x1, y1, x2 - x1, y2 - y1])
#                             confidences.append(float(confidence))
#                             class_ids.append(class_id.item())
#
#     # 使用 NMS 来移除重复的边界框
#     if len(boxes) > 0:
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
#         indices = indices.flatten()  # 扁平化索引
#         result = [(boxes[i], confidences[i], class_ids[i]) for i in indices]
#     else:
#         result = []
#
#     return result

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

'''
type(outputs[0][0])
<class 'torch.Tensor'>
outputs[0][0].shape
torch.Size([1, 255, 13, 13])
outputs[0][1].shape
torch.Size([1, 255, 26, 26])
outputs[0][2].shape
torch.Size([1, 255, 52, 52])
'''

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

        label = f'Class {class_id}: {score:.2f}'
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

show_result(img_path, detections)
