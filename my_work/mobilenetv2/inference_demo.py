import cv2
import torch

img_path = '/root/lanyun-tmp/openmmlab/mmdetection/demo/demo.jpg'
model_config_path = '/root/lanyun-tmp/openmmlab/mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py'
model_pth_path = '/root/lanyun-tmp/openmmlab/mmdetection/checkpoints/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'

from mmdet.apis import init_detector, inference_detector
import mmcv


# 初始化检测模型
model = init_detector(model_config_path, model_pth_path, device='cuda:0')

import cv2
import numpy as np
import torch

def preprocess_image(img_path, img_size=416):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # 归一化
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = torch.from_numpy(img).float()
    return img

input_tensor = preprocess_image(img_path)

def predict(model, img_tensor, device='cuda'):
    model = model.to(device)
    img_tensor = img_tensor.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
    return outputs

outputs = predict(model=model, img_tensor=input_tensor)

out1, out2, out3 = outputs[0][0], outputs[0][1], outputs[0][2]

# 假设class_names是一个包含类别名称的列表
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush']  # 根据实际情况填写类别名称

def sigmoid(x):
    return torch.sigmoid(x)

def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.4):
    boxes = xywh2xyxy(prediction[..., :4])
    conf = prediction[..., 4]
    class_conf, class_pred = torch.max(prediction[..., 5:], dim=-1, keepdim=True)

    mask = conf > conf_thres
    boxes, conf, class_conf, class_pred = boxes[mask.squeeze()], conf[mask.squeeze()], class_conf[mask.squeeze()], class_pred[mask.squeeze()]

    detections = torch.cat((boxes, conf.unsqueeze(1), class_conf, class_pred.float()), dim=1)
    
    unique_classes = torch.unique(class_pred)
    best_boxes = []
    for c in unique_classes:
        class_mask = (class_pred == c).squeeze()
        class_boxes = detections[class_mask]
        while class_boxes.shape[0]:
            max_index = torch.argmax(class_boxes[:, 4])
            best_box = class_boxes[max_index]
            best_boxes.append(best_box)
            if len(class_boxes) == 1:
                break
            class_boxes = torch.cat((class_boxes[:max_index], class_boxes[max_index+1:]), dim=0)
            ious = bbox_iou(best_box[:4], class_boxes[:, :4])
            class_boxes = class_boxes[ious < iou_thres]
    return torch.stack(best_boxes) if len(best_boxes) > 0 else torch.tensor([])

def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clip(inter_rect_x2 - inter_rect_x1, min=0) * torch.clip(inter_rect_y2 - inter_rect_y1, min=0)
    
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou

def process_yolo_output(output, anchors, num_classes, img_size, conf_thres, iou_thres):
    grid_size = output.shape[2]
    stride = img_size / grid_size
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = output.view(1, num_anchors, bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
    
    x = sigmoid(prediction[..., 0])  # Center x
    y = sigmoid(prediction[..., 1])  # Center y
    w = prediction[..., 2]  # Width
    h = prediction[..., 3]  # Height
    conf = sigmoid(prediction[..., 4])  # Conf
    pred_cls = sigmoid(prediction[..., 5:])  # Cls pred.
    
    grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).float().to(output.device)
    grid_y = grid_x.permute(0, 1, 3, 2)
    
    anchors = anchors.view(num_anchors, 1, 1, 2).to(prediction.device)
    pred_boxes = torch.zeros(prediction[..., :4].shape).to(prediction.device)
    pred_boxes[..., 0] = x + grid_x
    pred_boxes[..., 1] = y + grid_y
    pred_boxes[..., 2] = torch.exp(w) * anchors[:, :, :, 0]
    pred_boxes[..., 3] = torch.exp(h) * anchors[:, :, :, 1]

    pred_boxes[..., :4] *= stride

    pred_boxes = pred_boxes.view(-1, 4)
    conf = conf.view(-1, 1)
    pred_cls = pred_cls.view(-1, num_classes)

    mask = conf > conf_thres
    pred_boxes = pred_boxes[mask.squeeze()]
    conf = conf[mask].view(-1, 1)
    pred_cls = pred_cls[mask.squeeze()]

    output = torch.cat((pred_boxes, conf, pred_cls), 1)

    boxes = non_max_suppression(output, conf_thres, iou_thres)
    return boxes

import matplotlib.pyplot as plt

def draw_boxes(img, boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4].item()
        cls_conf = box[5].item()
        cls_pred = int(box[6].item())

        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f'{class_names[cls_pred]}: {cls_conf:.2f}'
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

# 使用在训练过程中配置的anchors
anchors1 = torch.tensor([(116, 90), (156, 198), (373, 326)])
anchors2 = torch.tensor([(30, 61), (62, 45), (59, 119)])
anchors3 = torch.tensor([(10, 13), (16, 30), (33, 23)])
num_classes = 80
img_size = 416
conf_thres = 0.5
iou_thres = 0.4

# 假设你已经有了输出结果
outputs = [out1, out2, out3]

boxes1 = process_yolo_output(outputs[0], anchors1, num_classes, img_size, conf_thres, iou_thres)
boxes2 = process_yolo_output(outputs[1], anchors2, num_classes, img_size, conf_thres, iou_thres)
boxes3 = process_yolo_output(outputs[2], anchors3, num_classes, img_size, conf_thres, iou_thres)

# 合并所有尺度的boxes
boxes = torch.cat((boxes1, boxes2, boxes3), dim=0)

img = cv2.imread(img_path)

img = draw_boxes(img, boxes, class_names)

# 使用matplotlib显示图像
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')  # 不显示坐标轴
# plt.show()

# 保存图像
cv2.imwrite('output.jpg', img)