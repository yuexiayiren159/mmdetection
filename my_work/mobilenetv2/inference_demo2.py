import torch
import copy
# from mmdet.models import build_detector
# from mmcv import Config
# from mmcv.runner import load_checkpoint
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mmdet.apis import init_detector

# 加载配置文件和模型权重
img_path = '/root/lanyun-tmp/openmmlab/mmdetection/demo/demo.jpg'
model_config_path = '/root/lanyun-tmp/openmmlab/mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py'
model_pth_path = '/root/lanyun-tmp/openmmlab/mmdetection/checkpoints/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'

# cfg = Config.fromfile(model_config_path)
# model = build_detector(cfg.model)
# checkpoint = load_checkpoint(model, model_pth_path, map_location='cuda:0')
# model.CLASSES = checkpoint['meta']['CLASSES']
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
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'] 

model = init_detector(model_config_path, model_pth_path, device='cuda:0')

# 预处理图像
def preprocess_image(img_path, img_size=416):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # 归一化
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = torch.from_numpy(img).float()
    return img, img_path

input_tensor, img_path = preprocess_image(img_path)

# 获取图像元数据
img_meta = dict(
    ori_shape=(416, 416, 3),  # 原始图像大小
    img_shape=(416, 416, 3),  # 处理后图像大小
    pad_shape=(416, 416, 3),  # 填充后的图像大小
    scale_factor=1.0,         # 缩放比例
    flip=False                # 是否翻转
)
batch_img_metas = [img_meta]

# 进行推理
with torch.no_grad():
    model = model.cuda()
    # outputs = model(input_tensor.cuda(), return_loss=False)
    outputs = model(input_tensor.cuda())

# 调用 predict_by_feat 方法
pred_maps = outputs[0]
result = model.bbox_head.predict_by_feat(
    pred_maps=pred_maps, 
    batch_img_metas=batch_img_metas, 
    cfg=None, 
    rescale=False, 
    with_nms=True
)

# 后处理和可视化
def draw_boxes(img, result, class_names, score_thr=0.5):
    bboxes = np.vstack(result.bboxes.cpu().numpy())
    scores = result.scores.cpu().numpy()
    labels = result.labels.cpu().numpy()
    for bbox, score, label in zip(bboxes, scores, labels):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = bbox.astype(int)
        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label_text = f'{class_names[label]}: {score:.2f}'
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

# 读取图像
img = cv2.imread(img_path)

# 绘制检测结果
img = draw_boxes(img, result[0], class_names)

# 使用matplotlib显示图像
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')  # 不显示坐标轴
plt.show()

# 保存图像
cv2.imwrite('output1.png', img)
