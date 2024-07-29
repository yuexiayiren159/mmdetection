from mmdet.apis import DetInferencer

model_config_path = '/root/lanyun-tmp/openmmlab/mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py'
model_pth_path = '/root/lanyun-tmp/openmmlab/mmdetection/checkpoints/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'
inferencer = DetInferencer(model=model_config_path, weights=model_pth_path)
img_path = '/root/lanyun-tmp/openmmlab/mmdetection/demo/demo.jpg'

outputs = inferencer(inputs=img_path, out_dir='./output')