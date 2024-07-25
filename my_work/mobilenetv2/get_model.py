# from mmpretrain import inference_model

image = '../../demo/dog.jpg'

# from mmpretrain.apis import inference_model, init_model, get_model

from mmdetection.apis import inference_detector, init_detector, get_model


# model = get_model('../../configs/mobilenet_v2/mobilenet-v2.py')
model = init_model('../../configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco_my_getmodel.py')
# print(model)
import torch
import torch.nn as nn
model.eval()
input_tensor = torch.randn(1,3,416,416)

output = model(input_tensor)
print(output[0].shape)

torch.onnx.export(
    model,
    input_tensor,
    "yolov3_mobilenetv2_w10.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11,
    training=2,
    do_constant_folding=False
)

from onnxsim import simplify
import onnx
model_simp, check = simplify(onnx.load("yolov3_mobilenetv2_w10.onnx"))
onnx.save(model_simp, "yolov3_mobilenetv2_w10_sim.onnx")




