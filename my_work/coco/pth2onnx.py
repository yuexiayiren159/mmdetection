# import torch
# import torch.onnx
# from mmdet.models import build_detector
# from mmcv import Config
#
# # 加载配置文件和训练好的权重文件
# config_file = '/path/to/your/config.py'
# checkpoint_file = '/path/to/your/model.pth'
#
# cfg = Config.fromfile(config_file)
# cfg.model.pretrained = None
#
# # 构建模型
# model = build_detector(cfg.model)
# checkpoint = torch.load(checkpoint_file, map_location='cpu')
# model.load_state_dict(checkpoint['state_dict'])
# model.eval()
#
# # 输入数据
# dummy_input = torch.randn(1, 3, 320, 320)
#
# # 导出ONNX模型
# torch.onnx.export(model, dummy_input, 'model.onnx', export_params=True, opset_version=11)
