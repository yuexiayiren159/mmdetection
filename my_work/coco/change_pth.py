import torch

# checkpoint_path = '/root/lanyun-tmp/my_work/mmdetection/yolov3_d53_8xb8-320-273e_coco_cat_my/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
checkpoint_path_new = '/root/lanyun-tmp/my_work/mmdetection/yolov3_d53_8xb8-320-273e_coco_cat_my/yolov3_mobilenetv2_320_300e_coco.pth'
# checkpoint = torch.load(checkpoint_path)
#
# # 确保 'message_hub' 键存在
# if 'message_hub' not in checkpoint:
#     checkpoint['message_hub'] = {}
# # 确保 'log_scalars' 和 'runtime_info' 键存在
# if 'log_scalars' not in checkpoint['message_hub']:
#     checkpoint['message_hub']['log_scalars'] = {}
# if 'runtime_info' not in checkpoint['message_hub']:
#     checkpoint['message_hub']['runtime_info'] = {}
# if 'resumed_keys' not in checkpoint['message_hub']:
#     checkpoint['message_hub']['resumed_keys'] = {}
#
# torch.save(checkpoint, checkpoint_path_new)

import torch

# 加载现有的权重文件
checkpoint_path_resume = '/root/lanyun-tmp/my_work/mmdetection/yolov3_d53_8xb8-320-273e_coco_cat_my/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
checkpoint_path_my = '/root/lanyun-tmp/my_work/mmdetection/yolov3_mobilenetv2_8xb24-320-300e_coco_my/epoch_1.pth'
checkpoint_my = torch.load(checkpoint_path_my)
print(checkpoint_my)
# print(checkpoint_my['message_hub'])

checkpoint_resume = torch.load(checkpoint_path_resume)
print(checkpoint_resume)
checkpoint_resume['message_hub'] =checkpoint_my['message_hub']

# torch.save(checkpoint_resume, checkpoint_path_new)
'''
# 确保 'message_hub' 键存在
if 'message_hub' not in checkpoint:
    checkpoint['message_hub'] = {}

# 确保 'log_scalars' 和 'runtime_info' 键存在
if 'log_scalars' not in checkpoint['message_hub']:
    checkpoint['message_hub']['log_scalars'] = {}
if 'runtime_info' not in checkpoint['message_hub']:
    checkpoint['message_hub']['runtime_info'] = {}
if 'resumed_keys' not in checkpoint['message_hub']:
    checkpoint['message_hub']['resumed_keys'] = {}

# 处理 GPU 数量不一致的问题
checkpoint['meta']['config']['env_cfg']['num_gpus'] = 1  # 设置为实际使用的 GPU 数量

# 处理学习率不一致的问题
base_batch_size = checkpoint['meta']['config']['auto_scale_lr']['base_batch_size']
current_batch_size = 24  # 设置为实际使用的批处理大小
lr = checkpoint['meta']['config']['optim_wrapper']['optimizer']['lr']
checkpoint['meta']['config']['optim_wrapper']['optimizer']['lr'] = lr * (current_batch_size / base_batch_size)

# 保存新的权重文件
new_checkpoint_path = '/path/to/new/weight/file.pth'
torch.save(checkpoint, new_checkpoint_path)
'''



