import torch

# 加载两个权重文件
checkpoint_path_resume = '/root/lanyun-tmp/my_work/mmdetection/yolov3_d53_8xb8-320-273e_coco_cat_my/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
checkpoint_path_my = '/root/lanyun-tmp/my_work/mmdetection/yolov3_mobilenetv2_8xb24-320-300e_coco_my/epoch_1.pth'

checkpoint_resume = torch.load(checkpoint_path_resume)
checkpoint_my = torch.load(checkpoint_path_my)

# 打印两个检查点的键，以便查看其中的内容
print("Keys in checkpoint_resume:")
print(checkpoint_resume.keys())
print("\nKeys in checkpoint_my:")
print(checkpoint_my.keys())

# 对比特定的键，例如 'state_dict' 和 'meta'
def compare_keys(dict1, dict2, key):
    keys1 = set(dict1[key].keys())
    keys2 = set(dict2[key].keys())
    print(f"\nKeys in {key} of checkpoint_resume but not in checkpoint_my:")
    print(keys1 - keys2)
    print(f"\nKeys in {key} of checkpoint_my but not in checkpoint_resume:")
    print(keys2 - keys1)

compare_keys(checkpoint_resume, checkpoint_my, 'state_dict')
compare_keys(checkpoint_resume, checkpoint_my, 'meta')

# 对比特定的参数，例如学习率和 GPU 数量
if 'meta' in checkpoint_resume and 'config' in checkpoint_resume['meta'] and 'optim_wrapper' in checkpoint_resume['meta']['config']:
    lr_resume = checkpoint_resume['meta']['config']['optim_wrapper']['optimizer']['lr']
    lr_my = checkpoint_my['meta']['config']['optim_wrapper']['optimizer']['lr']
    print(f"\nLearning rate in checkpoint_resume: {lr_resume}")
    print(f"Learning rate in checkpoint_my: {lr_my}")

if 'meta' in checkpoint_resume and 'config' in checkpoint_resume['meta'] and 'env_cfg' in checkpoint_resume['meta']['config']:
    num_gpus_resume = checkpoint_resume['meta']['config']['env_cfg']['num_gpus']
    num_gpus_my = checkpoint_my['meta']['config']['env_cfg']['num_gpus']
    print(f"\nNumber of GPUs in checkpoint_resume: {num_gpus_resume}")
    print(f"Number of GPUs in checkpoint_my: {num_gpus_my}")
