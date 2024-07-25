_base_ = ['./yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py']
# model settings

# yapf:disable
model = dict(
    bbox_head=dict(
        anchor_generator=dict(
            base_sizes=[[(220, 125), (128, 222), (264, 266)],
                        [(35, 87), (102, 96), (60, 170)],
                        [(10, 15), (24, 36), (72, 42)]])))
# yapf:enable

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/root/lanyun-tmp/coco3000/'
backend_args = None

input_size = (320, 320)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    # `mean` and `to_rgb` should be the same with the `preprocess_cfg`
    dict(
        type='Expand',
        mean=[123.675, 116.28, 103.53],
        to_rgb=True,
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=input_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=input_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(dataset=dict(dataset=dict(
    data_root=data_root,
    pipeline=train_pipeline)))
val_dataloader = dict(dataset=dict(
    data_root=data_root,
    pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator


auto_scale_lr = dict(base_batch_size=192, enable=True)
# auto_scale_lr = dict(base_batch_size=192)

train_cfg = dict(max_epochs=35)

# log_level='DEBUG'
log_level='INFO'


work_dir = '/root/lanyun-tmp/my_work/mmdetection/yolov3_mobilenetv2_8xb24-320-300e_coco_my_1'
