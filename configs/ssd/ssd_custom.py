_base_ = [
    # '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.tiny_ssd',
        'mmdet.models.backbones.tinyssd_custom',
        'mmdet.models.dense_heads.ssd_custom_head',
    ],
    allow_failed_imports=False
)

#############################
# datasets
dataset_type = 'CocoDataset'
data_root = 'data/banana-coco/'
metainfo = {
    'classes': ('banana'),  # 顺序必须和 JSON 文件的 id 对应
    'palette': [(255, 0, 0)]  # 可选的颜色，不影响训练
}

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=24,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,  # 加上这个
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=24,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,  # 加上这个
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator
#############################

model = dict(
    type='TinySSDDetector',
    backbone=dict(
        type='TinySSD_Custom',
        out_channels=[64, 128, 128, 128, 128]  # 举个例子
    ),
    neck=None,  # 没有特殊neck
    bbox_head=dict(
        type='TinySSDHead',
        num_classes=1,  # 香蕉只有1类,
        num_anchors=4,
        # in_channels=(16, 32, 64),
        in_channels=[64, 128, 128, 128, 128],  # ⭐这里一定要对齐！
    )
)

# dataset_type = 'CocoDataset'
# classes = ('banana',)

# data = dict(
#     samples_per_gpu=32,
#     workers_per_gpu=2,
#     train=dict(
#         img_prefix='data/banana-coco/train2017/',
#         classes=classes,
#         ann_file='data/banana-coco/annotations/instances_train2017.json'),
#     val=dict(
#         img_prefix='data/banana-coco/val2017/',
#         classes=classes,
#         ann_file='data/banana-coco/annotations/instances_val2017.json'),
#     test=dict(
#         img_prefix='data/banana-coco/val2017/',
#         classes=classes,
#         ann_file='data/banana-coco/annotations/instances_val2017.json'))

optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)

evaluation = dict(interval=1, metric='bbox')

# 修改学习率
lr_config = dict(
    policy='step',
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

log_config = dict(interval=50)

work_dir = './work_dirs/ssd_custom'
