print("DEBUG: LOADING ssd_custom.py CONFIG - VERSION <YOUR_CONFIG_UNIQUE_MARKER>")
_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.tiny_ssd',
        # 'mmdet.models.backbones.tinyssd_custom',
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
    # --- 修改这里 ---
    dict(type='Resize', scale=(256, 256), keep_ratio=False), # 强制所有图像都resize到 256x256
    # --- 结束修改 ---
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # dict(type='Resize', scale=(256, 256), keep_ratio=True),
    # --- 修改这里 ---
    dict(type='Resize', scale=(256, 256), keep_ratio=False), # 强制所有图像都resize到 256x256
    # --- 结束修改 ---
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,  # 从24减小到8
    num_workers=2,  # 从4减小到2
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=8,  # 从24减小到8
    num_workers=2,  # 从4减小到2
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
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

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1)

model = dict(
    type='TinySSDDetector',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='TinySSD_Custom',
        input_channels=3,
        # 这个 stage_configs 定义了网络的结构和哪些是输出特征
        # (out_c, num_convs, is_feature)
        # 这需要精确匹配你原始的期望：
        # 原始：num_filters = [3, 16, 32, 64]
        # Block 0: 3 -> 16 (非特征)
        # Block 1: 16 -> 32 (非特征)
        # Block 2: 32 -> 64 (特征, idx=2)
        # Block 3: 64 -> 128 (特征, idx=3)
        # Block 4: 128 -> 128 (特征, idx=4)
        # Block 5: 128 -> 128 (特征, idx=5)
        # Block 6: AdaptiveMaxPool2d (特征, idx=6, 输入是Block 5的输出128)
        stage_configs=[
            (16, 2, False), # blocks[0]
            (32, 2, False), # blocks[1]
            (64, 2, True),  # blocks[2] -> feature 0 (原始的 idx=2)
            (128, 2, True), # blocks[3] -> feature 1 (原始的 idx=3)
            (128, 2, True), # blocks[4] -> feature 2 (原始的 idx=4)
            (128, 2, True), # blocks[5] -> feature 3 (原始的 idx=5)
            # AdaptiveMaxPool2d 会在 _make_network_layers 中被自动添加为最后一个 block
            # 并且其输出会被作为最后一个特征 (对应你原始的 idx=6)
        ],
        # AdaptiveMaxPool2d 会作用于最后一个 stage_config (即 stage_configs[5]) 的输出 (128通道)
        # 并且 AdaptiveMaxPool2d 不改变通道数，所以最终输出也是128通道
        final_conv_out_channels=128, # 这个参数现在有点冗余，因为最后一个stage的out_c决定了池化前的通道
                                   # 除非你想在最后一个downsample block和池化层之间再加卷积调整通道
        init_cfg=None # 或者你的初始化配置
    ),
    neck=None,  # 没有特殊neck
    # bbox_head=dict(
    #     type='TinySSDHead',
    #     num_classes=1,  # 香蕉只有1类,
    #     num_anchors=4,
    #     # in_channels=(16, 32, 64),
    #     in_channels=[64, 128, 128, 128, 128],  # ⭐这里一定要对齐！
    #     #####################
    #     anchor_generator=dict( # 占位，但最终需要
    #         type='SSDAnchorGenerator',
    #         strides=[c], # 示例，需要与backbone的实际下采样对应
    #         ratios=([2], [2], [2], [2], [2]), # 简化
    #         min_sizes=[20, 50, 80, 110, 140], # 示例
    #         max_sizes=[c], # 示例
    #     ),
    #     bbox_coder=dict(
    #         type='DeltaXYWHBBoxCoder',
    #         target_means=[.0, .0, .0, .0],
    #         target_stds=[1.0, 1.0, 1.0, 1.0]),
    #     #######################
    # ),
    bbox_head=dict(
        type='TinySSDHead',
        in_channels=(64, 128, 128, 128, 128),
        stacked_convs=0,       # 确保没有额外的堆叠卷积层
        num_classes=1,
        use_depthwise=False,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.001),

        # set anchor size manually instead of using the predefined
        # SSD300 setting.
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            strides=[8, 16, 32, 64, 128],
            #  这个列表中的值必须是你的backbone输出的各个特征图相对于原始输入图像的下采样倍数（步幅）。你需要验证TinySSD_Custom中，输出in_channels对应特征图的实际下采样倍数是否是这些值。
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2]],
            # ratios: 每个特征图位置生成的锚框的宽高比。
            min_sizes=[20, 50, 80, 110, 140],
            max_sizes=[50, 80, 110, 140, 170]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
            # ... 其他参数 ...
        # --- ↓↓↓ 确保这些配置确实存在于你正在运行的文件中 ↓↓↓ ---
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=1.0,
            reduction='sum',
            loss_weight=1.0),
        train_cfg=dict(
            # train_cfg (bbox_head内部): 控制训练时正负样本分配、采样等。
            assigner=dict(
                type='MaxIoUAssigner', # assigner: MaxIoUAssigner，根据IoU分配正负锚框。
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.,
                ignore_iof_thr=-1,
                gt_max_assign_all=False),
            sampler=dict(type='PseudoSampler'), # sampler: PseudoSampler，表示不过滤由assigner分配的正负样本。
            allowed_border=-1,
            pos_weight=-1,
            neg_pos_ratio=3, # neg_pos_ratio=3: 负正样本比例，控制参与损失计算的负样本数量。
            smoothl1_beta=1.0, # MMDetection 2.x SSDHead train_cfg 参数
                              # MMDetection 3.x 中，loss_bbox 的 beta 在 loss_bbox 字典内定义
            debug=False
        ),
        test_cfg=dict( # test_cfg (bbox_head内部): 控制测试/推理时的行为。
            nms_pre=1000, # nms_pre: NMS前保留的候选框数量。
            min_bbox_size=0,
            score_thr=0.02, # score_thr: 分数阈值，低于此阈值的框被过滤。
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=100  # max_per_img: 每张图片最终输出的最大检测框数量。
        )
        # --- ↑↑↑ 确保这些配置确实存在于你正在运行的文件中 ↑↑↑ ---
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        sampler=dict(type='PseudoSampler'),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200)
    # test_cfg=dict(
    #     nms_pre=1000,
    #     score_thr=0.01,  # 降低分数阈值
    #     nms=dict(type='nms', iou_threshold=0.45),  # 降低NMS阈值
    #     min_bbox_size=0,
    #     max_per_img=200)
)


optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)  # lr从0.2减小到0.1
optimizer_config = dict(grad_clip=None) # 梯度裁剪配置，None表示不进行梯度裁剪。

evaluation = dict(interval=1, metric='bbox') # 每个epoch结束后进行一次评估，评估指标是bbox。

# 修改学习率
lr_config = dict(
    policy='step',
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
#  学习率调整策略。'step'表示在指定的epoch（第8和第11个epoch结束后）降低学习率（通常是乘以0.1）。

log_config = dict(interval=50) # 每50个iteration打印一次日志。

work_dir = './work_dirs/ssd_custom'
