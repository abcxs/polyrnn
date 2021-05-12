_base_ = './faster_rcnn_r50_fpn_1x_building.py'
model = dict(backbone=dict(in_channels=4))

dataset_type = 'BuildingDataset'
data_root = 'data/jinan/'
img_norm_cfg = dict(
    mean=[0,], std=[255,], to_rgb=False)
train_pipeline = [
    dict(type='LoadTifFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadTifFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'fusion/train/train.json',
        img_prefix=data_root + 'fusion/train/JPEGImages/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'fusion/val/val.json',
        img_prefix=data_root + 'fusion/val/JPEGImages/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'fusion/val/val.json',
        img_prefix=data_root + 'fusion/val/JPEGImages/',
        pipeline=test_pipeline))