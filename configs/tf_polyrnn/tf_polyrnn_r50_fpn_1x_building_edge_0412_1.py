_base_ = './tf_polyrnn_r50_fpn_1x_building_edge.py'
model=dict(roi_head=dict(polygon_head=dict(polyrnn_head=dict(use_coord=True, with_offset=True))))
load_from = './venus_last_tf/14/latest.pth'
img_norm_cfg = dict(
    mean=[0,], std=[255,], to_rgb=False)
train_pipeline = [
    dict(type='LoadTifFromFile', extra='fusion'),
    dict(type='LoadAnnotations', with_bbox=True, with_polygon=True),
    dict(type='Resize', img_scale=[(1333, 512), (1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800)], multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_polygons']),
]
test_pipeline = [
    dict(type='LoadTifFromFile', extra='fusion'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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
    samples_per_gpu=1,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))