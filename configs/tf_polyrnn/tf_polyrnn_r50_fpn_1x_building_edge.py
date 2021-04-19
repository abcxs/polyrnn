_base_ = [
    '../_base_/datasets/building_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='TFFasterRCNN',
    pretrained='./checkpoints/resnet50-19c8e357.pth',
    backbone=dict(
        type='TFResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        in_channels=1,
        extra_channels=4),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='PolygonRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        fusion_module=dict(
            type='FusionModule',
            in_channels=256,
            refine_level=1,
            refine_type=None),
        polygon_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=28, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]), 
        polygon_head=dict(
            type='PolygonHead', 
            vertex_head=dict(
                type='VertexEdgeHead',
                num_convs=4, 
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                polygon_size=28,
                conv_edge_channels=64, 
                conv_vertex_channels=64
            ),
            polyrnn_head=dict(
                type='PolyRnnHead',
                num_convs=4, 
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                feat_size=28,
                polygon_size=28,
                max_time_step=20,
                dilation_params=dict(with_dilation=False, dilations=[3, 3, 3, 3], num_convs=4),
                weight_kernel_params=dict(kernel_size=1, type='constant'),
                act_test='softmax',
                vertex_edge_params=dict(vertex_channels=64, edge_channels=64, type=5)
            ),
            loss_vertex=dict(
                type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=10.0),
            loss_polygon=dict(
                type='CrossEntropyLoss', use_mask=False, loss_weight=1.0), 
            loss_type=0)),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            epsilon=1,
            remove_abundant=False,
            filter_multi_part=True,
            poly_iou_thresh=0.0,
            poly_radius=1,
            polygon_size=28,
            max_polygon_len=20,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            polygon_size=28,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))


dataset_type = 'BuildingDataset'
data_root = 'data/jinan/'
img_norm_cfg = dict(
    mean=[0,], std=[255,], to_rgb=False)
train_pipeline = [
    dict(type='LoadTifFromFile', extra='fusion'),
    dict(type='LoadAnnotations', with_bbox=True, with_polygon=True),
    dict(type='Resize', img_scale=(1333, 512), keep_ratio=True),
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
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'pan/train/train.json',
        img_prefix=data_root + 'pan/train/JPEGImages/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'pan/val/val.json',
        img_prefix=data_root + 'pan/val/JPEGImages/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'pan/val/val.json',
        img_prefix=data_root + 'pan/val/JPEGImages/',
        pipeline=test_pipeline))


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

evaluation = dict(metric=['bbox', 'segm'])

optimizer = dict(lr=0.005)
lr_config = dict(step=[16, 22])
runner = dict(max_epochs=24)