_base_ = './tf_faster_rcnn_r50_fpn_1x_building.py'

model = dict(
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
    backbone_extra=None, 
    neck=[
        dict(
            type='TFFPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4),
        dict(
            type='DFPN',
            in_channels=256,
            num_levels=4,
            num_outs=5,
            refine_type='conv')
    ])
