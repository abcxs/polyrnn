_base_ = './polyrnn_r50_fpn_1x_building.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=False)))