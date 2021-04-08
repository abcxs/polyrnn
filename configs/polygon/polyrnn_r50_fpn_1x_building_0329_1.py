_base_ = './polyrnn_r50_fpn_1x_building.py'
model = dict(
    train_cfg=dict(
        rcnn=dict(
            epsilon=2)))
