_base_ = './polyrnn_r50_fpn_1x_building.py'
model = dict(
    train_cfg=dict(
        rcnn=dict(
            filter_multi_part=False,
            poly_iou_thresh=0.3)))
