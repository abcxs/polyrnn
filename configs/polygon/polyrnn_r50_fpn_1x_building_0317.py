_base_ = './polyrnn_r50_fpn_1x_building.py'
model = dict(
    roi_head=dict(
        polygon_head=dict(
            loss_vertex=dict(loss_weight=1.0),
            loss_polygon=dict(loss_weight=0.1))))
