_base_ = './polyrnn_r50_fpn_1x_building.py'
model = dict(
    roi_head=dict(
        polygon_head=dict(
            loss_vertex=dict(loss_weight=10.0))))
optimizer = dict(lr=0.005)