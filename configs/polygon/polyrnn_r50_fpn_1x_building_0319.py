_base_ = './polyrnn_r50_fpn_1x_building.py'
model=dict(roi_head=dict(polygon_scale_factor=1.2))