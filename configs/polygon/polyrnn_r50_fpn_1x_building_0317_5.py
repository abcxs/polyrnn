_base_ = './polyrnn_r50_fpn_1x_building.py'
model=dict(roi_head=dict(polygon_head=dict(polyrnn_head=dict(use_bn=True))))
optimizer = dict(lr=0.001)