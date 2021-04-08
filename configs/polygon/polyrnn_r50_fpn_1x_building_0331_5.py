_base_ = './polyrnn_r50_fpn_1x_building.py'
model=dict(roi_head=dict(polygon_head=dict(vertex_head=dict(num_convs=4), polyrnn_head=dict(num_convs=4))))