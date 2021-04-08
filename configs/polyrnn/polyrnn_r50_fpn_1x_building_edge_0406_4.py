_base_ = './polyrnn_r50_fpn_1x_building_edge.py'
model=dict(roi_head=dict(polygon_head=dict(polyrnn_head=dict(with_offset=True))))