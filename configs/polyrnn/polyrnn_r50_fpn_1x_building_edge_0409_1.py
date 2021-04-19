_base_ = './polyrnn_r50_fpn_1x_building_edge.py'
model=dict(roi_head=dict(polygon_head=dict(polyrnn_head=dict(use_coord=True, with_offset=True))))
load_from = './venus/0/latest.pth'