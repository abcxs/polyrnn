_base_ = './polyrnn_r50_fpn_1x_building.py'
model=dict(roi_head=dict(polygon_head=dict(polyrnn_head=dict(use_attention=True, attention_type=2))))