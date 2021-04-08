_base_ = './polyrnn_r50_fpn_1x_building.py'
model=dict(roi_head=dict(polygon_head=dict(loss_type=1)))
lr_config = dict(step=[32, 44])
runner = dict(max_epochs=48)