_base_ = './polyrnn_r50_fpn_1x_building.py'
lr_config = dict(step=[32, 44])
runner = dict(max_epochs=48)