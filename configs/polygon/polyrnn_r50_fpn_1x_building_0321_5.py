_base_ = './polyrnn_r50_fpn_1x_building.py'
model = dict(
    pretrained='./checkpoints/resnet101-5d3b4d8f.pth',
    backbone=dict(depth=101))