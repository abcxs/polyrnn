_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/building_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    pretrained='./checkpoints/resnet50-19c8e357.pth',
    roi_head=dict(
        bbox_head=dict(num_classes=1)))
optimizer = dict(lr=0.01)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])