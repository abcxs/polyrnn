_base_ = './tf_faster_rcnn_r50_fpn_1x_building.py'

model = dict(
    neck=[
        dict(
            type='TFFPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4),
        dict(
            type='DFPN',
            in_channels=256,
            num_levels=4,
            num_outs=5,
            refine_type='conv')
    ])