_base_ = './polyrnn_r50_fpn_1x_building.py'

model = dict(
    roi_head=dict(
        fusion_module=dict(
            type='FusionModule',
            in_channels=256,
            refine_level=0,
            refine_type=None),
        polygon_roi_extractor=dict(
            featmap_strides=[4])))