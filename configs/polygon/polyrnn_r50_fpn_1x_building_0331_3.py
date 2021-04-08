_base_ = './polyrnn_r50_fpn_1x_building.py'
model = dict(
    roi_head=dict(
        polygon_roi_extractor=dict(roi_layer=dict(output_size=24)), 
        polygon_head=dict(vertex_head=dict(polygon_size=24), polyrnn_head=dict(feat_size=24, polygon_size=24))),
    # model training and testing settings
    train_cfg=dict(rcnn=dict(polygon_size=24,)),
    test_cfg=dict(rcnn=dict(polygon_size=24)))