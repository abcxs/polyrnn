_base_ = './polyrnn_r50_fpn_1x_building_edge.py'
polygon_size=32
model = dict(
    roi_head=dict(
        polygon_roi_extractor=dict(roi_layer=dict(output_size=polygon_size)), 
        polygon_head=dict(vertex_head=dict(polygon_size=polygon_size), polyrnn_head=dict(feat_size=polygon_size, polygon_size=polygon_size))),
    # model training and testing settings
    train_cfg=dict(rcnn=dict(polygon_size=polygon_size)),
    test_cfg=dict(rcnn=dict(polygon_size=polygon_size)))