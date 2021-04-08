_base_ = './polyrnn_r50_fpn_1x_building_edge.py'

model = dict(
    roi_head=dict(
        polygon_head=dict(polyrnn_head=dict(vertex_edge_params=dict(vertex_channels=64, edge_channels=64, type=6)))))