_base_ = './polyrnn_r50_fpn_1x_building.py'
model=dict(
    roi_head=dict(
        polygon_head=dict(
            vertex_head=dict(num_convs=4, 
                             norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)), 
            polyrnn_head=dict(num_convs=4, 
                              norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),))))