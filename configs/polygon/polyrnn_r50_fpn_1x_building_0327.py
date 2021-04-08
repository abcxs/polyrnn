_base_ = './polyrnn_r50_fpn_1x_building.py'
model=dict(
    roi_head=dict(
        polygon_head=dict(
            loss_type=3, 
            loss_polygon=dict(_delete_=True, type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
            polyrnn_head=dict(act_test='sigmoid'))))