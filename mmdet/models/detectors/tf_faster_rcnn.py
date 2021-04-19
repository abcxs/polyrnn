from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class TFFasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None, 
                 backbone_extra=None, 
                 weight=None, 
                 fusion_type=1):
        super(TFFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        if backbone_extra:
            self.backbone_extra = builder.build_backbone(backbone_extra)
            self.backbone_extra.init_weights(pretrained)
        self.weight = weight
        self.fusion_type = fusion_type
        
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        img_pan = img[:, :1, :, :]
        img_extra = img[:, 1:, :, :]
        x = self.backbone(img_pan)
        if getattr(self, 'backbone_extra', None):
            x_extra = self.backbone_extra(img_extra)
        else:
            x_extra = self.backbone(img_extra)

        if self.fusion_type == 1:
            pan_weight = 1
            extra_weight = 1
            if self.weight:
                pan_weight = self.weight
                extra_weight = 1 - pan_weight
            assert pan_weight >= 0 and pan_weight <= 1
            assert extra_weight >= 0 and extra_weight <= 1
            x = [x_ * pan_weight + x_extra_ * extra_weight for x_, x_extra_ in zip(x, x_extra)]
            if self.with_neck:
                x = self.neck(x)
        else:
            x = self.neck((x, x_extra))
        return x 
