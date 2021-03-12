import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from .standard_roi_head import StandardRoIHead
from ..utils import FusionModule
from .test_mixins import PolygonTestMixin

@HEADS.register_module()
class PolygonRoIHead(StandardRoIHead, PolygonTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""
    def __init__(self, polygon_head=None, polygon_roi_extractor=None, polygon_scale_factor=1, *args, **kwargs):
        super(PolygonRoIHead, self).__init__(*args, **kwargs)
        if polygon_head is not None:
            self.init_polygon_head(polygon_roi_extractor, polygon_head)

    @property
    def with_polygon(self):
        """bool: whether the RoI head contains a `polygon_head`"""
        return hasattr(self, 'polygon_head') and self.polygon_head is not None

    def init_polygon_head(self, polygon_roi_extractor, polygon_head):
        self.polygon_roi_extractor = build_roi_extractor(polygon_roi_extractor)
        self.polygon_head = build_head(polygon_head)
        # 暂时手动添加配置
        self.fusion_module = FusionModule(256, refine_level=1)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(PolygonRoIHead, self).init_weights(pretrained=pretrained)
        if self.with_polygon:
            self.polygon_head.init_weights()
            self.polygon_roi_extractor.init_weights()

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_polygons=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            gt_polygons (None | list[list[ndarry]]) : true polygons for each instance
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask or self.with_polygon:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        if self.with_polygon:
            # 有些参数，部分模型需要
            # 但由于此处只有polygon模块，故暂时严格限定
            polygon_results = self._polygon_forward_train(x, sampling_results, gt_polygons)
            losses.update(polygon_results['loss_polygon'])
        return losses

    def _polygon_forward_train(self, inputs, sampling_results, gt_polygons):
        # 由于只能保留roi与gt的交，此时需要对pos_rois进行筛选
        # train_cfg 和 test_cfg 为rcnn部分
        proposal_inds_list, polygon_targets, polygon_masks, vertex_targets = self.polygon_head.get_targets(sampling_results, gt_polygons, self.train_cfg)
        pos_rois = bbox2roi([res.pos_bboxes[proposal_inds] for res, proposal_inds in zip(sampling_results, proposal_inds_list)])

        polygon_results = self._polygon_forward(inputs, pos_rois, polygon_targets)
        loss_polygon = self.polygon_head.loss(polygon_results['polygon_pred'], polygon_results['vertex_pred'], polygon_targets, polygon_masks, vertex_targets)
        polygon_results.update(loss_polygon=loss_polygon)
        return polygon_results

    def _polygon_forward(self, inputs, rois, gt_polygons=None):
        x = self.fusion_module(inputs)
        polygon_feats = self.polygon_roi_extractor([x], rois)
        polygon_results = self.polygon_head(polygon_feats, gt_polygons)
        return polygon_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        # 注释，通过polygon代替mask
        # if not self.with_mask:
        #     return bbox_results
        # else:
        #     segm_results = self.simple_test_mask(
        #         x, img_metas, det_bboxes, det_labels, rescale=rescale)
        #     return list(zip(bbox_results, segm_results))

        if not self.with_polygon:
            return bbox_results
        else:
            polygon_results = self.simple_test_polygon(x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, polygon_results))