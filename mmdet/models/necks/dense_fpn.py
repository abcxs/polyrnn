import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn.bricks import NonLocal2d

from ..builder import NECKS


@NECKS.register_module()
class DFPN(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    the paper `Libra R-CNN: Towards Balanced Learning for Object Detection
    <https://arxiv.org/abs/1904.02701>`_ for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
        add_extra_convs (bool): it decides whether to add conv
            layers on top of the original feature maps.
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 num_outs=5,
                 relu_before_extra_convs=False,
                 add_extra_convs=False):
        super(DFPN, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_type = refine_type
        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2d(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
            
        self.num_outs = num_outs
        self.add_extra_convs = add_extra_convs
        self.extra_convs = nn.ModuleList()
        extra_levels = num_outs - num_levels
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                extra_conv = ConvModule(
                    in_channels,
                    in_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.extra_convs.append(extra_conv)

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats = []
        for level in range(self.num_levels):
            gather_size = inputs[level].size()[2:]
            bfeats = []
            for i in range(self.num_levels):
                if i < level:
                    gathered = F.adaptive_max_pool2d(
                        inputs[i], output_size=gather_size)
                else:
                    gathered = F.interpolate(
                        inputs[i], size=gather_size, mode='nearest')
                bfeats.append(gathered)
            bsf = sum(bfeats) / len(bfeats)
            feats.append(bsf)
            
        outs = []
        if self.refine_type is not None:
            for i in range(len(feats)):
                feat = self.refine(feats[i]) + inputs[i]
                outs.append(feat)
        
        if self.num_outs > len(outs):
            extra_num_ous = self.num_outs - len(outs)
            if not self.add_extra_convs:
                for i in range(extra_num_ous):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                extra_source = outs[-1]
                outs.append(self.extra_convs[0](extra_source))
                for i in range(1, extra_num_ous):
                    if self.relu_before_extra_convs:
                        outs.append(self.extra_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.extra_convs[i](outs[-1]))

        return tuple(outs)

