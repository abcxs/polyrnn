import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule, xavier_init, Conv2d
from mmcv.cnn.bricks import NonLocal2d
from mmdet.core import class_to_grid, polygon_target
from mmdet.models.builder import HEADS, build_loss, build_head
from torch.nn.modules.utils import _pair
import cv2
@HEADS.register_module()
class VertexHead(nn.Module):
    def __init__(self, num_convs=2, in_channels=256, conv_kernel_size=3, conv_out_channels=256, conv_cfg=None, norm_cfg=None):
        super(VertexHead, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        self.conv_logits = Conv2d(self.conv_out_channels, 1, 1)

    def init_weights(self):
        # convs已被初始化
        nn.init.kaiming_normal_(self.conv_logits.weight,
                                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv_logits.bias, 0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return self.conv_logits(x)

@HEADS.register_module()
class PolyRnnHead(nn.Module):
    def __init__(self,
                 num_convs=2,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=128,
                 hidden_channels=64,
                 num_layers=2,
                 feat_size=7,
                 max_time_step=10,
                 use_attention=False,
                 conv_cfg=None,
                 norm_cfg=None):
        super(PolyRnnHead, self).__init__()

        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.feat_size = feat_size
        self.max_time_step = max_time_step
        self.use_attention = use_attention

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        self.conv_x = nn.ModuleList()
        self.conv_h = nn.ModuleList()

        padding = conv_kernel_size // 2
        for l in range(self.num_layers):
            if l != 0:
                in_channels = self.hidden_channels
            else:
                in_channels = self.conv_out_channels + 3

            self.conv_x.append(Conv2d(
                in_channels, 4 * hidden_channels, kernel_size=conv_kernel_size, padding=padding))
            self.conv_h.append(Conv2d(hidden_channels, 4 * hidden_channels,
                                      kernel_size=conv_kernel_size, padding=padding))

        if self.use_attention:
            self.conv_atten = ConvModule(
                hidden_channels * num_layers, 1, kernel_size=1)

        self.fc_out = nn.Linear(self.feat_size ** 2 *
                                hidden_channels, self.feat_size ** 2 + 1)

    def init_weights(self):
        for conv in self.conv_x:
            nn.init.kaiming_normal_(conv.weight,
                                    mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
        for conv in self.conv_h:
            nn.init.kaiming_normal_(conv.weight,
                                    mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
        nn.init.normal_(self.fc_out.weight, 0, 0.01)
        nn.init.constant_(self.fc_out.bias, 0)

    def rnn_step(self, input, cur_state):
        out_state = []
        for l in range(self.num_layers):
            hx, cx = cur_state[l]

            if l == 0:
                inp = input
            else:
                inp = out_state[-1][0]

            gates = self.conv_x[l](inp) + self.conv_h[l](hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = forgetgate * cx + ingate * cellgate
            hy = outgate * torch.tanh(cy)

            out_state.append([hy, cy])
        return out_state

    def attention(self, x, rnn_state):
        if not self.use_attention:
            return x
        n, _, h, w = x.shape
        h_cat = torch.cat([state[0] for state in rnn_state], dim=1)
        atten = self.conv_atten(h_cat)
        # nonlocal sigmoid 或者 softmax
        # 待测试
        atten = torch.softmax(atten.reshape(n, h * w), -1).view(n, 1, h, w)
        return x * atten

    def forward(self, x, first_vertex, gt_polygons=None):
        # 测试流程时，gt_polygons可为None
        n, c, h, w = x.shape
        device = x.device

        for conv in self.convs:
            x = conv(x)

        # predict poly
        v_pred2 = torch.zeros(n, 1, h, w, device=device)
        v_pred1 = torch.zeros(n, 1, h, w, device=device)
        v_first = torch.zeros(n, 1, h, w, device=device)

        # Initialize
        v_pred1 = class_to_grid(first_vertex, v_pred1)
        v_first = class_to_grid(first_vertex, v_first)

        # 第一个点通过其他模块得到
        # 因此，此时存储第二个step之后的结果
        logits = []
        rnn_state = []
        for _ in range(self.num_layers):
            h_i = torch.zeros(n, self.hidden_channels, h, w,
                            device=device, requires_grad=False)
            c_i = torch.zeros(n, self.hidden_channels, h, w,
                            device=device, requires_grad=False)
            rnn_state.append([h_i, c_i])

        for t in range(1, self.max_time_step):
            x = self.attention(x, rnn_state)
            inp = torch.cat([x, v_pred2, v_pred1, v_first], dim=1)

            rnn_state = self.rnn_step(inp, rnn_state)

            h_final = rnn_state[-1][0].view(n, -1)
            logits_t = self.fc_out(h_final)

            log_prob = F.log_softmax(logits_t, dim=-1)
            _, pred = torch.max(log_prob, dim=-1)

            logits.append(logits_t)

            v_pred2 = v_pred2.copy_(v_pred1)
            if self.training:
                v_pred1 = class_to_grid(gt_polygons[:, t], v_pred1)
            else:
                v_pred1 = class_to_grid(pred, v_pred1)

        logits = torch.stack(logits).permute(1, 0, 2)
        return logits

@HEADS.register_module()
class PolygonHead(nn.Module):
    def __init__(self,
                 vertex_head,
                 polyrnn_head,
                 loss_vertex=dict(
                    type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
                 loss_polygon=dict(
                     type='CrossEntropyLoss', use_mask=False, loss_weight=1.0)):
        super(PolygonHead, self).__init__()
        self.vertex_head = build_head(vertex_head)
        self.polyrnn_head = build_head(polyrnn_head)
        self.loss_vertex = build_loss(loss_vertex)
        self.loss_polygon = build_loss(loss_polygon)

    def init_weights(self):
        self.vertex_head.init_weights()
        self.polyrnn_head.init_weights()

    def forward(self, inputs, gt_polygons=None):
        x = inputs
        n, _, h, w = x.shape
        vertex_pred = self.vertex_head(x)

        first_vertex_prob = F.log_softmax(vertex_pred.reshape(n, -1), -1)
        _, first_vertex = torch.max(first_vertex_prob, -1)

        if self.training:
            first_vertex = gt_polygons[:, 0]

        polygon_pred = self.polyrnn_head(x, first_vertex, gt_polygons)
        results = dict(polygon_pred=polygon_pred, vertex_pred=vertex_pred, first_vertex=first_vertex)
        return results

    def get_targets(self, sampling_results, gt_polygons, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        # list[tensor] [batch, max_poly_len] [batch, max_poly_len]
        proposal_inds_list, polygon_targets, polygon_masks, vertex_targets = polygon_target(
            pos_proposals, pos_assigned_gt_inds, gt_polygons, rcnn_train_cfg)
        return proposal_inds_list, polygon_targets, polygon_masks, vertex_targets

    def loss(self, polygon_pred, vertex_pred, polygon_targets, polygon_masks, vertex_targets):
        batch_size = polygon_pred.size(0)

        polygon_pred = polygon_pred.reshape(-1, polygon_pred.size(-1))
        polygon_targets = polygon_targets[:, 1:].reshape(-1)
        polygon_masks = polygon_masks[:, 1:].reshape(-1)

        polygon_loss = self.loss_polygon(
            polygon_pred, polygon_targets, polygon_masks, reduction_override='none')
        polygon_loss = torch.sum(polygon_loss.view(batch_size, -1), dim=-1)
        polygon_loss = torch.mean(polygon_loss)

        vertex_pred = torch.sigmoid(vertex_pred)
        vertex_loss = self.loss_vertex(vertex_pred, vertex_targets[:, None])

        return dict(polygon_loss=polygon_loss, vertex_loss=vertex_loss)


    def get_polyons(self, first_vertex, polygon_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale, num_classes=1, mask_format=True, det_others=None):
        assert mask_format is True
        # [len(det_bboxes), time_step, class]
        polygon_pred = torch.softmax(polygon_pred, dim=-1)
        polygon_pred = torch.argmax(polygon_pred, dim=-1)
        polygon_pred = polygon_pred.detach().cpu().numpy()

        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            if isinstance(scale_factor, float):
                img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
                img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            else:
                w_scale, h_scale = scale_factor[0], scale_factor[1]
                img_h = np.round(ori_shape[0] * h_scale.item()).astype(
                    np.int32)
                img_w = np.round(ori_shape[1] * w_scale.item()).astype(
                    np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        grid_w, grid_h = _pair(rcnn_test_cfg.polygon_size)
        N = len(polygon_pred)
        im_mask = np.zeros((N, img_h, img_w), dtype=np.uint8)
        bboxes = bboxes.detach().cpu().numpy()

        for i in range(N):
            x1, y1, x2, y2 = bboxes[i]
            # [time_step]
            polygon_ins = polygon_pred[i]
            valid_poly = np.where(polygon_ins == grid_w * grid_h)[0]
            if len(valid_poly):
                valid_poly = valid_poly[0]
                polygon_ins = polygon_ins[:valid_poly]

            polygon_ins = [first_vertex[i].item()] + polygon_ins.tolist()
            poly_xy = []
            
            for p in polygon_ins:
                x = p % grid_w
                y = p // grid_w
                poly_xy.append([x, y])
            # if det_others[i] >= 0.9:
            #     print(det_others[i], len(poly_xy), poly_xy , [x1, y1])
            poly_xy = np.array(poly_xy).reshape(-1, 2) * ([(x2 - x1) / grid_w, (y2 - y1) / grid_w]) + [x1, y1]
            poly_xy = poly_xy.astype(np.int32)
            im_mask[i] = cv2.fillPoly(im_mask[i], [poly_xy], 255)

        polyon_segms = [[] for _ in range(num_classes)]  # BG is not included in num_classes

        for i in range(N):
            polyon_segms[labels[i]].append(im_mask[i])
        return polyon_segms
