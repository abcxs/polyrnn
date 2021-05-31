import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule, xavier_init, Conv2d, normal_init, bias_init_with_prob
from mmcv.cnn.bricks import NonLocal2d
from mmdet.core import class_to_grid, polygon_target
from mmdet.models.builder import HEADS, build_loss, build_head
from torch.nn.modules.utils import _pair
import cv2
from scipy.ndimage.morphology import distance_transform_cdt
from mmdet.models.utils import gen_gaussian_target, Bottleneck, gaussian2D

# 部分由于历史问题，去做兼容，导致额外的东西
@HEADS.register_module()
class VertexHead(nn.Module):
    def __init__(self, num_convs=2, in_channels=256, conv_kernel_size=3, conv_out_channels=256, polygon_size=None, conv_cfg=None, norm_cfg=None):
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
        self.polygon_size = polygon_size

    def init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_logits, std=0.01, bias=bias_cls)

    def forward(self, x):
        if self.polygon_size is not None and self.polygon_size != x.shape[-1]:
            size = _pair(self.polygon_size)
            x = F.interpolate(x, size=size)
        for conv in self.convs:
            x = conv(x)
        return self.conv_logits(x)
    
@HEADS.register_module()
class VertexEdgeHead(nn.Module):
    def __init__(self, num_convs=2, in_channels=256, conv_kernel_size=3, conv_out_channels=128, polygon_size=28, conv_edge_channels=64, conv_vertex_channels=64, conv_cfg=None, norm_cfg=None):
        super(VertexEdgeHead, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.conv_vertex_channels = conv_vertex_channels
        self.conv_edge_channels = conv_edge_channels

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
        self.conv_vertex = ConvModule(self.conv_out_channels, 
                                      self.conv_vertex_channels, 
                                      self.conv_kernel_size, 
                                      padding=padding, 
                                      conv_cfg=conv_cfg, 
                                      norm_cfg=norm_cfg)
        self.conv_vertex_logits = Conv2d(self.conv_vertex_channels, 1, 1)
        self.conv_edge = ConvModule(self.conv_out_channels, 
                                    self.conv_edge_channels, 
                                    self.conv_kernel_size, 
                                    padding=padding, 
                                    conv_cfg=conv_cfg, 
                                    norm_cfg=norm_cfg)
        self.conv_edge_logits = Conv2d(self.conv_edge_channels, 1, 1)

        self.polygon_size = polygon_size

    def init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_vertex_logits, std=0.01, bias=bias_cls)
        normal_init(self.conv_edge_logits, std=0.01, bias=bias_cls)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        conv_vertex = self.conv_vertex(x)
        vertex_logits = self.conv_vertex_logits(conv_vertex)
        conv_edge = self.conv_edge(x)
        edge_logits = self.conv_edge_logits(conv_edge)
        return vertex_logits, edge_logits, conv_vertex, conv_edge

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
                 polygon_size=None,
                 max_time_step=10,
                 use_attention=False,
                 attention_type=1,
                 use_coord=False,
                 coord_type=1,
                 use_bn=False,
                 conv_cfg=None,
                 norm_cfg=None, 
                 sample_vertex=None,
                 beam_step=0,
                 use_mask_pred=False,
                 weight_kernel_params=dict(kernel_size=1, type='constant'),
                 loss_type=0,
                 act_test='softmax',
                 with_offset=False,
                 dilation_params=dict(with_dilation=False, dilations=[3, 3, 3, 3], num_convs=4), 
                 vertex_edge_params=dict(vertex_channels=64, edge_channels=64, type=0)):
        super(PolyRnnHead, self).__init__()

        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.max_time_step = max_time_step
        self.use_attention = use_attention
        self.use_coord = use_coord
        self.coord_type = coord_type
        self.use_bn = use_bn
        if polygon_size is None:
            polygon_size = feat_size
        self.feat_size = feat_size
        self.polygon_size = polygon_size
        self.dilation_params = dilation_params
        self.beam_step = beam_step
        self.use_mask_pred = use_mask_pred
        self.weight_kernel_params = weight_kernel_params
        self.loss_type = loss_type
        self.act_test = act_test
        self.vertex_edge_params = vertex_edge_params
        self.with_offset = with_offset
        self.init_kernel()
        
        
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            if i == 0 and use_coord:
                in_channels += 2
            if i == 0 and (self.vertex_edge_params['type'] == 1 or self.vertex_edge_params['type'] == 3):
                in_channels += (self.vertex_edge_params['vertex_channels'] + self.vertex_edge_params['edge_channels'])
            if i == 0 and (self.vertex_edge_params['type'] == 4 or self.vertex_edge_params['type'] == 6):
                in_channels += 2
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        if dilation_params.get('with_dilation', False):
            dilations = dilation_params.get('dilations', None)
            num_convs = dilation_params.get('num_convs', 4)
            if dilations is None:
                dilations = [1 for _ in range(num_convs)]
            else:
                assert len(dilations) == num_convs
            for dilation in dilations:
                self.convs.append(Bottleneck(self.conv_out_channels, self.conv_out_channels // 2, dilation=dilation))

        self.conv_x = nn.ModuleList()
        self.conv_h = nn.ModuleList()
        self.bn_x = nn.ModuleList()
        self.bn_h = nn.ModuleList()
        self.bn_c = nn.ModuleList()
        

        padding = conv_kernel_size // 2
        for l in range(self.num_layers):
            if l != 0:
                in_channels = self.hidden_channels
            else:
                in_channels = self.conv_out_channels + 3
                if self.use_coord and self.coord_type == 2:
                    in_channels += 2
                if self.vertex_edge_params['type'] == 2 or self.vertex_edge_params['type'] == 3:
                    in_channels += (self.vertex_edge_params['vertex_channels'] + self.vertex_edge_params['edge_channels'])
                if self.vertex_edge_params['type'] == 5 or self.vertex_edge_params['type'] == 6:
                    in_channels += 2
            self.conv_x.append(Conv2d(
                in_channels, 4 * hidden_channels, kernel_size=conv_kernel_size, padding=padding))
            self.conv_h.append(Conv2d(hidden_channels, 4 * hidden_channels,
                                      kernel_size=conv_kernel_size, padding=padding))
            
            if self.use_bn:
                self.bn_x.append(nn.ModuleList([nn.BatchNorm2d(4*hidden_channels) for i in range(max_time_step - 1)]))
                self.bn_h.append(nn.ModuleList([nn.BatchNorm2d(4*hidden_channels) for i in range(max_time_step - 1)]))
                self.bn_c.append(nn.ModuleList([nn.BatchNorm2d(hidden_channels) for i in range(max_time_step - 1)]))

        if self.use_attention and attention_type != 3:
            self.conv_atten = ConvModule(
                hidden_channels * num_layers, 1, kernel_size=1, act_cfg=None)

        self.fc_out = nn.Linear(self.feat_size ** 2 *
                                hidden_channels, self.polygon_size ** 2 + 1)
        if self.with_offset:
            self.fc_offset = nn.Linear(self.feat_size ** 2 * hidden_channels, 2)
        self.attention = getattr(self, 'attention_%d' % attention_type)
        if attention_type == 3:
            self.conv_atten = ConvModule(hidden_channels * num_layers, self.conv_out_channels, kernel_size=1, act_cfg=None)
            self.atten_hidden = ConvModule(self.conv_out_channels, 1, kernel_size=1, act_cfg=None)
        if beam_step != 0:
            self.forward = self.forward_beam
            
    def init_kernel(self):
        kernel_size = self.weight_kernel_params['kernel_size']
        self.kernel_size = kernel_size
        assert kernel_size & 1
        ttype = self.weight_kernel_params['type']
        template = torch.ones(1, 1, self.polygon_size, self.polygon_size)
        if ttype == 'constant':
            kernel = torch.ones(kernel_size, kernel_size)[None][None]
            template = None
        elif ttype == 'avg_constant':
            kernel = torch.ones(kernel_size, kernel_size)[None][None]
            template = F.conv2d(template, kernel, padding=self.kernel_size // 2).squeeze()
        elif ttype == 'gaussian':
            kernel = gaussian2D(kernel_size // 2)[None][None]
            template = None
        elif ttype == 'avg_gaussian':
            kernel = gaussian2D(kernel_size // 2)[None][None]
            template = F.conv2d(template, kernel, padding=self.kernel_size // 2).squeeze()
#         kernel = kernel[None][None]
        self.score_kernel = kernel
        self.template = template
        
    def init_weights(self):
        for conv in self.conv_x:
            nn.init.kaiming_normal_(conv.weight,
                                    mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
        for conv in self.conv_h:
            nn.init.kaiming_normal_(conv.weight,
                                    mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
        if self.loss_type == 3:
            # 用focal_loss 损失，利用loss_type初始化
            bias_cls = bias_init_with_prob(0.01)
            normal_init(self.fc_out, std=0.01, bias=bias_cls)
        else:
            nn.init.normal_(self.fc_out.weight, 0, 0.01)
            nn.init.constant_(self.fc_out.bias, 0)
            if self.with_offset:
                nn.init.normal_(self.fc_offset.weight, 0, 0.01)
                nn.init.constant_(self.fc_offset.bias, 0)

    def rnn_step(self, input, cur_state, time):
        out_state = []
        for l in range(self.num_layers):
            hx, cx = cur_state[l]

            if l == 0:
                inp = input
            else:
                inp = out_state[-1][0]
            conv_x = self.conv_x[l](inp)
            if self.use_bn:
                conv_x = self.bn_x[l][time](conv_x)
            conv_h = self.conv_h[l](hx)
            if self.use_bn:
                conv_h = self.bn_h[l][time](conv_h)
            gates = conv_x + conv_h
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = forgetgate * cx + ingate * cellgate
            if self.use_bn:
                cy = self.bn_c[l][time](cy)
            hy = outgate * torch.tanh(cy)

            out_state.append([hy, cy])
        return out_state

    def attention_1(self, x, rnn_state):
        if not self.use_attention:
            return x
        n, _, h, w = x.shape
        h_cat = torch.cat([state[0] for state in rnn_state], dim=1)
        atten = self.conv_atten(h_cat)
        # nonlocal sigmoid 或者 softmax
        # 待测试
        atten = torch.softmax(atten.reshape(n, h * w), -1).view(n, 1, h, w)
        return x * atten
    
    def attention_2(self, x, rnn_state):
        if not self.use_attention:
            return x
        n, _, h, w = x.shape
        h_cat = torch.cat([state[0] for state in rnn_state], dim=1)
        atten = self.conv_atten(h_cat)
        # nonlocal sigmoid 或者 softmax
        # 待测试
        atten = torch.softmax(atten.reshape(n, h * w), -1).view(n, 1, h, w)
        return x * atten + x
    
    def attention_3(self, x, rnn_state):
        if not self.use_attention:
            return x
        n, _, h, w = x.shape
        h_cat = torch.cat([state[0] for state in rnn_state], dim=1)
        atten = F.relu(self.conv_atten(h_cat) + x)
        atten = self.atten_hidden(atten)
        atten = torch.softmax(atten.reshape(n, h * w), -1).view(n, 1, h, w)
        return x * atten
    
    def attention_4(self, x, rnn_state):
        if not self.use_attention:
            return x
        n, _, h, w = x.shape
        h_cat = torch.cat([state[0] for state in rnn_state], dim=1)
        atten = self.conv_atten(h_cat)
        # nonlocal sigmoid 或者 softmax
        # 待测试
        atten = torch.sigmoid(atten)
        return x * atten
    

    def forward(self, x, vertex_pred, gt_polygons=None, mask_pred=None, edge_pred=None, conv_vertex=None, conv_edge=None):
        # 测试流程时，gt_polygons可为None
        n, c, h, w = x.shape
        device = x.device
        # N, C
        self.score_kernel = self.score_kernel.to(device)
        if self.template is not None:
            self.template = self.template.to(device)
        
        v1 = torch.sigmoid(vertex_pred)
        if edge_pred is not None:
            v2 = torch.sigmoid(edge_pred)

        first_vertex_prob = F.sigmoid(vertex_pred)
        if self.kernel_size > 1:
            first_vertex_prob = first_vertex_prob.reshape(n, 1, self.polygon_size, self.polygon_size)
            first_vertex_prob = F.conv2d(first_vertex_prob, self.score_kernel, padding=self.kernel_size // 2)
            if self.template is not None:
                first_vertex_prob = first_vertex_prob / self.template
        first_vertex_prob = first_vertex_prob.reshape(-1, self.polygon_size ** 2)
        ppadding = vertex_pred.new_zeros([n, 1]) 
        first_vertex_prob = torch.cat([first_vertex_prob, ppadding], dim=-1)
        
        vertex_pred = vertex_pred.reshape(n, -1)
        ppadding = vertex_pred.new_zeros([n, 1]) 
        first_vertex_pred = torch.cat([vertex_pred, ppadding], dim=-1)
        
        _, first_vertex = torch.max(first_vertex_prob, -1)
        
        
        
        if self.training:
            first_vertex = gt_polygons[:, 0]
            first_vertex_prob = x.new_zeros([n, self.polygon_size ** 2 + 1])

            first_vertex_prob[range(n), first_vertex] = 1
        
       
        if self.use_coord:
            x_range = torch.linspace(-1, 1, x.shape[-1], device=x.device)
            y_range = torch.linspace(-1, 1, x.shape[-2], device=x.device)
            y_range, x_range = torch.meshgrid(y_range, x_range)
            y_range = y_range.expand([x.shape[0], 1, -1, -1])
            x_range = x_range.expand([x.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x_range, y_range], 1)
            x = torch.cat([x, coord_feat], 1)
        if self.vertex_edge_params['type'] == 1 or self.vertex_edge_params['type'] == 3:
            x = torch.cat([x, conv_vertex, conv_edge], 1)
        if self.vertex_edge_params['type'] == 4 or self.vertex_edge_params['type'] == 6:
            x = torch.cat([x, v1, v2], dim=1)
        for conv in self.convs:
            x = conv(x)
        if mask_pred is not None:
            # N, C, H, W * N, 1, H, W 
            x = x * torch.sigmoid(mask_pred)

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
        probs = []
        rnn_state = []
        offsets = []
        for _ in range(self.num_layers):
            h_i = torch.zeros(n, self.hidden_channels, h, w,
                            device=device, requires_grad=False)
            c_i = torch.zeros(n, self.hidden_channels, h, w,
                            device=device, requires_grad=False)
            rnn_state.append([h_i, c_i])

        for t in range(1, self.max_time_step):
            x = self.attention(x, rnn_state)
            if self.use_coord and self.coord_type == 2:
                inp = torch.cat([x, v_pred2, v_pred1, v_first, coord_feat], dim=1)
            else:
                inp = torch.cat([x, v_pred2, v_pred1, v_first], dim=1)
            if self.vertex_edge_params['type'] == 2 or self.vertex_edge_params['type'] == 3:
                inp = torch.cat([inp, conv_vertex, conv_edge], dim=1)
            if self.vertex_edge_params['type'] == 5 or self.vertex_edge_params['type'] == 6:
                inp = torch.cat([inp, v1, v2], dim=1)
            rnn_state = self.rnn_step(inp, rnn_state, t - 1)

            h_final = rnn_state[-1][0].view(n, -1)
            logits_t = self.fc_out(h_final)
            if self.with_offset:
                offset_t = self.fc_offset(h_final)
            
            if self.act_test == 'softmax':
                prob = F.softmax(logits_t, dim=-1)
            elif self.act_test == 'sigmoid':
                prob = F.sigmoid(logits_t)
            else:
                raise ValueError(f'{self.act_test} not in sigmoid or softmax')
            if self.kernel_size > 1:
                prob_pos = prob[:, :self.polygon_size ** 2].reshape(-1, 1, self.polygon_size, self.polygon_size)
                prob_pos = F.conv2d(prob_pos, self.score_kernel, padding=self.kernel_size // 2)
                if self.template is not None:
                    prob_pos = prob_pos / self.template
                prob_pos = prob_pos.reshape(-1, self.polygon_size ** 2)
                prob[:, :self.polygon_size ** 2] = prob_pos
            probs.append(prob)
            _, pred = torch.max(prob, dim=-1)

            logits.append(logits_t)
            if self.with_offset:
                offsets.append(offset_t)

            v_pred2 = v_pred2.copy_(v_pred1)
            if self.training:
                v_pred1 = class_to_grid(gt_polygons[:, t], v_pred1)
            else:
                v_pred1 = class_to_grid(pred, v_pred1)
        if offsets:
            offsets = torch.stack(offsets).permute(1, 0, 2)
        else:
            offsets = None
        logits = torch.stack(logits).permute(1, 0, 2)
        logits = torch.cat([first_vertex_pred.unsqueeze(1), logits], dim=1)
        probs = torch.stack(probs).permute(1, 0, 2)
        probs = torch.cat([first_vertex_prob.unsqueeze(1), probs], dim=1)
        
#         prob = F.softmax(logits, dim=-1)
#         vss, clses = prob.max(dim=-1)
#         import matplotlib.pyplot as plt
#         for logit, cs, vs in zip(prob, clses, vss):
#             for t, c, v in zip(logit, cs, vs):
#                 if c == self.polygon_size ** 2:
#                     break
#                 print(v)
#                 img = t[:-1].reshape(self.polygon_size, self.polygon_size) * 255
#                 img = img.detach().cpu().numpy().astype(np.uint8)
#                 plt.imshow(img)
#                 plt.show()
#             print('*' * 20)
        return logits, probs, offsets
    
    # 与forward不一致，未更新
    # 之前版本效果变差，可能实现有问题
    def forward_beam(self, x, vertex_pred, gt_polygons=None, mask_pred=None):
        # 测试流程时，gt_polygons可为None
        n, c, h, w = x.shape
        device = x.device
#         print(x.shape)
        
        if self.use_coord:
            x_range = torch.linspace(-1, 1, x.shape[-1], device=x.device)
            y_range = torch.linspace(-1, 1, x.shape[-2], device=x.device)
            y_range, x_range = torch.meshgrid(y_range, x_range)
            y_range = y_range.expand([x.shape[0], 1, -1, -1])
            x_range = x_range.expand([x.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x_range, y_range], 1)
            x = torch.cat([x, coord_feat], 1)
        for conv in self.convs:
            x = conv(x)
        if mask_pred and self.use_mask_pred:
            x = x * mask_pred
        
        hs = torch.zeros(self.beam_step, self.num_layers, n, self.hidden_channels, h, w, device=device)
        cs = torch.zeros(self.beam_step, self.num_layers, n, self.hidden_channels, h, w, device=device)
            
        first_vertex_prob = F.softmax(vertex_pred.reshape(n, -1), -1)
        ppadding = first_vertex_prob.new_zeros([n, 1]) 
        # n, c
        first_vertex_prob = torch.cat([first_vertex_prob, ppadding], dim=-1)
        # n, b
        first_vertex_score, first_vertex_cls = torch.topk(first_vertex_prob, self.beam_step, dim=-1)
        # n, c -> n, b, c
        first_vertex_prob = first_vertex_prob.unsqueeze(1).expand(-1, self.beam_step, -1)
        # n, time, b, [C]
        out_cls = first_vertex_cls.unsqueeze(1)
        out_score = first_vertex_score.unsqueeze(1)
        out_prob = first_vertex_prob.unsqueeze(1)
        
        # predict poly
        v_pred2 = torch.zeros(n, 1, h, w, device=device)
        v_pred1 = torch.zeros(n, 1, h, w, device=device)
        v_first = torch.zeros(n, 1, h, w, device=device)

        
        for t in range(1, self.max_time_step):
            last_pred = out_cls[:, -1]
            first_pred = out_cls[:, 0]
            last_pred2 = last_pred
            if t > 1:
                last_pred2 = out_cls[:, -2]
            
            
            cur_time_score = []
            cur_time_cls = []
            cur_time_prob = []
            cur_time_rnn_states = []
            for b in range(self.beam_step):
                v_pred1 = class_to_grid(last_pred[:, b], v_pred1)
                v_pred2 = class_to_grid(last_pred2[:, b], v_pred2)
                v_first = class_to_grid(first_pred[:, b], v_first)
                
                rnn_state = []
                for i in range(self.num_layers):
                    rnn_state.append([hs[b][i], cs[b][i]])
                
                x = self.attention(x, rnn_state)
                if self.use_coord and self.coord_type == 2:
                    inp = torch.cat([x, v_pred2, v_pred1, v_first, coord_feat], dim=1)
                else:
                    inp = torch.cat([x, v_pred2, v_pred1, v_first], dim=1)

                rnn_state = self.rnn_step(inp, rnn_state, t - 1)

                h_final = rnn_state[-1][0].view(n, -1)
                logits_t = self.fc_out(h_final)
                
                # n, c
                log_prob = F.softmax(logits_t, dim=-1)
                # n, b
                log_prob_score, log_prob_cls = torch.topk(log_prob, self.beam_step, dim=-1)
                cur_time_score.append(log_prob_score)
                cur_time_cls.append(log_prob_cls)
                # n, c
                cur_time_prob.append(log_prob)
                for i in range(self.num_layers):
                    hs[b][i] = rnn_state[i][0]
                    cs[b][i] = rnn_state[i][1]
                
            # b, n, b -> n, b, b -> n, 1, b, b
            cur_time_score = torch.stack(cur_time_score).permute(1, 0, 2).unsqueeze(1)
            cur_time_cls = torch.stack(cur_time_cls).permute(1, 0, 2).unsqueeze(1)
            # b, n, c -> n, b, c -> n, 1, b, c
            cur_time_prob = torch.stack(cur_time_prob).permute(1, 0, 2).unsqueeze(1)
            
            # n, t, beam -> n, t, beam, beam
            out_score_t = out_score.unsqueeze(-1).expand(-1, -1, -1, self.beam_step)
            out_cls_t = out_cls.unsqueeze(-1).expand(-1, -1, -1, self.beam_step)
            # n, t, beam, c
            out_prob_t = out_prob
            
            # n, t + 1, b, b -> n, t+1, b*b
            out_cls_t = torch.cat([out_cls_t, cur_time_cls], dim=1).flatten(2, 3)
            out_score_t = torch.cat([out_score_t, cur_time_score], dim=1).flatten(2, 3)
            # n, t+1, b, c -> n, t+1, b, b, c -> n, t+1, b*b, c
            out_prob_t = torch.cat([out_prob_t, cur_time_prob], dim=1).unsqueeze(-2).expand(-1, -1, -1, self.beam_step, -1).flatten(2, 3)
            
            # 寻找到第一个类别为polygon_size ** 2
            lengths = []
            for out_cls_t_b in out_cls_t:
                lengths_b = []
                for ii in range(self.beam_step ** 2):
                    ind_ = torch.where(out_cls_t_b[:, ii] == self.polygon_size ** 2)[0]
                    if len(ind_):
                        first_ = ind_[0] + 1
                    else:
                        first_ = len(out_cls_t_b)
                    lengths_b.append(first_)
                lengths.append(lengths_b)
#             lengths = np.array(lengths).reshape(n, self.beam_step ** 2)
            lengths = x.new_tensor(lengths).long()
            # n, t+1, b*b
            out_score_t_clone = out_score_t.clone()
            for out_score_t_clone_b, index_b in zip(out_score_t_clone, lengths):
                for ii in range(self.beam_step ** 2):
                    out_score_t_clone_b[index_b[ii]:, ii] = 0
            
            # n, b*b
            out_score_t_clone = out_score_t_clone.sum(dim=1)

            prob = (out_score_t_clone / lengths)
            # n, b
            b_ids = torch.arange(n)[:, None].to(device)
            _, index = torch.topk(prob, self.beam_step, dim=-1)
            out_cls_t = out_cls_t[b_ids, :, index].transpose(1, 2)
            out_score_t = out_score_t[b_ids, :, index].transpose(1, 2)
            
            out_prob = out_prob_t[b_ids, :, index].transpose(1, 2)
            out_cls = out_cls_t
            out_score = out_score_t
            
            # n, b * b
            rnn_indexs = x.new_tensor(list(range(self.beam_step))).long().unsqueeze(-1).unsqueeze(0).expand(n, self.beam_step, self.beam_step).flatten(1, 2).to(device)
            # n, b
            rnn_indexs = rnn_indexs[b_ids, index]
            # b, layers, n, c, h, w
            hs = hs.permute(2, 0, 1, 3, 4, 5)[b_ids, rnn_indexs]
            cs = cs.permute(2, 0, 1, 3, 4, 5)[b_ids, rnn_indexs]
            # n, b, layers, c, h, w
            hs = hs.permute(1, 2, 0, 3, 4, 5)
            cs = cs.permute(1, 2, 0, 3, 4, 5)

        index = out_cls == self.polygon_size ** 2
        out_score[index] = 0
        out_score_ = out_score.sum(dim=1)
        out_cls_ = torch.where(index, 1, 0)
        lengths = out_cls_.sum(dim=1)
        prob = (out_score_ / lengths)
        # n
        _, index = torch.max(prob, dim=-1)
        
        logits = out_prob[torch.arange(n), :, index]
        return logits, None

@HEADS.register_module()
class PolygonHead(nn.Module):
    def __init__(self,
                 vertex_head,
                 polyrnn_head,
                 loss_vertex=dict(
                    type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
                 loss_polygon=dict(
                     type='CrossEntropyLoss', use_mask=False, loss_weight=1.0),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_type=0, 
                 params=dict(dt_threshold=2, radius=1),):
        super(PolygonHead, self).__init__()
        self.vertex_head = build_head(vertex_head)
        polyrnn_head['loss_type'] = loss_type
        self.polyrnn_head = build_head(polyrnn_head)
        self.loss_vertex_cfg = loss_vertex
        self.loss_polygon_cfg = loss_polygon
        self.loss_vertex = build_loss(loss_vertex)
        self.loss_polygon = build_loss(loss_polygon)
        self.loss_offset = build_loss(loss_offset)
        self.loss_type = loss_type
        self.params = params
        
    
    def loss_1(self, polygon_pred, polygon_targets, polygon_masks, polygon_vertex_targets=None):
        h, w = _pair(self.train_cfg.polygon_size)
        dt_threshold = self.params['dt_threshold']
        batch_size = polygon_pred.shape[0]
        def get_weight(polygon_targets):
            batch_targets = []
            
            for b in range(polygon_targets.shape[0]):
                target_ins = []
                for p in polygon_targets[b]:
                    t = np.zeros(h * w + 1)
                    t[p] = 1
                    if p != h * w:
                        spatial_part = t[:-1].reshape(h, w)
                        spatial_part = -1 * (spatial_part - 1)
                        spatial_part = distance_transform_cdt(spatial_part, metric='taxicab').astype(np.float32)
                        spatial_part = np.clip(spatial_part, 0, dt_threshold)
                        spatial_part /= dt_threshold
                        spatial_part = -1. * (spatial_part - 1.)
                        spatial_part /= np.sum(spatial_part)
                        spatial_part = spatial_part.flatten()
                        t = np.concatenate([spatial_part, [0.]], axis=-1)
                    target_ins.append(t.astype(np.float32))
                batch_targets.append(target_ins)
            batch_targets = np.array(batch_targets).astype(np.float32)
            batch_targets = torch.from_numpy(batch_targets).reshape(batch_size, -1, h * w + 1).to(polygon_targets.device)
            return batch_targets
        targets = get_weight(polygon_targets)
        polygon_pred = polygon_pred.reshape(-1, polygon_pred.size(-1))
        targets = targets.reshape(-1, polygon_pred.size(-1))
        polygon_masks = polygon_masks.reshape(-1)
        
        polygon_loss = torch.sum(-targets * F.log_softmax(polygon_pred, dim=1), dim=1) * polygon_masks
        polygon_loss = torch.mean(torch.sum(polygon_loss.view(batch_size, -1), dim=1))
        return polygon_loss
    
    def loss_2(self, polygon_pred, polygon_targets, polygon_masks, polygon_vertex_targets=None):
        radius = self.params['radius']
        h, w = _pair(self.train_cfg.polygon_size)
        batch_size = polygon_pred.shape[0]
        def get_targets(polygon_targets):
            batch_targets = []
            for b in range(polygon_targets.shape[0]):
                target_ins = []
                for p in polygon_targets[b]:
                    t = np.zeros(h * w + 1)
                    
                    if p != h * w:
                        x = p % w
                        y = p // w
                        spatial_part = t[:-1].reshape(h, w)
                        spatial_part = gen_gaussian_target(torch.tensor(spatial_part), (x, y), radius).numpy()
                        spatial_part = spatial_part.flatten()
                        spatial_part[p] = 0
                        spatial_part = 1 - spatial_part
                        t = np.concatenate([spatial_part, [1.]], axis=-1)
                    target_ins.append(t.astype(np.float32))
                batch_targets.append(target_ins)
            batch_targets = np.array(batch_targets).astype(np.float32)
            batch_targets = torch.from_numpy(batch_targets).reshape(batch_size, -1, h * w + 1).to(polygon_targets.device)
            return batch_targets
        targets = get_targets(polygon_targets)
        targets = targets.reshape(-1, polygon_pred.size(-1))
        polygon_pred = polygon_pred.reshape(-1, polygon_pred.size(-1))
        polygon_targets = polygon_targets.reshape(-1)
        polygon_masks = polygon_masks.reshape(-1)
        
        polygon_loss = self.loss_polygon(
            polygon_pred * targets, polygon_targets, polygon_masks, reduction_override='none')
        polygon_loss = torch.sum(polygon_loss.view(batch_size, -1), dim=-1)
        polygon_loss = torch.mean(polygon_loss)
        return polygon_loss
    
    def loss_3(self, polygon_pred, polygon_targets, polygon_masks, polygon_vertex_targets=None):
        batch_size = polygon_pred.shape[0]
        polygon_pred = polygon_pred.reshape(-1, polygon_pred.size(-1))
        polygon_masks = polygon_masks.reshape(-1)
        polygon_vertex_targets = polygon_vertex_targets.reshape(-1, polygon_pred.size(-1))
        polygon_pred = torch.sigmoid(polygon_pred)
        assert self.loss_polygon_cfg['type'] == 'GaussianFocalLoss'
        # N * len, C
        # N, len, C
        
        polygon_loss = self.loss_polygon(polygon_pred, polygon_vertex_targets, polygon_masks[:, None], reduction_override='none')
        polygon_loss = torch.sum(polygon_loss.view(batch_size, -1), dim=-1)
        polygon_loss = torch.mean(polygon_loss)
        return polygon_loss
        
    def loss_0(self, polygon_pred, polygon_targets, polygon_masks, polygon_vertex_targets=None):
        batch_size = polygon_pred.shape[0]
        polygon_pred = polygon_pred.reshape(-1, polygon_pred.size(-1))
        polygon_targets = polygon_targets.reshape(-1)
        polygon_masks = polygon_masks.reshape(-1)
        polygon_loss = self.loss_polygon(
            polygon_pred, polygon_targets, polygon_masks, reduction_override='none')
        polygon_loss = torch.sum(polygon_loss.view(batch_size, -1), dim=-1)
        polygon_loss = torch.mean(polygon_loss)
        return polygon_loss

    def init_weights(self):
        self.vertex_head.init_weights()
        self.polyrnn_head.init_weights()

    def forward(self, inputs, gt_polygons=None, mask_pred=None):
        x = inputs
        n, _, h, w = x.shape
        result = self.vertex_head(x)
        edge_pred = None
        conv_vertex = None
        conv_edge = None
        if isinstance(result, tuple):
            vertex_pred, edge_pred, conv_vertex, conv_edge = result
        else:
            vertex_pred = result

        polygon_pred, polygon_prob, offset_pred = self.polyrnn_head(x, vertex_pred, gt_polygons, mask_pred, edge_pred, conv_vertex, conv_edge)
        results = dict(polygon_pred=polygon_pred, vertex_pred=vertex_pred, polygon_prob=polygon_prob, edge_pred=edge_pred, offset_pred=offset_pred)
        return results

    def get_targets(self, sampling_results, gt_polygons, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        # list[tensor] [batch, max_poly_len] [batch, max_poly_len]
        proposal_inds_list, polygon_targets, polygon_masks, vertex_targets, polygon_vertex_targets, edge_targets, offset_targets = polygon_target(
            pos_proposals, pos_assigned_gt_inds, gt_polygons, rcnn_train_cfg)
        return proposal_inds_list, polygon_targets, polygon_masks, vertex_targets, polygon_vertex_targets, edge_targets, offset_targets

    def loss(self, polygon_pred, vertex_pred, edge_pred, offset_pred, polygon_targets, polygon_masks, vertex_targets, polygon_vertex_targets, edge_targets, offset_targets):
        batch_size = polygon_pred.size(0)
        # b, T, C
        # b, T
        # b, T
        polygon_pred = polygon_pred[:, 1:]
        polygon_targets = polygon_targets[:, 1:]
        polygon_masks = polygon_masks[:, 1:]
        polygon_vertex_targets = polygon_vertex_targets[:, 1:]
        offset_targets = offset_targets[:, 1:]
        
        polygon_loss = getattr(self, f'loss_{self.loss_type}')(polygon_pred, polygon_targets, polygon_masks, polygon_vertex_targets)

        vertex_pred = torch.sigmoid(vertex_pred)
        # N, 1, H, W 
        # N, H, W
        vertex_loss = self.loss_vertex(vertex_pred, vertex_targets[:, None])
        
        ret_loss = dict(polygon_loss=polygon_loss, vertex_loss=vertex_loss)
        if offset_pred is not None:
            offset_loss = self.loss_offset(offset_pred, offset_targets, polygon_masks[..., None], reduction_override='none')
            offset_loss = torch.sum(offset_loss.reshape(batch_size, -1), dim=-1)
            offset_loss = torch.mean(offset_loss)
            ret_loss['offset_loss'] = offset_loss
        
        if edge_pred is not None:
            edge_pred = torch.sigmoid(edge_pred)
            edge_loss = self.loss_vertex(edge_pred, edge_targets[:, None])
            ret_loss['edge_loss'] = edge_loss

        return ret_loss


    def get_polyons(self, first_vertex, polygon_pred, polygon_prob, polygon_offset, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale, num_classes=1, mask_format=True, det_others=None):
        assert mask_format is True
        # [len(det_bboxes), time_step, class]
        polygon_pred = torch.softmax(polygon_pred, dim=-1)
        polygon_pred = torch.argmax(polygon_pred, dim=-1)
        
        if polygon_prob is not None:
            polygon_pred = torch.argmax(polygon_prob, dim=-1)
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

        all_poly_xy = []
        for i in range(N):
            x1, y1, x2, y2 = bboxes[i]
            # [time_step]
#             print(polygon_pred.shape)
            polygon_ins = polygon_pred[i]
#             polygon_ins = polygon_pred[i][1:]
            
            valid_poly = np.where(polygon_ins == grid_w * grid_h)[0]
            if len(valid_poly):
                valid_poly = valid_poly[0]
                polygon_ins = polygon_ins[:valid_poly]

#             polygon_ins = [first_vertex[i].item()] + polygon_ins.tolist()
            polygon_ins = polygon_ins.tolist()
            poly_xy = []
            
            for p in polygon_ins:
                x = p % grid_w
                y = p // grid_w
                poly_xy.append([x, y])
            # if det_others[i] >= 0.9:
            #     print(det_others[i], len(poly_xy), poly_xy , [x1, y1])
            
            poly_xy = np.array(poly_xy).reshape(-1, 2).astype(np.float32)
            if polygon_offset is not None and len(poly_xy) > 0:
                poly_offset = polygon_offset[i].detach().cpu().numpy().astype(np.float32)
                poly_offset = poly_offset[:len(poly_xy) - 1]
                poly_xy[1:] = poly_xy[1:] + poly_offset
            poly_xy = poly_xy * ([(x2 - x1) / grid_w, (y2 - y1) / grid_w]) + [x1, y1]
            all_poly_xy.append(poly_xy)
            poly_xy = poly_xy.astype(np.int32)
            if len(poly_xy) > 2:
                im_mask[i] = cv2.fillPoly(im_mask[i], [poly_xy], 255)

        polyon_segms = [[] for _ in range(num_classes)]  # BG is not included in num_classes
        # 0520 增加返回点的具体位置
        # mask再拟合会使得点数增多
        polygon_points = [[] for _ in range(num_classes)]
        for i in range(N):
            polyon_segms[labels[i]].append(im_mask[i])
            polygon_points[labels[i]].append(all_poly_xy[i].reshape(-1).tolist())
        return polyon_segms, polygon_points
