import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn.bricks import NonLocal2d

class FusionModule(nn.Module):
    def __init__(self, in_channels, refine_level=2, refine_type=None, conv_cfg=None,
                 norm_cfg=None):
        super(FusionModule, self).__init__()
        self.in_channels = in_channels
        self.refine_level = refine_level
        self.refine_type = refine_type
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

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

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        num_levels = len(inputs)

        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        if self.refine_type is not None:
            bsf = self.refine(bsf)

        return bsf

class ConvLstmCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True):
        super(ConvLstmCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.ih = nn.Conv2d(in_channels=self.input_size, 
                            out_channels=self.hidden_size * 4, 
                            kernel_size=self.kernel_size, 
                            padding=self.padding, 
                            bias=bias)
        self.hh = nn.Conv2d(in_channels=self.hidden_size, 
                            out_channels=self.hidden_size * 4, 
                            kernel_size=self.kernel_size, 
                            padding=self.padding, 
                            bias=bias)

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.ih(input) + self.hh(hx)
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)

        return hy, cy
        

class ConvLstm(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, num_layers=1, bias=True, bidirectional=False):
        super(ConvLstm, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layer = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions

        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                cell = ConvLstmCell(layer_input_size, hidden_size, kernel_size, bias)
                setattr(self, f'cell_{layer}_{direction}', cell)

    def forward(self, input, hx=None):
        
        def _forward(cell, reverse=False):
            def forward(input, hidden):
                output = []
                steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
                for i in steps:
                    hidden = cell(input[i], hidden)
                    output.append(hidden[0])

                if reverse:
                    output.reverse()

                output = torch.cat(output, 0).view(input.size(0), *output[0].size())
                return hidden, output
            return forward

        if hx is None:
            max_batch_size = input.size(1)
            hx = input.new_zeros(self.num_layer * self.num_directions, max_batch_size, self.hidden_size, input.size(3), input.size(4), requires_grad=False)
            hx = (hx, hx)

        hx = list(zip(*hx))

        next_hidden = []
        for i in range(self.num_layer):
            all_output = []
            for j in range(self.num_directions):
                reverse = False
                if j & 1:
                    reverse = True
                cell = _forward(getattr(self, f'cell_{i}_{j}'), reverse=reverse)

                l = i * self.num_directions + j
                hy, output = cell(input, hx[l])

                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, 2)
        
        next_h, next_c = zip(*next_hidden)
        total_layers = self.num_layer * self.num_directions
        next_hidden = (
            torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
            torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
        )

        return input, next_hidden

def test_conv_lstm():
    convlstm = ConvLstm(32, 32, 3, 2, bidirectional=False)
    input = torch.randn(10, 4, 32, 8, 8)
    output, (h, c) = convlstm(input)
    print(output.shape)
    print(h.shape)
    print(c.shape)

if __name__ == '__main__':
    test_conv_lstm()