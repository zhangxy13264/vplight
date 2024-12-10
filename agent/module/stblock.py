import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))
    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x
        return x
    
class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)
        return result
    
class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), dilation=1)
        
    def forward(self, x):   
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)
        x_p = x_causal_conv[:, : self.c_out, :, :]
        x_q = x_causal_conv[:, -self.c_out:, :, :]
        x = torch.mul((x_p + x_in), torch.sigmoid(x_q))
        return x
    
class SpatialConvLayer(nn.Module):
    def __init__(self, channels, factor=1):
        super(SpatialConvLayer, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        x2 = self.conv3x3(x)
        x11 = self.softmax(self.agp(x1).reshape(b, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b, c, -1)
        x21 = self.softmax(self.agp(x2).reshape(b, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b, c, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b, 1, h, w)
        return (x * weights.sigmoid()).reshape(b, c, h, w)
    
class STConvBlock(nn.Module):
    def __init__(self, Kt, n_vertex, last_block_channel, channels, droprate):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex)
        self.spatial_conv = SpatialConvLayer(channels=channels[0]*13)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        
    def forward(self, x):
        x = self.tmp_conv1(x)
        B, C, T, N = x.shape
        W, H = int(math.sqrt(N)), int(math.sqrt(N))
        x = x.view(B, C*T, H, W)
        x = self.spatial_conv(x)
        x = x.view(B, C, T, N)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        return x
    
class OutputBlock(nn.Module):
    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, bias, droprate):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        return x

class STEncoder(nn.Module):
    def __init__(self, blocks, n_vertex, **kwargs):
        super(STEncoder, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(kwargs['Kt'], n_vertex, blocks[l][-1], blocks[l+1], kwargs['droprate']))
        self.st_blocks = nn.Sequential(*modules)
        Ko = kwargs['action_interval'] - (len(blocks) - 3) * 2 * (kwargs['Kt'] - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, kwargs['enable_bias'], kwargs['droprate'])
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=kwargs['enable_bias'])
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=kwargs['enable_bias'])
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        return x
