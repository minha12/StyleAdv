import math
import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import (
    get_blocks,
    bottleneck_IR,
    bottleneck_IR_SE,
    _upsample_add,
)
from models.decoder.model import EqualLinear


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [
            Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        ]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode="ir", opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_layer(x)
        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = _upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = _upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class Encoder4Editing(Module):
    def __init__(self, num_layers, mode="ir", opts=None):
        super(Encoder4Editing, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def get_deltas_starting_dimensions(self):
        """Get a list of the initial dimension of every delta from which it is applied"""
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        features = c3
        for i in range(1, 18):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        return w

