from __future__ import absolute_import, division

import torch
from torch import nn
import torch.nn.functional as F


def my_grid_sample(inputs, grid):
    return F.grid_sample(inputs, grid, align_corners=True)


def interpolate2d(inputs, size, mode="bilinear"):
    return F.interpolate(inputs, size, mode=mode, align_corners=True)


def interpolate2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return interpolate2d(inputs, [h, w], mode=mode)


def _bchw2bhwc(tensor):
    return tensor.transpose(1,2).transpose(2,3)


def _bhwc2bchw(tensor):
    return tensor.transpose(2,3).transpose(1,2)


def upsample_flow_as(flow, output_as):

    size_inputs = flow.size()[2:4]
    size_targets = output_as.size()[2:4]
    resized_flow = F.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
    # correct scaling of flow
    u, v = resized_flow.chunk(2, dim=1)
    u = u * float(size_targets[1] / size_inputs[1])
    v = v * float(size_targets[0] / size_inputs[0])
    
    return torch.cat([u, v], dim=1)


def get_grid(x):

    b, _, h, w = x.size()
    grid_H = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(b, 1, h, w).to(device=x.device, dtype=x.dtype)
    grid_V = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(b, 1, h, w).to(device=x.device, dtype=x.dtype)
    grids = torch.cat([grid_H, grid_V], dim=1).requires_grad_(False)
    
    return grids

def get_coordgrid(x):

    b, _, h, w = x.size()
    grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w).to(device=x.device, dtype=x.dtype)
    grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w).to(device=x.device, dtype=x.dtype)
    ones = torch.ones_like(grid_h)
    coordgrid = torch.cat((grid_h, grid_v, ones), dim=1).requires_grad_(False)

    return coordgrid


class Meshgrid(nn.Module):
    def __init__(self):
        super(Meshgrid, self).__init__()
        self.width = 0
        self.height = 0
        self.xx = None
        self.yy = None

    def _compute_meshgrid(self, width, height):
        rangex = torch.arange(0, width)
        rangey = torch.arange(0, height)
        xx = rangex.repeat(height, 1).contiguous()
        yy = rangey.repeat(width, 1).t().contiguous()
        self.xx = xx.view(1, 1, height, width)
        self.yy = yy.view(1, 1, height, width)

    def forward(self, width, height, device=None, dtype=None):
        if self.width != width or self.height != height:
            self._compute_meshgrid(width=width, height=height)
            self.width = width
            self.height = height
        self.xx = self.xx.to(device=device, dtype=dtype)
        self.yy = self.yy.to(device=device, dtype=dtype)
        return self.xx, self.yy

