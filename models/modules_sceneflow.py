from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging 

from utils.interpolation import interpolate2d_as, my_grid_sample, get_grid, upsample_flow_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms
import utils.softsplat as softsplat


def merge_lists(input_list):
    """
    input_list = list:time[ list:level[4D tensor] ]
    output_list = list:level[ 4D tensor (batch*time, channel, height, width)]
    """
    len_tt = len(input_list)
    len_ll = len(input_list[0])

    output_list = []
    for ll in range(len_ll):
        list_ll = []
        for tt in range(len_tt):
            list_ll.append(input_list[tt][ll])            
        tensor_ll = torch.stack(list_ll, dim=1)
        tbb, ttt, tcc, thh, tww = tensor_ll.size()
        output_list.append(tensor_ll.reshape(tbb * ttt, tcc, thh, tww))

    return output_list


# https://github.com/google-research/google-research/blob/789d828d545dc35df8779ad4f9e9325fc2e3ceb0/uflow/uflow_model.py#L88
def compute_cost_volume(feat1, feat2, param_dict):
    """
    only implemented for:
        kernel_size = 1
        stride1 = 1
        stride2 = 1
    """

    max_disp = param_dict["max_disp"]

    _, _, height, width = feat1.size()
    num_shifts = 2 * max_disp + 1
    feat2_padded = F.pad(feat2, (max_disp, max_disp, max_disp, max_disp), "constant", 0)

    cost_list = []
    for i in range(num_shifts):
        for j in range(num_shifts):
            corr = torch.mean(feat1 * feat2_padded[:, :, i:(height + i), j:(width + j)], axis=1, keepdims=True)
            cost_list.append(corr)
    cost_volume = torch.cat(cost_list, axis=1)
    return cost_volume


# https://github.com/google-research/google-research/blob/789d828d545dc35df8779ad4f9e9325fc2e3ceb0/uflow/uflow_model.py#L44
def normalize_features(feature_list):

    statistics_mean = []
    statistics_var = []
    axes = [-3, -2, -1]

    for feature_image in feature_list:
        statistics_mean.append(feature_image.mean(dim=axes, keepdims=True))
        statistics_var.append(feature_image.var(dim=axes, keepdims=True))

    statistics_std = [torch.sqrt(v + 1e-16) for v in statistics_var]

    feature_list = [f - mean for f, mean in zip(feature_list, statistics_mean)]
    feature_list = [f / std for f, std in zip(feature_list, statistics_std)]

    return feature_list


class WarpingLayer_Flow(nn.Module):
    """
    Backward warp an input tensor "x" using the input optical "flow"
    """
    def __init__(self):
        super(WarpingLayer_Flow, self).__init__()

    def forward(self, x, flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(x.size(3) - 1, 1)
        flo_h = flow[:, 1] * 2 / max(x.size(2) - 1, 1)
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)        
        x_warp = my_grid_sample(x, grid)

        mask = torch.ones_like(x, requires_grad=False)
        mask = my_grid_sample(mask, grid)
        mask = (mask > 0.999).to(dtype=x_warp.dtype)

        return x_warp * mask


class WarpingLayer_SF(nn.Module):
    """
    Backward warp an input tensor "x" using the input "sceneflow" 
    To do so, it needs disparity (disp), camera intrinsic (k1), and input image size (input_size, for scaling the camera focal length)
    """
    def __init__(self):
        super(WarpingLayer_SF, self).__init__()
 
    def forward(self, x, sceneflow, disp, k1, input_size):

        _, _, h_x, w_x = x.size()
        disp = interpolate2d_as(disp, x) * w_x

        local_scale = torch.zeros_like(input_size)
        local_scale[:, 0] = h_x
        local_scale[:, 1] = w_x

        pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
        _, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

        grid = coord1.transpose(1, 2).transpose(2, 3)
        x_warp = my_grid_sample(x, grid)

        mask = torch.ones_like(x, requires_grad=False)
        mask = my_grid_sample(mask, grid)
        mask = (mask > 0.999).to(dtype=x_warp.dtype)

        return x_warp * mask


def initialize_msra(modules):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass


def upsample_outputs_as(input_list, ref_list):
    """
    upsample the tensor in the "input_list" with a size of tensor included in "ref_list"

    """
    output_list = []
    for ii in range(0, len(input_list)):
        output_list.append(interpolate2d_as(input_list[ii], ref_list[ii]))

    return output_list


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, padding_mode="zeros"):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=False)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)
        )


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale, padding_mode="zeros"):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size=kernel_size, padding_mode=padding_mode)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        return self.conv1(x)


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs, padding_mode="zeros"):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2, padding_mode=padding_mode),
                conv(ch_out, ch_out, padding_mode=padding_mode)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv_ii in self.convs:
            x = conv_ii(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class FeatureProp_Softsplat(nn.Module):
    """
    Forward-warp a input tensor "x" using input optical "flow"
    A depth order between colliding pixels is determined by the input "disp"
    Also, return only valid feature: it's valid only if the dot product between the feat1 and the corresponding feat2 is above a threshold, 0.5.
    """
    def __init__(self, padding_mode="zeros"):
        super(FeatureProp_Softsplat, self).__init__()
        
        self.warping_layer_flow = WarpingLayer_Flow()
        self.conv1x1 = conv(1, 1, kernel_size=1, padding_mode=padding_mode)

    def forward(self, x, flow, disp, feat1, feat2):

        # init
        flow = interpolate2d_as(flow, x, mode="bilinear")
        disp = interpolate2d_as(disp, x, mode="bilinear")

        b, _, h, w, = flow.size()
        mask = torch.ones(b, 1, h, w, dtype=flow.dtype, device=flow.device).requires_grad_(False)
        disocc = softsplat.FunctionSoftsplat(tenInput=mask, tenFlow=flow, tenMetric=None, strType='summation')
        disocc_map = (disocc > 0.5).to(dtype=flow.dtype)

        if disocc_map.sum() < (b * h * w / 2):
            return torch.zeros_like(x)
        else:
            x_warped = softsplat.FunctionSoftsplat(tenInput=x, tenFlow=flow, tenMetric=-20.0 * (0.4-disp), strType='softmax')
            feat1_warped = softsplat.FunctionSoftsplat(tenInput=feat1, tenFlow=flow, tenMetric=-20.0 * (0.4-disp), strType='softmax')
            valid_mask = (self.conv1x1((feat1_warped * feat2).sum(dim=1, keepdims=True)) > 0.5).to(dtype=x_warped.dtype)
            x_warped = x_warped * valid_mask

            return x_warped.contiguous()



class MonoSceneFlowDecoder_LSTM(nn.Module):
    """
    The split decoder model with LSTM
    """
    def __init__(self, ch_in, padding_mode="zeros"):
        super(MonoSceneFlowDecoder_LSTM, self).__init__()

        self.featprop_softsplat = FeatureProp_Softsplat(padding_mode)
        self.convs = nn.Sequential(
            conv(ch_in, 128, padding_mode=padding_mode),
            conv(128, 128, padding_mode=padding_mode),
            conv(128, 96, padding_mode=padding_mode)
        )
        self.conv_sf = nn.Sequential(
            conv(96, 64, padding_mode=padding_mode),
            conv(64, 32, padding_mode=padding_mode),
            conv(32, 3, isReLU=False, padding_mode=padding_mode)
            )
        self.conv_d1 = nn.Sequential(
            conv(96, 64, padding_mode=padding_mode),
            conv(64, 32, padding_mode=padding_mode),
            conv(32, 1, isReLU=False, padding_mode=padding_mode)
            )

        ## LSTM Cell
        self.input_dim = 96
        self.hidden_dim = 96

        self.conv_c_init =  conv(96, 96, padding_mode=padding_mode)
        self.conv_lstm = conv(self.input_dim + self.hidden_dim, 4 * self.hidden_dim, isReLU=False, padding_mode=padding_mode)
        self.cell_state = None

    def forward_lstm(self, input_tensor, h_cur, c_cur):

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv_lstm(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = nn.LeakyReLU(0.1, inplace=False)(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * nn.LeakyReLU(0.1, inplace=False)(c_next)

        return h_next, c_next


    def forward(self, x, x_out_pr=None, fl_pr=None, dp_pr=None, x0=None, x1=None):
        
        x_curr = self.convs(x)

        if x_out_pr is None:
            # initializing cell state in the begining
            self.cell_state = self.conv_c_init(x_curr)
        else:
            # forward-warp the hidden state and cell state using the estimated flow and disp
            h_pre = self.featprop_softsplat(x_out_pr, fl_pr, dp_pr, x0, x1)
            c_pre = self.featprop_softsplat(self.cell_state, fl_pr, dp_pr, x0, x1)

            x_curr, self.cell_state = self.forward_lstm(x_curr, h_pre, c_pre)

        sf = self.conv_sf(x_curr)
        disp1 = self.conv_d1(x_curr)

        return x_curr, sf, disp1