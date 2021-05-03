from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .modules_sceneflow import initialize_msra, upsample_outputs_as, compute_cost_volume, normalize_features, merge_lists, WarpingLayer_SF
from .modules_sceneflow import upconv
from .modules_sceneflow import FeatureExtractor
from .modules_sceneflow import MonoSceneFlowDecoder_LSTM

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, post_processing, projectSceneFlow2Flow


class MonoSceneFlow_Multi(nn.Module):
    def __init__(self, args):
        super(MonoSceneFlow_Multi, self).__init__()

        self._args = args

        self.num_chs = [3, 32, 64, 96, 128, 192]
        self.search_range = 4
        self.output_level = 3
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=False)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs, self._args.conv_padding_mode)
        self.warping_layer_sf = WarpingLayer_SF()

        self.flow_estimators = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        
        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = 2 * self.dim_corr + ch 
            else:
                num_ch_in = 2 * self.dim_corr + ch + 96 + 2 * 3 + 1
                self.upconv_layers.append(upconv(96, 96, 3, 2, self._args.conv_padding_mode))

            layer_sf = MonoSceneFlowDecoder_LSTM(num_ch_in, self._args.conv_padding_mode)            
            self.flow_estimators.append(layer_sf)

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}

        # using cuda-enabled correlation is up to ~1.5 times faster than the python operation
        if self._args.correlation_cuda_enabled:
            from .correlation_package.correlation import Correlation
            self.correlation_ftn = Correlation.apply
        else:
            self.correlation_ftn = compute_cost_volume
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

        # padding the input if necessary (default is "OFF")
        self._input_padding = 8 if self._args.calculate_disparity_scale else 0


    def run_multi(self, aug_size, feat_list, intrinsic, sf_pr=None, dp_pr=None, x_outs_f_pr=None, x_outs_b_pr=None):
            
        feat_pyr_0, feat_pyr_1, feat_pyr_2 = feat_list

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        x_outs_f = []
        x_outs_b = []

        # calculating flow for the forward-warping the hidden & cell state from LSTM
        if sf_pr != None:
            fl_pr = projectSceneFlow2Flow(intrinsic, sf_pr, dp_pr * dp_pr.size(-1), input_size=aug_size).detach()
        
        for l, (x0, x1, x2) in enumerate(zip(feat_pyr_0, feat_pyr_1, feat_pyr_2)):

            # warping feature maps
            if l == 0:
                x0_warp = x0
                x2_warp = x2
            else:
                sf_f = interpolate2d_as(sf_f, x1, mode="bilinear")
                sf_b = interpolate2d_as(sf_b, x1, mode="bilinear")
                dp_1 = interpolate2d_as(dp_1, x1, mode="bilinear")
                x_out_f = interpolate2d_as(self.upconv_layers[l-1](x_out_f), x1, mode="bilinear")
                x_out_b = interpolate2d_as(self.upconv_layers[l-1](x_out_b), x1, mode="bilinear")
                x2_warp = self.warping_layer_sf(x2, sf_f, dp_1, intrinsic, aug_size)  # becuase K can be changing when doing augmentation
                x0_warp = self.warping_layer_sf(x0, sf_b, dp_1, intrinsic, aug_size)

            # correlation
            x0, x0_warp, x1, x2_warp = normalize_features([x0, x0_warp, x1, x2_warp])
            out_corr_f = self.leakyRELU(self.correlation_ftn(x1, x2_warp, self.corr_params))
            out_corr_b = self.leakyRELU(self.correlation_ftn(x1, x0_warp, self.corr_params))

            # monosf LSTM decoder
            if l == 0:
                x_out_f, sf_f, dp_f = self.flow_estimators[l](torch.cat([x1, out_corr_f, out_corr_b], dim=1))
                x_out_b, sf_b, dp_b = self.flow_estimators[l](torch.cat([x1, out_corr_b, out_corr_f], dim=1))
            else:           
                if sf_pr == None:    
                    x_out_f, sf_f_res, dp_f = self.flow_estimators[l](torch.cat([x1, x_out_f, out_corr_f, out_corr_b, sf_f, -sf_b, dp_1], dim=1))
                    x_out_b, sf_b_res, dp_b = self.flow_estimators[l](torch.cat([x1, x_out_b, out_corr_b, out_corr_f, sf_b, -sf_f, dp_1], dim=1))
                else:
                    x_out_f, sf_f_res, dp_f = self.flow_estimators[l](torch.cat([x1, x_out_f, out_corr_f, out_corr_b, sf_f, -sf_b, dp_1], dim=1), x_outs_f_pr[l], fl_pr, dp_pr, x0, x1)
                    x_out_b, sf_b_res, dp_b = self.flow_estimators[l](torch.cat([x1, x_out_b, out_corr_b, out_corr_f, sf_b, -sf_f, dp_1], dim=1), x_outs_b_pr[l], fl_pr, dp_pr, x0, x1)
                sf_f = sf_f + sf_f_res
                sf_b = sf_b + sf_b_res

            # averaging disparity output, estimated in the bi-directional way
            dp_1 = (dp_f + dp_b) / 2.0

            # normalizing the dispairty (can be used when stereo baseline & focal length is not fixed), default is OFF
            if self._args.calculate_disparity_scale:
                dp_1 = nn.LayerNorm(dp_1.size()[1:], elementwise_affine=False)(dp_1)

            dp_1 = self.sigmoid(dp_1) * 0.3
            sceneflows_f.append(sf_f)
            sceneflows_b.append(sf_b)
            disps_1.append(dp_1)
            x_outs_f.append(x_out_f)
            x_outs_b.append(x_out_b)

            if l == self.output_level:                
                break

        x1_rev = feat_pyr_0[::-1]

        return upsample_outputs_as(sceneflows_f[::-1], x1_rev), upsample_outputs_as(sceneflows_b[::-1], x1_rev), upsample_outputs_as(disps_1[::-1], x1_rev), x_outs_f, x_outs_b


    def find_disparity_scale(self, feat_list_left, feat_list_right, intrinsic, aug_size):
        """
        estimate the optical flow between the left stereo image and the right stere image
        this will be used for scaling the output disparity into the actual disparity scale.
        Default: not using this function
        """
        feat_l = merge_lists(feat_list_left)
        feat_r = merge_lists(feat_list_right)

        # outputs
        sceneflows = []
        disps = []

        # bidirect
        for l, (f_l, f_r) in enumerate(zip(feat_l, feat_r)):

            xl, xr = torch.cat([f_l, f_r], dim=0), torch.cat([f_r, f_l], dim=0)
            xl_f, xr_f = torch.flip(xl, [-1]), torch.flip(xr, [-1])

            # warping
            if l == 0:
                xr_f_warp = xr_f
                xr_warp = xr
            else:
                sf_f = interpolate2d_as(sf_f, xl, mode="bilinear")
                sf_b = interpolate2d_as(sf_b, xl, mode="bilinear")
                dp_1 = interpolate2d_as(dp_1, xl, mode="bilinear")
                dp_1_flip = torch.flip(dp_1, [3])
                x_out_f = interpolate2d_as(self.upconv_layers[l-1](x_out_f), xl, mode="bilinear")
                x_out_b = interpolate2d_as(self.upconv_layers[l-1](x_out_b), xl, mode="bilinear")
                
                xr_warp = self.warping_layer_sf(xr, sf_f, dp_1, intrinsic, aug_size)  # becuase K can be changing when doing augmentation
                xr_f_warp = self.warping_layer_sf(xr_f, sf_b, dp_1_flip, intrinsic, aug_size)

            # correlation
            xl, xr_warp, xl_f, xr_f_warp = normalize_features([xl, xr_warp, xl_f, xr_f_warp])
            out_corr_f = self.leakyRELU(Correlation.apply(xl, xr_warp, self.corr_params))
            out_corr_b = self.leakyRELU(Correlation.apply(xl_f, xr_f_warp, self.corr_params))

            # monosf estimator
            if l == 0:
                x_out_f, sf_f, dp_f = self.flow_estimators[l](torch.cat([xl, out_corr_f, torch.flip(out_corr_b, [3])], dim=1))
                x_out_b, sf_b, dp_b = self.flow_estimators[l](torch.cat([xl_f, out_corr_b, torch.flip(out_corr_f, [3])], dim=1))
            else: 
                x_out_f, sf_f_res, dp_f = self.flow_estimators[l](torch.cat([xl, x_out_f, out_corr_f, torch.flip(out_corr_b, [3]), sf_f, flow_horizontal_flip(sf_b), dp_1], dim=1))
                x_out_b, sf_b_res, dp_b = self.flow_estimators[l](torch.cat([xl_f, x_out_b, out_corr_b, torch.flip(out_corr_f, [3]), sf_b, flow_horizontal_flip(sf_f), dp_1_flip], dim=1))
                sf_f = sf_f + sf_f_res
                sf_b = sf_b + sf_b_res

            dp_1 = (dp_f + torch.flip(dp_b, [3])) / 2.0
            dp_1 = nn.LayerNorm(dp_1.size()[1:], elementwise_affine=False)(dp_1)
            dp_1 = self.sigmoid(dp_1) * 0.3
            sceneflows.append(sf_f)
            disps.append(dp_1)

            if l == self.output_level:                
                break

        return upsample_outputs_as(sceneflows[::-1], feat_l[::-1]), upsample_outputs_as(disps[::-1], feat_l[::-1])


    def feature_extractor(self, images_5d):
        """
        Extracting feature.
        Output: list of feature pyramids of each image
        """
        b, t, c, h, w = images_5d.size()

        if self._input_padding != 0 and (not self.training):
            images_5d = images_5d.reshape(b*t, c, h, w)
            p_size = self._input_padding
            images_5d = F.pad(input=images_5d, pad=(p_size, p_size, p_size, p_size), mode='reflect')
            images_5d = images_5d.reshape(b, t, c, h+2*p_size, w+2*p_size)

        return [self.feature_pyramid_extractor(images_5d[:, tt]) + [images_5d[:, tt]] for tt in range(t)]


    def run_multiframe(self, aug_size, feat_list, intrinsic):
        """
        running multi-frame monocular scene flow
        """

        output_dict = {}

        sceneflow_forward = []
        sceneflow_backward = []
        disparity_1 = []

        for tt in range(1, len(feat_list) - 1):
            if tt == 1:
                sf_f, sf_b, disp_1, x_outs_f, x_outs_b = self.run_multi(aug_size, feat_list[0:3], intrinsic)
            else:
                sf_f, sf_b, disp_1, x_outs_f, x_outs_b = self.run_multi(aug_size, feat_list[tt-1:tt+2], intrinsic, sf_f[0], disp_1[0], x_outs_f, x_outs_b)
            
            sceneflow_forward.append(sf_f)
            sceneflow_backward.append(sf_b)
            disparity_1.append(disp_1)

        output_dict['sf_f'] = merge_lists(sceneflow_forward)
        output_dict['sf_b'] = merge_lists(sceneflow_backward)
        output_dict['disp_1'] = merge_lists(disparity_1)

        return output_dict


    def forward(self, input_dict):

        ## Left view
        aug_size = input_dict['aug_size']

        feat_list_left = self.feature_extractor(input_dict['input_left_aug'])
        output_dict = self.run_multiframe(aug_size, feat_list_left, input_dict['input_k_l_aug'])
        
        ## Right view: only used for obtaining the disocclusion map
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
  
            with torch.no_grad():
                feat_list_right_flip = self.feature_extractor(torch.flip(input_dict['input_right_aug'], [4]))
                output_dict_r = self.run_multiframe(aug_size, feat_list_right_flip, input_dict["input_k_r_flip_aug"])
                output_dict_r['disp_1'] = [torch.flip(dp, [3]) for dp in output_dict_r['disp_1']]
                output_dict['output_dict_r'] = output_dict_r

                if self._args.calculate_disparity_scale:
                    ## Find disparity scale, default is OFF
                    feat_list_right = self.feature_extractor(input_dict['input_right_aug'])
                    output_dict['sf_scale'], output_dict['dp_scale'] = self.find_disparity_scale(feat_list_left, feat_list_right, input_dict['input_k_l_aug'], aug_size)

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning:
            feat_list_left_flip = self.feature_extractor(torch.flip(input_dict['input_left_aug'], [4]))
            output_dict_flip = self.run_multiframe(aug_size, feat_list_left_flip, input_dict["input_k_l_flip_aug"])
            sf_f_pp = [ post_processing(sf, flow_horizontal_flip(sf_flip)) for sf, sf_flip in zip(output_dict['sf_f'], output_dict_flip['sf_f']) ]
            sf_b_pp = [ post_processing(sf, flow_horizontal_flip(sf_flip)) for sf, sf_flip in zip(output_dict['sf_b'], output_dict_flip['sf_b']) ]
            disp_1_pp = [ post_processing(dp, torch.flip(dp_flip, [3])) for dp, dp_flip in zip(output_dict['disp_1'], output_dict_flip['disp_1']) ]

            # removing the padded boundary when only needed. (default: not using)
            if self._input_padding != 0:
                p_size = self._input_padding
                sf_f_pp[0] = sf_f_pp[0][:,:, p_size:-p_size, p_size:-p_size]
                sf_b_pp[0] = sf_b_pp[0][:,:, p_size:-p_size, p_size:-p_size]
                disp_1_pp[0] = disp_1_pp[0][:,:, p_size:-p_size, p_size:-p_size]

            output_dict['sf_f_pp'] = sf_f_pp
            output_dict['sf_b_pp'] = sf_b_pp
            output_dict['disp_1_pp'] = disp_1_pp
            
        return output_dict
