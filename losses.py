from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.interpolation import interpolate2d_as, my_grid_sample, upsample_flow_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, reconstructImg, reconstructPts, projectSceneFlow2Flow, pts2pixel
from utils.sceneflow_util import disp2depth_kitti, depth2disp_kitti
from utils.monodepth_eval import compute_errors
from models.modules_sceneflow import WarpingLayer_Flow

import utils.softsplat as softsplat

###############################################
## Basic Module 
###############################################
def _square_norm(input_tensor):
    return torch.square(torch.norm(input_tensor, p=2, dim=1, keepdim=True))

def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)

def _elementwise_robust_epe_char(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)

def _robust_l1(diff):    
    return (diff ** 2 + 0.001 ** 2) ** 0.5

def _apply_disparity(img, disp):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = my_grid_sample(img, 2 * flow_field - 1)

    return output

def _generate_image_left(img, disp):
    return _apply_disparity(img, -disp)

def _adaptive_disocc_detection(flow_in):

    b, c, h, w = flow_in.size()

    assert(c in [1,2])

    # if input is disparity
    if c == 1:
        flow = torch.zeros(b, 2, h, w, dtype=flow_in.dtype, device=flow_in.device).requires_grad_(False)
        flow[:, 0:1, :, : ] = flow_in * w
    else:
        flow = flow_in

    # init mask
    mask = torch.ones(b, 1, h, w, dtype=flow.dtype, device=flow.device).requires_grad_(False)

    # forward waring using softsplat with the summation mode
    disocc = softsplat.FunctionSoftsplat(tenInput=mask, tenFlow=flow, tenMetric=None, strType='summation')
    disocc_map = (disocc > 0.499)

    # if a half of the map is empty, just return ones (for a better convergence in the early stage of training)
    if disocc_map.to(dtype=flow.dtype).sum() < (b * h * w / 2):
        disocc_map = torch.ones(b, 1, h, w, dtype=torch.bool, device=flow.device).requires_grad_(False)
        
    return disocc_map


def _image_grads(image_batch, stride=1):
    image_batch_gh = image_batch[:, :, stride:, :] - image_batch[:, :, :-stride, :]
    image_batch_gw = image_batch[:, :, :, stride:] - image_batch[:, :, :, :-stride]
    return image_batch_gh, image_batch_gw

def _masked_loss(loss, mask, eps=1e-8):
    """
    Average the loss only for the visible area (1: visible, 0: occluded)
    """
    return (loss * mask).sum() / (mask.sum() + eps)


###############################################
## Loss function
###############################################
class TernaryCensusLoss_OccAware(nn.Module):
    """
    Calculating ternary census only on the visible area
    """
    def __init__(self, kernel_size=7):
        super(TernaryCensusLoss_OccAware, self).__init__()
                
        self._weights = None
        self._mask = None
        self._kernel_size = kernel_size
        self._pad_size = kernel_size // 2

    def hamming_distance(self, t1, t2, mask):

        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)

        ## averaging over the valid pixels
        dist_sum = (dist_norm * mask).sum(dim=1, keepdims=True) / (mask.sum(dim=1, keepdims=True) + 1e-8)

        return dist_sum

    def ternary_transform(self, img, kernel_size=7):

        img_gray = img.mean(dim=1, keepdims=True) * 255
        padded = F.pad(img_gray, pad=(self._pad_size, self._pad_size, self._pad_size, self._pad_size), mode='replicate')
        patches = F.conv2d(padded, self._weights, bias=None, stride=1, padding=0, dilation=1, groups=1)
        transf = patches - img_gray    
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))

        return transf_norm

    def zero_mask_border(self, mask):

        mask[:, :, :self._pad_size, :] = 0  # up
        mask[:, :, :, :self._pad_size] = 0  # left
        mask[:, :, -self._pad_size:, :] = 0 # bottom
        mask[:, :, :, -self._pad_size:] = 0 # right

        return mask
        
    def robust_loss(self, diff, eps=0.01, q=0.4):

        return torch.pow((diff.abs() + eps), q)

    def forward(self, img1, img2, valid_mask, kernel_size=7):
        """
        img1 & img2 : two input images that we want to calculate the ternary census loss between
        valid_mask: input occlusion mask for img1 (0: occluded, 1: visible)
        kerner_size: size of the ternary census patch. (default: 7-by-7 patch)
        """

        if (self._kernel_size != kernel_size) or (self._weights == None):
            out_channels = kernel_size * kernel_size
            self._weights = torch.eye(out_channels).reshape((out_channels, 1, kernel_size, kernel_size)).requires_grad_(False).to(device=img1.device)
            self._pad_size = kernel_size // 2                        

        ## calcualting the ternary census signature for each image
        ternary1 = self.ternary_transform(img1)
        ternary2 = self.ternary_transform(img2)

        ## expanding mask
        mask_padded = F.pad(valid_mask.to(dtype=img1.dtype), pad=(self._pad_size, self._pad_size, self._pad_size, self._pad_size), mode='replicate')
        mask_expanded = F.conv2d(mask_padded, self._weights, bias=None, stride=1, padding=0, dilation=1, groups=1)
        
        ## occlusion aware hamming distance
        dist = self.hamming_distance(ternary1, ternary2, mask_expanded)
        dist = self.robust_loss(dist)
        valid_mask = self.zero_mask_border(valid_mask)

        ## returning loss only on the visible area
        loss = _masked_loss(dist, valid_mask)

        return loss


class Loss_SceneFlow_SemiSupFinetune_Multi(nn.Module):
    """
    Semi-supervised training loss = self-supervision loss + supervised loss using GT
    """
    def __init__(self, args):
        super(Loss_SceneFlow_SemiSupFinetune_Multi, self).__init__()        

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._selfsup_loss = TernaryCensusLoss_OccAware(args)

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        selsup_loss_dict = self._selfsup_loss(output_dict, target_dict)
        selsup_loss = selsup_loss_dict['total_loss']

        # Ground Truth
        gt_disp1 = target_dict['target_disp']
        gt_disp1_mask = (target_dict['target_disp_mask']==1).to(dtype=gt_disp1.dtype)  
        gt_disp2 = target_dict['target_disp2_occ']
        gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).to(dtype=gt_disp2.dtype)   
        gt_flow = target_dict['target_flow']
        gt_flow_mask = (target_dict['target_flow_mask']==1).to(dtype=gt_flow.dtype)  
        
        # when GT is not provided
        if gt_flow_mask.sum() == 0:
            loss_dict["total_loss"] = selsup_loss
            return loss_dict

        disp_loss = 0
        flow_loss = 0
        
        ibb, itt, icc, ihh, iww = target_dict['input_left_aug'].size()
        itt_e = itt - 2
        width_dp = gt_disp1.size(3)     

        for ii, (sf_f, disp_l1) in enumerate(zip(output_dict['sf_f_pp'], output_dict['disp_1_pp'])):

            _, _, shh, sww = sf_f.size()

            sf_f = sf_f.reshape(ibb, itt_e, 3, shh, sww)[:, -1, ...]
            disp_l1 = disp_l1.reshape(ibb, itt_e, 1, shh, sww)[:, -1, ...]

            ## Disp 1
            disp_l1 = interpolate2d_as(disp_l1, gt_disp1, mode="bilinear") * width_dp
            valid_abs_rel = torch.abs(gt_disp1 - disp_l1) * gt_disp1_mask
            disp_l1_loss = valid_abs_rel[gt_disp1_mask != 0].mean()

            ## Flow Loss
            sf_f_up = interpolate2d_as(sf_f, gt_flow, mode="bilinear")
            out_flow = projectSceneFlow2Flow(target_dict['input_k_l'], sf_f_up, disp_l1)
            valid_epe = _elementwise_robust_epe_char(out_flow, gt_flow) * gt_flow_mask
            flow_l1_loss = valid_epe[gt_flow_mask != 0].mean()

            ## Disp 2
            out_depth_l1 = disp2depth_kitti(disp_l1, target_dict['input_k_l'][:, 0, 0], depth_clamp=False)
            out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
            out_depth_l1_next = out_depth_l1 + sf_f_up[:, 2:3, :, :]
            disp_l1_next = depth2disp_kitti(out_depth_l1_next, target_dict['input_k_l'][:, 0, 0], depth_clamp=False)
            valid_abs_rel = torch.abs(gt_disp2 - disp_l1_next) * gt_disp2_mask
            disp_l2_loss = valid_abs_rel[gt_disp2_mask != 0].mean()
             
            disp_loss = disp_loss + (disp_l1_loss + disp_l2_loss) * self._weights[ii]
            flow_loss = flow_loss + flow_l1_loss * self._weights[ii]

        # dynamic weighting
        u_loss = selsup_loss.detach()
        d_loss = disp_loss.detach()
        f_loss = flow_loss.detach()

        max_val = torch.max(torch.max(f_loss, d_loss), u_loss)

        u_weight = max_val / u_loss
        d_weight = max_val / d_loss 
        f_weight = max_val / f_loss 

        total_loss = selsup_loss * u_weight + disp_loss * d_weight + flow_loss * f_weight
        loss_dict["selsup_loss"] = selsup_loss
        loss_dict["dp_loss"] = disp_loss
        loss_dict["fl_loss"] = flow_loss
        loss_dict["total_loss"] = total_loss

        return loss_dict


class Loss_SceneFlow_SelfSup_Multi(nn.Module):
    """
    Self-supervised loss consisting of disparity + scene flow loss.
    """
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_Multi, self).__init__()
        
        self._args = args
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 1000
        self._census_loss = TernaryCensusLoss_OccAware(kernel_size=7)
        self._beta = 150
        self._warping_layer = WarpingLayer_Flow()

    def smoothness_loss(self, img_l1, flow_f, norm_fact=None):

        img_gy, img_gx = _image_grads(img_l1, stride=2)

        # image-edge-aware weighting
        weights_x = torch.exp(-torch.mean(torch.abs(img_gx), 1, keepdim=True) * self._beta)
        weights_y = torch.exp(-torch.mean(torch.abs(img_gy), 1, keepdim=True) * self._beta)

        # compute second derivatives of the predicted smoothness.
        flow_gy, flow_gx = _image_grads(flow_f)
        flow_gyy, _ = _image_grads(flow_gy)
        _, flow_gxx = _image_grads(flow_gx)

        # compute weighted smoothness
        if norm_fact is None:
            loss_smoothness = ((weights_x * _robust_l1(flow_gxx)).mean() + (weights_y * _robust_l1(flow_gyy)).mean()) / 2.0
        else:
            loss_smoothness = ( (weights_x * _robust_l1(flow_gxx) / (norm_fact[:, :, :, 1:-1] + 1e-8) ).mean() + ((weights_y * _robust_l1(flow_gyy)) / (norm_fact[:, :, 1:-1, :] + 1e-8)).mean() ) / 2.0

        return loss_smoothness


    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, a, b, ii):

        # scaling disparity (default: a=1, b=0)
        disp_l_s = a * disp_l + b
        disp_r_s = a * disp_r + b

        # reconstructing left view from the right image
        img_r_warp = _generate_image_left(img_r_aug, disp_l_s)
        left_occ = _adaptive_disocc_detection(disp_r_s).detach()
        
        # photometric loss
        loss_img = self._census_loss(img_l_aug, img_r_warp, left_occ.bool())

        # disparities smoothness
        loss_smooth = self.smoothness_loss(img_l_aug, disp_l) / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth, left_occ


    def sceneflow_loss(self, sf_f, sf_b, disp_l, disp_occ_l, k_l_aug, img_src, img_tgt, aug_size, ii, ibb, itt_e):

        ## channel dimension of each tensor
        # sf:       itt_e - 1
        # disp:     itt_e
        # dis_occ:  itt_e
        # img_src:  itt_e - 1
        # img_tgt:  itt_e - 1
        # k_l_aug:  itt_e - 1
        # aug_size: itt_e - 1
        
        b, c_dp, h_dp, w_dp = disp_l.size()
        _, c_dpocc, _, _ = disp_occ_l.size()

        # reshaping the channel dimension of disp to itt_e-1
        disp_l = (disp_l * w_dp)
        disp_l_decom = disp_l.reshape(ibb, itt_e, c_dp, h_dp, w_dp)
        disp_f = disp_l_decom[:, :-1, :, :, :].reshape(ibb * (itt_e-1), c_dp, h_dp, w_dp)
        disp_b = disp_l_decom[:, 1: , :, :, :].reshape(ibb * (itt_e-1), c_dp, h_dp, w_dp)

        disp_occ_decom = disp_occ_l.reshape(ibb, itt_e, c_dp, h_dp, w_dp)
        disp_occ_f = disp_occ_decom[:, :-1, :, :, :].reshape(ibb * (itt_e-1), c_dpocc, h_dp, w_dp)
        disp_occ_b = disp_occ_decom[:, 1: , :, :, :].reshape(ibb * (itt_e-1), c_dpocc, h_dp, w_dp)

        # to scale the camera focal length for resized images
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp         

        pts1, k1_scale = pixel2pts_ms(k_l_aug, disp_f, local_scale / aug_size)
        pts2, k2_scale = pixel2pts_ms(k_l_aug, disp_b, local_scale / aug_size)

        _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
        _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_f)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_b)
        occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_b
        occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_f

        # Image reconstruction loss
        img_tgt_warp = reconstructImg(coord1, img_tgt)
        img_src_warp = reconstructImg(coord2, img_src)

        loss_im1 = self._census_loss(img_src, img_tgt_warp, occ_map_f)
        loss_im2 = self._census_loss(img_tgt, img_src_warp, occ_map_b)
        loss_im = loss_im1 + loss_im2

        # Point reconstruction Loss
        pts2_warp = reconstructPts(coord1, pts2)
        pts1_warp = reconstructPts(coord2, pts1) 

        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        
        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)

        loss_pts = _masked_loss(pts_diff1, occ_map_f) + _masked_loss(pts_diff2, occ_map_b)

        # 3D motion smoothness loss
        loss_3d_s = (self.smoothness_loss(img_src, sf_f, pts_norm1) + self.smoothness_loss(img_tgt, sf_b, pts_norm2)) / (2 ** ii)

        # Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
        
        return sceneflow_loss, loss_im, loss_pts, loss_3d_s

    def fb_consistency_mask(self, input1, input2):
        return _square_norm(input1 - input2) < 0.01 * (_square_norm(input1) + _square_norm(input2)) + 0.05

    def get_validity_mask(self, disp_l, disp_r):
        disp_r_w = self._warping_layer(disp_r, disp_l)
        return self.fb_consistency_mask(disp_l, -disp_r_w)

    def compute_disparity_scale(self, output_dict, input_img, intrinsic):
        """
        (only for training in the wild, not using when training on KITTI)
        scaling the estimated disparity to the actual disparity scale
        """
        with torch.no_grad():
            dp = interpolate2d_as(output_dict["dp_scale"][0], input_img, mode="bilinear") * input_img.size(-1)
            sf = interpolate2d_as(output_dict['sf_scale'][0], input_img, mode="bilinear")
            flow = projectSceneFlow2Flow(intrinsic, sf, dp)

            b_flow = flow.size(0) // 2
            flow[:, 1:2] *= 0
            actual_dp = flow[:b_flow]
            actual_dp_mask = self.get_validity_mask(actual_dp, flow[b_flow:])

            # least square fitting: y = ax + b
            # batch-wise implementation is not ready yet
            y = -actual_dp[:, :1][actual_dp_mask] # actual disparity
            x = dp[:b_flow][actual_dp_mask] # estimated by CNN

            n = x.size(0)
            x_mean = x.mean()
            y_mean = y.mean()
            xx_sum = (x * x).sum()
            xy_sum = (x * y).sum()
            denom = xx_sum - n * x_mean * x_mean
            a = ((xy_sum - n * x_mean * y_mean) / denom).detach()
            b = ((y_mean * xx_sum - x_mean * xy_sum) / denom).detach() / input_img.size(-1)
            # dividing it by input_img.size(-1): to scale it back to the ratio of image width

        return a, b

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        ibb, itt, icc, ihh, iww = target_dict['input_left_aug'].size()

        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_sf_sm = 0

        # scaling disparity map: estimated disp ([0, 0.3]) to the input image scale 
        # default: a=1, b=0, not used for training on KITTI
        if self._args.calculate_disparity_scale:
            a, b = self.compute_disparity_scale(output_dict, target_dict['input_left_aug'][:, 0, ...], target_dict['input_k_l_aug'])
        else:
            a = 1.
            b = 0.

        # effective temporal step == the number of time step of sf_f, sf_b, and disp
        itt_e = itt - 2

        img_left_aug = target_dict["input_left_aug"][:, 1:-1, :, :, :].reshape(ibb * itt_e, icc, ihh, iww)
        img_right_aug = target_dict["input_right_aug"][:, 1:-1, :, :, :].reshape(ibb * itt_e, icc, ihh, iww)

        k_l_aug = target_dict['input_k_l_aug'].unsqueeze(1).repeat(1, itt_e - 1, 1, 1).reshape(ibb * (itt_e - 1), 3, 3)
        aug_size = target_dict['aug_size'].unsqueeze(1).repeat(1, itt_e - 1, 1).reshape(ibb * (itt_e - 1), 2)

        for ii, (sf_f, sf_b, dp_l, dp_r) in enumerate(zip(output_dict['sf_f'], output_dict['sf_b'], output_dict['disp_1'], output_dict['output_dict_r']['disp_1'])):

            assert(sf_f.size() == sf_b.size())
            assert(sf_f.size()[2:4] == dp_l.size()[2:4])
            assert(dp_l.size() == dp_r.size())

            _, _, shh, sww = sf_f.size()

            ## For image reconstruction loss
            img_l = interpolate2d_as(img_left_aug, sf_f)
            img_r = interpolate2d_as(img_right_aug, sf_f)

            ## Disp Loss
            loss_dp_l, disp_occ_l1 = self.depth_loss_left_img(dp_l, dp_r, img_l, img_r, a, b, ii)
            loss_dp_sum = loss_dp_sum + loss_dp_l * self._weights[ii]

            ## Sceneflow Loss
            img_l_decom = img_l.reshape(ibb, itt_e, icc, shh, sww)
            img_l_src = img_l_decom[:, :-1, :, :, :].reshape(ibb * (itt_e-1), icc, shh, sww)
            img_l_tgt = img_l_decom[:, 1: , :, :, :].reshape(ibb * (itt_e-1), icc, shh, sww)

            sf_f_decom = sf_f.reshape(ibb, itt_e, 3, shh, sww)
            sf_f_valid = sf_f_decom[:, :-1, :, :, :].reshape(ibb * (itt_e-1) , 3, shh, sww)

            sf_b_decom = sf_b.reshape(ibb, itt_e, 3, shh, sww)
            sf_b_valid = sf_b_decom[:, 1:, :, :, :].reshape(ibb * (itt_e-1) , 3, shh, sww)

            if target_dict['curr_epoch'] <= 2:
                dp_l = dp_l.detach()

            loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f_valid, sf_b_valid, dp_l, disp_occ_l1, 
                                                                            k_l_aug, img_l_src, img_l_tgt, aug_size, ii, ibb, itt_e)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
            loss_sf_2d = loss_sf_2d + loss_im            
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss
       
        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        if self._args.calculate_disparity_scale:
            loss_dict["a"] = a
            loss_dict["b"] = b
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        loss_dict["s_3"] = loss_sf_3d
        loss_dict["total_loss"] = total_loss

        return loss_dict



###############################################
## Eval
###############################################

def eval_module_disp_depth(gt_disp, gt_disp_mask, output_disp, gt_depth, output_depth):
    """
    Evaluating monocular depth
    """

    loss_dict = {}
    batch_size = gt_disp.size(0)
    gt_dtype = gt_disp.dtype

    gt_disp_mask_f = gt_disp_mask.to(dtype=gt_dtype)

    ## KITTI disparity metric
    d_valid_epe = _elementwise_epe(output_disp, gt_disp) * gt_disp_mask_f
    d_outlier_epe = (d_valid_epe > 3).to(dtype=gt_dtype) * ((d_valid_epe / gt_disp) > 0.05).to(dtype=gt_dtype) * gt_disp_mask_f
    loss_dict["otl"] = (d_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
    loss_dict["otl_img"] = d_outlier_epe

    ## MonoDepth metric
    abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = compute_errors(gt_depth[gt_disp_mask], output_depth[gt_disp_mask])        
    loss_dict["abs_rel"] = abs_rel
    loss_dict["sq_rel"] = sq_rel
    loss_dict["rms"] = rms
    loss_dict["log_rms"] = log_rms
    loss_dict["a1"] = a1
    loss_dict["a2"] = a2
    loss_dict["a3"] = a3

    return loss_dict


class Eval_SceneFlow_KITTI_Test_Multi(nn.Module):
    def __init__(self, args):
        super(Eval_SceneFlow_KITTI_Test_Multi, self).__init__()

        self._seq_len = args.sequence_length

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        input_img = target_dict['input_left'][:, 0, ...]
        intrinsics = target_dict['input_k_l']                

        # Disp 1
        batch_size, _, _, width = input_img.size()
        out_disp_l1 = interpolate2d_as(output_dict["disp_1_pp"][0][-1:, ...], input_img, mode="bilinear") * width
        out_depth_l1 = disp2depth_kitti(out_disp_l1, intrinsics[:, 0, 0], depth_clamp=True)
        output_dict["out_disp_l_pp"] = out_disp_l1

        # Optical Flow
        out_sceneflow = interpolate2d_as(output_dict['sf_f_pp'][0][-1:, ...], input_img, mode="bilinear")
        out_flow = projectSceneFlow2Flow(intrinsics, out_sceneflow, output_dict["out_disp_l_pp"])
        output_dict["out_sceneflow_pp"] = out_sceneflow
        output_dict["out_flow_pp"] = out_flow

        # Disp 2
        out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
        out_disp_l1_next = depth2disp_kitti(out_depth_l1_next, intrinsics[:, 0, 0], depth_clamp=False)
        output_dict["out_disp_l_pp_next"] = out_disp_l1_next

        # as no GT available, just return 0 (not to cause the runtime error)
        loss_dict["sf"] = (out_disp_l1_next * 0).sum()

        return loss_dict


class Eval_SceneFlow_KITTI_Train_Multi(nn.Module):
    def __init__(self, args):
        super(Eval_SceneFlow_KITTI_Train_Multi, self).__init__()

        self._seq_len = args.sequence_length

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        gt_flow = target_dict['target_flow']
        gt_flow_mask = (target_dict['target_flow_mask']==1).to(dtype=gt_flow.dtype)

        gt_disp = target_dict['target_disp']
        gt_disp_mask = (target_dict['target_disp_mask']==1).to(dtype=gt_disp.dtype)

        gt_disp2_occ = target_dict['target_disp2_occ']
        gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).to(dtype=gt_disp2_occ.dtype)

        gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask


        intrinsics = target_dict['input_k_l']                

        # Disp 1
        batch_size, _, _, width = gt_disp.size()

        out_disp_l1 = interpolate2d_as(output_dict["disp_1_pp"][0][-1:, ...], gt_disp, mode="bilinear") * width
        out_depth_l1 = disp2depth_kitti(out_disp_l1, intrinsics[:, 0, 0], depth_clamp=True)
        gt_depth_l1 = disp2depth_kitti(gt_disp, intrinsics[:, 0, 0], depth_clamp=False)

        dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
        
        output_dict["out_disp_l_pp"] = out_disp_l1
        output_dict["out_depth_l_pp"] = out_depth_l1

        d0_outlier_image = dict_disp0_occ['otl_img']
        loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
        loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
        loss_dict["d1"] = dict_disp0_occ['otl']

        # Optical flow
        out_sceneflow = interpolate2d_as(output_dict['sf_f_pp'][0][-1:, ...], gt_flow, mode="bilinear")
        out_flow = projectSceneFlow2Flow(intrinsics, out_sceneflow, output_dict["out_disp_l_pp"])

        valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
        loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
        output_dict["out_flow_pp"] = out_flow
        output_dict["out_sceneflow_pp"] = out_sceneflow

        gt_dtype = gt_flow.dtype
        flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
        flow_outlier_epe = (valid_epe > 3).to(dtype=gt_dtype) * ((valid_epe / flow_gt_mag) > 0.05).to(dtype=gt_dtype) * gt_flow_mask
        loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68


        # Disp 2
        out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
        out_disp_l1_next = depth2disp_kitti(out_depth_l1_next, intrinsics[:, 0, 0], depth_clamp=False)
        gt_depth_l1_next = disp2depth_kitti(gt_disp2_occ, intrinsics[:, 0, 0], depth_clamp=False)

        dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
        
        output_dict["out_disp_l_pp_next"] = out_disp_l1_next
        output_dict["out_depth_l_pp_next"] = out_depth_l1_next

        d1_outlier_image = dict_disp1_occ['otl_img']
        loss_dict["d2"] = dict_disp1_occ['otl']
        
        # Scene Flow Eval
        outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() + d1_outlier_image.bool()).to(dtype=gt_dtype) * gt_sf_mask
        loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

        return loss_dict


