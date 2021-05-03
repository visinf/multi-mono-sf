from __future__ import absolute_import, division

import torch
import torch.nn as nn
import numpy as np

from utils.interpolation import interpolate2d, my_grid_sample
from utils.interpolation import Meshgrid


class PhotometricAugmentation(nn.Module):
    def __init__(self):
        super(PhotometricAugmentation, self).__init__()

        self._min_gamma = 0.8
        self._max_gamma = 1.2
        self._min_brght = 0.5
        self._max_brght = 2.0
        self._min_shift = 0.8
        self._max_shift = 1.2

        self._intv_gamma = self._max_gamma - self._min_gamma
        self._intv_brght = self._max_brght - self._min_brght
        self._intv_shift = self._max_shift - self._min_shift

    def forward(self, *args):

        _, orig_c, _, _ = args[0].size()
        num_splits = len(args)
        concat_data = torch.cat(args, dim=1)

        d_dtype = concat_data.dtype
        d_device = concat_data.device
        b, c, h, w = concat_data.size()
        num_images = int(c / orig_c)

        rand_gamma = torch.rand([b, 1, 1, 1], dtype=d_dtype, device=d_device, requires_grad=False) * self._intv_gamma + self._min_gamma
        rand_brightness = torch.rand([b, 1, 1, 1], dtype=d_dtype, device=d_device, requires_grad=False) * self._intv_brght + self._min_brght
        rand_shift = torch.rand([b, 3, 1, 1], dtype=d_dtype, device=d_device, requires_grad=False) * self._intv_shift + self._min_shift

        # gamma
        concat_data = concat_data ** rand_gamma.expand(-1, c, h, w)

        # brightness
        concat_data = concat_data * rand_brightness.expand(-1, c, h, w)

        # color shift
        rand_shift = rand_shift.expand(-1, -1, h, w)
        rand_shift = torch.cat([rand_shift for i in range(0, num_images)], dim=1)
        concat_data = concat_data * rand_shift

        # clip
        concat_data = torch.clamp(concat_data, 0, 1)
        split = torch.chunk(concat_data, num_splits, dim=1)

        return split


class _IdentityParams(nn.Module):
    def __init__(self):
        super(_IdentityParams, self).__init__()
        self._batch_size = 0
        self._device = None
        self._o = None
        self._i = None
        self._identity_params = None

    def _update(self, batch_size, device, dtype):
        self._o = torch.zeros([batch_size, 1, 1], device=device, dtype=dtype)
        self._i = torch.ones([batch_size, 1, 1], device=device, dtype=dtype)
        r1 = torch.cat([self._i, self._o, self._o], dim=2)
        r2 = torch.cat([self._o, self._i, self._o], dim=2)
        r3 = torch.cat([self._o, self._o, self._i], dim=2)
        return torch.cat([r1, r2, r3], dim=1)

    def forward(self, batch_size, device, dtype):
        if self._batch_size != batch_size or self._device != device or self._dtype != dtype:
            self._identity_params = self._update(batch_size, device, dtype)
            self._batch_size = batch_size
            self._device = device
            self._dtype = dtype

        return self._identity_params.clone()


def _intrinsic_scale(intrinsic, sx, sy):
    out = intrinsic.clone()
    out[:, 0, 0] *= sx
    out[:, 0, 2] *= sx
    out[:, 1, 1] *= sy
    out[:, 1, 2] *= sy
    return out


def _intrinsic_crop(intrinsic, str_x, str_y):    
    out = intrinsic.clone()
    out[:, 0, 2] -= str_x
    out[:, 1, 2] -= str_y
    return out



######################################################

class Augmentation_ScaleCrop(nn.Module):
    def __init__(self, args, photometric=True, trans=0.07, scale=[0.93, 1.0], resize=[256, 832]):
        super(Augmentation_ScaleCrop, self).__init__()

        # init
        self._args = args
        self._photometric = photometric
        self._photo_augmentation = PhotometricAugmentation()

        self._batch = None
        self._device = None
        self._dtype = None
        self._identity = _IdentityParams()
        self._meshgrid = Meshgrid()

        # Augmentation Parameters
        self._min_scale = scale[0]
        self._max_scale = scale[1]
        self._max_trans = trans
        self._resize = resize

    def compose_params(self, scale, rot, tx, ty):
        return torch.cat([scale, rot, tx, ty], dim=1)

    def decompose_params(self, params):
        return params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]

    def find_invalid(self, img_size, params):

        scale, _, tx, ty = self.decompose_params(params)

        ## Intermediate image
        intm_size_h = torch.floor(img_size[0] * scale)
        intm_size_w = torch.floor(img_size[1] * scale)

        ## 4 representative points of the intermediate images
        hf_h = (intm_size_h - 1.0) / 2.0
        hf_w = (intm_size_w - 1.0) / 2.0        
        hf_h.unsqueeze_(1)
        hf_w.unsqueeze_(1)
        hf_o = torch.zeros_like(hf_h)
        hf_i = torch.ones_like(hf_h)
        pt_mat = torch.cat([torch.cat([hf_w, hf_o, hf_o], dim=2), torch.cat([hf_o, hf_h, hf_o], dim=2), torch.cat([hf_o, hf_o, hf_i], dim=2)], dim=1)
        ref_mat = torch.ones(self._batch, 4, 3, device=self._device)
        ref_mat[:, 1, 1] = -1
        ref_mat[:, 2, 0] = -1
        ref_mat[:, 3, 0] = -1
        ref_mat[:, 3, 1] = -1
        ref_pts = torch.matmul(ref_mat, pt_mat).transpose(1,2)

        ## Perform trainsform
        tform_mat = self._identity(self._batch, self._device, self._dtype)
        tform_mat[:, 0, 2] = tx[:, 0]
        tform_mat[:, 1, 2] = ty[:, 0]   
        pts_tform = torch.matmul(tform_mat, ref_pts)

        ## Check validity: whether the 4 representative points are inside of the original images
        img_hf_h = (img_size[0] - 1.0) / 2.0
        img_hf_w = (img_size[1] - 1.0) / 2.0
        x_tf = pts_tform[:, 0, :]
        y_tf = pts_tform[:, 1, :]

        invalid = (((x_tf <= -img_hf_w) | (y_tf <= -img_hf_h) | (x_tf >= img_hf_w) | (y_tf >= img_hf_h)).sum(dim=1, keepdim=True) > 0).float()

        return invalid

    def calculate_tform_and_grids(self, img_size, resize, params):

        intm_scale, _, tx, ty = self.decompose_params(params)

        ## Intermediate image
        intm_size_h = torch.floor(img_size[0] * intm_scale)
        intm_size_w = torch.floor(img_size[1] * intm_scale)
        scale_x = intm_size_w / resize[1]
        scale_y = intm_size_h / resize[0]

        ## Coord of the resized image
        grid_ww, grid_hh = self._meshgrid(resize[1], resize[0], self._device, self._dtype)
        grid_ww = (grid_ww - (resize[1] - 1.0) / 2.0)
        grid_hh = (grid_hh - (resize[0] - 1.0) / 2.0)
        grid_pts = torch.cat([grid_ww, grid_hh, torch.ones_like(grid_hh)], dim=1).expand(self._batch, -1, -1, -1)

        ## 1st - scale_tform -> to intermediate image
        scale_tform = self._identity(self._batch, self._device, self._dtype)
        scale_tform[:, 0, 0] = scale_x[:, 0]
        scale_tform[:, 1, 1] = scale_y[:, 0]
        pts_tform = torch.matmul(scale_tform, grid_pts.view(self._batch, 3, -1))

        ## 2st - trans and rotate -> to original image (each pixel contains the coordinates in the original images)
        tr_tform = self._identity(self._batch, self._device, self._dtype)
        tr_tform[:, 0, 2] = tx[:, 0]
        tr_tform[:, 1, 2] = ty[:, 0]
        pts_tform = torch.matmul(tr_tform, pts_tform).view(self._batch, 3, resize[0], resize[1])

        grid_img_ww = pts_tform[:, 0, :, :] / img_size[1] * 2    # x2 is for scaling [-1. 1]
        grid_img_hh = pts_tform[:, 1, :, :] / img_size[0] * 2

        grid_img = torch.cat([grid_img_ww.unsqueeze(3), grid_img_hh.unsqueeze(3)], dim=3)

        return grid_img


    def find_aug_params(self, img_size, resize):

        ## Init
        scale = torch.zeros(self._batch, 1, device=self._device)
        rot = torch.zeros_like(scale)
        tx = torch.zeros_like(scale)
        ty = torch.zeros_like(scale)

        params = self.compose_params(scale, rot, tx, ty)

        invalid = torch.ones_like(scale)
        max_trans = torch.ones_like(scale) * self._max_trans 

        ## find params
        # scale: for the size of intermediate images (original * scale = intermediate image)
        # rot and trans: rotating and translating of the intermedinate image
        # then resize the augmented images into the resize image

        while invalid.sum() > 0:

            scale.uniform_(self._min_scale, self._max_scale)
            max_t = torch.min(torch.ones_like(scale) - scale, max_trans) * 0.5  # 0.5 because the translation rage is [-0.5*trans, 0.5*trans]
            tx = tx.uniform_(-1.0, 1.0) * max_t * img_size[1]
            ty = ty.uniform_(-1.0, 1.0) * max_t * img_size[0]
            params_new = self.compose_params(scale, rot, tx, ty)
            params = invalid * params_new + (1 - invalid) * params

            invalid = self.find_invalid(img_size, params)

        return params

    def augment_intrinsic_matrices(self, intrinsics, num_splits, img_size, resize, params):

        ### Finding the starting pt in the Original Image

        intm_scale, _, tx, ty = self.decompose_params(params)

        ## Intermediate image: finding scale from "Resize" to "Intermediate Image"
        intm_size_h = torch.floor(img_size[0] * intm_scale)
        intm_size_w = torch.floor(img_size[1] * intm_scale)
        scale_x = intm_size_w / resize[1]
        scale_y = intm_size_h / resize[0]

        ## Coord of the resized image
        pt_o = torch.zeros([1, 1], device=self._device, dtype=self._dtype)
        grid_ww = (pt_o - (resize[1] - 1.0) / 2.0).unsqueeze(0)
        grid_hh = (pt_o - (resize[0] - 1.0) / 2.0).unsqueeze(0)
        grid_pts = torch.cat([grid_ww, grid_hh, torch.ones_like(grid_hh)], dim=0).unsqueeze(0).expand(self._batch, -1, -1, -1)

        ## 1st - scale_tform -> to intermediate image
        scale_tform = self._identity(self._batch, self._device, self._dtype)
        scale_tform[:, 0, 0] = scale_x[:, 0]
        scale_tform[:, 1, 1] = scale_y[:, 0]
        pts_tform = torch.matmul(scale_tform, grid_pts.view(self._batch, 3, -1))

        ## 2st - trans and rotate -> to original image (each pixel contains the coordinates in the original images)
        tr_tform = self._identity(self._batch, self._device, self._dtype)
        tr_tform[:, 0, 2] = tx[:, 0]
        tr_tform[:, 1, 2] = ty[:, 0]
        pts_tform = torch.matmul(tr_tform, pts_tform)
        str_p_ww = pts_tform[:, 0, :] + torch.ones_like(pts_tform[:, 0, :]) * img_size[1] * 0.5 
        str_p_hh = pts_tform[:, 1, :] + torch.ones_like(pts_tform[:, 1, :]) * img_size[0] * 0.5

        ## Cropping
        intrinsics[:, :, 0, 2] -= str_p_ww[:, 0:1].expand(-1, num_splits)
        intrinsics[:, :, 1, 2] -= str_p_hh[:, 0:1].expand(-1, num_splits)

        ## Scaling        
        intrinsics[:, :, 0, 0] = intrinsics[:, :, 0, 0] / scale_x
        intrinsics[:, :, 1, 1] = intrinsics[:, :, 1, 1] / scale_y
        intrinsics[:, :, 0, 2] = intrinsics[:, :, 0, 2] / scale_x
        intrinsics[:, :, 1, 2] = intrinsics[:, :, 1, 2] / scale_y

        return intrinsics


class Augmentation_SceneFlow_MultiFrame(Augmentation_ScaleCrop):
    def __init__(self, args, photometric=True, trans=0.07, scale=[0.93, 1.0], resize=[256, 832]):
        super(Augmentation_SceneFlow_MultiFrame, self).__init__(
            args, 
            photometric=photometric, 
            trans=trans, 
            scale=scale, 
            resize=resize)


    def forward(self, example_dict):

        # Param init
        im_left = example_dict["input_left"]
        im_right = example_dict["input_right"]
        k_l = example_dict["input_k_l"].clone()
        k_r = example_dict["input_k_r"].clone()
        
        self._batch, seq_dim, ch_dim, h_orig, w_orig = im_left.size()
        self._device = im_left.device
        self._dtype = im_left.dtype

        ## Finding out augmentation parameters
        params = self.find_aug_params([h_orig, w_orig], self._resize)
        coords = self.calculate_tform_and_grids([h_orig, w_orig], self._resize, params)
        params_scale, _, _, _ = self.decompose_params(params)

        ## Augment images
        im_left_m = im_left.reshape(self._batch, seq_dim * ch_dim, h_orig, w_orig)
        im_left = my_grid_sample(im_left_m, coords).reshape(self._batch, seq_dim, ch_dim, self._resize[0], self._resize[1])
        im_right_m = im_right.view(self._batch, seq_dim * ch_dim, h_orig, w_orig)
        im_right = my_grid_sample(im_right_m, coords).reshape(self._batch, seq_dim, ch_dim, self._resize[0], self._resize[1])

        ## Augment intrinsic matrix         
        k_list = [k_l.unsqueeze(1), k_r.unsqueeze(1)]
        num_splits = len(k_list)
        intrinsics = torch.cat(k_list, dim=1)
        intrinsics = self.augment_intrinsic_matrices(intrinsics, num_splits, [h_orig, w_orig], self._resize, params)
        k_l, k_r = torch.chunk(intrinsics, num_splits, dim=1)
        k_l = k_l.squeeze(1)
        k_r = k_r.squeeze(1)


        if self._photometric and torch.rand(1) > 0.5:
            im_left = im_left.permute(0,2,3,4,1).reshape(self._batch, ch_dim, self._resize[0], self._resize[1] * seq_dim)
            im_right = im_right.permute(0,2,3,4,1).reshape(self._batch, ch_dim, self._resize[0], self._resize[1] * seq_dim)
            im_left, im_right = self._photo_augmentation(im_left, im_right)
            im_left = im_left.reshape(self._batch, ch_dim, self._resize[0], self._resize[1], seq_dim).permute(0,4,1,2,3).contiguous()
            im_right = im_right.reshape(self._batch, ch_dim, self._resize[0], self._resize[1], seq_dim).permute(0,4,1,2,3).contiguous()

        ## construct updated dictionaries        
        example_dict["input_coords"] = coords
        example_dict["input_aug_scale"] = params_scale
        
        example_dict["input_left_aug"] = im_left
        example_dict["input_right_aug"] = im_right
        
        example_dict["input_k_l_aug"] = k_l
        example_dict["input_k_r_aug"] = k_r
        
        k_l_flip = k_l.clone()
        k_r_flip = k_r.clone()
        width_aug = im_left.size(4)
        k_l_flip[:, 0, 2] = width_aug - k_l_flip[:, 0, 2]
        k_r_flip[:, 0, 2] = width_aug - k_r_flip[:, 0, 2]

        example_dict["input_k_l_flip_aug"] = k_l_flip
        example_dict["input_k_r_flip_aug"] = k_r_flip

        aug_size = torch.zeros_like(example_dict["input_size"])
        aug_size[:, 0] = self._resize[0]
        aug_size[:, 1] = self._resize[1]
        example_dict["aug_size"] = aug_size

        return example_dict

class Augmentation_Resize_Only_MultiFrame(nn.Module):
    def __init__(self, args, photometric=False, resize=[256, 832]):
        super(Augmentation_Resize_Only_MultiFrame, self).__init__()

        # init
        self._args = args
        self._imgsize = resize
        self._isRight = False
        self._photometric = photometric
        self._photo_augmentation = PhotometricAugmentation()

    def forward(self, example_dict):

        if 'input_right' in example_dict:
            self._isRight = True

        # Focal length rescaling
        im_left = example_dict["input_left"]

        if self._isRight:
            im_right = example_dict["input_right"]
            

        self._batch, seq_dim, ch_dim, hh, ww = im_left.size()
        self._device = im_left.device
        self._dtype = im_left.dtype
        sy = self._imgsize[0] / hh
        sx = self._imgsize[1] / ww

        # Image resizing
        im_left_merged = im_left.reshape(self._batch * seq_dim, ch_dim, hh, ww)
        im_left = interpolate2d(im_left_merged, self._imgsize).reshape(self._batch, seq_dim, ch_dim, self._imgsize[0], self._imgsize[1])
        k_l = _intrinsic_scale(example_dict["input_k_l"], sx, sy)
        
        if self._isRight:
            im_right_merged = im_right.reshape(self._batch * seq_dim, ch_dim, hh, ww)
            im_right = interpolate2d(im_right_merged, self._imgsize).reshape(self._batch, seq_dim, ch_dim, self._imgsize[0], self._imgsize[1])
            k_r = _intrinsic_scale(example_dict["input_k_r"], sx, sy)

        if self._photometric and torch.rand(1) > 0.5:
            im_left = im_left.permute(0,2,3,4,1).reshape(self._batch, ch_dim, self._imgsize[0], self._imgsize[1] * seq_dim)
            im_left = self._photo_augmentation(im_left)[0]
            im_left = im_left.reshape(self._batch, ch_dim, self._imgsize[0], self._imgsize[1], seq_dim).permute(0,4,1,2,3).contiguous()

            if self._isRight:
                im_right = im_right.permute(0,2,3,4,1).reshape(self._batch, ch_dim, self._imgsize[0], self._imgsize[1] * seq_dim)
                im_right = self._photo_augmentation(im_right)[0]
                im_right = im_right.reshape(self._batch, ch_dim, self._imgsize[0], self._imgsize[1], seq_dim).permute(0,4,1,2,3).contiguous()

        example_dict["input_left_aug"] = im_left
        example_dict["input_k_l_aug"] = k_l

        if self._isRight:
            example_dict["input_right_aug"] = im_right
            example_dict["input_k_r_aug"] = k_r

        k_l_flip = k_l.clone()
        k_l_flip[:, 0, 2] = im_left.size(4) - k_l_flip[:, 0, 2]
        example_dict["input_k_l_flip_aug"] = k_l_flip

        if self._isRight:
            k_r_flip = k_r.clone()
            k_r_flip[:, 0, 2] = im_right.size(4) - k_r_flip[:, 0, 2]
            example_dict["input_k_r_flip_aug"] = k_r_flip

        aug_size = torch.zeros_like(example_dict["input_size"])
        aug_size[:, 0] = self._imgsize[0]
        aug_size[:, 1] = self._imgsize[1]
        example_dict["aug_size"] = aug_size

        return example_dict

## Only for finetuning. Because the sparse GT cannot be interpolated, we just use cropping
class Augmentation_SceneFlow_Finetuning_MultiFrame(nn.Module):
    def __init__(self, args, photometric=True, resize=[256, 832]):
        super(Augmentation_SceneFlow_Finetuning_MultiFrame, self).__init__()

        # init
        self._args = args
        self._photometric = photometric
        self._photo_augmentation = PhotometricAugmentation()
        self._imgsize = resize


    def cropping(self, img, str_x, str_y, end_x, end_y):

        return img[..., str_y:end_y, str_x:end_x]

    def kitti_random_crop(self, example_dict):

        im_left = example_dict["input_left"]
        _, _, _, im_hh, im_ww = im_left.size()

        scale = np.random.uniform(0.94, 1.00)        
        crop_height = int(scale * im_hh)
        crop_width = int(scale * im_ww)

        # get starting positions
        x = np.random.uniform(0, im_ww - crop_width + 1)
        y = np.random.uniform(0, im_hh - crop_height + 1)
        str_x = int(x)
        str_y = int(y)
        end_x = int(x + crop_width)
        end_y = int(y + crop_height)

        ## Cropping
        example_dict["input_left_aug"] = self.cropping(example_dict["input_left"], str_x, str_y, end_x, end_y)
        example_dict["input_right_aug"] = self.cropping(example_dict["input_right"], str_x, str_y, end_x, end_y)

        example_dict["target_flow"] = self.cropping(example_dict["target_flow"], str_x, str_y, end_x, end_y)
        example_dict["target_flow_mask"] = self.cropping(example_dict["target_flow_mask"], str_x, str_y, end_x, end_y)
        example_dict["target_flow_noc"] = self.cropping(example_dict["target_flow_noc"], str_x, str_y, end_x, end_y)
        example_dict["target_flow_mask_noc"] = self.cropping(example_dict["target_flow_mask_noc"], str_x, str_y, end_x, end_y)
        
        example_dict["target_disp"] = self.cropping(example_dict["target_disp"], str_x, str_y, end_x, end_y)
        example_dict["target_disp_mask"] = self.cropping(example_dict["target_disp_mask"], str_x, str_y, end_x, end_y)
        example_dict["target_disp2_occ"] = self.cropping(example_dict["target_disp2_occ"], str_x, str_y, end_x, end_y)
        example_dict["target_disp2_mask_occ"] = self.cropping(example_dict["target_disp2_mask_occ"], str_x, str_y, end_x, end_y)

        example_dict["target_disp_noc"] = self.cropping(example_dict["target_disp_noc"], str_x, str_y, end_x, end_y)
        example_dict["target_disp_mask_noc"] = self.cropping(example_dict["target_disp_mask_noc"], str_x, str_y, end_x, end_y)
        example_dict["target_disp2_noc"] = self.cropping(example_dict["target_disp2_noc"], str_x, str_y, end_x, end_y)
        example_dict["target_disp2_mask_noc"] = self.cropping(example_dict["target_disp2_mask_noc"], str_x, str_y, end_x, end_y)

        ## will be used for finetuning supervised loss
        example_dict["input_k_l"] = _intrinsic_crop(example_dict["input_k_l"], str_x, str_y)
        example_dict["input_k_r"] = _intrinsic_crop(example_dict["input_k_r"], str_x, str_y)

        input_size = example_dict["input_size"].clone()
        input_size[:, 0] = crop_height
        input_size[:, 1] = crop_width
        example_dict["input_size"] = input_size

        return 


    def forward(self, example_dict):

        ## KITTI Random Crop
        self.kitti_random_crop(example_dict)

        # Image resizing
        self._batch, seq_dim, ch_dim, hh_crop, ww_crop = example_dict["input_left_aug"].size()
        im_left_merged = example_dict["input_left_aug"].reshape(self._batch * seq_dim, ch_dim, hh_crop, ww_crop)
        im_left = interpolate2d(im_left_merged, self._imgsize).reshape(self._batch, seq_dim, ch_dim, self._imgsize[0], self._imgsize[1])
       
        im_right_merged = example_dict["input_right_aug"].reshape(self._batch * seq_dim, ch_dim, hh_crop, ww_crop)
        im_right = interpolate2d(im_right_merged, self._imgsize).reshape(self._batch, seq_dim, ch_dim, self._imgsize[0], self._imgsize[1])

        # Focal length rescaling
        sy = self._imgsize[0] / hh_crop
        sx = self._imgsize[1] / ww_crop
        k_l = _intrinsic_scale(example_dict["input_k_l"], sx, sy)
        k_r = _intrinsic_scale(example_dict["input_k_r"], sx, sy)

        if self._photometric and torch.rand(1) > 0.5:

            im_left = im_left.permute(0,2,3,4,1).reshape(self._batch, ch_dim, self._imgsize[0], self._imgsize[1] * seq_dim)
            im_right = im_right.permute(0,2,3,4,1).reshape(self._batch, ch_dim, self._imgsize[0], self._imgsize[1] * seq_dim)
            im_left, im_right = self._photo_augmentation(im_left, im_right)
            im_left = im_left.reshape(self._batch, ch_dim, self._imgsize[0], self._imgsize[1], seq_dim).permute(0,4,1,2,3).contiguous()
            im_right = im_right.reshape(self._batch, ch_dim, self._imgsize[0], self._imgsize[1], seq_dim).permute(0,4,1,2,3).contiguous()

        ## Saving
        example_dict["input_left_aug"] = im_left
        example_dict["input_right_aug"] = im_right
        example_dict["input_k_l_aug"] = k_l
        example_dict["input_k_r_aug"] = k_r

        ## Flipping intrinsic
        k_l_flip = k_l.clone()
        k_r_flip = k_r.clone()
        k_l_flip[:, 0, 2] = im_left.size(4) - k_l_flip[:, 0, 2]
        k_r_flip[:, 0, 2] = im_right.size(4) - k_r_flip[:, 0, 2]
        example_dict["input_k_l_flip_aug"] = k_l_flip
        example_dict["input_k_r_flip_aug"] = k_r_flip

        aug_size = torch.zeros_like(example_dict["input_size"])
        aug_size[:, 0] = self._imgsize[0]
        aug_size[:, 1] = self._imgsize[1]
        example_dict["aug_size"] = aug_size

        return example_dict


