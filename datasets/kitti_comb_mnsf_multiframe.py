from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte
from .common import kitti_crop_image_list, kitti_adjust_intrinsic

## Combining datasets
from .kitti_2015_train_multiframe import KITTI_2015_MonoSceneFlow_Multi
from .kitti_raw_multiframe import KITTI_Raw_Multi
from torch.utils.data.dataset import ConcatDataset



class KITTI_Raw_for_Finetune_Multi(KITTI_Raw_Multi):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1,
                 index_file=""):
        super(KITTI_Raw_for_Finetune_Multi, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file=index_file)

    def __getitem__(self, index):
        
        self._seq_num = int(torch.randint(0, self._seq_dim, (1,)))
        
        index = index % self._size[self._seq_num]
        seq_list_l = self._seq_lists_l[self._seq_num][index]
        seq_list_r = self._seq_lists_r[self._seq_num][index]

        # read images
        img_list_l_np = [read_image_as_byte(img) for img in seq_list_l]
        img_list_r_np = [read_image_as_byte(img) for img in seq_list_r]

        # example filename
        im_l1_filename = seq_list_l[0]
        basename = os.path.basename(im_l1_filename)[:6]
        dirname = os.path.dirname(im_l1_filename)[-51:]
        datename = dirname[:10]
        k_l1 = torch.from_numpy(self.intrinsic_dict_l[datename]).float()
        k_r1 = torch.from_numpy(self.intrinsic_dict_r[datename]).float()
        
        # input size
        h_orig, w_orig, _ = img_list_l_np[0].shape
        input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

        # cropping 
        if self._preprocessing_crop:

            # get starting positions
            crop_height = self._crop_size[0]
            crop_width = self._crop_size[1]
            x = np.random.uniform(0, w_orig - crop_width + 1)
            y = np.random.uniform(0, h_orig - crop_height + 1)
            crop_info = [int(x), int(y), int(x + crop_width), int(y + crop_height)]

            # cropping images and adjust intrinsic accordingly
            img_list_l_np = kitti_crop_image_list(img_list_l_np, crop_info)
            img_list_r_np = kitti_crop_image_list(img_list_r_np, crop_info)
            k_l1, k_r1 = kitti_adjust_intrinsic(k_l1, k_r1, crop_info)
        
        # to tensors [t, c, h, w]
        imgs_l_tensor = torch.stack([self._to_tensor(img) for img in img_list_l_np], dim=0)
        imgs_r_tensor = torch.stack([self._to_tensor(img) for img in img_list_r_np], dim=0)
        
        ## void ground truth for the last frame being used for supervised loss
        void_tensor_ch1 = imgs_l_tensor[0, 0:1, ...] * 0
        void_tensor_ch2 = imgs_l_tensor[0, 0:2, ...] * 0


        common_dict = {
            "index": index,
            "basename": basename, 
            "input_size": input_im_size,
            "target_flow": void_tensor_ch2,
            "target_flow_mask": void_tensor_ch1,
            "target_flow_noc": void_tensor_ch2,
            "target_flow_mask_noc": void_tensor_ch1,
            "target_disp": void_tensor_ch1,
            "target_disp_mask": void_tensor_ch1,
            "target_disp2_occ": void_tensor_ch1,
            "target_disp2_mask_occ": void_tensor_ch1,
            "target_disp_noc": void_tensor_ch1,
            "target_disp_mask_noc": void_tensor_ch1,
            "target_disp2_noc": void_tensor_ch1,
            "target_disp2_mask_noc": void_tensor_ch1
        }

        # random flip
        if self._flip_augmentations is True and torch.rand(1) > 0.5:
            _, _, _, ww = imgs_l_tensor.size()
            imgs_l_tensor_flip = torch.flip(imgs_l_tensor, dims=[3])
            imgs_r_tensor_flip = torch.flip(imgs_r_tensor, dims=[3])

            k_l1[0, 2] = ww - k_l1[0, 2]
            k_r1[0, 2] = ww - k_r1[0, 2]

            example_dict = {
                "input_left": imgs_r_tensor_flip,
                "input_right": imgs_l_tensor_flip,
                "input_k_l": k_r1,
                "input_k_r": k_l1
            }
            example_dict.update(common_dict)

        else:
            example_dict = {
                "input_left": imgs_l_tensor,
                "input_right": imgs_r_tensor,
                "input_k_l": k_l1,
                "input_k_r": k_r1
            }
            example_dict.update(common_dict)

        return example_dict


class KITTI_Comb_Multi_Train(ConcatDataset):  
    def __init__(self, args, root):     
        
        self.dataset1 = KITTI_2015_MonoSceneFlow_Multi(
            args, 
            root + '/KITTI_flow/', 
            preprocessing_crop=True, 
            crop_size=[370, 1224], 
            dstype="train")

        self.dataset2 = KITTI_Raw_for_Finetune_Multi(
            args, 
            root + '/KITTI_raw/',
            flip_augmentations=True,
            preprocessing_crop=True,
            crop_size=[370, 1224],
            num_examples=-1,
            index_file='index_txt/kitti_train_scenes_all.txt')
      
        super(KITTI_Comb_Multi_Train, self).__init__(
            datasets=[self.dataset1, self.dataset2])


class KITTI_Comb_Multi_Val(KITTI_2015_MonoSceneFlow_Multi):
    def __init__(self,
                 args,
                 root,
                 preprocessing_crop=False,
                 crop_size=[370, 1224]):
        super(KITTI_Comb_Multi_Val, self).__init__(
            args,
            data_root=root + '/KITTI_flow/',          
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            dstype="valid")


class KITTI_Comb_Multi_Full(ConcatDataset):  
    def __init__(self, args, root):        

        self.dataset1 = KITTI_2015_MonoSceneFlow_Multi(
            args, 
            root + '/KITTI_flow/', 
            preprocessing_crop=True,
            crop_size=[370, 1224], 
            dstype="full")

        self.dataset2 = KITTI_Raw_for_Finetune_Multi(
            args, 
            root + '/KITTI_raw/',
            flip_augmentations=True,
            preprocessing_crop=True,
            crop_size=[370, 1224],
            num_examples=-1,
            index_file='index_txt/kitti_train_scenes_all.txt')

        super(KITTI_Comb_Multi_Full, self).__init__(
            datasets=[self.dataset1, self.dataset2])



