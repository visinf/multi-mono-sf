from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np
import copy

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_calib_into_dict, read_png_flow, read_png_disp, numpy2torch
from .common import kitti_crop_image_list, kitti_adjust_intrinsic, intrinsic_scale, get_date_from_width
from .common import list_flatten


class KITTI_2015_Test_Multi(data.Dataset):
    def __init__(self,
                 args,
                 root):

        self._args = args        

        self._to_tensor = vision_transforms.Compose([
            vision_transforms.ToPILImage(),
            vision_transforms.transforms.ToTensor()
        ])

        self._seq_len = args.sequence_length

        images_l_root = os.path.join(root, "data_scene_flow_multiview", "testing", "image_2_jpg")

        ## loading image -----------------------------------
        if not os.path.isdir(images_l_root):
            raise ValueError("Image directory {} not found!".format(images_l_root))

        # Save list of actual filenames for inputs and disp/flow
        path_dir = os.path.dirname(os.path.realpath(__file__))
        self._seq_lists_l = []
        img_ext = '.jpg'

        for ii in range(0,200):

            file_idx = '{:06d}'.format(ii)
            seq_list = []

            for ss in range(11 - self._seq_len, 11 + 1):

                seq_idx = '{:02d}'.format(ss)
                img_left = os.path.join(images_l_root, file_idx + "_" + seq_idx + img_ext)
                seq_list.append(img_left)

            self._seq_lists_l.append(seq_list)

        self._size = len(self._seq_lists_l)
        assert self._size != 0

        ## loading calibration matrix
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}        
        self.intrinsic_dict_l, self.intrinsic_dict_r = read_calib_into_dict(path_dir)

    def __getitem__(self, index):

        index = index % self._size

        # read images and flow
        img_list_l_np = [read_image_as_byte(img) for img in self._seq_lists_l[index]]
        
        # example filename
        basename = os.path.basename(self._seq_lists_l[index][0])[:6]
        k_l1 = torch.from_numpy(self.intrinsic_dict_l[get_date_from_width(img_list_l_np[0].shape[1])]).float()
        
        # input size
        h_orig, w_orig, _ = img_list_l_np[0].shape
        input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

        # to tensors [t, c, h, w]
        imgs_l_tensor = torch.stack([self._to_tensor(img) for img in img_list_l_np], dim=0)
        
        example_dict = {
            "input_left": imgs_l_tensor,
            "index": index,
            "basename": basename,
            "input_k_l": k_l1,
            "input_size": input_im_size
        }

        return example_dict

    def __len__(self):
        return self._size
