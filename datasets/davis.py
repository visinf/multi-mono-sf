from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import glob
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_calib_into_dict, get_date_from_width


class Davis_base(data.Dataset):
    def __init__(self,
                 args,
                 root,
                 dstype):

        self._args = args
        self._dstype = dstype

        images_l_root = os.path.join(root)

        ## loading image -----------------------------------
        if not os.path.isdir(images_l_root):
            raise ValueError("Image directory %s not found!", images_l_root)

        # Construct list of indices for training/validation
        self._seq_lists_l = []

        if dstype == "all":
            list_of_files = sorted(glob.glob(images_l_root + "*/*.jpg"))
            scene_dict = {}

            for fn in list_of_files:
                folder_name = os.path.basename(os.path.dirname(fn))
                if folder_name in scene_dict:
                    scene_dict[folder_name].append(fn)
                else:
                    scene_dict[folder_name] = [fn]
            
            for scene in scene_dict:
                for ii in range(len(scene_dict[scene])-4):
                    self._seq_lists_l.append(scene_dict[scene][ii:ii+5])

        else:
            list_of_files = sorted(glob.glob(images_l_root + "/*.jpg"))
            num_images = len(list_of_files)
            for ii in range(num_images-4):
                self._seq_lists_l.append(list_of_files[ii:ii+5])

        self._size = len(self._seq_lists_l)
        assert len(self._seq_lists_l) != 0
        assert self._size != 0

        ## loading calibration matrix
        self._to_tensor = vision_transforms.Compose([
            vision_transforms.ToPILImage(),
            vision_transforms.transforms.ToTensor()
        ])

    def __getitem__(self, index):

        index = index % self._size

        # read images and flow
        img_list_l_np = [read_image_as_byte(img) for img in self._seq_lists_l[index]]
        
        # example filename
        if self._dstype == "all":
            basename = os.path.join(os.path.basename(os.path.dirname(self._seq_lists_l[index][3][:-4])), os.path.basename(self._seq_lists_l[index][3][:-4]))
        else:
            basename = os.path.basename(self._seq_lists_l[index][3][:-4])
        
        # input size
        h_orig, w_orig, _ = img_list_l_np[0].shape
        input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()
        
        # intrinsic 
        intrinsics = np.array([[700.0, 0.0, w_orig / 2.], [0.0, 700.0, h_orig / 2.], [0.0, 0.0, 1.0]])
        k_l1 = torch.from_numpy(intrinsics).float()

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


class DavisAll(Davis_base):
    def __init__(self,
                 args,
                 root):
        super(DavisAll, self).__init__(
            args,
            root,
            dstype="all")

class Davis(Davis_base):
    def __init__(self,
                 args,
                 root):
        super(Davis, self).__init__(
            args,
            root,
            dstype="one")
