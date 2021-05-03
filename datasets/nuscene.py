from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import glob
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_calib_into_dict, get_date_from_width


class NuScenes(data.Dataset):
    def __init__(self,
                 args,
                 root):

        self._args = args

        images_l_root = os.path.join(root, "sweeps", "CAM_FRONT")

        ## loading image -----------------------------------
        if not os.path.isdir(images_l_root):
            raise ValueError("Image directory %s not found!", images_l_root)

        # Construct list of indices for training/validation
        list_of_files = sorted(glob.glob(images_l_root + "/*.jpg"))
        num_images = len(list_of_files)
        self._seq_lists_l = []

        scene_dict = {}

        for filename in list_of_files:
            key = os.path.split(filename)[-1][:15]
            if key in scene_dict:
                scene_dict[key].append(filename)
            else:
                scene_dict[key] = [filename]

        for key, file_list in scene_dict.items():
            for ii in range(len(file_list)-3):    
                self._seq_lists_l.append(file_list[ii:ii+5])

        self._size = len(self._seq_lists_l)
        assert len(self._seq_lists_l) != 0
        assert self._size != 0

        ## loading calibration matrix
        self.intrinsics = np.array([[1282.473573, 0.0, 826.03044], [0.0, 1282.110712, 483.91705], [0.0, 0.0, 1.0]])
        self._to_tensor = vision_transforms.Compose([
            vision_transforms.ToPILImage(),
            vision_transforms.transforms.ToTensor()
        ])

    def __getitem__(self, index):

        index = index % self._size

        # read images and flow
        img_list_l_np = [read_image_as_byte(img) for img in self._seq_lists_l[index]]
        
        # example filename
        basename = os.path.basename(self._seq_lists_l[index][3][:-4])
        k_l1 = torch.from_numpy(self.intrinsics).float()
        
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