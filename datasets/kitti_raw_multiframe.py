from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np
import copy
import glob

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_calib_into_dict
from .common import kitti_crop_image_list, kitti_adjust_intrinsic
from .common import list_chunks


class KITTI_Raw_Multi(data.Dataset):
    def __init__(self,
                 args,
                 images_root=None,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1,
                 index_file=None):

        self._args = args
        self._seq_dim = args.sequence_length
        self._seq_lists_l = [[]] * self._seq_dim

        self._flip_augmentations = flip_augmentations
        self._preprocessing_crop = preprocessing_crop
        self._crop_size = crop_size

        self._seq_num = 0

        ## loading index file
        path_dir = os.path.dirname(os.path.realpath(__file__))
        path_index_file = os.path.join(path_dir, index_file)

        if not os.path.exists(path_index_file):
            raise ValueError("Index File '%s' not found!", path_index_file)
        index_file = open(path_index_file, 'r')

        ## loading image
        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!")

        scene_list = [line.rstrip() for line in index_file.readlines()]
        view1 = 'image_02'
        view2 = 'image_03'
        ext = '.jpg'

        ## Lists
        for scene in scene_list:
            date = scene[:10]
            img_dir = os.path.join(images_root, date, scene, view1, 'data')
            img_list = sorted(glob.glob(img_dir + '/*' + ext))

            for ss in range(self._seq_dim):
                seqs = list_chunks(img_list[ss:], self._seq_dim)

                for seq in seqs:
                    last_img = seq[-1]
                    curridx = os.path.basename(last_img)[:-4]
                    nextidx = '{:010d}'.format(int(curridx) + 1)
                    seq.append(last_img.replace(curridx, nextidx))

                seqs.remove(seqs[-1]) # if sequence length satisfy: no last frame. else: remove it anyway

                self._seq_lists_l[ss] = self._seq_lists_l[ss] + seqs

        # truncate the seq lists
        min_num_examples = min([len(item) for item in self._seq_lists_l])
        if num_examples > 0:
            for ii in range(self._seq_dim):
                self._seq_lists_l[ii] = self._seq_lists_l[ii][:num_examples]
        else:
            for ii in range(self._seq_dim):
                self._seq_lists_l[ii] = self._seq_lists_l[ii][:min_num_examples]


        self._size = [len(seq) for seq in self._seq_lists_l]

        ## right images
        self._seq_lists_r = copy.deepcopy(self._seq_lists_l)
        for ii in range(len(self._seq_lists_r)):
            for jj in range(len(self._seq_lists_r[ii])):
                for kk in range(len(self._seq_lists_r[ii][jj])):
                    self._seq_lists_r[ii][jj][kk] = self._seq_lists_r[ii][jj][kk].replace(view1, view2)

        ## loading calibration matrix
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}        
        self.intrinsic_dict_l, self.intrinsic_dict_r = read_calib_into_dict(path_dir)

        self._to_tensor = vision_transforms.Compose([
            vision_transforms.ToPILImage(),
            vision_transforms.transforms.ToTensor()
        ])


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
        
        common_dict = {
            "index": index,
            "basename": basename,
            "datename": datename,
            "input_size": input_im_size
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

    def __len__(self):
        return self._size[self._seq_num]


class KITTI_Raw_Multi_KittiSplit_Train(KITTI_Raw_Multi):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1):
        super(KITTI_Raw_Multi_KittiSplit_Train, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_train_scenes.txt")


class KITTI_Raw_Multi_KittiSplit_Valid(KITTI_Raw_Multi):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=False,
                 preprocessing_crop=False,
                 crop_size=[370, 1224],
                 num_examples=-1):
        super(KITTI_Raw_Multi_KittiSplit_Valid, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_valid_scenes.txt")


class KITTI_Raw_Multi_KittiSplit_Full(KITTI_Raw_Multi):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1):
        super(KITTI_Raw_Multi_KittiSplit_Full, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_train_scenes_all.txt")
