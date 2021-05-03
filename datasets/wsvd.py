from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import glob
import numpy as np
import pickle
import copy

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, list_chunks_overlap

test_clip_id = ['4ss0qcbvJ-Q', 'aeW37ZgbwYg', 'EbNQFzlDzrc', 'Gfu31vB0dbI', 'gM_nUQLK4rA', 'kmYBQrbmZPM', 'P01ztxEiCOg', 'q6HbgPshE8E', '5DLA_0MZ7Uc', 'APsrbiUlGEw', 'fjTpY5ZanGE', 'GleHeqXY4cY', 'HDTHkqDiO-s', 'Md9jgyKpQ1g', 'pS0jdzpM2yg']
train_clip_id = ['0mnM7u45swU', '0QfN1wsl0yQ', '0YCXCzx_RIo', '1j1-9-CWPAk', '1kdd6j_TRyc', '2rdEzso-i2E', '3ZJrpq8iYSY', '5DLA_0MZ7Uc', '5J53betZS-Q', '7-p17CHB55s', '76F4lBZQdv4', '7txx9TyJI2g', '8CvW_9uX5o4', '8OZiDgaa_e8', '9ADCPQUmv8g', '9gONVYT_wDA', '9OwTjqIQPhA', 'ablmgvFjhYQ', 'AMU340JmYCk', 'AoEuAenqUgU', 'b5D5IWQFHcw', 'bfTuSLFzh2M', 'BItTHBScw6k', 'bY0xyUWnIH0', 'c0F0Hw9rOsE', 'cnEjRTRlVDI', 'cPyxEsf41wU', 'cVm-kmkXUns', 'DfFxmMPLTV0', 'DTALiU3LaHg', 'dvxjUqSIPpM', 'e7ZoClhVlyw', 'eM3tqdU44uo', 'EPJS3lOcUXY', 'Ev30J3SHfOo', 'fjTpY5ZanGE', 'FWHP_8rSSZo', 'FXjlJyZ4zwo', 'git_9TDi5MU', 'GjJQVX-g78c', 'GleHeqXY4cY', 'gn-6pMBy2RM', 'gWt3252npwk', 'HDTHkqDiO-s', 'HwZ_YYfZjk0', 'HzEo7fBHZ1k', 'i03E4e8MGQM', 'IEB_Ay8ipCI', 'IG6VsR61P3A', 'IWF8zVx3ogs', 'J4-zFz2v214', 'J6-girGi-HM', 'j9Mrg_e0yZE', 'JHVOwAC3VQw', 'JIhHNoirtfI', 'k2dNMBA23_I', 'kmYBQrbmZPM', 'lyUhAUjw-pU', 'mYibQS67BGA', 'neKuM2cL_BI', 'nU18tsqcOHk', 'O5oImJ4_g0A', 'ObySbaarOH8', 'P01ztxEiCOg', 'p3ye79Qu61M', 'pS0jdzpM2yg', 'QffabD-ekbM', 'qOsG2VhkYck', 'QtAhQZ9h-lA', 'qU1LOyZjWWM', 'QzgW0ATjeec', 'q_v9x7-dmMQ', 'r7X8Btx1NXM', 'rb8ahNwQPYQ', 'Rk-1lTvva4Y', 'ROKH3VgKNdU', 'SIIhKaNh7Qg', 'tYFYL27NX9U', 'U5ajbgrN_W8', 'V62bt7y49D8', 'VAp_zbH2mMo', 'VhzOg40qGzU', 'VrTZRlIgCmA', 'XL_2by6CdSQ', 'XS6juDVWKDg', 'XsL5fw40-uY', 'YQgufPXF31g', 'Ze8Em8GGFLU', '_hhOVQCGG10', '_kyS6Mxa0xU', '_yB7Q_2rGJo']
train_clip_id  = list( set(train_clip_id) - set(test_clip_id))

def read_wsvd_clip_pickle(filename):
    objects = []
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
        clip_dict = {}
        for video in objects[0]:
            key = video['name'][:-5] if '.f' in video['name'] else video['name']
            clip_dict[key] = video['clips']
        return clip_dict



class WSVD(data.Dataset):
    def __init__(self,
                 args,
                 root,
                 dstype=None,
                 flip_augmentations=False,
                 num_examples=-1):
        """
        Load WSVD datset
        root: root directory
        dstype : dataset split in in ["train", "test", "full"]
        """
        self._args = args
        self._seq_dim = args.sequence_length
        self._seq_lists_l = [[]] * self._seq_dim
        self._seq_num = 0

        self._frame_sampling = 4 # using every "N th" frame for training
        self._seq_sampling = 2 # using every "N th" sequence for training


        if not os.path.isdir(root):
            raise ValueError("Image directory {} not found!".format(images_l_root))

        # read video list
        dir_list = []
        dir_ds = os.path.join(root, 'data')
        if dstype == "full":                
            dir_list += [[os.path.join(dir_ds, v_id), v_id] for v_id in os.listdir(dir_ds)]
        elif dstype == "train":
            dir_list += [[os.path.join(dir_ds, v_id), v_id] for v_id in os.listdir(dir_ds) if v_id in train_clip_id]
        elif dstype == "test":
            dir_list = [[os.path.join(dir_ds, v_id), v_id] for v_id in os.listdir(dir_ds) if v_id in test_clip_id]
        else:
            raise ValueError("dstype should be among {}".format(dstype))

        train_clip_dict = read_wsvd_clip_pickle(os.path.join(root, "wsvd_train_clip_frame_ids.pkl"))
        test_clip_dict  = read_wsvd_clip_pickle(os.path.join(root, "wsvd_test_clip_frame_ids.pkl"))
        clip_dict = {**train_clip_dict, **test_clip_dict}

        # for each video dir, construct the sequence list
        for v_dir, v_id in dir_list:

            # collect images
            images = sorted(glob.glob(os.path.join(v_dir, "*.jpg")))
            file_dict = {}
            for img in images:
                file_dict[os.path.splitext(os.path.basename(img))[0]] = img
            
            # read clips
            clips = clip_dict[v_id]

            # for each clip (== a sequence of images)
            for clip in clips:
                
                img_path = []

                # check whether image exist
                for f_num in clip['frames']:
                    key = "{:06d}_l".format(f_num)
                    if key in file_dict:
                        img_path.append(file_dict[key])

                img_path = img_path[::self._frame_sampling]

                if len(img_path) < 5:
                    continue

                for ss in range(self._seq_dim):
                    seqs = list_chunks_overlap(img_path[ss:], self._seq_dim)[::self._seq_sampling]
                    self._seq_lists_l[ss] = self._seq_lists_l[ss] + seqs
      

        # truncate the seq lists
        min_num_examples = min([len(item) for item in self._seq_lists_l])
        assert min_num_examples != 0

        if num_examples > 0:
            for ii in range(self._seq_dim):
                self._seq_lists_l[ii] = self._seq_lists_l[ii][:num_examples]
        else:
            for ii in range(self._seq_dim):
                self._seq_lists_l[ii] = self._seq_lists_l[ii][:min_num_examples]

        self._size = [len(seq) for seq in self._seq_lists_l]
        assert min(self._size) != 0

        ## right images
        self._seq_lists_r = copy.deepcopy(self._seq_lists_l)
        for ii in range(len(self._seq_lists_r)):
            for jj in range(len(self._seq_lists_r[ii])):
                for kk in range(len(self._seq_lists_r[ii][jj])):
                    self._seq_lists_r[ii][jj][kk] = self._seq_lists_r[ii][jj][kk].replace("_l", "_r")

        ## loading calibration matrix
        self._to_tensor = vision_transforms.Compose([
            vision_transforms.ToPILImage(),
            vision_transforms.transforms.ToTensor()
        ])

        self._flip_augmentations = flip_augmentations

    def __getitem__(self, index):

        
        self._seq_num = int(torch.randint(0, self._seq_dim, (1,)))
        
        index = index % self._size[self._seq_num]

        # read images and flow
        img_list_l_np = [read_image_as_byte(img) for img in self._seq_lists_l[self._seq_num][index]]
        img_list_r_np = [read_image_as_byte(img) for img in self._seq_lists_r[self._seq_num][index]]
        
        # example filename
        ref_img_name = self._seq_lists_l[self._seq_num][index][3]
        basename = os.path.basename(os.path.dirname(ref_img_name)) + "_" + os.path.basename(ref_img_name[:-4])

        # input size
        h_orig, w_orig, _ = img_list_l_np[0].shape
        input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

        # intrinsic 
        intrinsics = np.array([[700.0, 0.0, w_orig / 2.], [0.0, 700.0, h_orig / 2.], [0.0, 0.0, 1.0]])
        k_l1 = torch.from_numpy(intrinsics).float()

        # to tensors [t, c, h, w]
        imgs_l_tensor = torch.stack([self._to_tensor(img) for img in img_list_l_np], dim=0)
        imgs_r_tensor = torch.stack([self._to_tensor(img) for img in img_list_r_np], dim=0)

        common_dict = {
            "index": index,
            "basename": basename,
            "input_size": input_im_size,
            "input_k_l": k_l1,
            "input_k_r": k_l1
        }

        # random flip
        if self._flip_augmentations is True and torch.rand(1) > 0.5:
            _, _, _, ww = imgs_l_tensor.size()
            imgs_l_tensor_flip = torch.flip(imgs_l_tensor, dims=[3])
            imgs_r_tensor_flip = torch.flip(imgs_r_tensor, dims=[3])

            example_dict = {
                "input_left": imgs_r_tensor_flip,
                "input_right": imgs_l_tensor_flip,
            }
            example_dict.update(common_dict)

        else:
            example_dict = {
                "input_left": imgs_l_tensor,
                "input_right": imgs_r_tensor,
            }
            example_dict.update(common_dict)

        return example_dict

    def __len__(self):
        return self._size[self._seq_num]


class WSVD_Train(WSVD):
    def __init__(self,
                 args,
                 root,
                 num_examples=-1):
        super(WSVD_Train, self).__init__(
            args,
            root=root,            
            dstype="train",
            flip_augmentations=True,
            num_examples=num_examples)

class WSVD_Test(WSVD):
    def __init__(self,
                 args,
                 root,
                 num_examples=-1):
        super(WSVD_Test, self).__init__(
            args,
            root=root,            
            dstype="test",
            flip_augmentations=False,
            num_examples=num_examples)