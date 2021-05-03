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

VALIDATE_INDICES = [2, 34, 35, 36, 37, 38, 39, 40, 41, 42, 77, 78, 79, 80, 81, 83, 99, 100, 101, 102, 105, 106, 112, 113, 114, 115, 116, 117, 133, 141, 144, 145, 167, 187, 190, 191, 192, 193, 195, 199]


class KITTI_2015_Train_Multi_Base(data.Dataset):
    def __init__(self,
                 args,
                 data_root=None,
                 dstype="full"):

        self._seq_len = args.sequence_length

        images_l_root = os.path.join(data_root, "data_scene_flow_multiview", "training", "image_2_jpg")
        images_r_root = os.path.join(data_root, "data_scene_flow_multiview", "training", "image_3_jpg")
        flow_root_occ = os.path.join(data_root, "data_scene_flow", "training", "flow_occ")
        flow_root_noc = os.path.join(data_root, "data_scene_flow", "training", "flow_noc")
        disp0_root_occ = os.path.join(data_root, "data_scene_flow", "training", "disp_occ_0")
        disp1_root_occ = os.path.join(data_root, "data_scene_flow", "training", "disp_occ_1")
        disp0_root_noc = os.path.join(data_root, "data_scene_flow", "training", "disp_noc_0")
        disp1_root_noc = os.path.join(data_root, "data_scene_flow", "training", "disp_noc_1")

        ## loading image -----------------------------------
        if not os.path.isdir(images_l_root):
            raise ValueError("Image directory {} not found!".format(images_l_root))
        if not os.path.isdir(images_r_root):
            raise ValueError("Image directory {} not found!".format(images_r_root))
        if not os.path.isdir(flow_root_occ):
            raise ValueError("Image directory {} not found!".format(flow_root_occ))
        if not os.path.isdir(flow_root_noc):
            raise ValueError("Image directory {} not found!".format(flow_root_noc))
        if not os.path.isdir(disp0_root_occ):
            raise ValueError("disparity directory {} not found!".format(disp0_root_occ))
        if not os.path.isdir(disp1_root_occ):
            raise ValueError("disparity directory {} not found!".format(disp1_root_occ))
        if not os.path.isdir(disp0_root_noc):
            raise ValueError("disparity directory {} not found!".format(disp0_root_noc))
        if not os.path.isdir(disp1_root_noc):
            raise ValueError("disparity directory {} not found!".format(disp1_root_noc))

        # ----------------------------------------------------------
        # Construct list of indices for training/validation
        # ----------------------------------------------------------
        num_images = 200
        validate_indices = [x for x in VALIDATE_INDICES if x in range(num_images)]
        if dstype == "train":
            list_of_indices = [x for x in range(num_images) if x not in validate_indices]
        elif dstype == "valid":
            list_of_indices = validate_indices
        elif dstype == "full":
            list_of_indices = range(num_images)
        else:
            raise ValueError("KITTI: dstype {} unknown!".format(dstype))

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and disp/flow
        # ----------------------------------------------------------
        path_dir = os.path.dirname(os.path.realpath(__file__))
        self._seq_lists_l = []
        self._flow_list = []
        self._disp_list = []
        img_ext = '.jpg'

        for ii in list_of_indices:

            file_idx = '{:06d}'.format(ii)
            seq_list = []

            for ss in range(11 - self._seq_len, 11 + 1):
            # for ss in range(11 - self._seq_len, 11 + 2):

                seq_idx = '{:02d}'.format(ss)
                img_left = os.path.join(images_l_root, file_idx + "_" + seq_idx + img_ext)
                seq_list.append(img_left)

            flow_occ = os.path.join(flow_root_occ, file_idx + "_10.png")
            flow_noc = os.path.join(flow_root_noc, file_idx + "_10.png")
            disparity0_occ = os.path.join(disp0_root_occ, file_idx + "_10.png")
            disparity1_occ = os.path.join(disp1_root_occ, file_idx + "_10.png")
            disparity0_noc = os.path.join(disp0_root_noc, file_idx + "_10.png")
            disparity1_noc = os.path.join(disp1_root_noc, file_idx + "_10.png")

            self._seq_lists_l.append(seq_list)
            self._flow_list.append([flow_occ, flow_noc])
            self._disp_list.append([disparity0_occ, disparity1_occ, disparity0_noc, disparity1_noc])

        ## right images
        self._seq_lists_r = copy.deepcopy(self._seq_lists_l)
        for ii in range(len(self._seq_lists_r)):
            for jj in range(len(self._seq_lists_r[ii])):
                self._seq_lists_r[ii][jj] = self._seq_lists_r[ii][jj].replace(images_l_root, images_r_root)

        self._size = len(self._seq_lists_l)
        assert self._size != 0

        ## loading calibration matrix
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}        
        self.intrinsic_dict_l, self.intrinsic_dict_r = read_calib_into_dict(path_dir)


class KITTI_2015_MonoSceneFlow_Multi(KITTI_2015_Train_Multi_Base):
    def __init__(self,
                 args,
                 data_root=None,
                 preprocessing_crop=False,
                 crop_size=[370, 1224],
                 dstype="full"):
        super(KITTI_2015_MonoSceneFlow_Multi, self).__init__(
            args,
            data_root=data_root,
            dstype=dstype)

        self._args = args        
        self._preprocessing_crop = preprocessing_crop
        self._crop_size = crop_size

        self._to_tensor = vision_transforms.Compose([
            vision_transforms.ToPILImage(),
            vision_transforms.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        index = index % self._size

        # read images and flow
        img_list_l_np = [read_image_as_byte(img) for img in self._seq_lists_l[index]]
        img_list_r_np = [read_image_as_byte(img) for img in self._seq_lists_r[index]]
        
        # flo_occ, mask_flo_occ, flo_noc, mask_flo_noc
        flo_list_np = [read_png_flow(img) for img in self._flow_list[index]]
        flo_list_np = list_flatten(flo_list_np)

        # disp0_occ, mask0_disp_occ, disp1_occ, mask1_disp_occ
        # disp0_noc, mask0_disp_noc, disp1_noc, mask1_disp_noc
        disp_list_np = [read_png_disp(img) for img in self._disp_list[index]]
        disp_list_np = list_flatten(disp_list_np)
        
        # example filename
        basename = os.path.basename(self._seq_lists_l[index][0])[:6]
        k_l1 = torch.from_numpy(self.intrinsic_dict_l[get_date_from_width(img_list_l_np[0].shape[1])]).float()
        k_r1 = torch.from_numpy(self.intrinsic_dict_r[get_date_from_width(img_list_r_np[0].shape[1])]).float()
        
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
            flo_list_np = kitti_crop_image_list(flo_list_np, crop_info)
            disp_list_np = kitti_crop_image_list(disp_list_np, crop_info)
            k_l1, k_r1 = kitti_adjust_intrinsic(k_l1, k_r1, crop_info)
            

        # to tensors [t, c, h, w]
        imgs_l_tensor = torch.stack([self._to_tensor(img) for img in img_list_l_np], dim=0)
        imgs_r_tensor = torch.stack([self._to_tensor(img) for img in img_list_r_np], dim=0)
        
        # convert np to tensor
        flo_list_tensor = [numpy2torch(img) for img in flo_list_np]
        disp_list_tensor = [numpy2torch(img) for img in disp_list_np]


        example_dict = {
            "input_left": imgs_l_tensor,
            "input_right": imgs_r_tensor,
            "index": index,
            "basename": basename,
            "target_flow": flo_list_tensor[0],
            "target_flow_mask": flo_list_tensor[1],
            "target_flow_noc": flo_list_tensor[2],
            "target_flow_mask_noc": flo_list_tensor[3],
            "target_disp": disp_list_tensor[0],
            "target_disp_mask": disp_list_tensor[1],
            "target_disp2_occ": disp_list_tensor[2],
            "target_disp2_mask_occ": disp_list_tensor[3],
            "target_disp_noc": disp_list_tensor[4],
            "target_disp_mask_noc": disp_list_tensor[5],
            "target_disp2_noc": disp_list_tensor[6],
            "target_disp2_mask_noc": disp_list_tensor[7],
            "input_k_l": k_l1,
            "input_k_r": k_r1,
            "input_size": input_im_size
        }

        return example_dict

    def __len__(self):
        return self._size



class KITTI_2015_MonoSceneFlow_Full_Multi(KITTI_2015_MonoSceneFlow_Multi):
    def __init__(self,
                 args,
                 root,
                 preprocessing_crop=False,
                 crop_size=[370, 1224]):
        super(KITTI_2015_MonoSceneFlow_Full_Multi, self).__init__(
            args,
            data_root=root,            
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            dstype="full")


class KITTI_2015_MonoSceneFlow_Train_Multi(KITTI_2015_MonoSceneFlow_Multi):
    def __init__(self,
                 args,
                 root,
                 preprocessing_crop=False,
                 crop_size=[370, 1224]):
        super(KITTI_2015_MonoSceneFlow_Train_Multi, self).__init__(
            args,
            data_root=root,            
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            dstype="train")


class KITTI_2015_MonoSceneFlow_Valid_Multi(KITTI_2015_MonoSceneFlow_Multi):
    def __init__(self,
                 args,
                 root,
                 preprocessing_crop=False,
                 crop_size=[370, 1224]):
        super(KITTI_2015_MonoSceneFlow_Valid_Multi, self).__init__(
            args,
            data_root=root,            
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            dstype="valid")