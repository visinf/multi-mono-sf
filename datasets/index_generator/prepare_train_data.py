from __future__ import absolute_import, division, print_function

import numpy as np
from glob import glob
import os
import copy


dir_path = os.path.dirname(os.path.realpath(__file__))


def extract_kitti_benchmark_scene():
    eigen_test_file = os.path.join(dir_path, 'provided/train_mapping.txt')
    if not os.path.exists(eigen_test_file):
        raise ValueError("KITTI Train File '%s' not found!", eigen_test_file)
    eigen_test = open(eigen_test_file, 'r')

    scene_name_set = set()
    filename_list = [line.split(' ') for line in eigen_test.readlines()]
    for item in filename_list:
        if len(item) != 3:
            continue
        _, scene_name, _ = item
        scene_name_set.add(scene_name)

    scene_name_set = sorted(scene_name_set)

    with open(os.path.join(dir_path, 'generated/kitti_test_scenes.txt'), 'w') as tf:
        for item in scene_name_set:
            tf.write('%s\n' % item)



def extract_eigen_test_scene():
    eigen_test_file = os.path.join(dir_path, 'provided/eigen_test_files.txt')
    if not os.path.exists(eigen_test_file):
        raise ValueError("Eigen Test File '%s' not found!", eigen_test_file)
    eigen_test = open(eigen_test_file, 'r')

    scene_name_set = set()
    filename_list = [line.split(' ') for line in eigen_test.readlines()]
    for item in filename_list:
        _, scene_name, _, _, _ = item[0].split('/')
        scene_name_set.add(scene_name)

    scene_name_set = sorted(scene_name_set)

    with open(os.path.join(dir_path, 'generated/eigen_test_scenes.txt'), 'w') as tf:
        for item in scene_name_set:
            tf.write('%s\n' % item)


def extract_eigen_test_kitti_benchmark_scene():

    ## Eigen
    eigen_test_file = os.path.join(dir_path, 'txt/eigen_test_files.txt')
    if not os.path.exists(eigen_test_file):
        raise ValueError("Eigen Test File '%s' not found!", eigen_test_file)
    eigen_test = open(eigen_test_file, 'r')

    scene_name_set = set()
    filename_list = [line.split(' ') for line in eigen_test.readlines()]
    for item in filename_list:
        _, scene_name, _, _, _ = item[0].split('/')
        scene_name_set.add(scene_name)


    ## KITTI
    kitti_test_file = os.path.join(dir_path, 'txt/train_mapping.txt')
    if not os.path.exists(kitti_test_file):
        raise ValueError("KITTI Train File '%s' not found!", kitti_test_file)
    kitti_test = open(kitti_test_file, 'r')
    
    filename_list = [line.split(' ') for line in kitti_test.readlines()]
    for item in filename_list:
        if len(item) != 3:
            continue
        _, scene_name, _ = item
        scene_name_set.add(scene_name)

    
    scene_name_set = sorted(scene_name_set)


    with open(os.path.join(dir_path, 'txt/eigen_kitti_test_scenes.txt'), 'w') as tf:
        for item in scene_name_set:
            tf.write('%s\n' % item)


class CollectDataList(object):
    def __init__(self, dataset_dir, split='eigen'):

        ## filename / variable definition
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        self.dataset_dir = dataset_dir
        self.excluded_frames = []
        self.train_frames = []
        self.num_train = -1
        self.split = split

        ## parsing data
        test_scene_file = dir_path + '/generated/' + self.split + '_test_scenes.txt'
        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t.rstrip() for t in test_scenes]

        self.split_scenes()


    def split_scenes(self):

        all_scenes = []

        for date in self.date_list:
            drive_set = os.listdir(self.dataset_dir + date + '/')
            for dr in drive_set:
                drive_dir = os.path.join(self.dataset_dir, date, dr)

                if os.path.isdir(drive_dir):
                    if dr in self.test_scenes:
                        continue
                    
                    img_dir = os.path.join(drive_dir, 'image_02', 'data')
                    num_images = len(glob(img_dir + '/*.jpg'))
                    all_scenes.append([dr, num_images])
                    print(img_dir)

        np.random.seed(1234)
        np.random.shuffle(all_scenes)

        cumsum = []
        total = 0
        for ii, scene in enumerate(all_scenes):
            total += scene[1]
            cumsum.append(total)

        break_num = cumsum[-1] * 0.2
        break_idx = 0
        for ii, num in enumerate(cumsum):
            if num > break_num:
                break_idx = ii
                break
        
        print(break_idx)
        print(cumsum[break_idx-1])
        print(cumsum[break_idx-1] / cumsum[-1])

        train_scenes = all_scenes[break_idx:]
        valid_scenes = all_scenes[:break_idx]

        sum_train = 0
        for nn in train_scenes:
            sum_train += nn[1]
            print(nn[0])

        print("")
        sum_valid = 0
        for nn in valid_scenes:
            sum_valid += nn[1]
            print(nn[0])

        print(sum_valid / (sum_valid + sum_train))

        with open(os.path.join(dir_path,  "generated/" + self.split + "_train_scenes.txt"), 'w') as tf:
            for item in train_scenes:
                tf.write('%s\n' % item[0])

        with open(os.path.join(dir_path,  "generated/" + self.split + "_valid_scenes.txt"), 'w') as tf:
            for item in valid_scenes:
                tf.write('%s\n' % item[0])

        with open(os.path.join(dir_path,  "generated/" + self.split + "_train_scenes_all.txt"), 'w') as tf:
            for item in all_scenes:
                tf.write('%s\n' % item[0])

def main():

    sequence_length = 1
    dataset_dir = '/fastdata/jhur/KITTI_raw_noPCL/'
    
    ## KITTI SPLIT
    extract_kitti_benchmark_scene()
    CollectDataList(dataset_dir=dataset_dir, split='kitti')
    # SplitTrainVal_even(dataset_dir=dataset_dir, file_name='generated/kitti_full.txt', seq_len=sequence_length, alias='kitti')

    # # EIGEN SPLIT
    # extract_eigen_test_scene()
    # CollectDataList(dataset_dir=dataset_dir, split='eigen', sequence_length=sequence_length)
    # SplitTrainVal_even(dataset_dir=dataset_dir, file_name='generated/eigen_full.txt', seq_len=sequence_length, alias='eigen')


main()
