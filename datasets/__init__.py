from . import kitti_raw_multiframe
from . import kitti_2015_train_multiframe
from . import kitti_2015_test_multiframe
from . import kitti_comb_mnsf_multiframe

from . import nuscene
from . import davis
from . import wsvd

# KITTI RAW multi-frame
KITTI_Raw_Multi_KittiSplit_Train 		= kitti_raw_multiframe.KITTI_Raw_Multi_KittiSplit_Train
KITTI_Raw_Multi_KittiSplit_Valid 		= kitti_raw_multiframe.KITTI_Raw_Multi_KittiSplit_Valid

# KITTI 2015 Train/Test
KITTI_2015_Train_Full_Multi 			= kitti_2015_train_multiframe.KITTI_2015_MonoSceneFlow_Full_Multi
KITTI_2015_Train_Train_Multi 			= kitti_2015_train_multiframe.KITTI_2015_MonoSceneFlow_Train_Multi
KITTI_2015_Train_Valid_Multi 			= kitti_2015_train_multiframe.KITTI_2015_MonoSceneFlow_Valid_Multi

KITTI_2015_Test_Multi 					= kitti_2015_test_multiframe.KITTI_2015_Test_Multi

# KITTI Raw + KITTI 2015 for fine-tuning
KITTI_Comb_Multi_Train 					= kitti_comb_mnsf_multiframe.KITTI_Comb_Multi_Train
KITTI_Comb_Multi_Val 					= kitti_comb_mnsf_multiframe.KITTI_Comb_Multi_Val
KITTI_Comb_Multi_Full 					= kitti_comb_mnsf_multiframe.KITTI_Comb_Multi_Full

# Datasets for testing generalization
NuScenes 								= nuscene.NuScenes										

DavisAll 								= davis.DavisAll
Davis									= davis.Davis

WSVD_Train								= wsvd.WSVD_Train
WSVD_Test								= wsvd.WSVD_Test

