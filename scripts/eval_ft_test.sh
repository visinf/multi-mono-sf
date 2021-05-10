#!/bin/bash

# DATASETS_HOME
DATA_HOME="(KITTI 2015 Scene Flow dataset path)/KITTI_flow"
CHECKPOINT="checkpoints/checkpoint_kitti_ft.ckpt"

# model
MODEL=MonoSceneFlow_Multi

Valid_Dataset=KITTI_2015_Test_Multi
Valid_Augmentation=Augmentation_Resize_Only_MultiFrame
Valid_Loss_Function=Eval_SceneFlow_KITTI_Test_Multi

# training configuration
SAVE_PATH="eval/kitti_test_ft/"
python ../main.py \
--batch_size=1 \
--batch_size_val=1 \
--sequence_length=4 \
--checkpoint=$CHECKPOINT \
--model=$MODEL \
--evaluation=True \
--num_workers=16 \
--save=$SAVE_PATH \
--start_epoch=1 \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_preprocessing_crop=False \
--validation_dataset_root=$DATA_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key=sf \
--calculate_disparity_scale=False \
--conv_padding_mode="zeros" \
--correlation_cuda_enabled=False

#--save_out=True \
#--save_vis=True