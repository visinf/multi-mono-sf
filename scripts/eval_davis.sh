#!/bin/bash

# DATASETS_HOME
DATA_HOME="(DAVID dataset path)/480p/rollerblade/"
CHECKPOINT="checkpoints/checkpoint_kitti_selfsup.ckpt"

# model
MODEL=MonoSceneFlow_Multi

Valid_Dataset=Davis
Valid_Augmentation=Augmentation_Resize_Only_MultiFrame
Valid_Loss_Function=Eval_SceneFlow_KITTI_Test_Multi

# training configuration
SAVE_PATH="eval/davis/rollerblade/"
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
--validation_augmentation_resize="[256, 832]" \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$DATA_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key=sf \
--calculate_disparity_scale=False \
--conv_padding_mode="zeros" \
--correlation_cuda_enabled=False \
--save_vis=True

# --save_out=True \
