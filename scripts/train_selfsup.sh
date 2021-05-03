#!/bin/bash


# experiments and datasets meta
DATA_HOME="(KITTI Raw dataset path)"
EXPERIMENTS_HOME="(Path where checkpoints and log files will be saved)"

# model
MODEL=MonoSceneFlow_Multi

# save path
ALIAS="-selfsup-"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL$ALIAS$TIME"
CHECKPOINT=None

# Loss and Augmentation
Train_Dataset=KITTI_Raw_Multi_KittiSplit_Train
Train_Augmentation=Augmentation_SceneFlow_MultiFrame
Train_Loss_Function=Loss_SceneFlow_SelfSup_Multi

Valid_Dataset=KITTI_Raw_Multi_KittiSplit_Valid
Valid_Augmentation=Augmentation_Resize_Only_MultiFrame
Valid_Loss_Function=Loss_SceneFlow_SelfSup_Multi

# training configuration
python ../main.py \
--batch_size=1 \
--batch_size_val=1 \
--sequence_length=4 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[23, 39, 47, 54]" \
--model=$MODEL \
--num_workers=16 \
--optimizer=Adam \
--optimizer_lr=2e-4 \
--save=$SAVE_PATH \
--total_epochs=62 \
--training_augmentation=$Train_Augmentation \
--training_augmentation_photometric=True \
--training_dataset=$Train_Dataset \
--training_dataset_root=$DATA_HOME \
--training_dataset_flip_augmentations=True \
--training_dataset_preprocessing_crop=True \
--training_dataset_num_examples=-1 \
--training_key=total_loss \
--training_loss=$Train_Loss_Function \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$DATA_HOME \
--validation_dataset_preprocessing_crop=False \
--validation_key=total_loss \
--validation_loss=$Valid_Loss_Function \
--calculate_disparity_scale=False \
--conv_padding_mode="zeros" \
--correlation_cuda_enabled=False