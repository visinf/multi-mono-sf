#!/bin/bash

# datasets
DATA_HOME="(Common path of KITTI 2015 Scene Flow + Raw dataset)"
EXPERIMENTS_HOME="(Path where checkpoints and log files will be saved)"

# model
MODEL=MonoSceneFlow_Multi

# save path
ALIAS="-kitti_ft-"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL$ALIAS$TIME"
CHECKPOINT="checkpoints/checkpoint_kitti_selfsup.ckpt.ckpt"

# Loss and Augmentation
Train_Dataset=KITTI_Comb_Multi_Full
Train_Augmentation=Augmentation_SceneFlow_Finetuning_MultiFrame
Train_Loss_Function=Loss_SceneFlow_SemiSupFinetune_Multi

Valid_Dataset=KITTI_Comb_Multi_Val
Valid_Augmentation=Augmentation_Resize_Only_MultiFrame
Valid_Loss_Function=Eval_SceneFlow_KITTI_Train_Multi

# training configuration
python ../main.py \
--batch_size=1 \
--batch_size_val=1 \
--sequence_length=4 \
--finetuning=True \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[25, 50, 75, 88, 100]" \
--model=$MODEL \
--num_workers=16 \
--optimizer=Adam \
--optimizer_lr=4e-5 \
--save=$SAVE_PATH \
--total_epochs=175 \
--training_augmentation=$Train_Augmentation \
--training_augmentation_photometric=True \
--training_dataset=$Train_Dataset \
--training_dataset_root=$DATA_HOME \
--training_loss=$Train_Loss_Function \
--training_key=total_loss \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$DATA_HOME \
--validation_key=sf \
--validation_loss=$Valid_Loss_Function \
--calculate_disparity_scale=False \
--conv_padding_mode="zeros" \
--correlation_cuda_enabled=False