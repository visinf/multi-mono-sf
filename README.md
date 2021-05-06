# Self-Supervised Multi-Frame Monocular Scene Flow

<img src=demo/demo.gif> 

> 3D visualization of estimated depth and scene flow (overlayed with input image) from temporally consecutive images.  
> Trained on KITTI in a self-supervised manner, and tested on DAVIS.

This repository is the official PyTorch implementation of the paper:  

&nbsp;&nbsp;&nbsp;**[Self-Supervised Multi-Frame Monocular Scene Flow](https://arxiv.org/abs/2105.02216)**  
&nbsp;&nbsp;&nbsp;[Junhwa Hur](https://hurjunhwa.github.io) and [Stefan Roth](https://www.visinf.tu-darmstadt.de/visinf/team_members/sroth/sroth.en.jsp)  
&nbsp;&nbsp;&nbsp;*CVPR*, 2021  
&nbsp;&nbsp;&nbsp;[Arxiv](https://arxiv.org/abs/2105.02216)

- Contact: junhwa.hur[at]gmail.com

## Installation
The code has been tested with Anaconda (Python 3.8), PyTorch 1.8.1 and CUDA 10.1 (Different Pytorch + CUDA version is also compatible).  
Please run the provided conda environment setup file:

  ```Shell
  conda env create -f environment.yml
  conda activate multi-mono-sf
  ```

**(Optional)** Using the CUDA implementation of the correlation layer accelerates training (~50% faster):
  ```Shell
  ./install_correlation.sh
  ```
After installing it, turn on this flag **`--correlation_cuda_enabled=True`** in training\/evaluation script files.

## Dataset

Please download the following to datasets for the experiment:
  - [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php) (synced+rectified data, please refer [MonoDepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data) for downloading all data more conveniently.)
  - merge [KITTI Scene Flow 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) and [Multi-view extension](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) in the same folder.
  
To save space, we convert the *KITTI Raw* **png** images to **jpeg**, following the convention from [MonoDepth](https://github.com/mrharicot/monodepth):
  ```
  find (data_folder)/ -name '*.png' | parallel 'convert {.}.png {.}.jpg && rm {}'
  ```   
We also converted images in *KITTI Scene Flow 2015* as well. Please convert the png images in `image_2` and `image_3` into jpg and save them into the seperate folder **`image_2_jpg`** and **`image_3_jpg`**.  
To save space further, you can delete the velodyne point data in KITTI raw data as we don't need it.

## Training and Inference

The **[scripts](scripts/)** folder contains training\/inference scripts.

**For self-supervised training**, you can simply run the following script files:

| Script                 | Training                   | Dataset                |
|------------------------|----------------------------|------------------------|
| `./train_selfsup.sh`   | Self-supervised            | KITTI Split            |


**Fine-tuning** is done with two stages: *(i)* first finding the stopping point using train\/valid split, and then *(ii)* fune-tuning using all data with the found iteration steps.  

| Script                 | Training                   | Dataset                |
|------------------------|----------------------------|------------------------|
| `./ft_1st_stage.sh`    | Semi-supervised finetuning | KITTI raw + KITTI 2015 |
| `./ft_2nd_stage.sh`    | Semi-supervised finetuning | KITTI raw + KITTI 2015 |

In the script files, please configure these following PATHs for experiments:
  - `DATA_HOME` : the directory where the training or test is located in your local system.
  - `EXPERIMENTS_HOME` : your own experiment directory where checkpoints and log files will be saved.
     
  
**To test pretrained models**, you can simply run the following script files:

| Script                    | Training        | Dataset            | 
|---------------------------|-----------------|--------------------|
| `./eval_selfsup_train.sh` | self-supervised | KITTI 2015 Train   |
| `./eval_ft_test.sh`       | fine-tuned      | KITTI 2015 Test    |
| `./eval_davis.sh`         | self-supervised | DAVIS (one scene)  |
| `./eval_davis_all.sh`     | self-supervised | DAVIS (all scenes) |

  - To save visuailization of outputs, please turn on **`--save_vis=True`** in the script.  
  - To save output images for [KITTI Scene Flow 2015 Benchmark](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php) submission,  please turn on **`--save_out=True`** in the script.  

## Pretrained Models 

The **[checkpoints](checkpoints/)** folder contains the checkpoints of the pretrained models.  


## Acknowledgement

Please cite our paper if you use our source code.  

```bibtex
@inproceedings{Hur:2021:SSM,  
  Author = {Junhwa Hur and Stefan Roth},  
  Booktitle = {CVPR},  
  Title = {Self-Supervised Multi-Frame Monocular Scene Flow},  
  Year = {2021}  
}
```

- Portions of the source code (e.g., training pipeline, runtime, argument parser, and logger) are from [Jochen Gast](https://scholar.google.com/citations?user=tmRcFacAAAAJ&hl=en)

