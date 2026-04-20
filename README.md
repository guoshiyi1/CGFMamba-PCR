# Preparation

## Environment

This codebase was tested with the following environment configurations. It may work with other versions.
- Ubuntu 20.04
- CUDA 11.7
- Python 3.9
- PyTorch 1.13.1 + cu117

## Color3DMatch/Color3DLoMatch datasets

### Data preparation

The Color3DMatch/Color3DLoMatch datasets can be downloaded following the article "ColorPCR: Color Point Cloud Registration with Multi-Stage Geometric-Color Fusion", which is published in CVPR2024. The data is organized as follows

```text
--dataset
         |--data--train--7-scenes-chess--cloud_bin_0.npy
               |      |               |--...
               |      |--...
               |--test--7-scenes-redkitchen--cloud_bin_0.npy
                      |                    |--...
                      |--...
```
## Test on Color3DMatch dataset

```shell
CUDA_VISIBLE_DEVICES=<GPU> python test.py --benchmark=3DMatch --snapshot=PATH
CUDA_VISIBLE_DEVICES=<GPU> python eval_C3DM.py --benchmark=3DMatch --method=csc2

```
Pre trained model is the pretrain.pth.tar in the current folder, you can replace PATH with the path of pretrain.pth.tar. 
##
The code in this folder can only be used for testing, and other code including training code will be open-sourced after the paper is accepted.