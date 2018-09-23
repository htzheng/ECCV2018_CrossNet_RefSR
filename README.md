## Description
This repository contains the pyTorch implementation of ECCV2018 paper ''CrossNet: An End-to-end Reference-based Super Resolution Network using Cross-scale Warping''. Note that we use a batchsize of 4 instead of 2, and a slightly different input for flownet (see usage 1), results with ~0.2dB higher PSNR is achieved.

We would like to thank Yang Tan (https://github.com/tanyang1231) for implementating the code using pyTorch.

## prerequisite library
pytorch
opencv
h5py

## usage
1. prepare an hdf5 file, which contains /img_HR, /img_LR, /img_MDSR, /img_LR_upsample. /img_HR is used as reference input and ground truth, /img_LR is used as low resolution input, /img_MDSR is the MDSR upsampled image, and /img_LR_upsample is bicubically upsampled image. (Different from the original paper, in this version of code, we use Flownet with bicubically upsampled image and reference image to generate optical flow)

2. mkdir checkpoints_charbonnier_3
3. python -u train_multi_warping.py  --dataset Flower   --display 50  --batch_size 2  --step_size 150000 --loss CharbonnierLoss --optim Adam --lr 0.0001 --checkpoints_dir ./checkpoints_charbonnier_3/  --checkpoint 195000 --checkpoint_file ./checkpoints_charbonnier_3/CP195000.pth

## citation
Please cite our paper if you find it interesting!

@inproceedings{zheng2018crossnet,  
  title={CrossNet: An End-to-end Reference-based Super Resolution Network using Cross-scale Warping},  
  author={Zheng, Haitian and Ji, Mengqi and Wang, Haoqian and Liu, Yebin and Fang, Lu},  
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},  
  pages={88--104},  
  year={2018}  
}

