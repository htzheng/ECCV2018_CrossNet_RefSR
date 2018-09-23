import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from random import getrandbits
##### following the paper ''Optical Flow Estimation using a Spatial Pyramid Network''
# in https://github.com/anuragranj/spynet/blob/master/transforms.lua
#  We randomly scale images by a factor of [1,2]
#  and apply rotations at random within [-17,17]
#  then random crop
# We include additive white Gaussian noise sampled uniformly from N(0,0.1). 
# We apply color jitter with additive brightness, contrast and saturation sampled from a Gaussian, N(0,0.4).
# We finally normalize the images using a mean and standard deviation computed from 
# a large corpus of ImageNet [33] data in [22].

##

def grayscale(img):
    dst = np.zeros(img.shape)
    dst[:,:,0] = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    dst[:,:,1] = dst[:,:,0]
    dst[:,:,2] = dst[:,:,0]
    return dst

def blend(img1,img2,alpha = 0.5):
    # if alpha > 1.5:
    #     alpha = 1.0
    # if alpha < 0.5:
    #     alpha = 0.5
    return img1 * alpha + img2 * (1-alpha)



def augmentation_LF_unflow(img,config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes, config_contrast, config_gamma):
    H = img.shape[0]
    W = img.shape[1]
    img_aug = img
    ## flipping
    if(config_flip==1):
        img_aug = img_aug[::-1,:]
    if(config_flip_lr==1):
        img_aug = img_aug[:,::-1]

    ## brightness changes
    img_aug = img_aug + config_brightness_changes

    ## multiplicative color changes
    img_aug = img_aug * config_multiplicative_color_changes

    # ## Contrast
    gs_2 = grayscale(img_aug)
    img_aug = blend(gs_2, img_aug, alpha = config_contrast)

    ## clip
    img_aug = np.clip(img_aug,0.0,1.0)

    # ## gama
    img_aug = np.power(img_aug,config_gamma)

    return img_aug


def batch_augmentation3_LF_UnFlow(img_batch1, img_batch2, img_batch3):
    # augmentation partically follows UnFlow: Unsupervised Learning of Optical Flow with a Bidirectional Census Loss
    # random scaling and additive Gaussian noise is not used here
    N = img_batch1.shape[0]
    new_batch1 = np.zeros(img_batch1.shape, dtype = np.float32)
    new_batch2 = np.zeros(img_batch2.shape, dtype = np.float32)
    new_batch3 = np.zeros(img_batch3.shape, dtype = np.float32)
    for i in range(N):
        img1 = img_batch1[i,:,:,:].transpose([1,2,0])
        img2 = img_batch2[i,:,:,:].transpose([1,2,0])
        img3 = img_batch3[i,:,:,:].transpose([1,2,0])

        # generate random config
        config_flip = 0
        config_flip_lr = random.randint(0, 1) > 0.5
        config_brightness_changes = np.random.normal(loc = 0, scale = 0.02)
        config_multiplicative_color_changes = np.random.uniform(0.9, 1.1)
        config_contrast = np.random.uniform(-0.3, 0.3)
        config_gamma = np.random.uniform(0.8, 1.3)

        # generate img
        img1_aug = augmentation_LF_unflow(img1, config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes,config_contrast, config_gamma)
        img2_aug = augmentation_LF_unflow(img2, config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes,config_contrast, config_gamma)
        img3_aug = augmentation_LF_unflow(img3, config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes,config_contrast, config_gamma)

        # plt.subplot(321)
        # plt.imshow(img1, vmin=-0, vmax=1, interpolation="nearest")
        # plt.subplot(323)
        # plt.imshow(img2, vmin=-0, vmax=1, interpolation="nearest")
        # plt.subplot(325)
        # plt.imshow(img3, vmin=-0, vmax=1, interpolation="nearest")
        # plt.subplot(322)
        # plt.imshow(img1_aug, vmin=-0, vmax=1, interpolation="nearest")
        # plt.subplot(324)
        # plt.imshow(img2_aug, vmin=-0, vmax=1, interpolation="nearest")
        # plt.subplot(326)
        # plt.imshow(img3_aug, vmin=-0, vmax=1, interpolation="nearest")
        # plt.show()

        new_batch1[i,:,:,:] = img1_aug.transpose([2,0,1])
        new_batch2[i,:,:,:] = img2_aug.transpose([2,0,1])
        new_batch3[i,:,:,:] = img3_aug.transpose([2,0,1])
    return [new_batch1, new_batch2, new_batch3]

def buffer_augmentation_UnFlow(buff, batchsize):
    # augmentation partically follows UnFlow: Unsupervised Learning of Optical Flow with a Bidirectional Census Loss
    # however, the random scaling and additive Gaussian noise is not used
    for i in range(batchsize):
        # generate random config for a sample
        config_flip = 0
        config_flip_lr = random.randint(0, 1) > 0.5
        config_brightness_changes = np.random.normal(loc = 0, scale = 0.02)
        config_multiplicative_color_changes = np.random.uniform(0.9, 1.1)
        config_contrast = np.random.uniform(-0.3, 0.3)
        config_gamma = np.random.uniform(0.8, 1.3)
        # for Array in buff:
        for k, v in buff.iteritems():
            print k
            
            # img = Array[i,:,:,:].transpose([1,2,0])
            # img_aug = augmentation_LF_unflow(img, config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes,config_contrast, config_gamma)


    N = img_batch1.shape[0]
    new_batch1 = np.zeros(img_batch1.shape, dtype = np.float32)
    new_batch2 = np.zeros(img_batch2.shape, dtype = np.float32)
    new_batch3 = np.zeros(img_batch3.shape, dtype = np.float32)
    for i in range(N):
        img1 = img_batch1[i,:,:,:].transpose([1,2,0])
        img2 = img_batch2[i,:,:,:].transpose([1,2,0])
        img3 = img_batch3[i,:,:,:].transpose([1,2,0])

        # generate random config
        config_flip = 0
        config_flip_lr = random.randint(0, 1) > 0.5
        config_brightness_changes = np.random.normal(loc = 0, scale = 0.02)
        config_multiplicative_color_changes = np.random.uniform(0.9, 1.1)
        config_contrast = np.random.uniform(-0.3, 0.3)
        config_gamma = np.random.uniform(0.8, 1.3)

        # generate img
        img1_aug = augmentation_LF_unflow(img1, config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes,config_contrast, config_gamma)
        img2_aug = augmentation_LF_unflow(img2, config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes,config_contrast, config_gamma)
        img3_aug = augmentation_LF_unflow(img3, config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes,config_contrast, config_gamma)

        # plt.subplot(321)
        # plt.imshow(img1, vmin=-0, vmax=1, interpolation="nearest")
        # plt.subplot(323)
        # plt.imshow(img2, vmin=-0, vmax=1, interpolation="nearest")
        # plt.subplot(325)
        # plt.imshow(img3, vmin=-0, vmax=1, interpolation="nearest")
        # plt.subplot(322)
        # plt.imshow(img1_aug, vmin=-0, vmax=1, interpolation="nearest")
        # plt.subplot(324)
        # plt.imshow(img2_aug, vmin=-0, vmax=1, interpolation="nearest")
        # plt.subplot(326)
        # plt.imshow(img3_aug, vmin=-0, vmax=1, interpolation="nearest")
        # plt.show()

        new_batch1[i,:,:,:] = img1_aug.transpose([2,0,1])
        new_batch2[i,:,:,:] = img2_aug.transpose([2,0,1])
        new_batch3[i,:,:,:] = img3_aug.transpose([2,0,1])

    return [new_batch1, new_batch2, new_batch3]