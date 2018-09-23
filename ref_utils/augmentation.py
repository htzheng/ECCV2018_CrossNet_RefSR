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

def augmentation(img,config_flip, config_scale, config_rotation, config_brightness_alpha, config_saturation_alpha, config_contrast_alpha, h_init = None, w_init = None):
    plt.subplot(211)
    plt.imshow(img, vmin=-0, vmax=1, interpolation="nearest")

    H = img.shape[0]
    W = img.shape[1]
    img_aug = img
    ## flipping
    if(config_flip==0):
        img_aug = img_aug[:,::-1]

    # ## scaling
    H_new = int(H*config_scale)
    W_new = int(W*config_scale)
    img_aug = cv2.resize(img_aug,(W_new,H_new),interpolation=cv2.INTER_CUBIC)

    ## rotation 
    M = cv2.getRotationMatrix2D((W_new/2,H_new/2), config_rotation, 1.0) #  config_scale
    img_aug = cv2.warpAffine(img_aug, M, (W_new,H_new))
    H_new = img_aug.shape[0]
    W_new = img_aug.shape[1]
    # print H_new, H, float(H_new)/float(H), config_scale

    ## cropping
    if h_init == None:
        h_init = random.randint(0, H_new - H)
    if w_init == None:
        w_init = random.randint(0, W_new - W)
    img_aug = img_aug[h_init:h_init+H, w_init:w_init+W]

    ## change format
    img_aug = img_aug.astype('float32')/255.0

    ## additive noise 
    # img_aug = img_aug + np.random.normal(0, 0.1, size = (H, W, 3))

    ## Brightness
    black = np.zeros(img_aug.shape)
    img_aug = blend(img_aug, black, alpha = config_brightness_alpha)

    ## Saturation
    gs = grayscale(img_aug)
    img_aug = blend(img_aug, gs, alpha = config_saturation_alpha)

    ## Contrast
    m = np.zeros(img_aug.shape)
    m[:,:,0] = np.mean(img_aug[:,:,0])
    m[:,:,1] = np.mean(img_aug[:,:,1])
    m[:,:,2] = np.mean(img_aug[:,:,2])
    img_aug = blend(img_aug, m, alpha = config_contrast_alpha)

    ## clip
    img_aug = np.clip(img_aug,0.0,1.0)

    # plt.subplot(212)
    # plt.imshow(img_aug, vmin=-0, vmax=1, interpolation="nearest")
    # plt.show()

    return [img_aug,h_init,w_init]

def augmentation_LF(img,config_flip,config_flip_lr, config_scale, config_rotation, config_brightness, config_saturation, config_contrast, config_brightness_alpha, config_saturation_alpha, config_contrast_alpha, h_init = None, w_init = None):
    H = img.shape[0]
    W = img.shape[1]
    img_aug = img
    ## flipping
    if(config_flip==1):
        img_aug = img_aug[::-1,:]
    if(config_flip_lr==1):
        img_aug = img_aug[:,::-1]

    # ## scaling
    # H_new = int(H*config_scale)
    # W_new = int(W*config_scale)
    # img_aug = cv2.resize(img_aug,(W_new,H_new),interpolation=cv2.INTER_CUBIC)

    ## rotation 
    # M = cv2.getRotationMatrix2D((W_new/2,H_new/2), config_rotation, 1.0) #  config_scale
    # img_aug = cv2.warpAffine(img_aug, M, (W_new,H_new))
    # H_new = img_aug.shape[0]
    # W_new = img_aug.shape[1]

    ## cropping
    # if h_init == None:
    #     h_init = random.randint(0, H_new - H)
    # if w_init == None:
    #     w_init = random.randint(0, W_new - W)
    # img_aug = img_aug[h_init:h_init+H, w_init:w_init+W]

    ## change format
    # img_aug = img_aug.astype('float32')/255.0

    ## additive noise 
    # img_aug = img_aug + np.random.normal(0, 0.1, size = (H, W, 3))

    ## Brightness
    if config_brightness:
        black = np.zeros(img_aug.shape)
        img_aug = blend(img_aug, black, alpha = config_brightness_alpha)

    ## Saturation
    if config_saturation:
        gs = grayscale(img_aug)
        img_aug = blend(img_aug, gs, alpha = config_saturation_alpha)

    ## Contrast
    if config_contrast:
        gs_2 = grayscale(img_aug)
        gs_2[:,:,:] = np.mean(gs_2)
        # m = np.zeros(img_aug.shape)
        # m[:,:,0] = np.mean(img_aug[:,:,0])
        # m[:,:,1] = np.mean(img_aug[:,:,1])
        # m[:,:,2] = np.mean(img_aug[:,:,2])
        img_aug = blend(img_aug, gs_2, alpha = config_contrast_alpha)

    ## clip
    img_aug = np.clip(img_aug,0.0,1.0)

    # plt.subplot(211)
    # plt.imshow(img, vmin=-0, vmax=1, interpolation="nearest")
    # plt.subplot(212)
    # plt.imshow(img_aug, vmin=-0, vmax=1, interpolation="nearest")
    # plt.show()

    return [img_aug,h_init,w_init]

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

def batch_augmentation(img_batch1, img_batch2):
    N = img_batch1.shape[0]
    new_batch1 = np.zeros(img_batch1.shape, dtype = np.float32)
    new_batch2 = np.zeros(img_batch2.shape, dtype = np.float32)

    for i in range(N):
        img1 = img_batch1[i,:,:,:].transpose([1,2,0])
        img2 = img_batch2[i,:,:,:].transpose([1,2,0])

        # generate random config
        config_flip = random.randint(0, 1) > 0.5
        config_scale = random.randint(1000, 2000) / 1000.0
        config_rotation = random.randint(-17, 17)
        var = 0.4
        config_brightness_alpha = np.random.normal(loc = 1.0, scale = var)  #np.random.uniform(-var, var)  np.random.normal(var) np.random.normal(0.4)
        config_saturation_alpha = np.random.normal(loc = 1.0, scale = var)  #np.random.uniform(-var, var)
        config_contrast_alpha   = np.random.normal(loc = 1.0, scale = var)  #np.random.uniform(-var, var)

        # generate img
        img1_aug,h_init,w_init = augmentation(img1,config_flip, config_scale, config_rotation,config_brightness_alpha,config_saturation_alpha,config_contrast_alpha, h_init = None  , w_init = None)
        img2_aug,_,_           = augmentation(img2,config_flip, config_scale, config_rotation,config_brightness_alpha,config_saturation_alpha,config_contrast_alpha, h_init = h_init, w_init = w_init)
                    #  img1.astype(np.float32)/255.0   #
        # img1_aug = img1
        # img2_aug = img2
        new_batch1[i,:,:,:] = img1_aug.transpose([2,0,1])
        new_batch2[i,:,:,:] = img2_aug.transpose([2,0,1])
    return [new_batch1, new_batch2]

def batch_augmentation3_LF(img_batch1, img_batch2, img_batch3):
    N = img_batch1.shape[0]
    new_batch1 = np.zeros(img_batch1.shape, dtype = np.float32)
    new_batch2 = np.zeros(img_batch2.shape, dtype = np.float32)
    new_batch3 = np.zeros(img_batch3.shape, dtype = np.float32)
    for i in range(N):
        img1 = img_batch1[i,:,:,:].transpose([1,2,0])
        img2 = img_batch2[i,:,:,:].transpose([1,2,0])
        img3 = img_batch3[i,:,:,:].transpose([1,2,0])

        # generate random config
        config_flip = 0   # random.randint(0, 1) > 0.5
        config_flip_lr = random.randint(0, 1) > 0.5
        config_scale = random.randint(1000, 2000) / 1000.0
        config_rotation = random.randint(-17, 17)
        var = 0.20 #0.4
        config_brightness = bool(random.getrandbits(1))
        config_saturation = bool(random.getrandbits(1))
        config_contrast = bool(random.getrandbits(1))
        config_brightness_alpha = np.random.normal(loc = 1.0, scale = var)  #np.random.uniform(-var, var)  np.random.normal(var) np.random.normal(0.4)
        config_saturation_alpha = np.random.normal(loc = 1.0, scale = var)  #np.random.uniform(-var, var)
        config_contrast_alpha   = np.random.uniform(-0.3, 0.3)  #np.random.normal(loc = 1.0, scale = var)  #np.random.uniform(-var, var)

        # generate img
        img1_aug,h_init,w_init = augmentation_LF(img1,config_flip, config_flip_lr, config_scale, config_rotation, config_brightness,config_saturation,config_contrast, config_brightness_alpha,config_saturation_alpha,config_contrast_alpha, h_init = None  , w_init = None)
        img2_aug,_,_           = augmentation_LF(img2,config_flip, config_flip_lr, config_scale, config_rotation, config_brightness,config_saturation,config_contrast, config_brightness_alpha,config_saturation_alpha,config_contrast_alpha, h_init = h_init, w_init = w_init)
        img3_aug,_,_           = augmentation_LF(img3,config_flip, config_flip_lr, config_scale, config_rotation, config_brightness,config_saturation,config_contrast, config_brightness_alpha,config_saturation_alpha,config_contrast_alpha, h_init = h_init, w_init = w_init)

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