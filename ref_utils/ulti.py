from six.moves import cPickle
import random
import numpy as np
import cv2
from scipy.signal import convolve2d,correlate2d
import matplotlib.pyplot as plt
import math

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# def psnr_with_mask_bidir(img1, img2, mask):

def psnr_with_mask(img1, img2, mask):
    mask_rep = np.repeat(mask, img1.shape[1], axis = 1)
    mse = np.sum( (img1 - img2) ** 2 * mask_rep ) / np.sum(mask_rep)
    if mse == 0:
        psnr_ = 100
    else:
        PIXEL_MAX = 1.0
        psnr_ = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    mask_ratio = np.mean(mask)
    return psnr_, mask_ratio

def epe(flow1, flow2):
    mse = np.abs(flow1 - flow2)
    return np.mean(mse)

def crop_function(arr, crop_size_H=1, crop_size_W=1):
    W = arr.shape[3]
    W_surround = (W - crop_size_W)/2
    H = arr.shape[2]
    H_surround = (H - crop_size_H)/2
    output = arr[:,:,H_surround:-H_surround,W_surround:-W_surround]
    return output

def my_imshow(image):
    if image.shape[1] == 3:
        plt.imshow(np.squeeze(np.transpose(image, axes=(0,2,3,1))), vmin=-0, vmax=1, interpolation="nearest")
    elif image.shape[1] == 1:
        plt.imshow(np.squeeze(np.transpose(image, axes=(0,2,3,1))), vmin=-0, vmax=1, interpolation="nearest", cmap='Greys')

def grayscale(img):
    dst = np.zeros((img.shape[0],1,img.shape[2],img.shape[3]), dtype=np.float32)
    dst[:,0,:,:] = 0.299 * img[:,0,:,:] + 0.587 * img[:,1,:,:] + 0.114 * img[:,2,:,:]
    dst = np.repeat(dst, 3, axis = 1)
    return dst

def blend(img1,img2,alpha = 0.5):
    # if alpha > 1.5:
    #     alpha = 1.0
    # if alpha < 0.5:
    #     alpha = 0.5
    return img1 * alpha + img2 * (1-alpha)

def normalize_image_LF(img):
    # normalize image such that every pixels follow normal distribution (the range of the original image is [0,1])
    # for ImageNet
    # imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((3,1,1))
    # imagenet_var = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((3,1,1))

    # for LF dataset
    imagenet_mean = np.array([0.179, 0.179, 0.179], dtype=np.float32).reshape((3,1,1))
    imagenet_var = np.array([0.217, 0.217, 0.217], dtype=np.float32).reshape((3,1,1))

    img_new = (img - imagenet_mean) / imagenet_var
    # use alex net normalization
    return img_new

def inverse_normalize_image_LF(img):
    # for ImageNet
    # imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((3,1,1))
    # imagenet_var = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((3,1,1))

    # for LF dataset
    imagenet_mean = np.array([0.179, 0.179, 0.179], dtype=np.float32).reshape((3,1,1))
    imagenet_var = np.array([0.217, 0.217, 0.217], dtype=np.float32).reshape((3,1,1))

    img_new = img*imagenet_var + imagenet_mean
    # use alex net normalization
    return img_new

def upsampling(img=None ,scale = 2):
    img = np.asarray(img*255.0,dtype = np.uint8)
    W = img.shape[3]*scale
    H = img.shape[2]*scale
    img_out = np.zeros((img.shape[0],img.shape[1],H,W),dtype=np.uint8)

    if(img.ndim == 4):
        img_out[:,0,:,:] = cv2.resize(img[0,0,:,:],(W,H),interpolation=cv2.INTER_CUBIC)
        img_out[:,1,:,:] = cv2.resize(img[0,1,:,:],(W,H),interpolation=cv2.INTER_CUBIC)
        img_out[:,2,:,:] = cv2.resize(img[0,2,:,:],(W,H),interpolation=cv2.INTER_CUBIC)
    img_out = np.asarray(img_out,dtype = np.float32) / 255.0
    # print img_out
    return img_out

def load_model(T_param,filename):
    f_l = open(filename, 'rb')
    params_l = cPickle.load(f_l)
    f_l.close()
    for ind, p in enumerate(params_l):
        T_param[ind].set_value(p.get_value())

# def load_params(T_param,filename):
#     f_l = open(filename, 'rb')
#     params_l = cPickle.load(f_l)
#     f_l.close()
#     for ind, p in enumerate(params_l):
#         T_param[ind].set_value(p.get_value())

# def save_params(T_param,filename):
#     f_params = open(filename, 'wb')
#     cPickle.dump(T_param, f_params, protocol=cPickle.HIGHEST_PROTOCOL)
#     f_params.close()

def yes_no_promt(str):
    Join = raw_input(str)
    while not(Join == 'y' or Join =='n'):
        Join = raw_input(str)
    return Join == 'y'

def load_params(T_params,filename):          # save list of numpy array
    f_l = open(filename, 'rb')
    params_l = cPickle.load(f_l)
    f_l.close()
    # print params_l
    # print T_params
    for ind in range(len(params_l)):
        print params_l[ind].shape, T_params[ind].get_value().shape
    print len(T_params), len(params_l)
    for ind in range(len(params_l)):
        T_params[ind].set_value(params_l[ind])
    # for ind, p in enumerate(T_params):
        # p.set_value(params_l[ind])

def save_params(T_params,filename):
    numpy_list = [p.get_value() for p in T_params]
    f_params = open(filename, 'wb')
    cPickle.dump(numpy_list, f_params, protocol=cPickle.HIGHEST_PROTOCOL)
    f_params.close()

def load_update(T_updates,filename):          # save list of numpy array
    f_l = open(filename, 'rb')
    params_l = cPickle.load(f_l)
    f_l.close()
    for ind, p in enumerate(T_updates.keys()):
        p.set_value(params_l[ind])

def save_update(T_updates,filename):
    numpy_list = [p.get_value() for p in T_updates.keys()]
    f_params = open(filename, 'wb')
    cPickle.dump(numpy_list, f_params, protocol=cPickle.HIGHEST_PROTOCOL)
    f_params.close()


def save_snapshot(T_params,T_updates,filename):
    save_params(T_params, filename+'.params')
    save_update(T_updates, filename+'.updates')

def load_snapshot(T_params,T_updates,filename):
    load_params(T_params, filename+'.params')
    load_update(T_updates, filename+'.updates')


def augment_data(input_, mirror_left_right = True, rotate_90_time = 0):
    if mirror_left_right:
        # input_.flip(axis=3)
        input_ = np.flip(input_,axis=3)
    if rotate_90_time!=0:
        # input_.rot90(input_, k=1, axes=(2, 3))
        input_ = np.rot90(input_, k=1, axes=(2, 3))
    return input_

def _load_img(filename):
    img = Image.open(filename)
    img = np.asarray(img, dtype=np.float32)
    img = img / 255.0
    # s = img.shape
    # if (img.ndim==3 and s[2]==3):
    #     y,u,v = _ycc(img[:,:,0],img[:,:,1],img[:,:,2])
    #     img = np.array(y, dtype=np.float32)
    return img

def selective_kernel(N,H,W):
    # compute a 4D kernel which rearanges features
    kernel = np.zeros((N*H*W,N,H,W),dtype = np.float32)
    count = 0;
    for n in range(N):
        for h in range(H):
            for w in range(W):
                kernel[count,n,h,w] = 1.0
                count = count + 1
    return kernel

import matplotlib.pyplot as plt

def bilinear_kernel(size,num_kernels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1.0
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    K = np.asarray((1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor),dtype=np.float32)
    print size, factor,center
    Ks = np.zeros((num_kernels,num_kernels,size,size),dtype=np.float32)
    for i in range(num_kernels):
        Ks[i,i,:,:] = K
    # normalize

    # plt.imshow(K, interpolation="nearest")
    # plt.show()
    # print Ks,np.sum(K)
    Ks=Ks/np.sum(K)
    # print Ks
    # exit()
    return Ks

def gradient_kernel():
    # compute a 4D kernel (4,1,5,5) which compute the gradient and second gradient
    unit_temp = np.asarray([[0,0,0,0,0],
                      [0,0,0,0,0],
                      [0,0,1,0,0],
                      [0,0,0,0,0],
                      [0,0,0,0,0]], dtype = np.float32)
    unit_temp2 = np.asarray([[0,0,0,0,0],
                            [0,1,1,1,0],
                            [0,1,1,1,0],
                            [0,1,1,1,0],
                            [0,0,0,0,0]], dtype = np.float32)/9.0
    first_grad_temp = np.asarray([[0,0,0,0,0],
                                  [0,0,1,0,0],
                                  [0,0,0,0,0],
                                  [0,0,-1,0,0],
                                  [0,0,0,0,0]], dtype = np.float32)
    second_grad_temp = np.asarray([[0,0, 1, 0,0],
                                   [0,0, 0, 0,0],
                                   [0,0,-2, 0,0],
                                   [0,0, 0, 0,0],
                                   [0,0, 1, 0,0]], dtype = np.float32)                   

    kernel = np.zeros((4,1,5,5),dtype = np.float32)
    kernel[0,0,:,:] = first_grad_temp
    kernel[1,0,:,:] = first_grad_temp.T
    kernel[2,0,:,:] = second_grad_temp
    kernel[3,0,:,:] = second_grad_temp.T
    # kernel[0,0,:,:] = unit_temp2
    # kernel[1,0,:,:] = unit_temp2
    # kernel[2,0,:,:] = unit_temp2
    # kernel[3,0,:,:] = unit_temp2
    return kernel

def identity_kernel(size,num_kernels):
    # use identity kernel to initialize the delated network (suggested by the paper)
    Ks = np.zeros((num_kernels,num_kernels,size,size),dtype=np.float32)
    for i in range(num_kernels):
        Ks[i,i,size/2,size/2] = 1.0
    return Ks

def retify_image(img):
    Img_res = img
    Img_res = np.select([Img_res>1.0,Img_res<=1.0],[np.ones_like(Img_res), Img_res])
    Img_res = np.select([Img_res<0.0,Img_res>=0.0],[np.zeros_like(Img_res), Img_res])
    return Img_res

def compute_gradient_feature(patch,conv_mode = 'valid'):
    # compute a 4D kernel (4,1,5,5) which compute the gradient and second gradient
    unit_temp = np.asarray([[0,0,0,0,0],
                      [0,0,0,0,0],
                      [0,0,1,0,0],
                      [0,0,0,0,0],
                      [0,0,0,0,0]], dtype = np.float32)
    unit_temp2 = np.asarray([[0,0,0,0,0],
                            [0,1,1,1,0],
                            [0,1,1,1,0],
                            [0,1,1,1,0],
                            [0,0,0,0,0]], dtype = np.float32)/9.0
    first_grad_temp = np.asarray([[0,0,0,0,0],
                                  [0,0,1,0,0],
                                  [0,0,0,0,0],
                                  [0,0,-1,0,0],
                                  [0,0,0,0,0]], dtype = np.float32)
    second_grad_temp = np.asarray([[0,0, 1, 0,0],
                                   [0,0, 0, 0,0],
                                   [0,0,-2, 0,0],
                                   [0,0, 0, 0,0],
                                   [0,0, 1, 0,0]], dtype = np.float32)       

    mode = conv_mode #'same' #'valid'
    feat1 = correlate2d(patch,first_grad_temp, mode = mode)
    feat2 = correlate2d(patch,first_grad_temp.T, mode = mode)
    feat3 = correlate2d(patch,second_grad_temp, mode = mode)
    feat4 = correlate2d(patch,second_grad_temp.T, mode = mode)
    return feat1, feat2, feat3, feat4
    
    
