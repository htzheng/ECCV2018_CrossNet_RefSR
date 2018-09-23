import sys
import matplotlib.pyplot as plt

import numpy as np

import h5py
from ulti import blend, grayscale, my_imshow, psnr

import time
import cv2

class Dataset(object):
    def __init__(self, filename = None, scale = 8, MDSR_as_bilinear = False):
        self.scale = scale
        # load a dictionary that contains all arrays
        self.arrays = dict()
        self.loadArrays(filename, scale = scale, MDSR_as_bilinear = MDSR_as_bilinear) 
        # set iterator counter
        self.rng_index = np.random.RandomState(100)           # random generator
        self.rng_augmentation = np.random.RandomState(200)
        self.rng_viewpoint = np.random.RandomState(300)
        self.rng_viewpoint_augmentation = np.random.RandomState(400)
        self.idx_counter = 0
        return

    def nextBatch(self):
        return

    def append_list(self,dictionary, str_list):
        #  append_list(dictionary, str_list):    generate a tuple of input data according to a string list (for feeding the tuple to train/test function)
        l = list()
        for i in range(len(str_list)):
            l.append(dictionary[str_list[i]])
        return l
        # return tuple(l)

    def loadArrays(self, filename, scale):
        #  loadArrays(filename):    should save data to self.dataset, and image size to self.W etc.
        return  

    def get_image(self, image_index, y1, x1, crop_shape = (320,512) ):
        H, W = crop_shape
        out_list = dict()
        out_list['HR'] = np.asarray(self.arrays['img_HR'][image_index:image_index+1,y1,x1,:,0:H,0:W], dtype = np.float32) / 255.0 
        out_list['LR_upsample'] = np.asarray(self.arrays['img_LR_upsample'] [image_index:image_index+1,y1,x1,:,0:H,0:W], dtype = np.float32) / 255.0 
        out_list['MDSR'] = np.asarray(self.arrays['img_MDSR'][image_index:image_index+1,y1,x1,:,0:H,0:W], dtype = np.float32) / 255.0 
        out_list['LR'] = np.asarray(self.arrays['img_LR'] [image_index:image_index+1,y1,x1,:,0:H/self.scale,0:W/self.scale], dtype = np.float32) / 255.0 
        return out_list

    def get_images(self, image_index, y1, x1, y2, x2):
        return self.get_image(image_index, y1, x1), self.get_image(image_index, y2, x2)

    def genIndex_list(self, batch_size, random, index_inc = True):
        #  __genIndex_list(batch_size, random, index_inc = True):   generate list of index to be sample with
        #       batch_size:          number of sample to be generated
        #       random:             set True if want to randomly generate sample
        #       index_inc:           increase index (affective if shuffle=False)

        idx_list = []
        for i in range(batch_size):
            if random:
                idx_list.append(self.rng_index.randint(0,self.size_N-1))
            else:
                idx_list.append((self.idx_counter+i)% self.size_N)
        if (index_inc):
            self.idx_counter = (self.idx_counter+batch_size) % self.size_N
        return idx_list

    def augmentation_array_config(self):
        #  __augmentation_array_config():   generate flipping, color based data_augmentation config
        config_flip = 0
        config_flip_lr = self.rng_augmentation.randint(0, 2) > 0.5
        config_brightness_changes = self.rng_augmentation.normal(loc = 0, scale = 0.02)
        config_multiplicative_color_changes = self.rng_augmentation.uniform(0.9, 1.1)
        config_contrast = self.rng_augmentation.uniform(-0.3, 0.3)
        config_gamma = self.rng_augmentation.uniform(0.8, 1.3)
        config = [config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes, config_contrast, config_gamma]
        # print 'augmentation config: ', config
        return config

    def augmentation_array(self, img, config):
        #  __augmentation_array( img, config):  perform flipping, color based data augmentation
        config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes, config_contrast, config_gamma = config
        H = img.shape[0]
        W = img.shape[1]
        img_aug = img
        
        time_total = time.time()
        ## flipping

        time_flip = time.time()
        if(config_flip==1):
            img_aug = img_aug[::-1,:]
        if(config_flip_lr==1):
            img_aug = img_aug[:,::-1]
        time_flip = time.time() - time_flip

        time_b = time.time()
        ## brightness changes
        img_aug = img_aug + config_brightness_changes
        ## multiplicative color changes
        img_aug = img_aug * config_multiplicative_color_changes
        time_b = time.time() - time_b

        # ## Contrast
        time_grey = time.time()
        gs_2 = grayscale(img_aug)
        img_aug = blend(gs_2, img_aug, alpha = config_contrast)
        time_grey = time.time() - time_grey
        ## clip
        img_aug = np.clip(img_aug,0.0,1.0)
        ## gama
        # time_gamma = time.time()
        # img_aug = np.power(img_aug, config_gamma, dtype=np.float32)
        # time_gamma = time.time() - time_gamma
        # time_total = time.time() - time_total

        # print 't_total: ', time_total, 't_flip',time_flip,'time_b',time_b,'time_grey', time_grey, 't_gamma: ', time_gamma
        return img_aug


    # def pycv_power(self, arr, exponent):
    #     """Raise the elements of a floating point matrix to a power. 
    #     It is 3-4 times faster than numpy's built-in power function/operator."""
    #     # if arr.dtype not in [numpy.float32, numpy.float64]:
    #     #     arr = arr.astype('f')
    #     res = np.empty_like(arr)
    #     if arr.flags['C_CONTIGUOUS'] == False:
    #         arr = np.ascontiguousarray(arr)        
    #     cv2.pow(cv2.fromarray(arr), float(exponent), cv2.fromarray(res))
    #     return res 