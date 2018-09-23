import sys
import matplotlib.pyplot as plt

import numpy as np

import h5py
from ulti import blend, grayscale, my_imshow, psnr
from Dataset import Dataset

class ChairsSDHomDataset(object):
    def __init__(self, filename = None):
        self.loadArrays(filename) 
        # set iterator counter
        self.rng_index = np.random.RandomState(100)           # random generator
        self.rng_augmentation = np.random.RandomState(200)
        self.idx_counter = 0
        return
        
    def loadArrays(self, filename):
        f = h5py.File(filename,'r')
        img_img0 = f.get('/img0')
        img_img0_LR_upsample = f.get('/img0_LR_upsample')
        img_img1 = f.get('/img1')
        img_flow = f.get('/flow')

        print 'loading training data'
        print 'img_HR             ', img_img0.shape
        print 'img_flow             ', img_flow.shape

        dataset = dict()
        dataset['img_img0'] = img_img0
        dataset['img_img0_LR_upsample'] = img_img0_LR_upsample
        dataset['img_img1'] = img_img1
        dataset['img_flow'] = img_flow

        self.array_dict = dataset
        self.size_N = img_img0.shape[0]
        self.size_C = img_img0.shape[1]
        self.size_H = img_img0.shape[2]
        self.size_W = img_img0.shape[3]
        self.size_flow_C = img_flow.shape[1]
        return 

    def nextBatch(self, batchsize = 8, view_mode = 0, shuffle = False, residue = False, augmentation = False, index_inc = True):
        buff = dict()
        # init 
        buffer_img0_LR_upsample = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
        buffer_img1 =             np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
        buffer_flow =             np.zeros([batchsize,2,self.size_H,self.size_W], dtype = np.float32)

        if residue:
            buffer_SRResNet_NN_LR =  np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
            buffer_SRResNet_NN_REF = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)

        # generate img number
        idx_img_list = self.__genIndex_list(batchsize, shuffle, index_inc = index_inc)

        for k in range(batchsize):
            idx_img = idx_img_list[k]
            # LR
            buffer_img0_LR_upsample[k,:,:,:] = np.asarray(self.array_dict['img_img0_LR_upsample'][idx_img,:,:,:], dtype = np.float32) / 255.0 
            # HR
            buffer_img1[k,:,:,:] = np.asarray(self.array_dict['img_img1'][idx_img,:,:,:], dtype = np.float32) / 255.0 
            # flow
            buffer_flow[k,:,:,:] = np.asarray(self.array_dict['img_flow'][idx_img,:,:,:], dtype = np.float32)
            if residue:
                buffer_SRResNet_NN_LR[k,:,:,:] = np.asarray(self.array_dict['img_SRResNet_NN'][idx_img,:,:,:], dtype = np.float32) / 255.0 
                buffer_SRResNet_NN_REF[k,:,:,:] = np.asarray(self.array_dict['img_SRResNet_NN'][idx_img,:,:,:], dtype = np.float32) / 255.0 

        # data augmentation
        if augmentation:
            augmentation_config = self.__augmentation_array_config()
            buffer_img0_LR_upsample = self.__augmentation_array(buffer_img0_LR_upsample, augmentation_config)
            buffer_img1 = self.__augmentation_array(buffer_img1, augmentation_config)
            if residue:
                buffer_SRResNet_NN_LR  = self.__augmentation_array(buffer_SRResNet_NN_LR, augmentation_config)
                buffer_SRResNet_NN_REF = self.__augmentation_array(buffer_SRResNet_NN_REF, augmentation_config)

        ## show
        # print '.'
        # print buffer_img0_LR_upsample.shape
        # for i in range(batchsize):
        #     plt.subplot(221)
        #     my_imshow(buffer_img0_LR_upsample[np.newaxis,i,:,:,:])
        #     plt.subplot(222)
        #     my_imshow(buffer_img1[np.newaxis,i,:,:,:])
        #     plt.subplot(223)
        #     plt.imshow(buffer_flow[i,0,:,:],vmin=-5, vmax=5)   # x flow
        #     plt.subplot(224)
        #     plt.imshow(buffer_flow[i,1,:,:],vmin=-5, vmax=5)   # y flow
        #     plt.show()

        # pack buffer
        buff['input_LR'] = buffer_img0_LR_upsample      #LR image
        buff['input_REF'] = buffer_img1
        buff['input_flow'] = buffer_flow

        if residue:
            buff['input_SRResNet_NN_LR' ] = buffer_SRResNet_NN_LR
            buff['input_SRResNet_NN_REF'] = buffer_SRResNet_NN_REF

        return buff