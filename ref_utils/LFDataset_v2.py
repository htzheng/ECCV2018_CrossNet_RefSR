import sys
import matplotlib.pyplot as plt

import numpy as np

import h5py
from ulti import blend, grayscale, my_imshow, psnr
from Dataset import Dataset
import time


import scipy
import scipy.io
import scipy.misc
import scipy.io as sio
import os

# dataset_test = ChairsSDHomDataset(filename = '/fileserver/haitian/dataset/ChairsSDHom/test_bicubic.h5')
# for i in range(0,100):
#     print 'next batch'
#     dataset_test.nextBatch(batchsize=4, shuffle=False, view_mode = 'Random', residue = False, augmentation = False, index_inc = False)
#     dataset_test.nextBatch(batchsize=4, shuffle=False, view_mode = 'Random', residue = False, augmentation = False, index_inc = True)


# dataset_train = Dataset(filename = '../HT_sr/flowdata/LF_video_Dataset/train_bicubic.h5')
# dataset_train.nextBatch(batchsize=8,shuffle=True,view_mode = 'Random',residue = True, augmentation = True)
# dataset_train.nextBatch(batchsize=8,shuffle=True,view_mode = 'Random',residue = True, augmentation = True)

# dataset_train.nextBatch(batchsize=8,shuffle=False,view_mode = 'Random',residue = False, augmentation = True)
# dataset_test = Dataset(filename = '../HT_sr/flowdata/LF_video_Dataset/test_bicubic.h5')


class LFDataset(Dataset):
    def loadArrays(self, filename):
        f = h5py.File(filename,'r')
        self.arrays['img_HR'] = f.get('/img_HR')
        self.arrays['img_LR'] = f.get('/img_LR')
        self.arrays['img_MDSR'] = f.get('/MDSR')    #  f.get('/SRResNet_NN')
        self.arrays['img_LR_antialiasing'] = f.get('/img_LR_antialiasing')

        print 'loading dataset: ', filename
        print 'img_HR             ', self.arrays['img_HR'].shape
        print 'img_LR_antialiasing', self.arrays['img_LR'].shape
        print 'img_LR             ', self.arrays['img_MDSR'].shape

        self.size_N = self.arrays['img_HR'].shape[0]
        self.size_C = self.arrays['img_HR'].shape[3]
        self.size_H = self.arrays['img_HR'].shape[4]
        self.size_W = self.arrays['img_HR'].shape[5]

    def __genViewPosition(self, view_mode, specified_view = None):
        if (view_mode == 'Random'):
            rnd_y = self.rng_viewpoint.randint(0,7)
            rnd_x = self.rng_viewpoint.randint(0,7)
            while True:
                rnd_y_ref = self.rng_viewpoint.randint(0,7)
                rnd_x_ref = self.rng_viewpoint.randint(0,7)
                if (rnd_y_ref!=rnd_y) or (rnd_x_ref!=rnd_x):
                    break
        elif (view_mode == 'Random_FixOffset'):        #  disparicy = (-3,-3)
            rnd_y = self.rng_viewpoint.randint(6,7)
            rnd_x = self.rng_viewpoint.randint(6,7)
            rnd_y_ref = rnd_y-6
            rnd_x_ref = rnd_x-6
        elif (view_mode == 'Fixed'):
            rnd_y = 0
            rnd_x = 0
            rnd_y_ref = 3
            rnd_x_ref = 3
        elif (view_mode == 'Fixed-inv'):
            rnd_y = 3
            rnd_x = 3
            rnd_y_ref = 0
            rnd_x_ref = 0
        elif (view_mode == 'Fixed-inv-large'):
            rnd_y = 7
            rnd_x = 7
            rnd_y_ref = 0
            rnd_x_ref = 0
        elif (view_mode == 'specified'):
            rnd_y, rnd_x, rnd_y_ref, rnd_x_ref = tuple(specified_view)
        return rnd_y,rnd_x,rnd_y_ref,rnd_x_ref

    def debugDataset(self):           # test the average psnr of 'random' view and 'Fixed' view
        sum_PSNR_fixed = 0
        sum_PSNR_random = 0
        sum_distance_random = 0
        import math
        for idx_img in range(268):
            idx_y, idx_x, idx_y_ref, idx_x_ref = self.__genViewPosition('Random')
            buffer_HR = np.asarray(self.array_dict['img_HR'][idx_img,idx_y,idx_x,:,:,:], dtype = np.float32) / 255.0 
            buffer_REF = np.asarray(self.array_dict['img_HR'][idx_img,idx_y_ref,idx_x_ref,:,:,:], dtype = np.float32) / 255.0 
            sum_PSNR_fixed += psnr(buffer_HR, buffer_REF)
            sum_distance_random += math.sqrt( (idx_y-idx_y_ref)*(idx_y-idx_y_ref) + (idx_x-idx_x_ref)*(idx_x-idx_x_ref) )

            idx_y, idx_x, idx_y_ref, idx_x_ref = self.__genViewPosition('Fixed')
            buffer_HR = np.asarray(self.array_dict['img_HR'][idx_img,idx_y,idx_x,:,:,:], dtype = np.float32) / 255.0 
            buffer_REF = np.asarray(self.array_dict['img_HR'][idx_img,idx_y_ref,idx_x_ref,:,:,:], dtype = np.float32) / 255.0 
            sum_PSNR_random += psnr(buffer_HR, buffer_REF)

            print 'psnr:     ',sum_PSNR_fixed/(idx_img+1), sum_PSNR_random/(idx_img+1)
            print 'distance: ', 4.2426, sum_distance_random/(idx_img+1)
    

    def nextBatch_new(self, batchsize = 8, shuffle = False, view_mode = 'Random', augmentation = False, index_inc = True, crop_shape = None, SR=True, Dual = False):
        # nextBatch_new(batchsize = 8, shuffle = False, view_mode = 'Random', augmentation = False, index_inc = True)
        #       generate a dictionary that contains HR, LR and SR images of two views
        buff = dict()
        # init 
        input_img1_LR = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
        input_img2_LR = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
        input_img1_HR = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
        input_img2_HR = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
        if SR:
            input_img1_SR = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
            if Dual:
                input_img2_SR = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)

        t_read = time.time()

        idx_list = self.genIndex_list(batchsize, shuffle, index_inc = index_inc)
        for k in range(batchsize):
            # generate img number
            idx_img = idx_list[k]

            # generate view position
            y1, x1, y2, x2 = self.__genViewPosition(view_mode)

            # LR
            input_img1_LR[k,:,:,:] = np.asarray(self.arrays['img_LR'][idx_img,y1,x1,:,:,:], dtype = np.float32) / 255.0 
            if Dual:
                input_img2_LR[k,:,:,:] = np.asarray(self.arrays['img_LR'][idx_img,y2,x2,:,:,:], dtype = np.float32) / 255.0 
            # HR
            input_img1_HR[k,:,:,:] = np.asarray(self.arrays['img_HR'][idx_img,y1,x1,:,:,:], dtype = np.float32) / 255.0 
            input_img2_HR[k,:,:,:] = np.asarray(self.arrays['img_HR'][idx_img,y2,x2,:,:,:], dtype = np.float32) / 255.0 
            # SR
            if SR:
                input_img1_SR[k,:,:,:] = np.asarray(self.arrays['img_MDSR'][idx_img,y1,x1,:,:,:], dtype = np.float32) / 255.0 
                if Dual:
                    input_img2_SR[k,:,:,:] = np.asarray(self.arrays['img_MDSR'][idx_img,y2,x2,:,:,:], dtype = np.float32) / 255.0 

        t_aug = time.time()

        # data augmentation
        if augmentation:
            augmentation_config = self.augmentation_array_config()
            input_img1_LR = self.augmentation_array(input_img1_LR, augmentation_config)
            if Dual:
                input_img2_LR = self.augmentation_array(input_img2_LR, augmentation_config)
            input_img1_HR = self.augmentation_array(input_img1_HR, augmentation_config) 
            input_img2_HR = self.augmentation_array(input_img2_HR, augmentation_config)
            if SR:
                input_img1_SR = self.augmentation_array(input_img1_SR, augmentation_config)
                if Dual:
                    input_img2_SR = self.augmentation_array(input_img2_SR, augmentation_config)

        t_crop = time.time()
        
        # crop image
        if not crop_shape is None:
            input_img1_LR = input_img1_LR[:,:,0:crop_shape[0],0:crop_shape[1]]
            if Dual:
                input_img2_LR = input_img2_LR[:,:,0:crop_shape[0],0:crop_shape[1]]
            input_img1_HR = input_img1_HR[:,:,0:crop_shape[0],0:crop_shape[1]]
            input_img2_HR = input_img2_HR[:,:,0:crop_shape[0],0:crop_shape[1]]
            if SR:
                input_img1_SR = input_img1_SR[:,:,0:crop_shape[0],0:crop_shape[1]]
                if Dual:
                    input_img2_SR = input_img2_SR[:,:,0:crop_shape[0],0:crop_shape[1]]
        t_end = time.time()
        # print 'g_read time: ', t_aug - t_read, 'g_aug time: ', t_crop - t_aug, 'g_aug time: ', t_end - t_crop

        # pack buffer
        buff['input_img1_LR'] = input_img1_LR
        if Dual:
            buff['input_img2_LR'] = input_img2_LR
        buff['input_img1_HR'] = input_img1_HR
        buff['input_img2_HR'] = input_img2_HR
        if SR:
            buff['input_img1_SR'] = input_img1_SR
            if Dual:
                buff['input_img2_SR'] = input_img2_SR
        return buff

    def generate_compare_experiment_data(self, save_path):
        crop_shape = [320,512]
        # init 
        input_img1_LR = np.zeros([3,self.size_H,self.size_W], dtype = np.float32)
        input_img1_HR = np.zeros([3,self.size_H,self.size_W], dtype = np.float32)
        input_img2_HR = np.zeros([3,self.size_H,self.size_W], dtype = np.float32)
        input_img1_SR = np.zeros([3,self.size_H,self.size_W], dtype = np.float32)

        for v in range(1,8):
            folder = save_path+'/LR('+str(v)+','+str(v)+')-REF(0,0)'
            print folder
            if not os.path.exists(folder):
                os.mkdir(folder)
            if not os.path.exists(folder+'/LR/'):
                os.mkdir(folder+'/LR/')
            if not os.path.exists(folder+'/GT/'):
                os.mkdir(folder+'/GT/')
            if not os.path.exists(folder+'/REF/'):
                os.mkdir(folder+'/REF/')
            if not os.path.exists(folder+'/MDSR/'):
                os.mkdir(folder+'/MDSR/')
            if not os.path.exists(folder+'/LR_bicubic/'):
                os.mkdir(folder+'/LR_bicubic/')

            for idx_img in range(self.size_N):
                y1, x1, y2, x2 = (v, v, 0, 0)
                input_img1_LR_bicubic = np.asarray(self.arrays['img_LR'][idx_img,y1,x1,:,0:crop_shape[0],0:crop_shape[1]], dtype = np.float32) / 255.0
                input_img1_LR = np.asarray(self.arrays['img_LR_antialiasing'][idx_img,y1,x1,:,0:crop_shape[0],0:crop_shape[1]], dtype = np.float32) / 255.0
                # input_img1_HR = np.asarray(self.arrays['img_HR'][idx_img,y1,x1,:,0:crop_shape[0],0:crop_shape[1]], dtype = np.float32) / 255.0 
                # input_img2_HR = np.asarray(self.arrays['img_HR'][idx_img,y2,x2,:,0:crop_shape[0],0:crop_shape[1]], dtype = np.float32) / 255.0 
                # input_img1_SR = np.asarray(self.arrays['img_MDSR'][idx_img,y1,x1,:,0:crop_shape[0],0:crop_shape[1]], dtype = np.float32) / 255.0

                scipy.misc.toimage(np.squeeze(np.transpose(input_img1_LR_bicubic,axes=(1,2,0))), cmin=0.0, cmax=1.0).save(folder+'/LR_bicubic/'+str(idx_img)+'.png')
                scipy.misc.toimage(np.squeeze(np.transpose(input_img1_LR,axes=(1,2,0))), cmin=0.0, cmax=1.0).save(folder+'/LR/'+str(idx_img)+'.png')
                # scipy.misc.toimage(np.squeeze(np.transpose(input_img1_HR,axes=(1,2,0))), cmin=0.0, cmax=1.0).save(folder+'/GT/'+str(idx_img)+'.png')
                # scipy.misc.toimage(np.squeeze(np.transpose(input_img2_HR,axes=(1,2,0))), cmin=0.0, cmax=1.0).save(folder+'/REF/'+str(idx_img)+'.png')
                # scipy.misc.toimage(np.squeeze(np.transpose(input_img1_SR,axes=(1,2,0))), cmin=0.0, cmax=1.0).save(folder+'/MDSR/'+str(idx_img)+'.png')

    def generate_MDSR_input_data(self, save_path):
        crop_shape = [320,512]
        # init 
        input_img1_LR = np.zeros([3,self.size_H,self.size_W], dtype = np.float32)

        for idx_img in range(self.size_N):
            for idx_y in range(8):
                for idx_x in range(8):
                    input_img1_LR = np.asarray(dataset['img_LR'][idx_img,idx_y,idx_x,:,:,:], dtype = np.float32) / 255.0 

                    